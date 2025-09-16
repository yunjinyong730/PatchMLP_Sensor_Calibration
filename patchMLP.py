from typing import List, Optional
import tensorflow as tf
from keras import layers, Model

# --------------------
# Utilities
# --------------------

def gelu(x):
    return tf.keras.activations.gelu(x, approximate=True)


class LayerNorm(layers.Layer):
    def __init__(self, axis=-1, eps=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.eps = eps

    def build(self, input_shape):
        dim = input_shape[self.axis]
        self.gamma = self.add_weight("gamma", shape=(dim,), initializer="ones", trainable=True)
        self.beta = self.add_weight("beta", shape=(dim,), initializer="zeros", trainable=True)

    def call(self, x):
        mean = tf.reduce_mean(x, axis=self.axis, keepdims=True)
        var = tf.reduce_mean(tf.square(x - mean), axis=self.axis, keepdims=True)
        x_hat = (x - mean) / tf.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta


# ----------------------------------------
# 1) Multi-Scale Patch Embedding (MPE)
# ----------------------------------------
class MultiScalePatchEmbedding(layers.Layer):

    def __init__(
        self,
        patch_sizes: List[int],
        d_each: int,
        d_fuse: Optional[int] = None,
        interpolation: str = "linear",
        flatten_tokens: bool = True,
        d_model: Optional[int] = None,
        name: str = "mpe",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        assert len(patch_sizes) >= 1
        self.patch_sizes = list(patch_sizes)
        self.d_each = d_each
        self.d_fuse = d_fuse or (len(patch_sizes) * d_each)
        self.interpolation = interpolation
        self.flatten_tokens = flatten_tokens
        self.d_model = d_model

        self.proj_per_scale = [layers.Dense(d_each, name=f"{name}_proj_s{i}") for i, _ in enumerate(patch_sizes)]
        self.fuse = layers.Dense(self.d_fuse, name=f"{name}_fuse")
        if self.flatten_tokens:
            assert self.d_model is not None, "d_model must be provided when flatten_tokens=True"
            self.flat_proj = layers.Dense(self.d_model, name=f"{name}_flat_proj")

    def _linear_resample(self, emb, max_tokens):
        # emb: [B, N, M, D] → [B, max_tokens, M, D]
        B = tf.shape(emb)[0]
        N = tf.shape(emb)[1]
        M = tf.shape(emb)[2]
        D = tf.shape(emb)[3]
        if emb.shape[1] == 1:
            return tf.repeat(emb, repeats=max_tokens, axis=1)

        pos_src = tf.linspace(0.0, tf.cast(N - 1, tf.float32), N)
        pos_tgt = tf.linspace(0.0, tf.cast(N - 1, tf.float32), max_tokens)
        i0 = tf.cast(tf.floor(pos_tgt), tf.int32)
        i1 = tf.minimum(i0 + 1, N - 1)
        w1 = pos_tgt - tf.cast(i0, tf.float32)
        w0 = 1.0 - w1

        # reshape for gather: [B*M*D, N]
        BMD = B * M * D
        e2 = tf.reshape(tf.transpose(emb, [0, 2, 3, 1]), [BMD, N])
        g0 = tf.gather(e2, i0, axis=1)
        g1 = tf.gather(e2, i1, axis=1)
        e_lin = w0[None, :] * g0 + w1[None, :] * g1  # [BMD, max_tokens]
        out = tf.transpose(tf.reshape(e_lin, [B, M, D, max_tokens]), [0, 3, 1, 2])
        return out

    def call(self, x):  # x: [B, L, M]
        B = tf.shape(x)[0]
        L = tf.shape(x)[1]
        M = tf.shape(x)[2]
        L_int = x.shape[1]

        scale_tokens = []
        max_tokens = 0
        for i, p in enumerate(self.patch_sizes):
            pad = (p - (L_int % p)) % p if L_int is not None else 0
            x_pad = tf.pad(x, [[0, 0], [0, pad], [0, 0]]) if pad > 0 else x
            Lp = tf.shape(x_pad)[1]
            N = Lp // p  # tokens per scale

            # [B, N, p, M] → flatten patch over time p and project to d_each
            patches = tf.reshape(x_pad, [B, N, p, M])
            patch_vec = tf.reshape(patches, [B * N * M, p])  # [B*N*M, p]
            emb = self.proj_per_scale[i](patch_vec)          # [B*N*M, d_each]
            emb = tf.reshape(emb, [B, N, M, self.d_each])    # [B, N, M, d_each]

            max_tokens = tf.maximum(max_tokens, N)
            scale_tokens.append(emb)

        # resample to same token length and fuse feature dims
        upsampled = []
        for emb in scale_tokens:
            N = tf.shape(emb)[1]
            if self.interpolation == "nearest" or (tf.executing_eagerly() and emb.shape[1] == max_tokens):
                if not tf.equal(N, max_tokens):
                    reps = (max_tokens + N - 1) // N
                    emb_rep = tf.repeat(emb, repeats=reps, axis=1)[:, :max_tokens, :, :]
                    upsampled.append(emb_rep)
                else:
                    upsampled.append(emb)
            else:
                upsampled.append(self._linear_resample(emb, max_tokens))

        feats = tf.concat(upsampled, axis=-1)  # [B, N_max, M, sum d_each]
        feats = self.fuse(feats)               # [B, N_max, M, d_fuse]

        if self.flatten_tokens:
            B = tf.shape(feats)[0]
            N = tf.shape(feats)[1]
            M = tf.shape(feats)[2]
            D = tf.shape(feats)[3]
            z = tf.reshape(feats, [B, N, M * D])     # [B, N, M*D]
            # simple aggregation over tokens (mean) before projection to d_model
            z = tf.reduce_mean(z, axis=1)            # [B, M*D]
            z = tf.reshape(z, [B, M, D])             # treat D as per-variable feature bucket
            z = self.flat_proj(z)                    # [B, M, d_model]
            return z
        return feats


# ---------------------------------------------------
# 2) Feature Decomposition with repeat-edge padding
# ---------------------------------------------------
class FeatureDecomposition(layers.Layer):
    """AvgPool smoothing over token axis with repeat-edge padding.
    Input:  [B, N, M, D]
    Output: (Xs, Xr) both [B, N, M, D]
    """
    def __init__(self, pool_size: int = 13, **kwargs):
        super().__init__(**kwargs)
        assert pool_size % 2 == 1, "pool_size should be odd to maintain length"
        self.pool_size = pool_size
        self.avg = layers.AveragePooling1D(pool_size=self.pool_size, strides=1, padding="valid")

    def call(self, X):  # [B, N, M, D]
        B = tf.shape(X)[0]
        N = tf.shape(X)[1]
        M = tf.shape(X)[2]
        D = tf.shape(X)[3]

        # reshape to [B*M*D, N, 1] for 1D pooling along tokens
        X_ = tf.transpose(X, [0, 2, 3, 1])        # [B, M, D, N]
        X_ = tf.reshape(X_, [B * M * D, N, 1])

        k = self.pool_size
        pad = (k - 1) // 2
        left = tf.repeat(X_[:, :1, :], repeats=pad, axis=1)
        right = tf.repeat(X_[:, -1:, :], repeats=pad, axis=1)
        Xpad = tf.concat([left, X_, right], axis=1)  # [B*M*D, N+2*pad, 1]

        Xs = self.avg(Xpad)                          # [B*M*D, N, 1]
        Xs = tf.reshape(Xs, [B, M, D, N])
        Xs = tf.transpose(Xs, [0, 3, 1, 2])          # [B, N, M, D]
        Xr = X - Xs
        return Xs, Xr


# --------------------------------------------------------
# 3) MLP Blocks with selectable axes and interactions
# --------------------------------------------------------
class MLPBlock(layers.Layer):
    def __init__(self, hidden_dim: int, dropout: float, **kwargs):
        super().__init__(**kwargs)
        self.fc1 = layers.Dense(hidden_dim)
        self.fc2 = None  # set at build
        self.drop = layers.Dropout(dropout)

    def build(self, input_shape):
        d = input_shape[-1]
        self.fc2 = layers.Dense(d)

    def call(self, x, training=False):
        y = self.fc1(x); y = gelu(y); y = self.drop(y, training=training)
        y = self.fc2(y); y = self.drop(y, training=training)
        return y


class IntraVariableMLP(layers.Layer):
    """Mix along either token axis (N) for 4D input, or feature axis (D) for 3D/4D input.
    axis: "token" | "feature"
    Accepts x as [B,N,M,D] or [B,M,D].
    """
    def __init__(self, hidden_mult: float = 2.0, dropout: float = 0.0, axis: str = "feature", **kwargs):
        super().__init__(**kwargs)
        assert axis in ("token", "feature")
        self.axis = axis
        self.hidden_mult = hidden_mult
        self.drop = dropout
        self.norm = LayerNorm(axis=-1, name=f"{self.name}_ln")
        self._mlp = None

    def build(self, input_shape):
        if self.axis == "token":
            if len(input_shape) == 4:
                N = input_shape[1]
                hidden_dim = int(self.hidden_mult * N)
            else:
                raise ValueError("axis='token' requires 4D input [B,N,M,D]")
        else:  # feature
            D = input_shape[-1]
            hidden_dim = int(self.hidden_mult * D)
        self._mlp = MLPBlock(hidden_dim=hidden_dim, dropout=self.drop, name=f"{self.name}_mlp")

    def call(self, x, training=False):
        y = self.norm(x)
        if self.axis == "token":
            # [B,N,M,D] -> [B,M,D,N]
            y = tf.transpose(y, [0, 2, 3, 1])
            y = self._mlp(y, training=training)
            y = tf.transpose(y, [0, 3, 1, 2])  # back to [B,N,M,D]
            return x + y
        else:
            # feature mixing over last dim (works for [B,N,M,D] or [B,M,D])
            y = self._mlp(y, training=training)
            return x + y


class InterVariableMLP(layers.Layer):
    """Mix along variable axis M, with interaction mode:
    - "elem": elementwise gating (y * x + x), aligns with reference
    - "dot":  dot-product scalar gating across feature dim

    Accepts [B,N,M,D] or [B,M,D].
    """
    def __init__(self, hidden_mult: float = 2.0, dropout: float = 0.0, interaction: str = "elem", **kwargs):
        super().__init__(**kwargs)
        assert interaction in ("elem", "dot")
        self.hidden_mult = hidden_mult
        self.drop = dropout
        self.interaction = interaction
        self.norm = LayerNorm(axis=-1, name=f"{self.name}_ln")
        self._mlp = None

    def build(self, input_shape):
        if len(input_shape) == 4:
            M = input_shape[2]
        elif len(input_shape) == 3:
            M = input_shape[1]
        else:
            raise ValueError("InterVariableMLP expects 3D or 4D input")
        hidden_dim = int(self.hidden_mult * M)
        self._mlp = MLPBlock(hidden_dim=hidden_dim, dropout=self.drop, name=f"{self.name}_mlp")

    def _mix_over_M(self, z, training=False):
        # z last dim must be M for Dense to act over variables
        z = self._mlp(z, training=training)
        return z

    def call(self, x, training=False):
        z_in = self.norm(x)
        if len(x.shape) == 4:
            # [B,N,M,D] -> [B,N,D,M]
            y_in = tf.transpose(z_in, [0, 1, 3, 2])
            y = self._mix_over_M(y_in, training=training)  # mix over M
            if self.interaction == "dot":
                dot = tf.reduce_sum(y * y_in, axis=2, keepdims=True)  # [B,N,1,M]
                y = y + dot * y_in
            else:
                y = y * y_in + tf.transpose(x, [0, 1, 3, 2])
            y = tf.transpose(y, [0, 1, 3, 2])
            return y
        else:
            # [B,M,D] -> [B,D,M]
            y_in = tf.transpose(z_in, [0, 2, 1])
            y = self._mix_over_M(y_in, training=training)
            if self.interaction == "dot":
                dot = tf.reduce_sum(y * y_in, axis=1, keepdims=True)  # [B,1,M]
                y = y + dot * y_in
            else:
                y = y * y_in + tf.transpose(x, [0, 2, 1])
            y = tf.transpose(y, [0, 2, 1])
            return y


class PatchMLPBlock(layers.Layer):
    def __init__(self, intra_hidden_mult=2.0, inter_hidden_mult=2.0, dropout=0.0,
                 intra_axis="feature", interaction="elem", name="block", **kwargs):
        super().__init__(name=name, **kwargs)
        self.intra = IntraVariableMLP(hidden_mult=intra_hidden_mult, dropout=dropout,
                                      axis=intra_axis, name=f"{name}_intra")
        self.inter = InterVariableMLP(hidden_mult=inter_hidden_mult, dropout=dropout,
                                      interaction=interaction, name=f"{name}_inter")

    def call(self, x, training=False):
        x = self.intra(x, training=training)
        x = self.inter(x, training=training)
        return x


# ----------------------------------------
# 4) Predictor / Projection head to horizon T
#    (변수 축 M을 내부에서 집약하여 [B, T] 출력)
# ----------------------------------------
class Predictor(layers.Layer):
    def __init__(self, T: int, use_weighted_sum: bool = False, **kwargs):
        """
        Args:
            T: 예측 길이 (보통 1)
            use_weighted_sum: True면 변수 축 M에 대해 Dense(1) 가중합, False면 평균.
        """
        super().__init__(**kwargs)
        self.T = T
        self.use_weighted_sum = use_weighted_sum
        self.var_pool = None  # 변수 축 집약 레이어 (옵션)
        self.proj = None      # 최종 T로 사상

    def build(self, input_shape):
        # 입력은 [B,N,M,D] 또는 [B,M,D]
        if self.use_weighted_sum:
            # 변수축 M에 대해 가중합을 하기 위해 Dense(1) 준비
            # 적용을 쉽게 하려면 (마지막 축이 M이 되도록) 전치하여 Dense(1) 사용
            self.var_pool = layers.Dense(1)  # 가중합
        else:
            self.var_pool = None  # 평균 사용

        # var_pool 이후에는 [B,N,D] 또는 [B,D]가 되고,
        # 이를 펼쳐서 Dense(T) 전달
        if len(input_shape) == 4:
            N = input_shape[1]
            D = input_shape[-1]
            in_dim = N * D
        else:
            D = input_shape[-1]
            in_dim = D
        self.proj = layers.Dense(self.T)

    def _pool_over_M(self, x):
        # x: [B,N,M,D] 또는 [B,M,D]
        if len(x.shape) == 4:
            if self.use_weighted_sum:
                # [B,N,M,D] -> [B,N,D,M] -> Dense(1) -> [B,N,D,1] -> [B,N,D]
                y = tf.transpose(x, [0, 1, 3, 2])
                y = self.var_pool(y)
                y = tf.squeeze(y, axis=-1)
                return y  # [B,N,D]
            else:
                return tf.reduce_mean(x, axis=2)  # [B,N,D]
        else:
            if self.use_weighted_sum:
                # [B,M,D] -> [B,D,M] -> Dense(1) -> [B,D,1] -> [B,D]
                y = tf.transpose(x, [0, 2, 1])
                y = self.var_pool(y)
                y = tf.squeeze(y, axis=-1)
                return y  # [B,D]
            else:
                return tf.reduce_mean(x, axis=1)  # [B,D]

    def call(self, x):
        # 변수 축을 먼저 집약
        z = self._pool_over_M(x)  # [B,N,D] or [B,D]

        # 토큰 축이 남아있다면 펼친 뒤 T로 사상
        if len(z.shape) == 3:  # [B,N,D]
            B = tf.shape(z)[0]
            N = tf.shape(z)[1]
            D = tf.shape(z)[2]
            zm = tf.reshape(z, [B, N * D])   # [B, N*D]
            out = self.proj(zm)              # [B, T]
            return out
        else:  # [B,D]
            out = self.proj(z)               # [B, T]
            return out


# -------------------------
# Full PatchMLP model
# -------------------------
class PatchMLP(Model):
    def __init__(
        self,
        L: int = 360,
        M: int = 1,
        T: int = 1,
        patch_sizes: List[int] = (4, 8, 16),
        d_each: int = 8, #32
        d_fuse: Optional[int] = None,
        d_model: int = 8, #128
        pool_size: int = 13,
        num_blocks: int = 3,
        intra_hidden_mult: float = 2.0,
        inter_hidden_mult: float = 2.0,
        dropout: float = 0.0,
        interaction: str = "elem",   # "elem" aligns with reference; "dot" aligns text description
        intra_axis: str = "feature", # "feature" aligns with reference
        flatten_tokens: bool = True,
        use_norm: bool = False,
        use_weighted_sum: bool = False,  # Predictor에서 M축 가중합 사용 여부
        name: str = "PatchMLP",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.L, self.M, self.T = L, M, T
        self.use_norm = use_norm
        self.flatten_tokens = flatten_tokens

        # Embedding
        self.mpe = MultiScalePatchEmbedding(
            patch_sizes=list(patch_sizes),
            d_each=d_each,
            d_fuse=d_fuse,
            interpolation="linear",
            flatten_tokens=flatten_tokens,
            d_model=d_model,
            name="mpe",
        )

        # Decomposition (only if we keep token axis). If flattened, we'll lift to tokens via a fake N=1 to reuse logic
        self.decomp = FeatureDecomposition(pool_size=pool_size, name="decomposition")

        # Separate stacks for seasonal & trend
        self.seasonal_blocks = [
            PatchMLPBlock(
                intra_hidden_mult=intra_hidden_mult,
                inter_hidden_mult=inter_hidden_mult,
                dropout=dropout,
                intra_axis=intra_axis,
                interaction=interaction,
                name=f"sea_blk_{i}",
            ) for i in range(num_blocks)
        ]
        self.trend_blocks = [
            PatchMLPBlock(
                intra_hidden_mult=intra_hidden_mult,
                inter_hidden_mult=inter_hidden_mult,
                dropout=dropout,
                intra_axis=intra_axis,
                interaction=interaction,
                name=f"trd_blk_{i}",
            ) for i in range(num_blocks)
        ]

        # 수정된 Predictor: [B,T] 출력
        self.pred_head = Predictor(T=T, use_weighted_sum=use_weighted_sum, name="predictor")

    def _maybe_to_tokens4d(self, z):
        """Ensure tensor is 4D [B,N,M,D] to run decomposition & tokenwise blocks.
        If flatten_tokens=True, we have [B,M,D]; we can insert N=1.
        """
        if len(z.shape) == 3:  # [B,M,D] → [B,1,M,D]
            return tf.expand_dims(z, axis=1)
        return z

    def _maybe_from_tokens4d(self, z):
        """Back to original rank if we artificially added N=1.
        [B,1,M,D] → [B,M,D]
        """
        if len(z.shape) == 4 and z.shape[1] == 1 and self.flatten_tokens:
            return tf.squeeze(z, axis=1)
        return z

    def call(self, x_enc, training=False):  # x_enc: [B,L,M]
        # Optional normalization as in the reference
        means = None; stdev = None
        if self.use_norm:
            means = tf.reduce_mean(x_enc, axis=1, keepdims=True)
            stdev = tf.sqrt(tf.math.reduce_variance(x_enc, axis=1, keepdims=True) + 1e-5)
            x_proc = (x_enc - means) / stdev
        else:
            x_proc = x_enc

        # Embedding
        z = self.mpe(x_proc)  # [B,M,d_model] or [B,N,M,D]

        # Ensure 4D for decomposition & blocks
        z4 = self._maybe_to_tokens4d(z)

        # Decomposition on embedding tokens
        Xs, Xr = self.decomp(z4)

        # Seasonal & trend stacks (separate parameters)
        z_s, z_r = Xs, Xr
        for b in self.seasonal_blocks:
            z_s = b(z_s, training=training)
        for b in self.trend_blocks:
            z_r = b(z_r, training=training)

        z_out = z_s + z_r
        z_out = self._maybe_from_tokens4d(z_out)  # back to [B,M,D] if needed

        # Projection to horizon T  → [B, T]
        y = self.pred_head(z_out)

        # De-normalization
        if self.use_norm:
            # means/stdev: [B,1,M]; pred_head가 M축을 집약했으므로
            # 변수축 평균을 이용해 역정규화(간단히 평균 사용)
            m_mean = tf.reduce_mean(means[:, 0, :], axis=-1, keepdims=True)   # [B,1]
            m_std  = tf.reduce_mean(stdev[:, 0, :], axis=-1, keepdims=True)   # [B,1]
            y = y * m_std + m_mean  # [B,T]
        return y



