from typing import List, Optional
import tensorflow as tf
from keras import layers, Model

# --------------------
# Utilities
# --------------------

def gelu(x):
    return tf.keras.activations.gelu(x, approximate=True)


# ----------------------------------------
# 1) Multi-Scale Patch Embedding (MPE)
#    - einsum 기반 패치 투영
#    - tf.image.resize로 선형 보간 대체
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

        # per-scale projection weights: [p, d_each]
        self.W_per_scale = []
        for i, p in enumerate(self.patch_sizes):
            self.W_per_scale.append(
                self.add_weight(
                    name=f"{name}_W_s{i}",
                    shape=(p, d_each),
                    initializer="glorot_uniform",
                    trainable=True,
                )
            )

        self.fuse = layers.Dense(self.d_fuse, name=f"{name}_fuse")
        if self.flatten_tokens:
            assert self.d_model is not None, "d_model must be provided when flatten_tokens=True"
            self.flat_proj = layers.Dense(self.d_model, name=f"{self.name}_flat_proj")

    def _linear_resample(self, emb, max_tokens):
        # emb: [B, N, M, D] → [B, max_tokens, M, D]
        B = tf.shape(emb)[0]
        N = tf.shape(emb)[1]
        M = tf.shape(emb)[2]
        D = tf.shape(emb)[3]

        if emb.shape[1] == 1:
            return tf.repeat(emb, repeats=max_tokens, axis=1)

        # tf.image.resize는 4D(NHWC) 입력 필요
        y = tf.reshape(emb, [B * M, N, D, 1])                 # [B*M, N, D, 1]
        y = tf.image.resize(y, size=(max_tokens, D), method="bilinear", antialias=False)
        y = tf.reshape(y, [B, M, max_tokens, D])              # [B, M, Nmax, D]
        out = tf.transpose(y, [0, 2, 1, 2])                   # [B, Nmax, M, D]
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

            # [B, N, p, M]
            patches = tf.reshape(x_pad, [B, N, p, M])
            # einsum으로 p→d_each 투영: 'bnpm,pd->bnmd'
            W = self.W_per_scale[i]
            emb = tf.einsum('bnpm,pd->bnmd', patches, W)      # [B, N, M, d_each]

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
            # 토큰 축 평균 → [B, M, d_fuse] 후 선형사상
            z = tf.reduce_mean(feats, axis=1)           # [B, M, d_fuse]
            z = self.flat_proj(z)                       # [B, M, d_model]
            return z
        return feats


# ---------------------------------------------------
# 2) Feature Decomposition with repeat-edge padding
#     - tf.nn.avg_pool1d + SYMMETRIC pad
#     - 전치/왕복 축 제거
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

    def call(self, X):  # [B, N, M, D]
        B = tf.shape(X)[0]
        N = tf.shape(X)[1]
        M = tf.shape(X)[2]
        D = tf.shape(X)[3]

        # [B*M*D, N, 1]  (전치 없이 바로 reshape)
        X_ = tf.reshape(X, [B * M * D, N, 1])

        k = self.pool_size
        pad = (k - 1) // 2
        Xpad = tf.pad(X_, [[0, 0], [pad, pad], [0, 0]], mode="SYMMETRIC")  # repeat-edge ~= SYMMETRIC

        Xs = tf.nn.avg_pool1d(Xpad, ksize=k, strides=1, padding="VALID")   # [B*M*D, N, 1]
        Xs = tf.reshape(Xs, [B, N, M, D])                                  # [B, N, M, D]
        Xr = X - Xs
        return Xs, Xr


# --------------------------------------------------------
# 3) MLP Blocks with selectable axes and interactions
#    - LN: fused LayerNormalization
#    - InterVariableMLP: einsum으로 M축만 변환 (전치 제거)
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
        self.norm = layers.LayerNormalization(axis=-1, epsilon=1e-5, name=f"{self.name}_ln")
        self._mlp = None

    def build(self, input_shape):
        if self.axis == "token":
            if len(input_shape) == 4:
                N = int(input_shape[1])
                hidden_dim = int(self.hidden_mult * N)
            else:
                raise ValueError("axis='token' requires 4D input [B,N,M,D]")
        else:  # feature
            D = int(input_shape[-1])
            hidden_dim = int(self.hidden_mult * D)
        self._mlp = MLPBlock(hidden_dim=hidden_dim, dropout=self.drop, name=f"{self.name}_mlp")

    def call(self, x, training=False):
        y = self.norm(x)
        if self.axis == "token":
            # token 축 혼합을 위해 [B,N,M,D] → [B,M,D,N] 전치는 그대로 필요
            y = tf.transpose(y, [0, 2, 3, 1])
            y = self._mlp(y, training=training)
            y = tf.transpose(y, [0, 3, 1, 2])  # back to [B,N,M,D]
            return x + y
        else:
            # feature mixing over last dim (works for [B,N,M,D] or [B,M,D])
            y = self._mlp(y, training=training)
            return x + y


class InterVariableMLP(layers.Layer):
    """Mix along variable axis M with einsum (no transposes).
    interaction: "elem" (y * x + x) | "dot" (y + <y,x> * x)  (원 의미 유지)
    Accepts [B,N,M,D] or [B,M,D].
    """
    def __init__(self, hidden_mult: float = 2.0, dropout: float = 0.0, interaction: str = "elem", **kwargs):
        super().__init__(**kwargs)
        assert interaction in ("elem", "dot")
        self.hidden_mult = hidden_mult
        self.drop_rate = dropout
        self.interaction = interaction
        self.norm = layers.LayerNormalization(axis=-1, epsilon=1e-5, name=f"{self.name}_ln")

    def build(self, input_shape):
        if len(input_shape) == 4:
            self.M = int(input_shape[2])
            self.rank4 = True
        elif len(input_shape) == 3:
            self.M = int(input_shape[1])
            self.rank4 = False
        else:
            raise ValueError("InterVariableMLP expects 3D or 4D input")
        H = int(self.hidden_mult * self.M)
        # MLP over M-axis: M->H->M
        self.W1 = self.add_weight("W1", shape=(self.M, H), initializer="glorot_uniform")
        self.b1 = self.add_weight("b1", shape=(H,), initializer="zeros")
        self.W2 = self.add_weight("W2", shape=(H, self.M), initializer="glorot_uniform")
        self.b2 = self.add_weight("b2", shape=(self.M,), initializer="zeros")
        self.drop = layers.Dropout(self.drop_rate)

    def _mlp_over_M(self, x, training):
        # x: [B,N,M,D] or [B,M,D]; 변환은 M축에만 적용
        if self.rank4:
            y = tf.einsum('bnmd,mh->bnhd', x, self.W1) + self.b1    # M->H
            y = gelu(y); y = self.drop(y, training=training)
            y = tf.einsum('bnhd,hm->bnmd', y, self.W2) + self.b2    # H->M
            y = self.drop(y, training=training)
            return y
        else:
            y = tf.einsum('bmd,mh->bhd', x, self.W1) + self.b1
            y = gelu(y); y = self.drop(y, training=training)
            y = tf.einsum('bhd,hm->bmd', y, self.W2) + self.b2
            y = self.drop(y, training=training)
            return y

    def call(self, x, training=False):
        z_in = self.norm(x)
        y = self._mlp_over_M(z_in, training=training)
        if self.interaction == "dot":
            # dot-product scalar gating across feature dim
            if self.rank4:
                dot = tf.reduce_sum(y * z_in, axis=3, keepdims=True)  # [B,N,M,1]
                y = y + dot * z_in
            else:
                dot = tf.reduce_sum(y * z_in, axis=2, keepdims=True)  # [B,M,1]
                y = y + dot * z_in
        else:
            # elementwise gating
            y = y * z_in + x
        return y


class PatchMLPBlock(layers.Layer):
    def __init__(self, intra_hidden_mult=2.0, inter_hidden_mult=2.0, dropout=0.0,
                 intra_axis="feature", interaction="elem", name="block", **kwargs):
        super().__init__(name=name, **kwargs)
        self.intra = IntraVariableMLP(hidden_mult=intra_hidden_mult, dropout=dropout,
                                      axis=intra_axis, name=f"{self.name}_intra")
        self.inter = InterVariableMLP(hidden_mult=inter_hidden_mult, dropout=dropout,
                                      interaction=interaction, name=f"{self.name}_inter")

    def call(self, x, training=False):
        x = self.intra(x, training=training)
        x = self.inter(x, training=training)
        return x


# ----------------------------------------
# 4) Predictor / Projection head to horizon T
#    - 거대 flatten 제거: N/M 평균 풀링 + Dense(T)
#    - 옵션: M축 가중합 유지
# ----------------------------------------
class Predictor(layers.Layer):
    def __init__(self, T: int, use_weighted_sum: bool = False, pool: str = "mean", **kwargs):
        """
        Args:
            T: 예측 길이 (보통 1)
            use_weighted_sum: True면 변수 축 M 가중합(Dense(1)), False면 평균.
            pool: N 축 풀링 방식 ("mean" | "max")
        """
        super().__init__(**kwargs)
        self.T = T
        self.use_weighted_sum = use_weighted_sum
        self.pool = pool
        self.var_pool = None
        self.proj = layers.Dense(self.T)

    def build(self, input_shape):
        if self.use_weighted_sum:
            self.var_pool = layers.Dense(1)  # M축 가중합

    def _pool_over_M(self, x):
        # x: [B,N,M,D] 또는 [B,M,D]
        if len(x.shape) == 4:
            if self.use_weighted_sum:
                # [B,N,M,D] -> [B,N,D,M] -> Dense(1) -> squeeze -> [B,N,D]
                y = tf.transpose(x, [0, 1, 3, 2])
                y = self.var_pool(y)
                y = tf.squeeze(y, axis=-1)
                return y
            else:
                return tf.reduce_mean(x, axis=2)  # [B,N,D]
        else:
            if self.use_weighted_sum:
                y = tf.transpose(x, [0, 2, 1])
                y = self.var_pool(y)
                y = tf.squeeze(y, axis=-1)        # [B,D]
                return y
            else:
                return tf.reduce_mean(x, axis=1)  # [B,D]

    def _pool_over_N(self, z):
        # z: [B,N,D] 또는 [B,D]
        if len(z.shape) == 3:
            if self.pool == "max":
                return tf.reduce_max(z, axis=1)   # [B,D]
            return tf.reduce_mean(z, axis=1)      # [B,D]
        return z

    def call(self, x):
        z = self._pool_over_M(x)     # [B,N,D] or [B,D]
        z = self._pool_over_N(z)     # [B,D]
        out = self.proj(z)           # [B,T]
        return out


# -------------------------
# Full PatchMLP model
# -------------------------
class PatchMLPTwo(Model):
    def __init__(
        self,
        L: int = 360,
        M: int = 1,
        T: int = 1,
        patch_sizes: List[int] = (4, 8, 16),
        d_each: int = 8,  # 32
        d_fuse: Optional[int] = None,
        d_model: int = 8,  # 128
        pool_size: int = 13,
        num_blocks: int = 3,
        intra_hidden_mult: float = 2.0,
        inter_hidden_mult: float = 2.0,
        dropout: float = 0.0,
        interaction: str = "elem",
        intra_axis: str = "feature",
        flatten_tokens: bool = True,
        use_norm: bool = False,
        use_weighted_sum: bool = False,
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

        # Decomposition (token 축이 남아있는 경우에만 실질 의미 있음)
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

        self.pred_head = Predictor(T=T, use_weighted_sum=use_weighted_sum, name="predictor")

    def _maybe_to_tokens4d(self, z):
        """Ensure tensor is 4D [B,N,M,D] to run decomposition & tokenwise blocks.
        If flatten_tokens=True, we have [B,M,D]; we can insert N=1.
        """
        if len(z.shape) == 3:
            return tf.expand_dims(z, axis=1)  # [B,1,M,D]
        return z

    def _maybe_from_tokens4d(self, z):
        """Back to original rank if we artificially added N=1.
        [B,1,M,D] → [B,M,D]
        """
        if len(z.shape) == 4 and z.shape[1] == 1 and self.flatten_tokens:
            return tf.squeeze(z, axis=1)
        return z

    def call(self, x_enc, training=False):  # x_enc: [B,L,M]
        # Optional normalization
        if self.use_norm:
            means = tf.reduce_mean(x_enc, axis=1, keepdims=True)  # [B,1,M]
            stdev = tf.sqrt(tf.math.reduce_variance(x_enc, axis=1, keepdims=True) + 1e-5)
            x_proc = (x_enc - means) / stdev
        else:
            means = None; stdev = None
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
            m_mean = tf.reduce_mean(means[:, 0, :], axis=-1, keepdims=True)   # [B,1]
            m_std  = tf.reduce_mean(stdev[:, 0, :], axis=-1, keepdims=True)   # [B,1]
            y = y * m_std + m_mean  # [B,T]
        return y
