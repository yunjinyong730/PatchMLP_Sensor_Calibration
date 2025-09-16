# PatchMLP_Sensor_Calibration
```
# 실험 결과
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_4 (InputLayer)        [(1, 360, 5)]             0         
                                                                 
 normalizer_3 (Normalizer)   (1, 360, 5)               0         
                                                                 
 PatchMLP (PatchMLP)         (1, 1)                    3619      
                                                                 
 denormalizer_3 (Denormaliz  (1, 1)                    0         
 er)                                                             
...
_________________________________________________________________
934/934 [==============================] - 7s 6ms/step
Inference time: 7.062 seconds
Throughput: 4231.30 samples/second
```

## PM10 정확도
<img width="1000" height="400" alt="Antwerp_pm10_w360" src="https://github.com/user-attachments/assets/b4137a4d-c6e9-4693-ae37-2fe06b2ea7dd" />
<b>Antwerp: val rmse : 8.31779956817627, test rmse : 11.96527099609375</b> <br>
<img width="1000" height="400" alt="oslo_pm10_w360" src="https://github.com/user-attachments/assets/6474d05b-ff7d-479e-80ac-649d504b54bb" />
<b>Oslo: val rmse : 9.048280715942383, test rmse : 12.056038856506348</b> <br>
<img width="1000" height="400" alt="Zagreb_pm10_w360" src="https://github.com/user-attachments/assets/0068141b-d2fb-46aa-b10e-43074f4a96be" />
<b>Zagreb: val rmse : 16.360061645507812, test rmse : 13.834146499633789</b>


## SensEURCity 데이터 셋 
### (3개 유럽 도시 대규모 미세먼지 데이터| 저비용 센서 | 고비용 센서 데이터 포함)
사용한 데이터 셋:
- https://www.nature.com/articles/s41597-023-02135-w

Paper Link: 
- https://arxiv.org/html/2405.13575v3 
- https://github.com/TangPeiwang/PatchMLP

## PatchMLP: Patch 기반 MLP로 장기 시계열 예측을 재정의함

patchMLP를 논문을 기반으로 구현하고, 해당 모델을 센서 보정(저비용 센서를 고비용 센서 데이터만큼 딥러닝 모델을 이용해서 보정) 모델로 변환 후  
온-디바이스 성능 측정하기

### TL;DR

Transformer가 LTSF(Long-Term Time Series Forecasting)에서 강력해 보이는 이유는 자체가 아니라 ‘Patch’ 표현 덕일 수 있으며, 멀티스케일 패치 + 임베딩 공간에서의 단순 분해 + Intra/Inter-variable MLP 혼합만으로도 SOTA를 달성하는 PatchMLP를 제안함

### 문제의식과 핵심 메시지

- 자기회귀적이지 않은 Transformer의 Permutation-invariant self-attention은 절대적 시간질서를 희석시키며, 원시 시계열의 고주파 잡음과 중복 특징에 취약함을 보임
- 반면 Patch는 지역성(locality)을 강화하고 차원을 축소하며 스무딩 효과로 잡음을 줄여, 시계열에 더 적합한 입력 표현을 제공함
- 최근 유행한 채널 독립(channel independence) 가설은 과대평가되었고, 올바른 방식의 변수 간 상호작용(channel mixing) 은 다변량 예측 성능에 필수적임을 보임
- 위 통찰을 바탕으로, 복잡한 어텐션 없이도 단순 MLP로 최첨단 성능을 달성하는 PatchMLP를 설계함

### 모델 개요

<img width="1536" height="342" alt="image" src="https://github.com/user-attachments/assets/67f69c20-4606-4d86-8a22-b39c07b78023" />


PatchMLP는 네 개의 구성요소로 이뤄짐

- **Multi-Scale Patch Embedding (MPE)**: 서로 다른 길이의 패치로 원시 시계열을 비중첩 분할한 뒤, 각 패치를 선형 임베딩하여 멀티스케일 정보를 결합함
- **Feature Decomposition in Embedding Space**: 원신호가 아니라 임베딩 토큰을 평균풀링(AvgPool) 기반으로 스무스 성분(Xs) 과 잔차 성분(Xr) 로 분리하여, 랜덤 요동을 억제하고 유의미한 패턴을 부각함
- **MLP Layer with Dual Mixing**:
  - Intra-variable MLP로 시간축 내 패턴을, Inter-variable MLP로 변수 간 상호작용을 학습함
  - Inter-variable 경로에서 점곱(dot-product) 결합을 도입하여 비선형 상호작용을 강화함
  - 각 블록 뒤에 Residual connection과 정규화를 적용해 학습 안정성을 확보함
- **Projection Layer & Loss**: 잠재표현을 원공간으로 투영해 멀티스텝 예측을 산출하고, MSE 손실로 학습함

### 왜 Patch가 효과적인가

- 고빈도 샘플링으로 인한 중복·잡음 특성이 많은 시계열에서, Patch는 입력을 압축·평활해 노이즈 민감도를 낮추고 지역적 의미 구조를 강화함
- 패치 크기는 커질수록 항상 유리하지 않으며, 모델 용량(d_model) 과의 균형이 중요함을 실험으로 보임
- 입력 길이가 길어질수록 최적 패치 크기도 커지는 경향이 있으며, 지나친 압축은 정보 손실을 유발할 수 있음을 보고함

### 설계 포인트

- **임베딩 후 분해**: 전통적 추세/계절 분해를 원신호에서 수행하는 대신, 임베딩 공간에서 평균풀링으로 스무스/잔차를 나눠 간단하면서도 효과적인 잡음 억제를 달성함
- **Dual Mixing**: 채널 독립이 만능이 아니며, 적절한 변수 간 혼합이 예측력을 일관되게 끌어올림을 보임
- **점곱 결합 이점**: Inter-variable 경로에서 단순 합보다 점곱 결합이 상호작용 표현력을 높여 우수한 성능을 보임

### 실험 결과 요약

- ETT 시리즈, ECL, Traffic, Weather, Solar 등 8개 표준 벤치마크에서 4개 예측 지평(96/192/336/720) 평균 성능 기준으로 전 항목 SOTA를 보고함
- iTransformer, PatchTST, Crossformer, FEDformer 등 Transformer 계열과 TimeMixer, DLinear, TiDE, TimesNet 등 CNN/MLP 계열을 폭넓게 상회함을 제시함
- 입력 길이 증가 시 다수 모델이 장기 구간에서 성능 저하를 겪는 반면, PatchMLP와 DLinear는 안정적 개선을 보이며 장기 패턴 포착에 유리함을 시사함

### 어블레이션 인사이트

- MPE 제거 시 멀티스케일 관계 학습이 약화되어 성능 하락이 발생함
- 임베딩 분해 제거 시 잡음 억제가 어려워져 오류가 증가함
- Inter-variable 점곱 제거 또는 변수 혼합 자체 제거 시 다변량 상호작용을 잃어 유의한 성능 저하가 발생함
- 결론적으로 멀티스케일 패치 + 임베딩 분해 + 점곱 기반 변수 혼합의 조합이 성능의 핵심 동력임을 확인함

### 한계와 향후 과제

- 평균풀링 기반 분해는 비정상성·구조적 변화가 매우 큰 도메인에서 최적이 아닐 수 있으며, 적응적 분해 커널이나 학습형 스무딩으로 확장이 필요함
- 멀티스케일 패치의 크기·비율 선택은 도메인 주기성과 상호작용하므로, 자동 스케일 선택 혹은 메타러닝 기법의 도입이 유망함
- Inter-variable 점곱 결합은 단순하고 효율적이지만, 희소·가변적 상관구조를 가진 데이터에서는 가중 마스킹이나 조건부 혼합으로 더 나은 적응성을 기대할 수 있음

### 결론

PatchMLP는 복잡한 어텐션 설계 없이도 Patch 표현의 본질적 이점과 임베딩 공간 분해, 이중 혼합 MLP만으로 LTSF에서 간결함·효율성·정확성을 동시에 달성함을 보였음  
이는 LTSF에서 Transformer의 우수성이 어텐션 그 자체가 아니라 입력 표현(패치) 에 기인했을 수 있음을 시사하며, 단순하지만 올바른 구조적 선택이 대안이 될 수 있음을 입증함

---

# 구현 코드

### 데이터 흐름
[B, L, M] 입력 → MPE(멀티스케일 패치 임베딩) → 임베딩 공간 분해(Xs/Xr) → Dual Mixing MLP 블록(Intra→Inter) → Predictor → 출력 [B, T]

### 상세 개요

- 입력 `x_enc ∈ ℝ^{B×L×M}`에서 L은 입력 윈도 길이, M은 변수 수, B는 배치 크기
- MPE가 원시 시계열을 스케일별로 패치화 → 각 패치를 선형 임베딩 → 스케일 결합
- 임베딩 공간 분해가 이동평균으로 스무스 성분 Xs와 잔차 성분 Xr 분리
- Dual Mixing MLP 블록이 Intra-variable → Inter-variable 순으로 시간/특징 축과 변수 축을 혼합
- Predictor가 변수 축을 요약하고 예측 지평 T로 사상해 최종 출력 생성

### 전체 흐름을 큰 그림으로 보기

입력: `x = [배치, 길이 L, 변수 M]`

#### MPE(멀티스케일 패치 임베딩)

- 여러 패치 길이(예: 4, 8, 16)로 x를 조각내고, 각 조각을 간단한 Dense에 통과시켜 토큰으로 바꿈
- 스케일마다 토큰 개수가 다를 수 있으니 보간해서 길이를 맞춘 다음, 합쳐서 하나의 표현으로 만듦

#### 임베딩 공간 분해(FeatureDecomposition)

- 방금 만든 토큰열을 이동평균으로 부드럽게 만든 것(Xs)과, 원본에서 그걸 뺀 잔차(Xr) 로 나눔  
- 즉 “추세/느린 파동”과 “빠른 변화/노이즈에 가까운 부분”을 임베딩에서 분리한다고 보면 됨

#### Dual Mixing MLP 블록(여러 층)

- **Intra-variable(변수 내부) MLP**: 각 변수 안에서 시간/특징을 섞어 그 변수 자체의 패턴을 더 잘 표현
- **Inter-variable(변수 간) MLP**: 변수 축을 기준으로 MLP를 돌려 변수들 사이 상호작용을 학습
  - `interaction="elem"`: 원소별 게이팅(y * x + x) — 안정적
  - `interaction="dot"`: 점곱 게이트 — 표현력↑(가끔 민감)

> 포인트: 각 블록 뒤에 Residual/정규화를 적용해 학습 안정성 확보

#### Predictor 헤드

- 변수 축을 평균(또는 가중합)으로 요약한 뒤, Dense(T)로 T 스텝 예측을 뱉어냄  
- 이 구현은 기본이 단일 시계열 출력 `[B, T]` 이야. 다변량 예측이 필요하면 헤드를 바꾸면 됨

#### 옵션) 정규화(`use_norm`)

- 입력을 윈도 길이 기준으로 표준화했다가, 예측을 낼 때 원척도로 되돌려줌

### 각 블록을 “왜/어떻게” 중심으로 이해하기

1) **MultiScalePatchEmbedding**  
   - **왜?** 긴 시계열에는 빠른 변화도 있고 느린 주기도 있어. 여러 길이의 패치로 보면 두 영역을 같이 잡기 쉬워짐  
   - **어떻게?**  
     - 길이 `p`로 자름 → `[B, N, p, M]`  
     - `p` 길이 패치를 펴서 `Dense(d_each)` → `[B, N, M, d_each]`  
     - 스케일마다 `N(토큰 수)`이 다르면 보간으로 맞춤  
     - 스케일들을 특징 차원으로 합치고 `Dense(d_fuse)`로 정리  
     - `flatten_tokens=True`라면 토큰을 평균내서 `[B, M, d_model]`로 압축(시간 해상도 ↓, 계산 효율 ↑)  
     - 팁: 세밀한 시간 패턴이 중요하면 `flatten_tokens=False`로 두고 토큰을 유지함

2) **FeatureDecomposition (이동평균 분해)**  
   - **왜?** 임베딩에도 여전히 노이즈/빠른 요동이 있음. 이동평균으로 부드럽게 만든 것과 잔차로 나누면, 다음 블록들이 더 안정적으로 배울 수 있음  
   - **어떻게?** 토큰 축 `N` 방향으로 `AveragePooling1D`를 적용하는데, 양끝을 반복해서 패딩해서 길이가 줄지 않도록 했음  
   - 결과는 `(Xs, Xr)` = 같은 모양의 두 토큰열

3) **Dual Mixing MLP (Intra → Inter 순서)**  
   - **Intra-variable MLP**: 각 변수 안에서 시간/특징을 섞어 해당 변수의 표현을 업그레이드  
     - `axis="feature"`가 기본: 마지막 특징 차원만 MLP로 돌려 가볍고 안정적  
     - `axis="token"`도 가능: 토큰 축을 마지막으로 옮겨 시간 방향 혼합도 할 수 있음  
   - **Inter-variable MLP**: 변수 축을 마지막으로 옮겨 Dense가 변수 간을 섞게 함  
     - `interaction` 모드로 상호작용 강도를 고를 수 있음  
       - `"elem"`: 안정적, 기본값으로 무난  
       - `"dot"`: 점곱 게이트로 변수 간 관계를 더 강하게 표현(학습률/정규화에 다소 민감)  
   - **포인트**: Intra로 각 변수 내부를 다듬고, Inter로 변수 간 관계를 잡는다 — **순서를 유지**하는 게 안정적

4) **Predictor (출력)**  
   - 이 구현은 변수 축을 먼저 요약(평균 또는 가중합)하고, 남은 표현을 펴서 `Dense(T)`에 넣어 `[B, T]` 를 출력  
   - **다변량 출력이 필요하면?**  
     - 집약하기 전에 변수별로 `Dense(T)`를 적용해서 `[B, T, M]`을 만들거나  
     - `Dense(T*M)` 후 `reshape`하는 방식으로 바꿀 수 있음

### 설정을 고를 때의 직관

- **flatten_tokens**
  - True: 빠르고 가벼움, 긴 예측 지평/고잡음 도메인에 유리(시간 해상도는 희생)
  - False: 토큰 유지로 세밀한 패턴 포착(연산/메모리 ↑)
- **interaction**
  - "elem"으로 시작 → 안정화 후 "dot" 실험
- **pool_size(이동평균 커널)**
  - 데이터 주기와 창 길이를 보고 9~25 정도에서 튜닝(너무 크면 과하게 부드러워질 수 있음)
- **정규화(`use_norm=True`)**
  - 변수 스케일이 제각각이면 거의 필수. 수렴과 일반화에 도움

---
