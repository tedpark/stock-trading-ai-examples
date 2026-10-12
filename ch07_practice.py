"""07장 학습 예제 — 더미 데이터 전용. 원문: 07-cross-asset-joint-hmm.md"""
import numpy as np
from hmmlearn.hmm import GaussianHMM

SEED, T = 42, 500
N_MACRO_STATES = 3
rng = np.random.default_rng(SEED)

# 20차원 관측 벡터 500개 (합성)
X = rng.normal(0, 1, size=(T, 20))

# 60% 훈련, 40% out-of-sample
train_len = int(T * 0.6)
macro_hmm = GaussianHMM(
    n_components=N_MACRO_STATES,
    covariance_type="full",
    n_iter=200,
    random_state=SEED,
)
macro_hmm.fit(X[:train_len])

# OOS를 한 시점씩 공개하고 마지막 행만 읽는다.
# 마지막 행 뒤에는 관측치가 없으므로 미래 정보가 섞이지 않는다.
alpha_oos = []
for t in range(train_len, T):
    alpha_t = macro_hmm.predict_proba(X[:t + 1])[-1]
    alpha_oos.append(alpha_t)

alpha_oos = np.asarray(alpha_oos)
regime_oos = alpha_oos.argmax(axis=1)

print(f"관측 벡터 shape: {X.shape}")
print(f"학습 완료: {N_MACRO_STATES} 상태")
print(f"OOS 레짐 분포: {np.bincount(regime_oos)} (OOS {T - train_len}봉)")
