"""4장 보완 실습: 훈련 구간으로만 HMM을 학습하고 분산 순서로 상태 정렬."""
import numpy as np
from hmm_demo import dummy_bars, features, fit_model, forward

x = features(dummy_bars()).to_numpy()
train_len = int(len(x) * 0.6)
model, scaler, order = fit_model(x, train_len)
probabilities = forward(model, scaler.transform(x))[:, order]
assert np.allclose(probabilities.sum(axis=1), 1)
print('state order (low / middle / high variance):', order)
print('latest probabilities:', probabilities[-1].round(4))
print('train / evaluation:', train_len, len(x) - train_len)
# 분산 순위는 상태 이름을 붙이는 규칙이며 평균회귀/추세의 실증 증거는 아니다.
