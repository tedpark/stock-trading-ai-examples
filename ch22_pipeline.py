"""22장 보완 실습: 더미 가격 → 피처 → HMM → 신호 → 비용 차감 평가."""
import numpy as np
from hmm_demo import dummy_bars, features, fit_model, forward

bars = dummy_bars()
frame = features(bars)
x = frame.to_numpy()
split = int(len(x) * 0.6)
model, scaler, order = fit_model(x, split)
probabilities = forward(model, scaler.transform(x))[:, order]
close = bars.close.reindex(frame.index)
zscore = ((close - close.rolling(20).mean()) / close.rolling(20).std()).to_numpy()
# 종가 t의 정보로 결정하고 수익은 t → t+1에만 적용한다.
position, equity, peak, mdd, turnover = 0.0, 1000.0, 1000.0, 0.0, 0.0
fee = 0.001
for t in range(split, len(x) - 1):
    p, z = probabilities[t], zscore[t]
    regime = p.argmax()
    target = 0.0
    if np.isfinite(z) and p.max() >= 0.65:
        if regime == 0 and abs(z) > 1.5:
            target = -np.sign(z)
        elif regime == 2 and abs(z) > 1.0:
            target = np.sign(z)
    change = abs(target - position)
    net_return = target * (close.iloc[t + 1] / close.iloc[t] - 1) - fee * change
    equity *= 1 + net_return
    turnover += change
    position = target
    peak = max(peak, equity)
    mdd = max(mdd, 1 - equity / peak)
equity *= 1 - fee * abs(position)  # 최종 청산 비용
turnover += abs(position)
mdd = max(mdd, 1 - equity / peak)
assert np.isfinite(equity) and equity > 0 and 0 <= mdd < 1
print(f'dummy OOS equity={equity:.2f}, MDD={mdd:.2%}, turnover={turnover:.0f}')
