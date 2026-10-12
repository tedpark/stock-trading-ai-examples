"""11장 학습 예제 — 더미 데이터 전용. 원문: 11-stat-arb-pairs.md"""
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import coint

MIN_CORR, MAX_PVALUE, MAX_HALFLIFE = 0.40, 0.05, 60

def estimate_halflife(spread):
    y = spread[1:] - spread[:-1]
    x = spread[:-1] - spread[:-1].mean()
    slope, *_ = stats.linregress(x, y)
    if slope >= 0:
        return float("inf")
    return float(-np.log(2) / slope)

def screen_pair(prices_a, prices_b):
    log_a, log_b = np.log(prices_a), np.log(prices_b)
    # 1) 공통 움직임의 1차 필터: 로그수익률 상관관계
    ret_corr = np.corrcoef(np.diff(log_a), np.diff(log_b))[0, 1]
    if ret_corr < MIN_CORR:
        return None
    # 2) OLS β + 스프레드
    slope, intercept, *_ = stats.linregress(log_b, log_a)
    spread = log_a - slope * log_b - intercept
    # 3) 잔차 추정을 반영한 Engle-Granger 공적분 검정
    _, eg_pvalue, _ = coint(log_a, log_b)
    if eg_pvalue >= MAX_PVALUE:
        return None
    # 4) 반감기
    hl = estimate_halflife(spread)
    if hl <= 0 or hl > MAX_HALFLIFE:
        return None
    return {"return_corr": float(ret_corr),
            "eg_p": float(eg_pvalue),
            "hl": hl}

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    common = np.cumsum(rng.normal(0, 1, 300))
    a = np.exp(common * 0.9 + rng.normal(0, 0.2, 300)) * 100
    b = np.exp(common * 1.1 + rng.normal(0, 0.2, 300)) * 150
    print(screen_pair(a, b))
