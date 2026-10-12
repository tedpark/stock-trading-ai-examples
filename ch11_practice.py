"""Chapter 11 learning example; dummy data only. Source: 11-stat-arb-pairs.md"""
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
    # 1) Initial co-movement filter: log-return correlation
    ret_corr = np.corrcoef(np.diff(log_a), np.diff(log_b))[0, 1]
    if ret_corr < MIN_CORR:
        return None
    # 2) OLS hedge ratio and spread
    slope, intercept, *_ = stats.linregress(log_b, log_a)
    spread = log_a - slope * log_b - intercept
    # 3) Engle-Granger cointegration test accounting for estimated residuals
    _, eg_pvalue, _ = coint(log_a, log_b)
    if eg_pvalue >= MAX_PVALUE:
        return None
    # 4) Half-life
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
