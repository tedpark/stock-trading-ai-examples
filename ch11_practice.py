"""Chapter 11 learning example; dummy data only. Source: 11-stat-arb-pairs.md"""
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import coint

MIN_CORR, MAX_PVALUE, MAX_HALFLIFE = 0.40, 0.05, 60

def estimate_halflife(spread, min_obs=20):
    values = np.asarray(spread, dtype=float)
    if not isinstance(min_obs, int) or min_obs < 3:
        raise ValueError("min_obs must be an integer >= 3")
    if (values.ndim != 1 or len(values) < min_obs
            or not np.isfinite(values).all()):
        return float("inf")
    lag, change = values[:-1], np.diff(values)
    x = lag - lag.mean()
    denominator = float(x @ x)
    if denominator <= 0:
        return float("inf")
    phi = float(x @ (change - change.mean()) / denominator)
    # Positive, stationary AR(1): a = 1 + phi lies strictly in (0, 1).
    if not -1 < phi < 0:
        return float("inf")
    return float(np.log(0.5) / np.log1p(phi))

def screen_pair(prices_a, prices_b):
    prices_a, prices_b = np.asarray(prices_a, dtype=float), np.asarray(prices_b, dtype=float)
    if (prices_a.ndim != 1 or prices_b.ndim != 1 or len(prices_a) != len(prices_b)
            or len(prices_a) < 20 or not np.isfinite(prices_a).all()
            or not np.isfinite(prices_b).all() or (prices_a <= 0).any() or (prices_b <= 0).any()):
        raise ValueError("aligned positive finite price arrays with at least 20 observations required")
    log_a, log_b = np.log(prices_a), np.log(prices_b)
    if np.std(np.diff(log_a)) == 0 or np.std(np.diff(log_b)) == 0:
        return None
    # 1) Initial co-movement filter: log-return correlation
    ret_corr = np.corrcoef(np.diff(log_a), np.diff(log_b))[0, 1]
    if not np.isfinite(ret_corr) or ret_corr < MIN_CORR:
        return None
    # 2) OLS hedge ratio and spread
    slope, intercept, *_ = stats.linregress(log_b, log_a)
    spread = log_a - slope * log_b - intercept
    if np.std(spread) <= 1e-12:
        return None
    # 3) Engle-Granger cointegration test accounting for estimated residuals
    _, eg_pvalue, _ = coint(log_a, log_b)
    if not np.isfinite(eg_pvalue) or eg_pvalue >= MAX_PVALUE:
        return None
    # 4) Half-life
    hl = estimate_halflife(spread)
    if not np.isfinite(hl) or hl <= 0 or hl > MAX_HALFLIFE:
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
