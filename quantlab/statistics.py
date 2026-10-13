import numpy as np
import pandas as pd

def compute_zscore(close, window=20):
    T = len(close)
    z = np.full(T, np.nan, dtype=np.float64)
    for i in range(window, T):
        win = close[i - window: i]
        mu = float(np.mean(win))
        sigma = float(np.std(win))
        if sigma > 1e-10:
            z[i] = (close[i] - mu) / sigma
    return z


def _kalman_innovations(log_prices, delta=1e-4, obs_noise=1.0):
    values = np.asarray(log_prices, dtype=float)
    if (values.ndim != 1 or np.isinf(values).any()
            or not np.isfinite([delta, obs_noise]).all()
            or delta < 0 or obs_noise <= 0):
        raise ValueError("1D prices, Q >= 0, R > 0; NaN denotes missing")
    innovations = np.full(len(values), np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    if not len(valid):
        return innovations
    first = valid[0]
    x, P = values[first], 1.0
    # Each row is one equal time step, including missing observations.
    for t in range(first + 1, len(values)):
        P += delta
        if np.isnan(values[t]):
            continue
        innov = values[t] - x
        innovations[t] = innov
        K = P / (P + obs_noise)
        x += K * innov
        P *= 1.0 - K
    return innovations


def _build_z_series(spreads, window=60):
    s = pd.Series(spreads)
    roll_mu = s.rolling(window).mean().shift(1)        
    roll_sd = s.rolling(window).std(ddof=0).shift(1)
    z = ((s - roll_mu) / roll_sd).to_numpy(copy=True)
    z[~np.isfinite(z)] = np.nan
    return z


def kelly_fraction(p_up, macro_bias, asset_class):
    import math
    if not (math.isfinite(p_up) and 0.0 <= p_up <= 1.0):
        raise ValueError("p_up must be in [0, 1]")
    if not (math.isfinite(macro_bias) and -1.0 <= macro_bias <= 1.0):
        raise ValueError("macro_bias must be in [-1, 1]")
    # Illustrative long-only, unit-payoff-odds allocation.
    base = max(p_up - 0.5, 0.0)
    sensitivity = {"equity": 0.5, "safe_haven": -0.5}.get(asset_class, 0.0)
    multiplier = max(0.0, 1.0 + macro_bias * sensitivity)
    return min(base * multiplier, 1.0)


# hmm_v5_core.py (excerpt)
def fractional_kelly(mu, sigma2, kelly_frac=0.25, max_size=1.0) -> float:
    """f* = kelly_frac × |μ| / σ²
    Reduce size for higher uncertainty; increase it for stronger signals.
    """
    if (not np.isfinite([mu, sigma2, kelly_frac, max_size]).all()
            or sigma2 <= 1e-12 or not 0 <= kelly_frac <= 1
            or not 0 <= max_size <= 1):
        return 0.0
    k = kelly_frac * abs(mu) / (sigma2 + 1e-12)
    return float(np.clip(k, 0.0, max_size))


def calc_halflife(spread, min_obs=20):
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


def compute_sortino(bar_returns, periods_per_year=252, target_per_period=0.0):
    returns = np.asarray(bar_returns, dtype=float)
    if (returns.ndim != 1 or len(returns) < 2
            or not np.isfinite(returns).all()
            or not np.isfinite([periods_per_year, target_per_period]).all()
            or periods_per_year <= 0):
        raise ValueError("finite returns, target and positive annualization required")
    excess = returns - target_per_period
    downside = np.minimum(excess, 0.0)
    downside_dev = np.sqrt(np.mean(downside ** 2))
    if downside_dev == 0:
        return None
    return float(np.mean(excess) / downside_dev
                 * np.sqrt(periods_per_year))
