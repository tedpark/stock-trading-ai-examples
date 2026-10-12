"""Chapter 16 learning example; dummy data only. Source: 16-risk-management.md"""
from math import ceil, isfinite, sqrt
from statistics import mean, stdev


def compute_risk_metrics(equity, periods=252):
    """Use equal-period equity samples and a zero risk-free rate.

    VaR/CVaR use signed losses: negative values indicate gains in the tail.
    CVaR integrates exactly 5% of the empirical probability mass.
    """
    if len(equity) < 3 or not isfinite(periods) or periods <= 0:
        raise ValueError("at least three equity observations required")
    if any(not isfinite(x) or x <= 0 for x in equity):
        raise ValueError("positive finite equity required")
    returns = [b / a - 1 for a, b in zip(equity, equity[1:])]
    mu, sigma = mean(returns), stdev(returns)
    downside = sqrt(mean([min(r, 0)**2 for r in returns]))
    peak, mdd = equity[0], 0.0
    for value in equity:
        peak = max(peak, value)
        mdd = max(mdd, 1 - value / peak)
    annual = (equity[-1] / equity[0])**(periods / len(returns)) - 1
    ordered = sorted(returns)
    index = max(0, ceil(0.05 * len(ordered)) - 1)
    q05 = ordered[index]
    mass = 0.05 * len(ordered)
    whole = int(mass)
    tail_mean = (sum(ordered[:whole]) + (mass - whole) * ordered[whole]) / mass
    return {
        "sharpe": mu / sigma * sqrt(periods) if sigma else None,
        "sortino": mu / downside * sqrt(periods) if downside else None,
        "mdd": mdd,
        "calmar": annual / mdd if mdd else None,
        "var95_loss": -q05,
        "cvar95_loss": -tail_mean,
    }


def pnl_gini(contributions):
    if not contributions or any(not isfinite(x) or x < 0
                                for x in contributions):
        raise ValueError("nonnegative finite contributions required")
    total, n = sum(contributions), len(contributions)
    if total == 0:
        return None
    return sum(abs(a - b) for a in contributions
               for b in contributions) / (2 * n * total)


metrics = compute_risk_metrics([100, 80, 100])
assert abs(metrics["mdd"] - 0.2) < 1e-12
assert pnl_gini([10, 10]) == 0
assert pnl_gini([20, 0]) == 0.5
assert compute_risk_metrics([100, 100, 100])["sharpe"] is None
print({k: None if v is None else round(v, 4)
       for k, v in metrics.items()})
