"""Chapter 18 learning example; dummy data only. Source: 18-monitoring.md"""
import json
from math import ceil, isfinite, sqrt
from statistics import mean, stdev


def monitor(pnl, starting_equity=1000.0, mdd_limit=0.08):
    if (not pnl or not isfinite(starting_equity)
            or starting_equity <= 0
            or any(not isfinite(x) for x in pnl)):
        raise ValueError("finite PnL and positive starting equity required")
    equity, peak, mdd, returns = starting_equity, starting_equity, 0.0, []
    for amount in pnl:
        returns.append(amount / equity)
        equity += amount
        if equity <= 0:
            raise ValueError("insolvent equity path")
        peak = max(peak, equity)
        mdd = max(mdd, 1 - equity / peak)
    sigma = stdev(returns) if len(returns) > 1 else 0.0
    score = mean(returns) / sigma * sqrt(252) if sigma else None
    violations = ["mdd"] if mdd > mdd_limit else []
    if score is None:
        violations.append("sharpe_undefined")
    return {"status": "warn" if violations else "ok",
            "violations": violations,
            "metrics": {"sharpe": score, "mdd": mdd,
                        "positive_day_rate": sum(x > 0 for x in pnl) / len(pnl),
                        "total_pnl": sum(pnl)}}


if __name__ == "__main__":
    result = monitor([-200, 200])
    assert result["status"] == "warn"
    assert abs(result["metrics"]["mdd"] - 0.2) < 1e-12
    print(json.dumps(result, allow_nan=False, indent=2))
