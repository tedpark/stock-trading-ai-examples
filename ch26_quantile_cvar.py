"""26장 학습 예제 — 더미 데이터 전용. 원문: 26-distributional-rl-cvar.md"""
from math import ceil, isfinite


def cvar_size(quantiles, alpha=0.2, loss_budget=0.175):
    if (not quantiles or not 0 < alpha <= 1
            or not isfinite(loss_budget) or loss_budget <= 0
            or any(not isfinite(q) for q in quantiles)):
        raise ValueError("invalid quantiles or risk budget")
    ordered = sorted(quantiles)
    count = max(1, ceil(alpha * len(ordered)))
    cvar = sum(ordered[:count]) / count
    loss = max(0.0, -cvar)
    size = min(1.0, loss_budget / loss) if loss else 1.0
    return cvar, size


cvar, size = cvar_size([-0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5])
assert abs(cvar + 0.35) < 1e-12 and abs(size - 0.5) < 1e-12
print(round(cvar, 3), round(size, 3))
