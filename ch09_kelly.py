"""Chapter 09 learning example; dummy data only. Source: 09-kelly-sizing.md"""
from math import isfinite


def kelly_fraction(p, odds, fraction=0.5, cap=1.0):
    if (not all(isfinite(x) for x in (p, odds, fraction, cap))
            or not 0 <= p <= 1 or odds <= 0
            or not 0 <= fraction <= 1 or not 0 <= cap <= 1):
        raise ValueError("invalid Kelly inputs")
    raw = p - (1 - p) / odds
    return min(cap, max(0.0, fraction * raw))


assert abs(kelly_fraction(0.55, 1) - 0.05) < 1e-12
assert kelly_fraction(0.45, 1) == 0
print(round(kelly_fraction(0.55, 1), 3))
