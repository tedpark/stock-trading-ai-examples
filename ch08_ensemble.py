"""Chapter 08 learning example; dummy data only. Source: 08-two-layer-hmm.md"""
from math import isfinite


def combine_two_layers(probabilities, models, half_kelly=0.5):
    """Blend active models and apply fractional Kelly with equal win/loss odds."""
    if not isfinite(half_kelly) or not 0 <= half_kelly <= 1:
        raise ValueError("Kelly multiplier must be between zero and one")
    if (not probabilities or any(not isfinite(p) or p < 0
                                for p in probabilities)
            or abs(sum(probabilities) - 1) > 1e-8):
        raise ValueError("invalid regime probabilities")
    dominant = max(range(len(probabilities)), key=probabilities.__getitem__)
    active = [m for m in models if dominant in m["regimes"]]
    if not active:
        return 0.0, 0.0
    if any(not isfinite(m["weight"]) or not 0 <= m["p_up"] <= 1 or m["weight"] < 0
           for m in active):
        raise ValueError("invalid model probability or weight")
    scale = max(m["weight"] for m in active)
    if scale == 0:
        return 0.0, 0.0
    weights = [m["weight"] / scale for m in active]
    p_up = sum(m["p_up"] * w for m, w in zip(active, weights)) / sum(weights)
    size = min(1.0, max(0.0, 2 * p_up - 1) * half_kelly)
    return p_up, size


models = [{"regimes": [0], "p_up": 0.6, "weight": 1.0},
          {"regimes": [0, 1], "p_up": 0.8, "weight": 1.0}]
p_up, size = combine_two_layers([0.9, 0.1], models)
assert abs(p_up - 0.7) < 1e-12 and abs(size - 0.2) < 1e-12
assert combine_two_layers([0.9, 0.1], []) == (0.0, 0.0)
print(round(p_up, 3), round(size, 3))
