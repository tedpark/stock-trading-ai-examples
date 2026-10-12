"""Chapter 25 learning example; dummy data only. Source: 25-rl-algorithm-comparison.md"""
from math import isfinite


def replay_weights(errors, alpha=0.6, beta=0.4):
    """Return probabilities and weights normalized over the entire toy buffer.

    Alpha controls priority strength; beta controls importance correction.
    A sampled minibatch may use a different maximum-weight normalization.
    """
    if not all(isfinite(x) and 0 <= x <= 1 for x in (alpha, beta)):
        raise ValueError("alpha and beta must be between zero and one")
    if not errors or any(not isfinite(e) for e in errors):
        raise ValueError("finite TD errors required")
    priority = [(abs(e) + 1e-6)**alpha for e in errors]
    total = sum(priority)
    probabilities = [p / total for p in priority]
    raw = [(len(errors) * p)**(-beta) for p in probabilities]
    maximum = max(raw)
    weights = [w / maximum for w in raw]
    return probabilities, weights


probabilities, weights = replay_weights([0.1, 1.0, 2.0])
assert abs(sum(probabilities) - 1) < 1e-12
assert probabilities[0] < probabilities[1] < probabilities[2]
assert weights[0] > weights[1] > weights[2]
print([round(p, 3) for p in probabilities])
