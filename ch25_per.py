"""25장 학습 예제 — 더미 데이터 전용. 원문: 25-rl-algorithm-comparison.md"""
from math import isfinite


def replay_weights(errors, alpha=0.6, beta=0.4):
    if not errors or any(not isfinite(e) for e in errors):
        raise ValueError("finite TD errors required")
    priority = [(abs(e) + 1e-6)**alpha for e in errors]
    probabilities = [p / sum(priority) for p in priority]
    raw = [(len(errors) * p)**(-beta) for p in probabilities]
    weights = [w / max(raw) for w in raw]
    return probabilities, weights


probabilities, weights = replay_weights([0.1, 1.0, 2.0])
assert abs(sum(probabilities) - 1) < 1e-12
assert probabilities[0] < probabilities[1] < probabilities[2]
assert weights[0] > weights[1] > weights[2]
print([round(p, 3) for p in probabilities])
