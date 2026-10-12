"""Chapter 05 learning example; dummy data only. Source: 05-hmm-forward-only.md"""
from math import exp, pi, sqrt


def forward_step(prior, observation, transition, means, variances):
    if len(prior) != len(means) or len(means) != len(variances):
        raise ValueError("state dimensions differ")
    predicted = [sum(prior[i] * transition[i][j]
                     for i in range(len(prior)))
                 for j in range(len(prior))]
    weights = [p * exp(-0.5 * (observation - m)**2 / v)
               / sqrt(2 * pi * v)
               for p, m, v in zip(predicted, means, variances)]
    total = sum(weights)
    if total <= 0:
        raise ValueError("likelihood underflow; use log space")
    return [w / total for w in weights]


transition = [[0.95, 0.05], [0.10, 0.90]]
means, variances = [0.0, 2.0], [1.0, 1.0]

def filter_prefix(observations):
    posterior = [0.5, 0.5]
    result = []
    for value in observations:
        posterior = forward_step(
            posterior, value, transition, means, variances)
        result.append(posterior)
    return result

first = filter_prefix([0.0, 0.2, 2.0])
extended = filter_prefix([0.0, 0.2, 2.0, -4.0])
assert first == extended[:3]
assert all(abs(sum(p) - 1) < 1e-12 for p in first)
print([[round(p, 3) for p in row] for row in first])
