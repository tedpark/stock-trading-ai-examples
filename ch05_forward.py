"""Chapter 05 learning example; dummy data only. Source: 05-hmm-forward-only.md"""
from math import exp, fsum, isfinite, log, pi


def forward_step(prior, observation, transition, means, variances):
    """Apply one transition then one emission update to a normalized prior."""
    n = len(prior)
    if (not n or len(means) != n or len(variances) != n
            or len(transition) != n or any(len(row) != n for row in transition)):
        raise ValueError("state dimensions differ")
    if (not isfinite(observation) or any(not isfinite(m) for m in means)
            or any(not isfinite(v) or v <= 0 for v in variances)):
        raise ValueError("finite observations/means and positive variances required")
    for probabilities in [prior, *transition]:
        if (any(not isfinite(p) or not 0 <= p <= 1 for p in probabilities)
                or abs(fsum(probabilities) - 1) > 1e-8):
            raise ValueError("normalized nonnegative probabilities required")
    predicted = [fsum(prior[i] * transition[i][j] for i in range(n))
                 for j in range(n)]
    # Log weights avoid density underflow; structural zeros remain impossible.
    try:
        weights = [log(p) - .5 * (log(2 * pi) + log(v)
                   + (observation - m)**2 / v) if p > 0 else -float("inf")
                   for p, m, v in zip(predicted, means, variances)]
    except OverflowError as exc:
        raise ValueError("observation exceeds the numeric range") from exc
    maximum = max(weights)
    if not isfinite(maximum):
        raise ValueError("no finite likelihood in reachable states")
    weights = [exp(w - maximum) for w in weights]
    total = fsum(weights)
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
