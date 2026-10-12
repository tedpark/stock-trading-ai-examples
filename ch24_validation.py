"""24장 학습 예제 — 더미 데이터 전용. 원문: 24-production-validation.md"""
from math import isfinite


def validate(timestamp, probabilities, previous=None):
    if previous is not None and timestamp <= previous:
        return "duplicate_or_old", previous
    if (len(probabilities) != 3
            or any(not isfinite(p) or not 0 <= p <= 1
                   for p in probabilities)
            or abs(sum(probabilities) - 1) > 1e-8):
        return "invalid_probability", previous
    return "ok", timestamp


status, last = validate(1, [0.7, 0.2, 0.1])
duplicate, _ = validate(1, [0.7, 0.2, 0.1], last)
invalid, unchanged = validate(2, [float("nan"), 0.2, 0.1], last)
assert unchanged == last
assert validate(2, [0.7, 0.4, -0.1], last)[0] == "invalid_probability"
assert validate(2, [0.7, 0.4, 0.1], last)[0] == "invalid_probability"
print(status, duplicate, invalid)
