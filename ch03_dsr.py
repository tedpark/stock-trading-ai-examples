"""03장 학습 예제 — 더미 데이터 전용. 원문: 03-backtest-engine.md"""
from math import sqrt
from statistics import NormalDist


def dsr_probability(sr, n, trials, trial_sr_std,
                    skew=0.0, excess_kurtosis=0.0):
    if n < 3 or trials < 1 or trial_sr_std < 0:
        raise ValueError("invalid sample or trial inputs")
    normal = NormalDist()
    benchmark = 0.0
    if trials > 1:
        gamma = 0.5772156649
        benchmark = trial_sr_std * (
            (1 - gamma) * normal.inv_cdf(1 - 1 / trials)
            + gamma * normal.inv_cdf(1 - 1 / (trials * 2.718281828)))
    variance = 1 - skew * sr + (excess_kurtosis + 2) * sr**2 / 4
    if variance <= 0:
        raise ValueError("invalid Sharpe variance estimate")
    z = (sr - benchmark) * sqrt(n - 1) / sqrt(variance)
    return normal.cdf(z)


one = dsr_probability(0.10, 252, 1, 0.05)
many = dsr_probability(0.10, 252, 100, 0.05)
assert many < one
print(round(one, 3), round(many, 3))
