"""Chapter 19 learning example; dummy data only. Source: 19-bamberger-pairs-theory.md"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

def fit_and_test_pair(price_a, price_b):
    pair = pd.concat(
        [price_a.rename("a"), price_b.rename("b")], axis=1
    ).dropna()

    # price_a = alpha + beta * price_b + ε
    X = sm.add_constant(pair["b"])
    fit = sm.OLS(pair["a"], X).fit()
    alpha, beta = fit.params
    spread = pair["a"] - (alpha + beta * pair["b"])

    # MacKinnon approximate p-value for the Engle-Granger test
    eg_stat, eg_pvalue, critical_values = coint(pair["a"], pair["b"])
    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "spread": spread,
        "eg_stat": float(eg_stat),
        "eg_pvalue": float(eg_pvalue),
        "critical_values": critical_values,
        "is_candidate": eg_pvalue < 0.05,
    }

rng = np.random.default_rng(42)
common = np.cumsum(rng.normal(size=300))
price_b = pd.Series(100 + common)
price_a = pd.Series(5 + 1.2 * price_b + rng.normal(0, 0.3, 300))
result = fit_and_test_pair(price_a, price_b)
assert abs(result["beta"] - 1.2) < 0.1
assert result["eg_pvalue"] < 0.05
print(round(result["beta"], 3), result["is_candidate"])
