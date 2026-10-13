import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import coint
from .statistics import calc_halflife


def cointegration_pvalue(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or len(a) < 20 or not np.isfinite([a, b]).all():
        raise ValueError("aligned finite series of length >= 20 required")
    if np.std(a) == 0 or np.std(b) == 0:
        return 1.0
    return float(coint(a, b)[1])


def find_cointegrated_pairs(prices, sectors, min_correlation=0.4, alpha=0.05, max_halflife=60):
    """Apply BH once to every tested pair in the supplied training universe."""
    if not 0 < alpha < 1 or not -1 <= min_correlation <= 1 or max_halflife <= 0:
        raise ValueError("invalid selection thresholds")
    tested, seen = [], set()
    for symbols in sectors.values():
        for i, a in enumerate(symbols):
            for b in symbols[i+1:]:
                key = tuple(sorted((a, b)))
                if a == b or key in seen:
                    continue
                seen.add(key)
                pair = prices[[a, b]].dropna()
                if len(pair) < 20 or not np.isfinite(pair.to_numpy()).all() or (pair <= 0).any().any():
                    continue
                corr = float(pair.pct_change().dropna().corr().iloc[0, 1])
                if not np.isfinite(corr) or corr < min_correlation:
                    continue
                pvalue = cointegration_pvalue(pair[a], pair[b])
                if not np.isfinite(pvalue):
                    continue
                fit = sm.OLS(pair[a], sm.add_constant(pair[b])).fit()
                intercept, beta = fit.params
                spread = pair[a] - (intercept + beta * pair[b])
                tested.append(dict(symbols=(a, b), corr=corr, pvalue=pvalue,
                                   beta=float(beta), halflife=calc_halflife(spread)))
    if not tested:
        return []
    reject, qvalues, _, _ = multipletests([p['pvalue'] for p in tested], alpha=alpha, method='fdr_bh')
    return sorted([{**p, 'qvalue':float(q)} for p, keep, q in zip(tested, reject, qvalues)
                   if keep and np.isfinite(p['halflife']) and p['halflife'] <= max_halflife],
                  key=lambda p:(p['qvalue'], p['halflife']))


def log_spread(log_a, log_b, beta):
    a, b, beta = np.asarray(log_a), np.asarray(log_b), np.asarray(beta)
    if a.shape != b.shape or beta.shape not in [(), a.shape] or not all(np.isfinite(x).all() for x in (a,b,beta)):
        raise ValueError("aligned finite log prices and beta required")
    return a - beta * b


def holding_limit(correlation, spread, corr_min=0.4, hl_max=60, max_hold_days=30, hl_hold_mult=2):
    hl = calc_halflife(spread)
    if (not np.isfinite(correlation) or correlation < corr_min or not np.isfinite(hl)
            or hl > hl_max or min(max_hold_days, hl_hold_mult) <= 0):
        return None
    return max(1, min(max_hold_days, int(hl_hold_mult * hl)))
