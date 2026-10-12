"""Shared by Chapters 04 and 22: dummy closes/volume, training-only scaling, and forward filtering."""
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler


def dummy_bars(n=360, seed=42):
    rng = np.random.default_rng(seed)
    volatility = np.tile(np.repeat([0.004, 0.012, 0.025], 40), (n + 119) // 120)[:n]
    close = 100 * np.exp(np.cumsum(rng.normal(0, volatility)))
    return pd.DataFrame({'close': close, 'volume': rng.integers(1000, 5000, n)},
                        index=pd.date_range('2020-01-01', periods=n, freq='B'))


def features(bars):
    close, volume = bars.close, bars.volume
    returns = np.log(close / close.shift())
    return pd.DataFrame({
        'return': returns,
        'volatility': returns.rolling(5).std() * np.sqrt(252),
        'momentum': close / close.rolling(20).mean() - 1,
        'volume': np.log(volume / volume.rolling(20).mean()),
    }).dropna()


def fit_model(x, train_len):
    scaler = StandardScaler().fit(x[:train_len])
    model = GaussianHMM(n_components=3, covariance_type='full',
                        n_iter=100, random_state=42, min_covar=0.001)
    model.fit(scaler.transform(x[:train_len]))
    order = np.argsort([np.trace(cov) for cov in model.covars_])
    return model, scaler, order


def forward(model, x):
    """Filter in log space without future observations. Columns retain the original state order."""
    emissions = np.column_stack([
        multivariate_normal.logpdf(x, mean=mean, cov=cov)
        for mean, cov in zip(model.means_, model.covars_)])
    transition = np.log(np.maximum(model.transmat_, 1e-300))
    log_alpha = np.log(np.maximum(model.startprob_, 1e-300))
    result = []
    for t, emission in enumerate(emissions):
        if t:
            log_alpha = logsumexp(log_alpha[:, None] + transition, axis=0)
        log_alpha += emission
        log_alpha -= logsumexp(log_alpha)
        result.append(np.exp(log_alpha))
    return np.asarray(result)
