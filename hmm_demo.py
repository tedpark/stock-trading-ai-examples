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
    """Return uses log(close[t]/close[t-1]); it is not an arithmetic return."""
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
    # Match the book: rank total variance across standardized features.
    # This is not a ranking of return volatility alone or proof of mean reversion.
    order = np.argsort([np.trace(cov) for cov in model.covars_])
    return model, scaler, order


def forward(model, x):
    """Filter in log space without future observations. Columns retain the original state order."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError('nonempty finite feature matrix required')
    emissions = np.column_stack([
        multivariate_normal.logpdf(x, mean=mean, cov=cov)
        for mean, cov in zip(model.means_, model.covars_)])
    transition_probs = np.asarray(model.transmat_, dtype=float)
    start_probs = np.asarray(model.startprob_, dtype=float)
    n_states = len(model.means_)
    if (transition_probs.shape != (n_states, n_states) or start_probs.shape != (n_states,)
            or not np.isfinite(transition_probs).all() or not np.isfinite(start_probs).all()
            or (transition_probs < 0).any() or (start_probs < 0).any()
            or not np.allclose(transition_probs.sum(axis=1), 1)
            or not np.isclose(start_probs.sum(), 1)):
        raise ValueError('normalized nonnegative HMM probabilities required')
    # A structural zero means impossible, not a tiny positive probability.
    with np.errstate(divide='ignore'):
        transition = np.log(transition_probs)
        log_alpha = np.log(start_probs)
    result = []
    for t, emission in enumerate(emissions):
        if t:
            log_alpha = logsumexp(log_alpha[:, None] + transition, axis=0)
        log_alpha += emission
        normalizer = logsumexp(log_alpha)
        if not np.isfinite(normalizer):
            raise ValueError('observation has no finite likelihood in reachable states')
        log_alpha -= normalizer
        result.append(np.exp(log_alpha))
    return np.asarray(result)
