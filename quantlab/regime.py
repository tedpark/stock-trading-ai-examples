import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from hmm_demo import fit_model, forward


def rank_states(model):
    """Covariance trace ranks standardized feature variance, not proven regimes."""
    return np.argsort([np.trace(cov) for cov in model.covars_])


class DailyHMM:
    def __init__(self):
        self.model = self.scaler = self.order = self.alpha = None

    def fit(self, features):
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or np.isinf(x).any():
            raise ValueError("finite feature matrix required; NaN warmup allowed")
        x = x[np.isfinite(x).all(axis=1)]
        if len(x) < 60:
            return False
        model, scaler, order = fit_model(x, len(x))
        posterior = forward(model, scaler.transform(x))[-1]
        self.model, self.scaler, self.order, self.alpha = model, scaler, order, posterior
        return True

    def predict_proba(self, features):
        if self.model is None:
            raise RuntimeError("fit first")
        return forward(self.model, self.scaler.transform(features))[:, self.order]

    def step_forward(self, feat_row):
        if self.model is None:
            raise RuntimeError("fit first")
        row = np.asarray(feat_row, dtype=float)
        if row.shape != self.model.means_.shape[1:] or not np.isfinite(row).all():
            raise ValueError("one finite feature row required")
        x = self.scaler.transform(row.reshape(1, -1))[0]
        emissions = np.array([multivariate_normal.logpdf(x, mean=mu, cov=cov)
                              for mu, cov in zip(self.model.means_, self.model.covars_)])
        with np.errstate(divide="ignore"):
            log_alpha = np.log(self.alpha @ self.model.transmat_) + emissions
        normalizer = logsumexp(log_alpha)
        if not np.isfinite(normalizer):
            return None
        self.alpha = np.exp(log_alpha - normalizer)
        return self.alpha[self.order].copy()


def walk_forward(features, train_bars=60, refit_days=20):
    """Refit strictly before t, then consume observation t once."""
    if train_bars < 60 or refit_days < 1:
        raise ValueError("at least 60 training bars and positive refit interval required")
    model = DailyHMM()
    result = []
    for t in range(train_bars, len(features)):
        if (t - train_bars) % refit_days == 0:
            if not model.fit(features[:t]):
                raise ValueError("insufficient training observations")
        result.append(model.step_forward(features[t]))
    return result


def regime_signal(probabilities, z, entry_z=1.5, short_entry_z=2.0):
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (3,) or not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(), 1):
        raise ValueError("three normalized probabilities required")
    if not np.isfinite([z, entry_z, short_entry_z]).all() or min(entry_z, short_entry_z) <= 0:
        raise ValueError("finite z and positive entry thresholds required")
    if p[0] < 0.3 or p[2] > 0.4:
        return 0
    return 1 if z < -entry_z else (-1 if z > short_entry_z else 0)


def regime_multiplier(probabilities, multipliers=(1.5, 1.0, 0.5)):
    p, m = np.asarray(probabilities, dtype=float), np.asarray(multipliers, dtype=float)
    if (p.ndim != 1 or p.shape != m.shape or not np.isfinite(p).all()
            or not np.isfinite(m).all() or (p < 0).any() or not np.isclose(p.sum(), 1)):
        raise ValueError("normalized probabilities and matching finite multipliers required")
    return float(np.clip(p @ m, 0, 2))


def blend_votes(forecast, regime_scale, long_votes, total_votes, vote_threshold=0.6):
    if (not np.isfinite([forecast, regime_scale, long_votes, total_votes, vote_threshold]).all()
            or not 0 <= long_votes <= total_votes or not 0.5 <= vote_threshold <= 1
            or regime_scale < 0):
        raise ValueError("invalid forecast or votes")
    ratio = long_votes / total_votes if total_votes else 0.5
    return 0.0 if 1 - vote_threshold < ratio < vote_threshold else forecast * regime_scale


def state_return_means(scaler, model, return_index=0):
    return scaler.inverse_transform(model.means_)[:, return_index]


def pipeline_direction(alpha, z, confidence=0.65, entry_z=1.5):
    p = np.asarray(alpha, dtype=float)
    if p.shape != (3,) or not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(), 1):
        raise ValueError("three normalized probabilities required")
    if not np.isfinite([z, confidence, entry_z]).all() or not 0 <= confidence <= 1 or entry_z <= 0:
        raise ValueError("valid thresholds required")
    if p.max() < confidence:
        return 0
    state = int(p.argmax())
    if state == 0:
        return 1 if z < -entry_z else (-1 if z > entry_z else 0)
    if state == 2:
        return 1 if z > 1 else (-1 if z < -1 else 0)
    return 0
