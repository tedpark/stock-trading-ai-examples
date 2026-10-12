"""Chapter 02 learning example; dummy data only. Source: 02-data-pipeline.md"""
import numpy as np

def compute_daily_features(close, volume):
    close = np.asarray(close, dtype=float)
    volume = np.asarray(volume, dtype=float)
    if close.ndim != 1 or volume.shape != close.shape:
        raise ValueError("equal-length 1D arrays required")
    if not np.all(np.isfinite(close)) or np.any(close <= 0):
        raise ValueError("close must be finite and positive")
    if not np.all(np.isfinite(volume)) or np.any(volume < 0):
        raise ValueError("volume must be finite and nonnegative")
    n = len(close)
    feat = np.full((n, 4), np.nan)
    ret = np.full(n, np.nan)
    ret[1:] = np.diff(np.log(close))
    feat[:, 0] = ret
    for t in range(n):
        if t >= 5:
            feat[t, 1] = np.std(ret[t-4:t+1], ddof=1)
        if t >= 20:
            feat[t, 2] = close[t] / close[t-20] - 1
            base = np.mean(volume[t-20:t])
            if base > 0:
                feat[t, 3] = volume[t] / base
    return feat

features = compute_daily_features([100.0] * 25, [10.0] * 25)
assert features.shape == (25, 4)
assert np.allclose(features[20], [0.0, 0.0, 0.0, 1.0])
assert np.isfinite(features).all(axis=1).sum() == 5

# Fit scaling statistics on training rows, then reuse them unchanged for OOS rows.
close = 100 * np.exp(np.arange(60) * 0.001)
volume = np.full(60, 10.0)
features = compute_daily_features(close, volume)
X = features[np.isfinite(features).all(axis=1)]
cut = int(len(X) * 0.7)
X_train, X_oos = X[:cut], X[cut:]
mean = X_train.mean(axis=0)
scale = X_train.std(axis=0)
scale[scale < 1e-12] = 1.0
train_scaled = (X_train - mean) / scale
oos_scaled = (X_oos - mean) / scale
print(train_scaled.shape, oos_scaled.shape)
assert (len(X_train), len(X_oos)) == (28, 12)
