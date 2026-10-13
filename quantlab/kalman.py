import numpy as np
from .statistics import _kalman_innovations, _build_z_series


class KalmanFilter1D:
    def __init__(self, Q=1e-4, R=1e-2):
        if not np.isfinite([Q, R]).all() or Q < 0 or R <= 0:
            raise ValueError("Q >= 0 and R > 0 required")
        self.Q, self.R, self.x, self.P = Q, R, 0.0, 1.0

    def update(self, z):
        if not np.isfinite(z):
            raise ValueError("finite observation required")
        self.P += self.Q
        gain = self.P / (self.P + self.R)
        self.x += gain * (z - self.x)
        self.P *= 1 - gain
        return self.x


def kalman_zscores(close, window=60, delta=1e-4, obs_noise=1.0):
    close = np.asarray(close, dtype=float)
    if close.ndim != 1 or not np.isfinite(close).all() or (close <= 0).any():
        raise ValueError("positive finite prices required")
    innovations = _kalman_innovations(np.log(close), delta, obs_noise)
    return _build_z_series(innovations, window)
