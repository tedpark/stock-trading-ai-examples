import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def configure_logging():
    logging.getLogger("quant.data").setLevel(logging.DEBUG)


def scale_training_only(train, test):
    """Fit statistics on training observations and reuse them for the holdout."""
    train, test = np.asarray(train, dtype=float), np.asarray(test, dtype=float)
    if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]:
        raise ValueError("matching two-dimensional feature arrays required")
    if not len(train) or not np.isfinite(train).all() or not np.isfinite(test).all():
        raise ValueError("finite observations required")
    scaler = StandardScaler().fit(train)
    return scaler.transform(train), scaler.transform(test), scaler


def discard_warmup(features):
    features = np.asarray(features, dtype=float)
    if features.ndim != 2 or np.isinf(features).any():
        raise ValueError("feature matrix required; only NaN may mark warmup")
    return features[np.isfinite(features).all(axis=1)]


def download_ohlcv(symbol, interval="1d", downloader=None):
    if downloader is None:
        import yfinance as yf
        downloader = yf.download
    df = downloader(symbol, period="5y", interval=interval,
                    auto_adjust=True, multi_level_index=False)
    if df.empty:
        raise ValueError("No bars returned")
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        if df.columns.nlevels != 2 or list(df.columns.get_level_values(1).unique()) != [symbol]:
            raise ValueError("Expected only the requested ticker")
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    required = ["open", "high", "low", "close", "volume"]
    if df.columns.duplicated().any() or not set(required) <= set(df.columns):
        raise ValueError("Missing or duplicate OHLCV columns")
    df.index = pd.to_datetime(df.index, utc=True)
    if df.index.hasnans or df.index.has_duplicates or not df.index.is_monotonic_increasing:
        raise ValueError("Expected unique, ordered bar timestamps")
    data = df[required].astype(float)
    if (not np.isfinite(data.to_numpy()).all() or (data.iloc[:, :4] <= 0).any().any()
            or (data.volume < 0).any() or (data.high < data.low).any()
            or (data.high < data[['open', 'close']].max(axis=1)).any()
            or (data.low > data[['open', 'close']].min(axis=1)).any()):
        raise ValueError("invalid OHLCV values")
    return data
