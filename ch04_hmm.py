"""Chapter 04: fit the HMM on training data only and order states by variance."""
import numpy as np
from hmm_demo import dummy_bars, features, fit_model, forward

x = features(dummy_bars()).to_numpy()
train_len = int(len(x) * 0.6)
model, scaler, order = fit_model(x, train_len)
probabilities = forward(model, scaler.transform(x))[:, order]
assert np.allclose(probabilities.sum(axis=1), 1)
print('state order (low / middle / high variance):', order)
print('latest probabilities:', probabilities[-1].round(4))
print('train / evaluation:', train_len, len(x) - train_len)
# Variance ranks label states; they do not establish mean reversion or trends.
