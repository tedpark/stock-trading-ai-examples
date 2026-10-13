import threading
import numpy as np
import optuna
from hmm_demo import fit_model, forward
from tiny_env import TinyTradingEnv

# source-region: actions
ACTION_HOLD = 0
ACTION_ENTER = 1
ACTION_EXIT = -1
# end-source-region: actions


def model(today_features):
    """Deterministic teaching rule, not a learned price predictor."""
    row = np.asarray(today_features, dtype=float)
    if row.ndim != 1 or not len(row) or not np.isfinite(row).all():
        raise ValueError("finite feature vector required")
    return "hold" if row[0] >= 0 else "cash"


def decision_demo():
    return [model([0.01]), model([-0.01])]


def make_environment(returns, fee=0.001):
    return TinyTradingEnv(returns, fee=fee)


def rollout(env, agent, seed=7):
    obs, info = env.reset(seed=seed)
    rewards = []
    done = False
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        done = terminated or truncated
    return float(sum(rewards))


def load_and_act(path, state):
    from stable_baselines3 import SAC
    agent = SAC.load(path, device="cpu")
    state = np.asarray(state, dtype=np.float32)
    if state.shape != agent.observation_space.shape or not np.isfinite(state).all():
        raise ValueError("state must match the checkpoint observation schema")
    action, _ = agent.predict(state, deterministic=True)
    return action


def threshold_objective(trial, forecasts, next_returns, std_e, val_end, fee_rate=0.001):
    """Tune only the explicitly supplied validation prefix; no held-out returns."""
    f, r = np.asarray(forecasts, dtype=float), np.asarray(next_returns, dtype=float)
    if (f.ndim != 1 or f.shape != r.shape or not isinstance(val_end, int)
            or not 0 < val_end <= len(f) or not np.isfinite([std_e, fee_rate]).all()
            or std_e <= 0 or fee_rate < 0 or not np.isfinite(f[:val_end]).all()
            or not np.isfinite(r[:val_end]).all()):
        raise ValueError("aligned validation arrays and valid costs required")
    scale = trial.suggest_float("scale", 0.15, 2.5)
    thresh = std_e * scale
    signals = np.where(f[:val_end] > thresh, 1.0, np.where(f[:val_end] < -thresh, -1.0, 0.0))
    changes = np.abs(np.diff(np.r_[0.0, signals]))
    changes[-1] += abs(signals[-1])
    return float(np.sum(signals * r[:val_end] - fee_rate * changes))


def vote_objective(trial, vote_ratios, next_returns, fee_rate=0.001, periods_per_year=252):
    votes, returns = np.asarray(vote_ratios, dtype=float), np.asarray(next_returns, dtype=float)
    if (votes.ndim != 1 or len(votes) < 2 or votes.shape != returns.shape
            or not np.isfinite(votes).all() or not np.isfinite(returns).all()
            or ((votes < 0) | (votes > 1)).any()
            or not np.isfinite([fee_rate, periods_per_year]).all() or fee_rate < 0 or periods_per_year <= 0):
        raise ValueError("aligned votes and next returns required")
    threshold = trial.suggest_float("vote_thresh", 0.50, 0.85)
    signals = np.where(votes > threshold, 1., np.where(votes < 1-threshold, -1., 0.))
    changes = np.abs(np.diff(np.r_[0., signals]))
    changes[-1] += abs(signals[-1])
    net = signals * returns - fee_rate * changes
    downside = np.sqrt(np.mean(np.minimum(net, 0.) ** 2))
    if downside == 0:
        return -float('inf')
    return float(net.mean() / downside * np.sqrt(periods_per_year)
                 - max(0., (changes.mean() - 0.05) * 3))


def create_study():
    return optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))


class SafeSignal:
    """Fit outside the lock, then atomically publish a complete model/scaler pair."""
    def __init__(self):
        self._state = None
        self._lock = threading.RLock()

    def retrain(self, features):
        state = fit_model(features, len(features))
        with self._lock:
            self._state = state

    def predict(self, features):
        with self._lock:
            state = self._state
        if state is None:
            raise RuntimeError("fit before prediction")
        model, scaler, order = state
        return forward(model, scaler.transform(features))[:, order]
