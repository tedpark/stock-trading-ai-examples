"""10장 학습 예제 — 더미 데이터 전용. 원문: 10-hmm-ensemble.md"""
import numpy as np
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


def net_returns(predictions, next_returns, threshold, fee=0.0005):
    position = np.where(predictions > threshold, 1.0,
                        np.where(predictions < -threshold, -1.0, 0.0))
    turnover = np.abs(np.diff(np.r_[0.0, position]))
    result = position * next_returns - fee * turnover
    result[-1] -= fee * abs(position[-1])
    return result


def evaluate_objective(returns):
    sigma = float(returns.std(ddof=1))
    return float(returns.mean() / sigma) if sigma > 0 else -999.0


rng = np.random.default_rng(42)
predictions = rng.normal(0, 0.003, 300)
next_returns = 0.2 * predictions + rng.normal(0, 0.008, 300)
# Features/predictions are available BEFORE the corresponding next return.
scale = float(predictions[:150].std(ddof=1))

def objective(trial):
    threshold = scale * trial.suggest_float("scale", 0.2, 2.0)
    return evaluate_objective(net_returns(
        predictions[150:225], next_returns[150:225], threshold))

study = optuna.create_study(
    direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=12)
threshold = scale * study.best_params["scale"]
holdout = net_returns(predictions[225:], next_returns[225:], threshold)
assert len(holdout) == 75 and np.isfinite(holdout).all()
print(study.best_params, "holdout_score", evaluate_objective(holdout))
