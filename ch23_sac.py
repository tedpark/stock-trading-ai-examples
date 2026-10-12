"""Chapter 23 learning example; dummy data only. Source: 23-sac-rl-preview.md"""
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from tiny_env import TinyTradingEnv

rng = np.random.default_rng(7)
train_returns = rng.normal(0, 0.01, 80)
test_returns = rng.normal(0, 0.01, 80)
train_env = TinyTradingEnv(train_returns)
check_env(train_env)
model = SAC("MlpPolicy", train_env, seed=7, device="cpu",
            learning_starts=32, batch_size=16, buffer_size=500,
            policy_kwargs={"net_arch": [16, 16]}, verbose=0)
model.learn(total_timesteps=128)
test_env = TinyTradingEnv(test_returns)
obs, _ = test_env.reset(seed=8)
done, total, steps = False, 0.0, 0
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = test_env.step(action)
    done = terminated or truncated
    total += reward
    steps += 1
assert steps == 70 and np.isfinite(total)
print("evaluation_steps", steps, "reward_sum", round(total, 6))
