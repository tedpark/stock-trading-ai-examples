"""Chapter 13 learning example; dummy data only. Source: 13-rl-environment.md"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class TinyTradingEnv(gym.Env):
    def __init__(self, returns, fee=0.001):
        super().__init__()
        self.returns = np.asarray(returns, dtype=np.float64)
        if (self.returns.ndim != 1 or len(self.returns) <= 10
                or not np.isfinite(self.returns).all()
                or np.any(np.abs(self.returns) >= 0.5)):
            raise ValueError("more than 10 finite toy returns required")
        if not 0 <= fee < 0.1:
            raise ValueError("invalid fee")
        self.fee = fee
        self.action_space = spaces.Box(-1.0, 1.0, (1,), np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, (11,), np.float32)

    def _observation(self):
        return np.r_[self.returns[self.t-10:self.t],
                     self.position].astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t, self.position, self.done = 10, 0.0, False
        return self._observation(), {}

    def step(self, action):
        if self.done:
            raise RuntimeError("reset after termination")
        action = np.asarray(action, dtype=float)
        if action.shape != (1,) or not np.isfinite(action).all():
            raise ValueError("one finite action required")
        target = float(np.clip(action[0], -1, 1))
        reward = (target * self.returns[self.t]
                  - self.fee * abs(target - self.position))
        self.position = target
        self.t += 1
        self.done = self.t == len(self.returns)
        if self.done:
            reward -= self.fee * abs(self.position)
            self.position = 0.0
        return self._observation(), float(reward), self.done, False, {}


if __name__ == "__main__":
    env = TinyTradingEnv([0.0] * 12)
    observation, _ = env.reset(seed=7)
    _, r1, done, _, _ = env.step([1.0])
    _, r2, done, _, _ = env.step([1.0])
    assert done and abs(r1 + r2 + 0.002) < 1e-12
    print(observation.shape, round(r1 + r2, 3), done)
