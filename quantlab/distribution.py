import numpy as np
import torch
from torch import nn


class QuantileNet(nn.Module):
    def __init__(self, state_dim=28, n_actions=3, n_quantiles=51, hidden=256):
        super().__init__()
        self.n_actions = n_actions
        self.n_quantiles = n_quantiles
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions * n_quantiles),
        )

    def forward(self, state):
        logits = self.net(state)
        return logits.view(-1, self.n_actions, self.n_quantiles)


def cvar_position_scaler(cvar, config):
    """Map Q-tail reward to [size_floor, 1]; all thresholds use reward units."""
    values = [cvar, config.cvar_target, config.cvar_scale, config.size_floor]
    if (not np.isfinite(values).all() or config.cvar_scale <= 0
            or not 0 <= config.size_floor <= 1):
        raise ValueError("finite CVaR and valid scaling configuration required")
    excess = max(0.0, config.cvar_target - cvar)
    return max(config.size_floor, 1.0 - excess / config.cvar_scale)


class CVaRSizer:
    def __init__(self, agent, config):
        self.agent = agent
        self.config = config
        self.last_cvar = None

    def scale(self, state, action):
        # This signal hook accepts one state, not a batch of different trades.
        value = self.agent.cvar(state, action, alpha=0.05)
        if value.numel() != 1:
            raise ValueError("one state required")
        self.last_cvar = float(value.item())
        return cvar_position_scaler(self.last_cvar, self.config)


def apply_cvar_to_signal(signal, sizer, state):
    # Signal values: EXIT=-1, HOLD=0, ENTER=1.
    # Network columns: EXIT=0, HOLD=1, ENTER=2.
    action_to_index = {-1: 0, 0: 1, 1: 2}
    if signal.action not in action_to_index:
        raise ValueError("unknown trading action")
    proposed_scale = sizer.scale(state, action_to_index[signal.action])
    # Scale new entry size; never reduce a requested exit or alter HOLD.
    applied_scale = proposed_scale if signal.action == 1 else 1.0
    signal.size *= applied_scale
    signal.metadata["cvar_5pct"] = sizer.last_cvar
    signal.metadata["cvar_scale"] = applied_scale
    return signal

class QuantileAgent:
    def __init__(self, net):
        self.net = net.eval()

    def act(self, state):
        with torch.no_grad():
            q_dist = self.net(state)              # (B, 3, 51)
            q_expectation = q_dist.mean(dim=-1)   # (B, 3)
            return q_expectation.argmax(dim=-1)

    def cvar(self, state, action, alpha=0.05):
        """Return the lower tail of equal-mass Q atoms for one shared action."""
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        with torch.no_grad():
            q_dist = self.net(state)
            if not isinstance(action, int) or not 0 <= action < q_dist.shape[1]:
                raise ValueError("one valid action index required")
            quantiles = q_dist[:, action, :]
            if not torch.isfinite(quantiles).all() or quantiles.shape[-1] == 0:
                raise ValueError("finite nonempty Q atoms required")
            ordered = quantiles.sort(dim=-1).values
            mass = alpha * ordered.shape[-1]
            if mass <= 1:
                return ordered[:, 0]
            whole = int(mass)
            included = ordered[:, :whole + (mass > whole)]
            scale = included.abs().amax(dim=-1)
            divisor = torch.where(scale > 0, scale, torch.ones_like(scale))
            # Normalize first so a finite mean cannot overflow its partial sum.
            tail = ((ordered[:, :whole] / divisor[:, None]) / mass).sum(dim=-1)
            if mass > whole:
                tail = tail + ((mass - whole) / mass) * (ordered[:, whole] / divisor)
            return tail.clamp(-1.0, 1.0) * scale
