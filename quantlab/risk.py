import numpy as np
from dataclasses import dataclass, field



# source-region: kelly
KELLY_MIN, KELLY_MAX = 0.10, 1.0
THRESHOLD = 0.53  # Positive-edge normalization reference, not an entry gate.

def calculate_kelly_size(
    p_up, direction,
    vol_mult=1.0, macro_gate=1.0, novelty_gate=1.0,
) -> float:
    """Return a normalized size allowance; the caller supplies an exposure budget."""
    if not np.isfinite(p_up) or not 0 <= p_up <= 1 or direction not in (-1, 1):
        raise ValueError("probability in [0, 1] and direction -1 or 1 required")
    edge = (p_up - 0.5) if direction > 0 else ((1.0 - p_up) - 0.5)
    if edge <= 0:
        return 0.0


    f = edge


    if not np.isfinite([vol_mult, macro_gate, novelty_gate]).all():
        return 0.0
    if min(vol_mult, macro_gate, novelty_gate) <= 0:
        return 0.0
    f *= vol_mult * macro_gate * novelty_gate



    edge_max = max(THRESHOLD - 0.5, 1e-6)
    return float(np.clip(f / edge_max, KELLY_MIN, KELLY_MAX))


# end-source-region: kelly

# source-region: trailing

def _get_tiered_trailing_stop(self, pnl_pct: float) -> float:
    if pnl_pct >= 0.10:    
        return -0.03
    elif pnl_pct >= 0.03:  # +3% ~ +10% → -2%
        return -0.02
    else:                  
        return -0.01


def ratchet_long_stop(previous_stop, high_watermark, distance):
    """Keep a long-position stop from falling when a tier widens its distance."""
    if (not np.isfinite([high_watermark, distance]).all()
            or high_watermark <= 0 or not -1 < distance < 0
            or (previous_stop is not None
                and (not np.isfinite(previous_stop) or previous_stop <= 0))):
        raise ValueError("positive prices and a stop distance in (-1, 0) required")
    candidate = high_watermark * (1 + distance)
    return candidate if previous_stop is None else max(previous_stop, candidate)

# end-source-region: trailing


@dataclass
class Signal:
    action: int
    size: float = 1.0
    metadata: dict = field(default_factory=dict)


def hard_stop(pnl_pct, threshold=-0.02):
    if not np.isfinite([pnl_pct, threshold]).all() or not -1 < threshold < 0:
        raise ValueError("finite P&L and negative fractional threshold required")
    if pnl_pct <= threshold:
        return Signal(-1, metadata={"sell_type": "hard_stop"})
    return None


def atr_stop_loss(atr, entry_price):
    if not np.isfinite([atr, entry_price]).all() or atr < 0 or entry_price <= 0:
        raise ValueError("nonnegative ATR and positive entry price required")
    return -min(max(1.5 * atr / entry_price, 0.02), 0.05)


def drawdowns(equity_curve):
    equity = np.asarray(equity_curve, dtype=float)
    if equity.ndim != 1 or not len(equity) or not np.isfinite(equity).all() or (equity <= 0).any():
        raise ValueError("positive finite equity required")
    drawdown = equity / np.maximum.accumulate(equity) - 1
    return drawdown, float(-drawdown.min())


def return_tail(bar_returns, alpha=0.05):
    ordered = np.sort(np.asarray(bar_returns, dtype=float))
    if ordered.ndim != 1 or not len(ordered) or not np.isfinite(ordered).all() or not 0 < alpha <= 1:
        raise ValueError("finite return vector and tail mass in (0,1] required")
    mass = alpha * len(ordered)
    var = ordered[max(0, int(np.ceil(mass)) - 1)]
    if mass <= 1:
        cvar = ordered[0]
    else:
        whole = int(mass)
        # Scale the included atoms before averaging to keep finite tails finite.
        scale = np.max(np.abs(ordered[:whole + (mass > whole)]))
        if scale == 0:
            cvar = 0.0
        else:
            terms = (ordered[:whole] / scale) / mass
            total = float(np.sum(terms))
            if mass > whole:
                total += ((mass - whole) / mass) * (ordered[whole] / scale)
            cvar = np.clip(total, -1.0, 1.0) * scale
    return float(var), float(cvar)


class HighWatermarks:
    def __init__(self, collection=None):
        self.values = {}
        self.collection = collection

    def save(self, symbol, price):
        if not symbol or not np.isfinite(price) or price <= 0:
            raise ValueError("symbol and positive price required")
        value = max(self.values.get(symbol, price), price)
        # Persist first: failed persistence must not be reported as durable success.
        if self.collection is not None:
            self.collection.update_one({"symbol": symbol},
                {"$max": {"high_watermark": value}}, upsert=True)
        self.values[symbol] = value
        return value
