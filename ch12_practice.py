"""Chapter 12 learning example; dummy data only. Source: 12-stat-arb-signals.md"""
import numpy as np

Z_ENTRY, HL_MAX, CORR_MIN, VOL_MIN = 2.0, 60, 0.40, 0.90

def compute_entry_score(z, hl, corr, vol=1.0):
    if not np.isfinite([z, hl, corr, vol]).all() or hl <= 0:
        raise ValueError("finite inputs and positive half-life required")
    return abs(z) / max(hl, 1.0) * corr * vol

def check_entry(z, z_prev, hl, corr, vol_mult, z_entry_adj=Z_ENTRY):
    """Apply the five entry checks from Section 12.1. Return (go, reason)."""
    if not np.isfinite([z, z_prev, hl, corr, vol_mult, z_entry_adj]).all():
        return False, "nonfinite input"
    if z_entry_adj <= 0 or not -1 <= corr <= 1 or vol_mult < 0:
        return False, "invalid threshold, correlation or volume"
    # 1. Half-life
    if hl <= 0:                      return False, "hl must be positive"
    if hl > HL_MAX:                   return False, f"hl={hl}>{HL_MAX}"
    # 2. Correlation
    if corr < CORR_MIN:               return False, f"corr={corr:.2f}<{CORR_MIN}"
    # 3. Z-score threshold (use the BW-HMM-adjusted z_entry_adj)
    if abs(z) < z_entry_adj:          return False, f"abs(z)<{z_entry_adj}"
    # 4. Convergence direction
    if abs(z) >= abs(z_prev):         return False, "z still rising"
    # 5. Volume microstructure
    if vol_mult < VOL_MIN:            return False, f"vol_mult={vol_mult:.2f}<{VOL_MIN}"
    return True, "ok"

if __name__ == "__main__":
    cases = [
        ("Valid entry",   dict(z=-2.8, z_prev=-3.1, hl=25, corr=0.65, vol_mult=0.95)),
        ("Half-life too long",     dict(z=-2.5, z_prev=-2.9, hl=80, corr=0.60, vol_mult=0.95)),
        ("Weak correlation", dict(z=-2.6, z_prev=-3.0, hl=30, corr=0.30, vol_mult=0.95)),
        ("Z below threshold",      dict(z=-1.5, z_prev=-1.8, hl=30, corr=0.70, vol_mult=0.95)),
        ("Spread diverging",    dict(z=-3.0, z_prev=-2.5, hl=30, corr=0.65, vol_mult=0.95)),
        ("Low volume", dict(z=-2.8, z_prev=-3.1, hl=25, corr=0.65, vol_mult=0.80)),
    ]
    for label, c in cases:
        go, reason = check_entry(**c)
        score = compute_entry_score(c["z"], c["hl"], c["corr"], vol=c["vol_mult"]) if go else 0.0
        print(f"{label:<10} go={go!s:<5} score={score:.4f} reason={reason}")
