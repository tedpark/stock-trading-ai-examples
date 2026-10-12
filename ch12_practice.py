"""12장 학습 예제 — 더미 데이터 전용. 원문: 12-stat-arb-signals.md"""
import numpy as np

Z_ENTRY, HL_MAX, CORR_MIN, VOL_MIN = 2.0, 60, 0.40, 0.90

def compute_entry_score(z, hl, corr, vol=1.0):
    if not np.isfinite([z, hl, corr, vol]).all() or hl <= 0:
        raise ValueError("finite inputs and positive half-life required")
    return abs(z) / max(hl, 1.0) * corr * vol

def check_entry(z, z_prev, hl, corr, vol_mult, z_entry_adj=Z_ENTRY):
    """12.1의 진입 자격 5단계 체크. (go, reason) 반환."""
    if not np.isfinite([z, z_prev, hl, corr, vol_mult, z_entry_adj]).all():
        return False, "nonfinite input"
    if z_entry_adj <= 0 or not -1 <= corr <= 1 or vol_mult < 0:
        return False, "invalid threshold, correlation or volume"
    # 1. 반감기
    if hl <= 0:                      return False, "hl must be positive"
    if hl > HL_MAX:                   return False, f"hl={hl}>{HL_MAX}"
    # 2. 상관관계
    if corr < CORR_MIN:               return False, f"corr={corr:.2f}<{CORR_MIN}"
    # 3. Z-score 임계값 (BW-HMM으로 보정된 z_entry_adj 사용)
    if abs(z) < z_entry_adj:          return False, f"abs(z)<{z_entry_adj}"
    # 4. 수렴 방향
    if abs(z) >= abs(z_prev):         return False, "z still rising"
    # 5. 거래량 미시구조
    if vol_mult < VOL_MIN:            return False, f"vol_mult={vol_mult:.2f}<{VOL_MIN}"
    return True, "ok"

if __name__ == "__main__":
    cases = [
        ("정상 진입",   dict(z=-2.8, z_prev=-3.1, hl=25, corr=0.65, vol_mult=0.95)),
        ("HL 과다",     dict(z=-2.5, z_prev=-2.9, hl=80, corr=0.60, vol_mult=0.95)),
        ("상관관계 약", dict(z=-2.6, z_prev=-3.0, hl=30, corr=0.30, vol_mult=0.95)),
        ("Z 미달",      dict(z=-1.5, z_prev=-1.8, hl=30, corr=0.70, vol_mult=0.95)),
        ("Z 상승중",    dict(z=-3.0, z_prev=-2.5, hl=30, corr=0.65, vol_mult=0.95)),
        ("거래량 부족", dict(z=-2.8, z_prev=-3.1, hl=25, corr=0.65, vol_mult=0.80)),
    ]
    for label, c in cases:
        go, reason = check_entry(**c)
        score = compute_entry_score(c["z"], c["hl"], c["corr"], vol=c["vol_mult"]) if go else 0.0
        print(f"{label:<10} go={go!s:<5} score={score:.4f} reason={reason}")
