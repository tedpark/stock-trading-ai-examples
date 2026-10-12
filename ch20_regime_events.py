"""20장 학습 예제 — 더미 데이터 전용. 원문: 20-pair-selection-regime.md"""
def reversion_counts(zscores, labels, horizon=20):
    if len(zscores) != len(labels) or horizon < 1:
        raise ValueError("aligned inputs and positive horizon required")
    counts = {r: {"success": 0, "failure": 0, "incomplete": 0}
              for r in set(labels)}
    t = 1
    while t < len(zscores):
        if not (abs(zscores[t - 1]) <= 2 < abs(zscores[t])):
            t += 1
            continue
        regime, stop = labels[t], min(t + horizon, len(zscores) - 1)
        outcome, end = "incomplete", stop
        for j in range(t + 1, stop + 1):
            if labels[j] != regime:
                end = j
                break
            if abs(zscores[j]) < 0.5:
                outcome, end = "success", j
                break
            if abs(zscores[j]) > 4:
                outcome, end = "failure", j
                break
        else:
            if t + horizon < len(zscores):
                outcome = "failure"
        counts[regime][outcome] += 1
        t = end + 1
    return counts


counts = reversion_counts([0, 2.5, 1.5, 0.2, 0, 2.5, 4.2], [0] * 7)
assert counts[0] == {"success": 1, "failure": 1, "incomplete": 0}
print(counts)
