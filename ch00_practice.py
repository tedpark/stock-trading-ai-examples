"""00장 학습 예제 — 더미 데이터 전용. 원문: 00a-vibe-coding-quickstart.md"""
prices = [100.0, 80.0, 100.0]
initial = 1000.0
quantity = initial / prices[0]
cash = initial - quantity * prices[0]
equity = [initial] + [cash + quantity * p for p in prices]
peak = initial
mdd = 0.0
for value in equity:
    peak = max(peak, value)
    mdd = max(mdd, (peak - value) / peak)
final_cash = cash + quantity * prices[-1]
assert final_cash == 1000.0
assert mdd == 0.2
print(final_cash, mdd)
