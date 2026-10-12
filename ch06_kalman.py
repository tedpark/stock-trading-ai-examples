"""06장 학습 예제 — 더미 데이터 전용. 원문: 06-kalman-filter.md"""
def filter_prices(prices, q=0.01, r=1.0):
    if not prices or q < 0 or r <= 0:
        raise ValueError("nonempty prices, q >= 0 and r > 0 required")
    estimate, variance = float(prices[0]), 1.0
    result = []
    for price in prices:
        predicted_variance = variance + q
        gain = predicted_variance / (predicted_variance + r)
        estimate += gain * (price - estimate)
        variance = (1 - gain) * predicted_variance
        result.append(estimate)
    return result


prices = [100, 101, 99, 102, 100]
filtered = filter_prices(prices)
assert filter_prices(prices + [120])[:len(prices)] == filtered
assert filter_prices([100] * 5) == [100.0] * 5
print([round(x, 3) for x in filtered])
