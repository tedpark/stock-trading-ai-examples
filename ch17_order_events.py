"""17장 학습 예제 — 더미 데이터 전용. 원문: 17-ibkr.md"""
class OrderLedger:
    def __init__(self, quantity):
        self.quantity, self.filled = quantity, 0
        self.status, self.executions = "submitted", set()

    def request_cancel(self):
        if self.status != "filled":
            self.status = "cancel_pending"

    def on_fill(self, execution_id, quantity):
        if execution_id in self.executions:
            return
        if quantity <= 0 or self.filled + quantity > self.quantity:
            raise ValueError("invalid fill quantity")
        self.executions.add(execution_id)
        self.filled += quantity
        if self.filled == self.quantity:
            self.status = "filled"

    def on_cancelled(self):
        if self.filled < self.quantity:
            self.status = "cancelled"


order = OrderLedger(10)
order.request_cancel()
order.on_fill("execution-1", 4)
order.on_fill("execution-1", 4)
order.on_cancelled()
assert (order.filled, order.status) == (4, "cancelled")
print(order.filled, order.status)
