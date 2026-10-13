import asyncio
import os
import time
import threading
from math import isfinite


class BrokerSession:
    """Injected account reader and order preview. No placeOrder call is implemented."""
    def __init__(self, backend, host=None, port=None, client_id=None, cooldown=30):
        self.backend = backend
        self.host = host or os.environ.get("IBKR_HOST", "127.0.0.1")
        self.port = int(os.environ.get("IBKR_PORT", "4002")) if port is None else port
        self.client_id = int(os.environ.get("IBKR_CLIENT_ID", "1")) if client_id is None else client_id
        if not 1 <= self.port <= 65535 or self.client_id < 0 or not isfinite(cooldown) or cooldown < 0:
            raise ValueError("valid port, client ID and cooldown required")
        self.cooldown = cooldown
        self.connected = False
        self.failed_at = None
        self.lock = asyncio.Lock()

    async def connect(self):
        async with self.lock:
            if self.connected and self.backend.is_connected():
                return True
            now = time.monotonic()
            if self.failed_at is not None and now - self.failed_at < self.cooldown:
                return False
            self.connected = False
            try:
                await asyncio.wait_for(self.backend.connect(self.host, self.port, self.client_id), 10)
                if not self.backend.is_connected():
                    raise ConnectionError("not connected")
                accounts = await asyncio.wait_for(self.backend.accounts(), 10)
                if not accounts:
                    raise ValueError("no managed account")
                self.connected, self.failed_at = True, None
            except Exception:
                self.failed_at = time.monotonic()
            return self.connected

    async def session_info(self, symbol):
        if not self.connected or not symbol:
            raise ValueError("connected session and symbol required")
        rows = await asyncio.wait_for(self.backend.contract_details(symbol), 10)
        if len(rows) != 1:
            raise ValueError("one resolved contract required")
        return {key: rows[0][key] for key in ("symbol", "timezone", "trading_hours", "liquid_hours")}

    async def preview_order(self, symbol, action, qty, limit_price=None, allow_outside_rth=False):
        if (not self.connected or action not in ("BUY", "SELL") or not isfinite(qty)
                or isinstance(qty, bool) or qty <= 0 or int(qty) != qty):
            raise ValueError("connected session, valid side and positive whole quantity required")
        if limit_price is not None and (not isfinite(limit_price) or limit_price <= 0):
            raise ValueError("positive finite limit price required")
        if not await asyncio.wait_for(self.backend.session_open(symbol, allow_outside_rth), 10):
            return {"success": False, "error": "session_closed"}
        contract = await asyncio.wait_for(self.backend.qualify(symbol), 10)
        if not contract.get("con_id"):
            return {"success": False, "error": "contract_not_qualified"}
        return {"success": True, "mode": "preview", "contract": contract,
                "action": action, "qty": int(qty), "limit_price": limit_price,
                "outside_rth": allow_outside_rth}

    async def positions(self, account):
        if not self.connected or not account:
            raise ValueError("connected session and explicit account required")
        rows = await asyncio.wait_for(self.backend.positions(), 8)
        return [p for p in rows if p["account"] == account and p["size"] != 0]

    async def account_summary(self, account):
        if not self.connected or not account:
            raise ValueError("connected session and explicit account required")
        rows = await asyncio.wait_for(self.backend.summary(account), 10)
        tags = {"NetLiquidation", "TotalCashValue", "AvailableFunds", "GrossPositionValue", "Leverage", "MaintMarginReq"}
        result = {}
        for row in rows:
            if row["account"] != account or row["tag"] not in tags:
                continue
            value = float(row["value"])
            if not isfinite(value):
                raise ValueError("nonfinite account value")
            result.setdefault(row["tag"], {})[row["currency"]] = value
        if "NetLiquidation" not in result:
            raise ValueError("account summary is not ready")
        return result


class PaperBrokerBackend:
    """Deterministic local backend for protocol exercises; no network or money."""
    def __init__(self):
        self.connected = False
        self.open = True
    async def connect(self, host, port, client_id):
        self.connected = True
    def is_connected(self):
        return self.connected
    async def accounts(self):
        return ["paper"]
    async def contract_details(self, symbol):
        return [{"symbol":symbol, "timezone":"UTC", "trading_hours":"SIMULATED", "liquid_hours":"SIMULATED"}]
    async def session_open(self, symbol, outside):
        return self.open
    async def qualify(self, symbol):
        return {"con_id":1, "symbol":symbol}
    async def positions(self):
        return [{"account":"paper", "symbol":"DEMO", "size":2, "avg_cost":100.0}]
    async def summary(self, account):
        return [{"account":account, "tag":"NetLiquidation", "currency":"USD", "value":10000.0}]

async def main():
    session = BrokerSession(PaperBrokerBackend())
    if not await session.connect():
        raise RuntimeError('paper backend unavailable')
    print(await session.preview_order('DEMO', 'BUY', 2))

if __name__ == '__main__':
    asyncio.run(main())
