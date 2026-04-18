import time


class SpotPosition:
    def __init__(
        self,
        symbol: str,
        qty: float,
        max_hold_hours: float = 4.0,
        min_hold_sec: int = 0,
        trail_pct: float = 0.018,
        take_profit_pct: float = 0.035,
    ):
        self.symbol = symbol
        self.qty = qty
        self.entry_price = None
        self.max_price = None
        self.mode = "ENTRY"
        self.created_at = time.time()
        self.trail_pct = trail_pct
        self.take_profit_pct = take_profit_pct
        self.hard_timeout_s = int(max_hold_hours * 3600)
        self.min_hold_sec = int(min_hold_sec)

        self.entry_diag = {}
        self.entry_conf = None

    def on_filled_buy(self, price: float):
        self.entry_price = price
        self.max_price = price
        self.mode = "EXIT"

    def update_max(self, price: float):
        if self.max_price is None or price > self.max_price:
            self.max_price = price

    def trailing_levels(self):
        if self.entry_price is None or self.max_price is None:
            return None
        tp = self.entry_price * (1 + self.take_profit_pct)
        stop = self.max_price * (1 - self.trail_pct)
        stop_limit = stop * 0.999
        return tp, stop, stop_limit

    def expired(self) -> bool:
        return (time.time() - self.created_at) > self.hard_timeout_s

    def min_hold_passed(self) -> bool:
        return (time.time() - self.created_at) >= self.min_hold_sec
