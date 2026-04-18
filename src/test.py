# src/test.py
import os
import time
import random
import uuid
from dataclasses import dataclass
from typing import List
from dotenv import load_dotenv

from .utils import SYMBOL, log_trade, IS_TESTNET
from .decision_engine import decide_with_filters
from .executor import (
    make_client,
    market_order,
    quantize_qty,
    check_min_notional,
    cancel_all_orders,
)

load_dotenv()

API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")

RUN_LOGIC_SCENARIOS = int(os.getenv("TEST_RUN_LOGIC_SCENARIOS", "1"))
RUN_LIVE_TRADE = int(os.getenv("TEST_RUN_LIVE_TRADE", "1"))

TEST_WAIT_SEC = int(os.getenv("TEST_WAIT_SEC", "5"))
TEST_RISK_FRAC = float(os.getenv("TEST_RISK_FRAC", "0.01"))
TEST_QTY = float(os.getenv("TEST_QTY", "0"))

RUN_ID = os.getenv("RUN_ID", "test-run")

def _now_ms() -> int:
    return int(time.time() * 1000)

def make_klines_from_prices(prices: List[float], step_ms: int = 3600_000) -> List[list]:
    if len(prices) < 10:
        raise ValueError("Need at least 10 prices for indicators")

    start = _now_ms() - step_ms * len(prices)
    klines = []
    for i, c in enumerate(prices):
        open_time = start + i * step_ms
        o = prices[i - 1] if i > 0 else c
        h = max(o, c) * 1.001
        l = min(o, c) * 0.999
        v = 10.0
        klines.append([
            open_time,
            f"{o:.2f}", f"{h:.2f}", f"{l:.2f}", f"{c:.2f}",
            f"{v:.6f}", 0, 0, 0, 0, 0, "0",
        ])
    return klines

def make_flat_prices(base: float, n: int, noise_abs: float = 1.0, seed: int = 42) -> List[float]:
    rng = random.Random(seed)
    return [base + rng.uniform(-noise_abs, noise_abs) for _ in range(n)]

def make_flat_then_breakout_prices(
    base: float,
    n_flat: int = 260,
    n_break: int = 60,
    noise_abs: float = 1.0,
    breakout_step: float = 12.0,
    direction: str = "up",
    seed: int = 43
) -> List[float]:
    flat = make_flat_prices(base, n_flat, noise_abs=noise_abs, seed=seed)
    last = flat[-1]
    out = flat[:]
    sign = 1.0 if direction == "up" else -1.0
    for _ in range(n_break):
        last = last + sign * breakout_step
        out.append(last)
    return out

def print_case(title: str):
    print("\n" + "=" * 72, flush=True)
    print(f"[TEST] {title}", flush=True)
    print("=" * 72, flush=True)

@dataclass
class LogicScenario:
    name: str
    prices: List[float]
    expect: str

def run_logic_scenarios():
    """
    LOGIC scenariusze NIE zapisują NIC do transactions.csv.
    To jest test logiki, a nie dataset ML.
    """
    print_case("LOGIC SCENARIOS (NO CSV WRITES)")

    base = 67000.0
    scenarios = [
        LogicScenario("ENTRY_BUY_uptrend", [base + i * 40 for i in range(320)], "BUY"),
        LogicScenario("ENTRY_SELL_downtrend", [base - i * 40 for i in range(320)], "SELL"),
        LogicScenario("ENTRY_HOLD_flat", make_flat_prices(base, 320, noise_abs=1.0, seed=42), "HOLD"),
        LogicScenario("ENTRY_BREAKOUT_up", make_flat_then_breakout_prices(base, direction="up", seed=43), "BUY"),
        LogicScenario("ENTRY_BREAKOUT_down", make_flat_then_breakout_prices(base, direction="down", seed=44), "SELL"),
    ]

    for sc in scenarios:
        klines = make_klines_from_prices(sc.prices)
        d, conf, diag = decide_with_filters(klines)
        print(f"[SCENARIO] {sc.name} expect~{sc.expect} -> got={d} conf={conf:.3f}", flush=True)

def last_price(client, symbol: str) -> float:
    t = client.get_symbol_ticker(symbol=symbol)
    return float(t["price"])

def usdt_balance(client) -> float:
    bal = client.get_asset_balance(asset="USDT")
    return float(bal["free"]) if bal else 0.0

def asset_balance(client, asset: str) -> float:
    bal = client.get_asset_balance(asset=asset)
    return float(bal["free"]) if bal else 0.0

def run_live_trade_buy_sell():
    """
    LIVE TRADE zapisuje WYŁĄCZNIE:
    - TRADE_ENTRY
    - TRADE_EXIT
    Dokładnie jak ustaliliśmy.
    """
    print_case("LIVE TRADE (TESTNET) BUY -> SELL (CSV: ENTRY/EXIT ONLY)")

    if not API_KEY or not API_SECRET:
        raise RuntimeError("Missing BINANCE_API_KEY / BINANCE_API_SECRET in .env")

    client = make_client(API_KEY, API_SECRET)
    base_asset = SYMBOL.replace("USDT", "")

    price = last_price(client, SYMBOL)
    usdt = usdt_balance(client)

    try:
        cancel_all_orders(client, SYMBOL)
    except Exception:
        pass

    if TEST_QTY > 0:
        qty_raw = TEST_QTY
    else:
        notional = usdt * TEST_RISK_FRAC
        qty_raw = notional / price if price > 0 else 0.0

    qty = quantize_qty(client, SYMBOL, qty_raw)
    if qty <= 0:
        raise RuntimeError(f"[LIVE] qty too small after quantize (raw={qty_raw})")

    if not check_min_notional(client, SYMBOL, qty, price):
        raise RuntimeError(f"[LIVE] minNotional not met: qty={qty} price={price:.2f}")

    trade_id = uuid.uuid4().hex
    entry_ts = time.time()

    buy_res = market_order(client, SYMBOL, "BUY", qty)
    fills = buy_res.get("fills", []) if isinstance(buy_res, dict) else []
    entry_price = float(fills[0]["price"]) if fills else price

    # CSV: TRADE_ENTRY
    log_trade("logs/transactions.csv", {
        "channel": "TEST",
        "run_id": RUN_ID,
        "trade_id": trade_id,
        "event": "TRADE_ENTRY",
        "symbol": SYMBOL,
        "side": "BUY",
        "qty": float(qty),
        "entry_price": float(entry_price),
        "final_decision": "BUY",
        "gpt_sent": "",
    })

    print(f"[LIVE] BUY filled: qty={qty} entry_price={entry_price:.2f}. Waiting {TEST_WAIT_SEC}s...", flush=True)
    time.sleep(TEST_WAIT_SEC)

    base_now = asset_balance(client, base_asset)
    qty_sell = quantize_qty(client, SYMBOL, base_now)
    exit_price = last_price(client, SYMBOL)

    if qty_sell <= 0:
        raise RuntimeError(f"[LIVE] No {base_asset} to sell (free={base_now})")

    if not check_min_notional(client, SYMBOL, qty_sell, exit_price):
        raise RuntimeError(f"[LIVE] minNotional not met for SELL: qty={qty_sell} price={exit_price:.2f}")

    market_order(client, SYMBOL, "SELL", qty_sell)

    hold_sec = int(time.time() - entry_ts)
    pnl_pct = (exit_price - entry_price) / entry_price if entry_price > 0 else 0.0
    pnl_usdt = (exit_price - entry_price) * float(qty_sell)

    # CSV: TRADE_EXIT
    log_trade("logs/transactions.csv", {
        "channel": "TEST",
        "run_id": RUN_ID,
        "trade_id": trade_id,
        "event": "TRADE_EXIT",
        "symbol": SYMBOL,
        "side": "SELL",
        "qty": float(qty_sell),
        "entry_price": float(entry_price),
        "exit_price": float(exit_price),
        "hold_sec": hold_sec,
        "pnl_usdt": float(pnl_usdt),
        "pnl_pct": float(pnl_pct),
        "exit_reason": "TEST_CLOSE",
        "final_decision": "SELL",
        "gpt_sent": "",
    })

    print(f"[LIVE] SOLD: qty={qty_sell} exit_price={exit_price:.2f} pnl_pct={pnl_pct*100:.3f}% hold={hold_sec}s", flush=True)

def main():
    print_case("BINANCEBOT TEST SUITE")
    print(f"[TEST] IS_TESTNET={IS_TESTNET} SYMBOL={SYMBOL}", flush=True)
    print(f"[TEST] logic={RUN_LOGIC_SCENARIOS} live={RUN_LIVE_TRADE}", flush=True)
    print("[TEST] NOTE: CSV will contain only TRADE_ENTRY and TRADE_EXIT.", flush=True)

    if RUN_LOGIC_SCENARIOS:
        run_logic_scenarios()
    if RUN_LIVE_TRADE:
        run_live_trade_buy_sell()

    print("\n[TEST] Done. Check logs/transactions.csv", flush=True)

if __name__ == "__main__":
    main()