import os
import time
import uuid
import logging
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

from .ml_gating import MLGating
from .utils import SYMBOL, log_trade, MIN_TRADE_INTERVAL_SEC, RISK_FRAC, IS_TESTNET
from .executor import (
    make_client,
    market_order,
    place_oco_takeprofit_stop,
    cancel_all_orders,
    quantize_qty,
    check_min_notional,
)
from .data_feed import history_client, get_klines
from .decision_engine import decide_with_filters
from .risk import size_by_balance_usdt
from .position_manager import SpotPosition
from .analysis.chat_gpt_query import analyze_market_with_gpt

AGGRESIVE_MODE = int(os.getenv("AGGRESSIVE_MODE", "1"))
TREND_HARD_BLOCK = float(os.getenv("TREND_HARD_BLOCK", "0.20"))

API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")

LOOP_SEC = int(os.getenv("LOOP_SEC", "60"))
ENTRY_INTERVAL = os.getenv("ENTRY_INTERVAL", "5m")
TREND_INTERVAL = os.getenv("TREND_INTERVAL", "15m")
MAX_HOLD_HOURS = float(os.getenv("MAX_HOLD_HOURS", "1.0"))
MIN_HOLD_SEC = int(os.getenv("MIN_HOLD_SEC", "180"))

BOT_VERBOSE = int(os.getenv("BOT_VERBOSE", "1"))
GPT_ON_HOLD = int(os.getenv("GPT_ON_HOLD", "0"))
RUN_ID = os.getenv("RUN_ID", "thesis_demo_public")
ML_USE_GATING = os.getenv("ML_USE_GATING", "1") == "1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("binancebot")


def utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def vinfo(msg: str):
    if BOT_VERBOSE:
        logger.info(msg)


def last_price(client):
    t = client.get_symbol_ticker(symbol=SYMBOL)
    return float(t["price"])


def usdt_balance(client):
    bal = client.get_asset_balance(asset="USDT")
    return float(bal["free"]) if bal else 0.0


def asset_balance(client, asset):
    bal = client.get_asset_balance(asset=asset)
    return float(bal["free"]) if bal else 0.0


def combine_decisions(dec_1h: str, dec_4h: str, conf_4h: float, hard_block=0.35) -> str:
    if dec_1h == "HOLD":
        return "HOLD"

    if conf_4h >= hard_block:
        if dec_1h == "BUY" and dec_4h == "SELL":
            return "HOLD"
        if dec_1h == "SELL" and dec_4h == "BUY":
            return "HOLD"

    return dec_1h


def recover_position_state(client_exec, base_asset):
    try:
        qty_free_raw = asset_balance(client_exec, base_asset)
        qty_free = quantize_qty(client_exec, SYMBOL, qty_free_raw)

        if qty_free > 0:
            p = SpotPosition(
                SYMBOL,
                qty_free,
                max_hold_hours=MAX_HOLD_HOURS,
                min_hold_sec=MIN_HOLD_SEC,
            )
            current = last_price(client_exec)
            p.on_filled_buy(current)
            logger.warning(f"[RECOVERY] detected existing asset balance: qty={qty_free}")
            return p
    except Exception as e:
        logger.warning(f"[RECOVERY] failed: {e}")

    return None


def build_entry_features(diag: dict, conf: float) -> dict:
    return {
        "rsi": float(diag.get("rsi", 0.0) or 0.0),
        "macd_hist": float(diag.get("macd_hist", 0.0) or 0.0),
        "ema50": float(diag.get("ema50", 0.0) or 0.0),
        "ema200": float(diag.get("ema200", 0.0) or 0.0),
        "bb_width": float(diag.get("bb_width", 0.0) or 0.0),
        "atr_pct": float(diag.get("atr_pct", 0.0) or 0.0),
        "conf": float(conf or 0.0),
    }


def run_loop():
    client_hist = history_client(API_KEY, API_SECRET)
    client_exec = make_client(API_KEY, API_SECRET)
    ml_gate = MLGating("logs/transactions.csv")

    logger.info(f"[START {utc_ts()}] connecting OK testnet={IS_TESTNET}")
    logger.info(f"[START] symbol={SYMBOL} entry_interval={ENTRY_INTERVAL} trend_interval={TREND_INTERVAL}")
    logger.info(f"[START] usdt_free={usdt_balance(client_exec):.2f}")

    position: SpotPosition | None = None
    last_trade_ts = 0.0
    base_asset = SYMBOL.replace("USDT", "")

    recovered = recover_position_state(client_exec, base_asset)
    if recovered is not None:
        position = recovered

    trade_id = None
    entry_ts_epoch = None
    entry_features = {}
    entry_meta = {}
    entry_qty = None

    last_entry_candle_time = None

    while True:
        try:
            price = last_price(client_hist)
            now = time.time()

            vinfo(f"[LOOP] mode={'ENTRY' if position is None else position.mode} price={price:.2f}")

            if position is None:
                kl_1h = get_klines(client_hist, SYMBOL, interval=ENTRY_INTERVAL, limit=300)
                d1, c1, diag1 = decide_with_filters(kl_1h)

                entry_candle_time = diag1.get("candle_time_utc")
                if last_entry_candle_time == entry_candle_time:
                    vinfo(f"[SKIP] same entry candle already analyzed: {entry_candle_time}")
                    time.sleep(LOOP_SEC)
                    continue
                last_entry_candle_time = entry_candle_time

                kl_4h = get_klines(client_hist, SYMBOL, interval=TREND_INTERVAL, limit=300)
                d4, c4, diag4 = decide_with_filters(kl_4h)

                decision = combine_decisions(
                    d1,
                    d4,
                    c4,
                    hard_block=1.0 if AGGRESIVE_MODE else TREND_HARD_BLOCK,
                )
                conf = c1

                if decision == "SELL":
                    qty_free_raw = asset_balance(client_exec, base_asset)
                    qty_free = quantize_qty(client_exec, SYMBOL, qty_free_raw)
                    if qty_free <= 0:
                        decision = "HOLD"

                if (now - last_trade_ts) < MIN_TRADE_INTERVAL_SEC:
                    vinfo(f"[COOLDOWN] {int(MIN_TRADE_INTERVAL_SEC - (now - last_trade_ts))}s left")
                    time.sleep(LOOP_SEC)
                    continue

                gpt_sent = ""
                if decision != "HOLD" or GPT_ON_HOLD:
                    g = analyze_market_with_gpt(
                        symbol=SYMBOL,
                        rsi=diag1.get("rsi", 0),
                        macd=diag1.get("macd", 0),
                        macd_signal=diag1.get("macd_signal", 0),
                        ema50=diag1.get("ema50", 0),
                        ema200=diag1.get("ema200", 0),
                        boll_up=diag1.get("bb_up", 0),
                        boll_down=diag1.get("bb_dn", 0),
                        current_price=price,
                        trend=decision,
                        mode="sentiment",
                    )
                    gpt_sent = str(g) if g else ""

                ml_result = {
                    "model_ready": False,
                    "prob_win": None,
                    "threshold": ml_gate.gate_threshold,
                    "allow": True,
                    "reason": "not_used",
                }

                if ML_USE_GATING and decision == "BUY":
                    train_info = ml_gate.train_if_needed()
                    if train_info.get("trained"):
                        logger.info(
                            f"[ML] trained rows={train_info['rows']} train={train_info['train_rows']} "
                            f"test={train_info['test_rows']} acc={train_info['acc']:.3f}"
                        )
                    else:
                        logger.info(f"[ML] skip train: {train_info.get('reason')}")

                    features = build_entry_features(diag1, conf)
                    ml_result = ml_gate.allow_trade(features)

                    if not ml_result["model_ready"]:
                        logger.info("[ML] gate bypass: model not trained yet")
                    else:
                        logger.info(
                            f"[ML] prob_win={ml_result['prob_win']:.3f} "
                            f"threshold={ml_result['threshold']:.3f} allow={ml_result['allow']}"
                        )

                    if not ml_result["allow"]:
                        logger.info(
                            f"[SKIP] ML gate blocked BUY: prob_win={ml_result['prob_win']:.3f} "
                            f"< {ml_result['threshold']:.3f}"
                        )
                        log_trade(
                            "logs/transactions.csv",
                            {
                                "channel": "PROD",
                                "run_id": RUN_ID,
                                "trade_id": uuid.uuid4().hex,
                                "event": "ML_FILTER_BLOCK",
                                "symbol": SYMBOL,
                                "side": "BUY",
                                "price": float(price),
                                "decision": decision,
                                "ml_gate_used": 1,
                                "ml_model_ready": int(ml_result["model_ready"]),
                                "ml_prob": float(ml_result["prob_win"] or 0.0),
                                "ml_threshold": float(ml_result["threshold"]),
                                "ml_allow": int(ml_result["allow"]),
                                "ml_reason": ml_result["reason"],
                                **features,
                            },
                        )
                        time.sleep(LOOP_SEC)
                        continue

                if decision == "BUY":
                    usdt = usdt_balance(client_exec)
                    qty_raw = size_by_balance_usdt(usdt, risk_frac=RISK_FRAC, price=price)
                    qty = quantize_qty(client_exec, SYMBOL, qty_raw)

                    if qty > 0 and check_min_notional(client_exec, SYMBOL, qty, price):
                        logger.info(f"[BUY {utc_ts()}] qty={qty} {SYMBOL} @ {price:.2f}")
                        res = market_order(client_exec, SYMBOL, "BUY", qty)

                        fills = res.get("fills", [])
                        fill_price = float(fills[0]["price"]) if fills else price

                        position = SpotPosition(
                            SYMBOL,
                            qty,
                            max_hold_hours=MAX_HOLD_HOURS,
                            min_hold_sec=MIN_HOLD_SEC,
                        )
                        position.on_filled_buy(fill_price)
                        position.entry_diag = diag1.copy()
                        position.entry_conf = conf
                        last_trade_ts = now

                        trade_id = uuid.uuid4().hex
                        entry_ts_epoch = now
                        entry_qty = float(qty)
                        entry_features = build_entry_features(diag1, conf)
                        entry_meta = {
                            "d1": d1,
                            "c1": c1,
                            "d4": d4,
                            "c4": c4,
                            "final_decision": decision,
                            "gpt_sent": gpt_sent,
                            "ml_gate_used": int(ML_USE_GATING),
                            "ml_model_ready": int(ml_result["model_ready"]),
                            "ml_prob": float(ml_result["prob_win"] or 0.0),
                            "ml_threshold": float(ml_result["threshold"]),
                            "ml_allow": int(ml_result["allow"]),
                            "ml_reason": ml_result["reason"],
                        }

                        log_trade(
                            "logs/transactions.csv",
                            {
                                "channel": "PROD",
                                "run_id": RUN_ID,
                                "trade_id": trade_id,
                                "event": "TRADE_ENTRY",
                                "symbol": SYMBOL,
                                "side": "BUY",
                                "qty": entry_qty,
                                "entry_price": fill_price,
                                **entry_features,
                                **entry_meta,
                            },
                        )

                        levels = position.trailing_levels()
                        if levels is not None:
                            tp, stop, stop_limit = levels
                            try:
                                cancel_all_orders(client_exec, SYMBOL)
                            except Exception as e:
                                logger.warning(f"[WARN] cancel before OCO failed: {e}")

                            try:
                                place_oco_takeprofit_stop(client_exec, SYMBOL, qty, tp, stop, stop_limit)
                                logger.info(f"[OCO] TP={tp:.2f} STOP={stop:.2f} STOP_LIMIT={stop_limit:.2f}")
                            except Exception as e:
                                logger.warning(f"[WARN] OCO placement failed after BUY: {e}")

                elif decision == "SELL":
                    qty_free_raw = asset_balance(client_exec, base_asset)
                    qty_free = quantize_qty(client_exec, SYMBOL, qty_free_raw)

                    if qty_free > 0 and check_min_notional(client_exec, SYMBOL, qty_free, price):
                        logger.info(f"[SELL {utc_ts()}] close spot qty={qty_free} @ {price:.2f}")
                        cancel_all_orders(client_exec, SYMBOL)
                        market_order(client_exec, SYMBOL, "SELL", qty_free)
                        last_trade_ts = now

            else:
                min_hold_passed = position.min_hold_passed()
                if not min_hold_passed:
                    vinfo(f"[MIN_HOLD] waiting {MIN_HOLD_SEC}s before allowing normal exit logic")

                qty_free_raw = asset_balance(client_exec, base_asset)
                qty_free = quantize_qty(client_exec, SYMBOL, qty_free_raw)

                if min_hold_passed and qty_free <= 0:
                    exit_price = price
                    hold_sec = int(time.time() - float(entry_ts_epoch or time.time()))

                    pnl_usdt = 0.0
                    pnl_pct = 0.0
                    if position.entry_price and position.entry_price > 0 and entry_qty:
                        pnl_pct = (exit_price - position.entry_price) / position.entry_price
                        pnl_usdt = (exit_price - position.entry_price) * float(entry_qty)

                    log_trade(
                        "logs/transactions.csv",
                        {
                            "channel": "PROD",
                            "run_id": RUN_ID,
                            "trade_id": trade_id or "",
                            "event": "TRADE_EXIT",
                            "symbol": SYMBOL,
                            "side": "SELL",
                            "qty": float(entry_qty or 0.0),
                            "entry_price": float(position.entry_price or 0.0),
                            "exit_price": float(exit_price),
                            "hold_sec": hold_sec,
                            "pnl_usdt": pnl_usdt,
                            "pnl_pct": pnl_pct,
                            "exit_reason": "OCO_OR_MANUAL",
                            **entry_features,
                            **entry_meta,
                        },
                    )

                    try:
                        cancel_all_orders(client_exec, SYMBOL)
                    except Exception as e:
                        logger.warning(f"[WARN] cancel after detected exit failed: {e}")

                    position = None
                    trade_id = None
                    entry_ts_epoch = None
                    entry_features = {}
                    entry_meta = {}
                    entry_qty = None

                    logger.info("[CLOSED] exit detected -> back to ENTRY mode")
                    time.sleep(LOOP_SEC)
                    continue

                if position.expired():
                    logger.info(f"[TIMEOUT {utc_ts()}] max_hold={MAX_HOLD_HOURS}h reached, closing by market")

                    exit_qty_raw = asset_balance(client_exec, base_asset)
                    exit_qty = quantize_qty(client_exec, SYMBOL, exit_qty_raw)

                    exit_price = price
                    if exit_qty > 0:
                        try:
                            cancel_all_orders(client_exec, SYMBOL)
                        except Exception as e:
                            logger.warning(f"[WARN] cancel before timeout sell failed: {e}")

                        try:
                            market_order(client_exec, SYMBOL, "SELL", exit_qty)
                        except Exception as e:
                            logger.warning(f"[WARN] market sell on timeout failed: {e}")

                    hold_sec = int(time.time() - float(entry_ts_epoch or time.time()))
                    pnl_usdt = 0.0
                    pnl_pct = 0.0
                    if position.entry_price and position.entry_price > 0:
                        calc_qty = float(exit_qty or entry_qty or 0.0)
                        pnl_pct = (exit_price - position.entry_price) / position.entry_price
                        pnl_usdt = (exit_price - position.entry_price) * calc_qty

                    log_trade(
                        "logs/transactions.csv",
                        {
                            "channel": "PROD",
                            "run_id": RUN_ID,
                            "trade_id": trade_id or "",
                            "event": "TRADE_EXIT",
                            "symbol": SYMBOL,
                            "side": "SELL",
                            "qty": float(exit_qty or entry_qty or 0.0),
                            "entry_price": float(position.entry_price or 0.0),
                            "exit_price": float(exit_price),
                            "hold_sec": hold_sec,
                            "pnl_usdt": pnl_usdt,
                            "pnl_pct": pnl_pct,
                            "exit_reason": "TIMEOUT",
                            **entry_features,
                            **entry_meta,
                        },
                    )

                    position = None
                    trade_id = None
                    entry_ts_epoch = None
                    entry_features = {}
                    entry_meta = {}
                    entry_qty = None

                    logger.info("[CLOSED] timeout -> back to ENTRY mode")
                    time.sleep(LOOP_SEC)
                    continue

                position.update_max(price)
                levels = position.trailing_levels()
                if not levels:
                    logger.warning("[WARN] trailing levels unavailable, skipping refresh")
                    time.sleep(LOOP_SEC)
                    continue

                tp, stop, stop_limit = levels

                open_orders = client_exec.get_open_orders(symbol=SYMBOL)
                have_stop = any(o["type"] in ("STOP_LOSS_LIMIT", "STOP_LOSS") for o in open_orders)

                need_refresh = not have_stop
                if have_stop:
                    for o in open_orders:
                        if o["type"] in ("STOP_LOSS_LIMIT", "STOP_LOSS"):
                            cur = float(o.get("stopPrice", o.get("price", 0.0)))
                            if stop > cur * 1.001:
                                need_refresh = True
                            break

                if need_refresh and min_hold_passed:
                    try:
                        cancel_all_orders(client_exec, SYMBOL)

                        refresh_qty_raw = asset_balance(client_exec, base_asset)
                        refresh_qty = quantize_qty(client_exec, SYMBOL, refresh_qty_raw)

                        if refresh_qty > 0:
                            place_oco_takeprofit_stop(client_exec, SYMBOL, refresh_qty, tp, stop, stop_limit)
                            vinfo(f"[TRAIL {utc_ts()}] TP={tp:.2f} STOP={stop:.2f} STOP_LIMIT={stop_limit:.2f}")
                        else:
                            logger.warning("[WARN] skip trail refresh: no asset balance available")
                    except Exception as e:
                        logger.warning(f"[WARN] TRAIL OCO refresh failed: {e}")

        except Exception as e:
            logger.exception(f"[ERROR] {e}")

        time.sleep(LOOP_SEC)


if __name__ == "__main__":
    run_loop()
