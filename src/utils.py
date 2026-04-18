# src/utils.py
import os
import time
import csv
from dotenv import load_dotenv
from datetime import timezone, datetime

load_dotenv()

def cfg(key, default=None, cast=str):
    v = os.getenv(key, default)
    return cast(v) if v is not None and cast is not str else v

IS_TESTNET = cfg("USE_TESTNET", "true", str).lower() == "true"
SYMBOL = cfg("SYMBOL", "BTCUSDT")
MIN_TRADE_INTERVAL_SEC = cfg("MIN_TRADE_INTERVAL_SEC", 180, int)
RISK_FRAC = cfg("RISK_FRAC", 0.02, float)

# ======== STABILNY SCHEMAT CSV: TYLKO ENTRY/EXIT ========
TRANSACTIONS_FIELDS = [
    # czas
    "ts_epoch", "ts_utc",

    # meta
    "channel",          # PROD / TEST
    "is_testnet",       # 1/0
    "run_id",           # opcjonalnie

    # trade identity
    "trade_id",         # wspólny dla ENTRY i EXIT
    "event",            # TRADE_ENTRY / TRADE_EXIT
    "symbol",

    # execution
    "side",             # BUY / SELL
    "qty",
    "entry_price",
    "exit_price",
    "hold_sec",

    # wynik
    "pnl_usdt",
    "pnl_pct",
    "exit_reason",      # TIMEOUT / OCO_OR_MANUAL / MANUAL etc.

    # ML features (z momentu wejścia, kopiowane też na EXIT)
    "rsi",
    "macd_hist",
    "ema50",
    "ema200",
    "bb_width",
    "atr_pct",
    "conf",

    # decyzje pomocnicze (opcjonalnie, ale stałe kolumny)
    "d1", "c1",
    "d4", "c4",
    "final_decision",
    "gpt_sent",
]

def _utc_str(ts_epoch: float) -> str:
    return datetime.fromtimestamp(ts_epoch, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def _normalize_row(row: dict) -> dict:
    """
    Wymusza:
    - stałe kolumny
    - brak losowych dodatkowych kolumn
    - channel/is_testnet/symbol zawsze ustawione
    """
    if "ts_epoch" not in row or row["ts_epoch"] in (None, ""):
        row["ts_epoch"] = time.time()

    if "ts_utc" not in row or row["ts_utc"] in (None, ""):
        row["ts_utc"] = _utc_str(float(row["ts_epoch"]))

    if "channel" not in row or row["channel"] in (None, ""):
        row["channel"] = "PROD"

    if "is_testnet" not in row or row["is_testnet"] in (None, ""):
        row["is_testnet"] = 1 if IS_TESTNET else 0

    if "symbol" not in row or row["symbol"] in (None, ""):
        row["symbol"] = SYMBOL

    clean = {k: "" for k in TRANSACTIONS_FIELDS}
    for k in TRANSACTIONS_FIELDS:
        if k in row and row[k] is not None:
            clean[k] = row[k]
    return clean

def log_trade(path: str, row: dict):
    """
    CSV jest pod ML i analizę transakcji.
    Zapisujemy TYLKO:
    - TRADE_ENTRY
    - TRADE_EXIT
    Reszta ma iść do logów (stdout/journalctl), nie do danych.
    """
    allowed = {"TRADE_ENTRY", "TRADE_EXIT"}
    ev = str(row.get("event", "")).strip()

    if ev not in allowed:
        return  # ignorujemy wszystko inne

    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.isfile(path)

    clean = _normalize_row(row)

    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TRANSACTIONS_FIELDS)
        if not file_exists:
            w.writeheader()
        w.writerow(clean)