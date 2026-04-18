import pandas as pd
import numpy as np
import os

MIN_ATR_PCT = float(os.getenv("MIN_ATR_PCT", "0.003"))
MAX_ATR_PCT = float(os.getenv("MAX_ATR_PCT", "0.12"))
MIN_EXPECTED_MOVE_PCT = float(os.getenv("MIN_EXPECTED_MOVE_PCT", "0.004"))


# -----------------------
# PODSTAWOWE WSKAŹNIKI
# -----------------------

def _make_df(klines):
    df = pd.DataFrame(klines, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "ct", "qv", "n", "tb", "tqv", "ignore"
    ])

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df


def add_ema(df):
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()
    return df


def add_bollinger(df, window=20, k=2.0):
    ma = df["close"].rolling(window).mean()
    std = df["close"].rolling(window).std(ddof=0)
    df["bb_mid"] = ma
    df["bb_up"] = ma + k * std
    df["bb_dn"] = ma - k * std
    df["bb_width"] = (df["bb_up"] - df["bb_dn"]) / df["bb_mid"]
    return df


# -----------------------
# LOGIKA WSKAŹNIKÓW
# -----------------------

def ema_signal(df):
    aggressive = os.getenv("AGGRESSIVE_MODE", "0") == "1"

    if len(df) < 200:
        return "HOLD", 0.0

    last = df.iloc[-1]
    prev = df.iloc[-2]

    if prev["ema_50"] <= prev["ema_200"] and last["ema_50"] > last["ema_200"]:
        return "BUY", 0.75
    if prev["ema_50"] >= prev["ema_200"] and last["ema_50"] < last["ema_200"]:
        return "SELL", 0.75

    if not aggressive:
        return "HOLD", 0.20

    if last["ema_50"] > last["ema_200"]:
        return "BUY", 0.35
    if last["ema_50"] < last["ema_200"]:
        return "SELL", 0.35

    return "HOLD", 0.20


def bb_signal(df):
    last = df.iloc[-1]
    recent = df["bb_width"].tail(200).dropna()
    if len(recent) < 50:
        return "NEUTRAL", 0.0

    aggressive = os.getenv("AGGRESSIVE_MODE", "0") == "1"

    squeeze_q = 0.25 if aggressive else 0.20
    squeeze = last["bb_width"] <= recent.quantile(squeeze_q)

    breakout_up = last["close"] > last["bb_up"]
    breakout_down = last["close"] < last["bb_dn"]

    if squeeze and breakout_up:
        return "BUY", 0.60 if aggressive else 0.55
    if squeeze and breakout_down:
        return "SELL", 0.60 if aggressive else 0.55

    return "NEUTRAL", 0.10


def cycle_filter_score(df):
    ts = df.iloc[-1]["open_time"]
    hour = ts.hour
    weekday = ts.weekday()

    hour_score = 0.4
    if 13 <= hour <= 17:
        hour_score = 1.0
    elif 20 <= hour <= 23:
        hour_score = 0.8
    elif 4 <= hour <= 6:
        hour_score = 0.2

    day_score = 0.7
    if weekday in (5, 6):
        day_score = 0.5

    return round(0.5 * hour_score + 0.5 * day_score, 3)


# -----------------------
# ELLIOTT / ZIGZAG
# -----------------------

def zigzag_pivots(series, pct=0.015):
    if len(series) < 10:
        return []
    prices = series.values
    idxs = series.index
    pivots = []
    trend = 0
    anchor_i = 0
    anchor_p = prices[0]

    for i in range(1, len(prices)):
        change = (prices[i] - anchor_p) / anchor_p
        if trend >= 0 and change >= pct:
            trend = 1
            pivots.append((idxs[anchor_i], anchor_p))
            anchor_i, anchor_p = i, prices[i]
        elif trend <= 0 and change <= -pct:
            trend = -1
            pivots.append((idxs[anchor_i], anchor_p))
            anchor_i, anchor_p = i, prices[i]
        else:
            if trend >= 0 and prices[i] > anchor_p:
                anchor_i, anchor_p = i, prices[i]
            if trend <= 0 and prices[i] < anchor_p:
                anchor_i, anchor_p = i, prices[i]

    pivots.append((idxs[anchor_i], anchor_p))
    return pivots[-7:]


def add_rsi(df, period: int = 14):
    close = df["close"].astype(float)
    delta = close.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    df["rsi"] = rsi.clip(0, 100).fillna(50.0)

    print(
        "RSI sanity:",
        "close=", float(close.iloc[-1]),
        "delta=", float(delta.iloc[-1]),
        "gain=", float(gain.iloc[-1]),
        "loss=", float(loss.iloc[-1]),
        "rsi=", float(df["rsi"].iloc[-1]),
        flush=True
    )

    return df


def add_macd(df, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()

    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def elliott_strength(df, pct=0.015):
    piv = zigzag_pivots(df["close"], pct=pct)
    if len(piv) < 5:
        return 0.0, 0.2
    prices = [p for _, p in piv]
    hh = sum(prices[i] > prices[i - 2] for i in range(2, len(prices), 2))
    hl = sum(prices[i] > prices[i - 2] for i in range(3, len(prices), 2))
    lh = sum(prices[i] < prices[i - 2] for i in range(2, len(prices), 2))
    ll = sum(prices[i] < prices[i - 2] for i in range(3, len(prices), 2))

    bull = hh + hl
    bear = lh + ll
    total = bull + bear if (bull + bear) > 0 else 1
    bias = (bull - bear) / total
    strength = min(1.0, total / 6)
    return round(bias, 3), round(strength, 3)


def add_atr(df, period: int = 14):
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    prev_close = close.shift(1)

    tr = pd.DataFrame(index=df.index)
    tr["hl"] = high - low
    tr["hc"] = (high - prev_close).abs()
    tr["lc"] = (low - prev_close).abs()
    tr["tr"] = tr[["hl", "hc", "lc"]].max(axis=1)

    df["atr"] = tr["tr"].ewm(alpha=1 / period, adjust=False).mean()

    if os.getenv("ATR_DEBUG", "0") == "1" and len(df) >= 3:
        last = df.iloc[-1]
        prev = df.iloc[-2]

        print("─────────────── ATR DEBUG ───────────────", flush=True)
        print(
            f"LAST CANDLE UTC: {last['open_time']} | "
            f"open={last['open']:.2f} high={last['high']:.2f} "
            f"low={last['low']:.2f} close={last['close']:.2f}",
            flush=True
        )
        print(
            f"PREV CLOSE: {prev['close']:.2f} | "
            f"HL={tr['hl'].iloc[-1]:.2f} "
            f"HC={tr['hc'].iloc[-1]:.2f} "
            f"LC={tr['lc'].iloc[-1]:.2f} "
            f"TR={tr['tr'].iloc[-1]:.2f}",
            flush=True
        )
        print(
            f"ATR={df['atr'].iloc[-1]:.2f} | "
            f"ATR%={(df['atr'].iloc[-1] / last['close']) * 100:.2f}%",
            flush=True
        )
        print("LAST 3 CANDLES:", flush=True)
        print(
            df[["open_time", "open", "high", "low", "close", "atr"]].tail(3).to_string(index=False),
            flush=True
        )
        print("─────────────────────────────────────────", flush=True)

    return df


# -----------------------
# GŁÓWNA DECYZJA Z LOGAMI
# -----------------------

def decide_with_filters(klines):
    df = _make_df(klines)
    df = add_ema(df)
    df = add_bollinger(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_atr(df)

    last = df.iloc[-1]
    diag_candle_utc = last["open_time"]
    price = float(last["close"])
    atr = float(last["atr"])
    atr_pct = atr / price if price > 0 else 0.0

    # filtr jakości rynku
    atr_regime = "normal"
    if atr_pct < MIN_ATR_PCT:
        atr_regime = "too_low"
    elif atr_pct > MAX_ATR_PCT:
        atr_regime = "too_high"

    expected_move_pct = atr_pct * 0.5

    d_ema, c_ema = ema_signal(df)
    d_bb, c_bb = bb_signal(df)
    cyc = cycle_filter_score(df)
    ell_bias, ell_strength_val = elliott_strength(df, pct=0.015)

    score = 0.0
    if d_ema == "BUY":
        score += 0.6 * c_ema
    if d_ema == "SELL":
        score -= 0.6 * c_ema
    if d_bb == "BUY":
        score += 0.4 * c_bb
    if d_bb == "SELL":
        score -= 0.4 * c_bb

    score *= (0.6 + 0.4 * cyc)
    score += 0.2 * ell_bias * ell_strength_val

    # VOLATILITY regime
    if atr_pct < 0.0015:
        score *= 0.3
    elif atr_pct < 0.003:
        score *= 0.6
    elif atr_pct < 0.006:
        score *= 1.0
    else:
        score *= 1.25

    ema50 = float(last["ema_50"])
    ema200 = float(last["ema_200"])
    rsi = float(last["rsi"])
    macd_hist = float(last["macd_hist"])

    score += 0.18 if ema50 > ema200 else -0.18
    score += 0.006 * (rsi - 50.0)
    score += 40.0 * (macd_hist / price) if price > 0 else 0.0

    bb_up = float(last["bb_up"])
    bb_dn = float(last["bb_dn"])
    if price <= bb_dn:
        score += 0.25
    elif price >= bb_up:
        score -= 0.25

    aggressive = os.getenv("AGGRESSIVE_MODE", "0") == "1"
    if atr_pct < 0.002:
        TH = 0.20 if aggressive else 0.25
    elif atr_pct < 0.004:
        TH = 0.15 if aggressive else 0.18
    else:
        TH = 0.12 if aggressive else 0.14

    diag = {
        "cyc": float(cyc),
        "ell": float(ell_bias),
        "ell_s": float(ell_strength_val),
        "rsi": rsi,
        "macd": float(last["macd"]),
        "macd_signal": float(last["macd_signal"]),
        "macd_hist": macd_hist,
        "ema50": ema50,
        "ema200": ema200,
        "bb_width": float(last["bb_width"]),
        "bb_up": float(last["bb_up"]),
        "bb_dn": float(last["bb_dn"]),
        "bb_mid": float(last["bb_mid"]),
        "atr": float(last["atr"]),
        "atr_pct": float(atr_pct),
        "atr_regime": atr_regime,
        "expected_move_pct": float(expected_move_pct),
        "th": float(TH),
        "candle_time_utc": str(diag_candle_utc),
        "score": float(score),
        "aggressive": int(aggressive),
    }

    print("─────────────── MARKET SNAPSHOT ───────────────", flush=True)
    print(f"EMA: {d_ema:<5}  conf={c_ema:.2f}", flush=True)
    print(f"BOLL: {d_bb:<7}  conf={c_bb:.2f}", flush=True)
    print(f"CYCLE: {cyc:.2f}", flush=True)
    print(f"ELLIOTT: bias={ell_bias:+.2f}  strength={ell_strength_val:.2f}", flush=True)
    print(f"RSI: {rsi:.2f} | MACD hist: {macd_hist:.4f}", flush=True)
    print(
        f"ATR: {atr:.2f} ({atr_pct*100:.2f}%) | regime={atr_regime} | "
        f"expected_move={expected_move_pct*100:.2f}% | TH: {TH:.2f}",
        flush=True
    )
    print(f"→ SCORE={score:.3f}", flush=True)
    print("───────────────────────────────────────────────", flush=True)

    if atr_regime == "too_low":
        print(f"[FILTER] HOLD: volatility too low (ATR%={atr_pct*100:.2f}%)", flush=True)
        return "HOLD", round(abs(score), 3), diag

    if atr_regime == "too_high":
        print(f"[FILTER] HOLD: volatility too high (ATR%={atr_pct*100:.2f}%)", flush=True)
        return "HOLD", round(abs(score), 3), diag

    if expected_move_pct < MIN_EXPECTED_MOVE_PCT:
        print(
            f"[FILTER] HOLD: expected move too small "
            f"(expected={expected_move_pct*100:.2f}%, min={MIN_EXPECTED_MOVE_PCT*100:.2f}%)",
            flush=True
        )
        return "HOLD", round(abs(score), 3), diag

    if score >= TH:
        return "BUY", round(score, 3), diag
    if score <= -TH:
        return "SELL", round(-score, 3), diag
    return "HOLD", round(abs(score), 3), diag