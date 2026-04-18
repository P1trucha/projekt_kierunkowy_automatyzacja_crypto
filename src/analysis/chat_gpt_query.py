# src/analysis/chat_gpt_query.py
import os, re, json, time
from datetime import datetime
from typing import Dict, Any, Optional

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GPT_TEMPERATURE = float(os.getenv("GPT_TEMPERATURE", "0.1"))
GPT_MIN_PERIOD_SEC = int(os.getenv("GPT_MIN_PERIOD_SEC", "120"))  # ochronnie co >= 2 min
NEWS_ENABLED = os.getenv("NEWS_ENABLED", "true").lower() == "true"
NEWS_TIMEOUT = float(os.getenv("NEWS_TIMEOUT_SEC", "8"))
NEWS_CACHE_SEC = int(os.getenv("NEWS_CACHE_SEC", "900"))

_client: Optional[OpenAI] = None
_last_gpt_call_ts: Optional[float] = None
_news_cache = {"ts": 0.0, "text": "No headlines"}

def _client_ok() -> Optional[OpenAI]:
    global _client
    if not OPENAI_KEY:
        return None
    if _client is None:
        _client = OpenAI(api_key=OPENAI_KEY)
    return _client

def _now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

def _throttle_ok() -> bool:
    global _last_gpt_call_ts
    now = time.time()
    if _last_gpt_call_ts is None or (now - _last_gpt_call_ts) >= GPT_MIN_PERIOD_SEC:
        _last_gpt_call_ts = now
        return True
    return False

def _safe_json(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, re.S | re.I)
    if m:
        t = m.group(1).strip()
    if not t.startswith("{"):
        m2 = re.search(r"(\{.*\})", t, re.S)
        if m2:
            t = m2.group(1).strip()
    try:
        return json.loads(t)
    except Exception:
        return {}

def _fetch_news() -> str:
    global _news_cache
    if not NEWS_ENABLED:
        return "News disabled"
    if time.time() - _news_cache["ts"] < NEWS_CACHE_SEC:
        return _news_cache["text"]
    try:
        url = (
            "https://newsdata.io/api/1/news"
            "?apikey=pub_49247c1dd2349efc0d5b64b8b1cbf65db4"
            "&q=bitcoin+crypto+market&language=en"
        )
        r = requests.get(url, timeout=NEWS_TIMEOUT)
        r.raise_for_status()
        items = (r.json() or {}).get("results", [])[:5]
        titles = [ (it.get("title") or "").strip()[:140] for it in items if it.get("title") ]
        txt = " | ".join(titles) if titles else "No fresh headlines"
        _news_cache = {"ts": time.time(), "text": txt}
        return txt
    except Exception as e:
        txt = f"News error: {e}"
        _news_cache = {"ts": time.time(), "text": txt}
        return txt

def analyze_market_with_gpt(
    symbol: str,
    rsi: float,
    macd: float,
    macd_signal: float,
    ema50: float,
    ema200: float,
    boll_up: float,
    boll_down: float,
    current_price: float,
    trend: str,
    *,
    mode: str = "sentiment",       # "sentiment" | "entry_advice"
    volume: float = 0.0,           # dla XTB (lot/”BTC”), opcjonalnie
    spread_pct: float = 0.0,       # ważne na XTB
    avg_price_7d: float = 0.0,     # jeżeli chcesz użyć w ocenie
    hour: Optional[int] = None,
    weekday: Optional[int] = None
) -> Dict[str, Any]:
    """
    Zwraca ustandaryzowany JSON:
    {
      "mode": "sentiment|entry_advice",
      "decision": "LONG|SHORT|HOLD",        # tylko dla entry_advice
      "confidence": 0.0-1.0,                # tylko dla entry_advice
      "sentiment": "bullish|bearish|neutral",
      "explanation": "..."
    }
    Brak klucza/limitów/timeoutów => neutral.
    """
    client = _client_ok()
    if client is None:
        return {
            "mode": mode, "decision": "HOLD", "confidence": 0.0,
            "sentiment": "neutral", "explanation": "Missing OPENAI_API_KEY"
        }

    if not _throttle_ok():
        return {
            "mode": mode, "decision": "HOLD", "confidence": 0.0,
            "sentiment": "neutral", "explanation": "Throttled"
        }

    headlines = _fetch_news() if mode == "sentiment" else ""
    now = _now()

    if mode == "sentiment":
        prompt = (
            "You are a crypto market analyst. Return only strict JSON.\n"
            f"Time: {now}\nSymbol: {symbol}\n\n"
            "--- Technical Indicators ---\n"
            f"RSI: {rsi}\nMACD: {macd}, Signal: {macd_signal}\n"
            f"EMA50: {ema50}, EMA200: {ema200}\n"
            f"Bollinger: upper={boll_up}, lower={boll_down}\n"
            f"Price: {current_price}\nTrend (from TA/ML): {trend}\n\n"
            "--- Recent Headlines ---\n"
            f"{headlines}\n\n"
            'Respond strictly in JSON like: {"sentiment":"bullish|bearish|neutral","explanation":"<=2 sentences"}'
        )
    else:  # entry_advice  (zastępuje Twój XTB prompt, ale zwraca JSON)
        cautious = "true" if (avg_price_7d and current_price < 0.97 * avg_price_7d) else "false"
        prompt = (
            "You are an experienced trader. Return only strict JSON.\n"
            f"Time: {now} | TF: H1/H4 | Hour: {hour} | Weekday: {weekday}\n"
            f"Symbol: {symbol} (can be CFD or spot; spread may matter)\n\n"
            "--- Technical Data ---\n"
            f"RSI: {rsi:.2f}\nMACD: {macd:.2f}, Signal: {macd_signal:.2f}\n"
            f"Trend(SMA50 vs SMA200): {trend}\n"
            f"Spread(%): {spread_pct:.2f}\n"
            f"Volume(unit): {volume:.6f}\n"
            f"Price: {current_price:.2f}\n"
            f"AvgPrice7d: {avg_price_7d:.2f}\n"
            f"CautiousMode: {cautious}\n\n"
            "Rules (soft):\n"
            "- LONG if momentum aligns: RSI<30 rising OR MACD>Signal & rising & trend up; price << avg7d with rising momentum allowed but cautious.\n"
            "- SHORT if overbought: RSI>70 falling OR MACD<Signal & trend down; price >> avg7d with weakening indicators.\n"
            "- HOLD if signals conflict or spread is high.\n\n"
            'Respond strictly in JSON like: {"decision":"LONG|SHORT|HOLD","confidence":0.0-1.0,"explanation":"<=2 sentences","sentiment":"bullish|bearish|neutral"}'
        )

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=GPT_TEMPERATURE,
            max_tokens=160
        )
        content = (resp.choices[0].message.content or "").strip()
        data = _safe_json(content)

        # Normalizacja wyjścia:
        out = {
            "mode": mode,
            "decision": str(data.get("decision", "HOLD")).upper(),
            "confidence": float(data.get("confidence", 0.0)),
            "sentiment": str(data.get("sentiment", "neutral")).lower(),
            "explanation": str(data.get("explanation", ""))[:280]
        }
        if out["decision"] not in ("LONG", "SHORT", "HOLD"):
            out["decision"] = "HOLD"
        if out["sentiment"] not in ("bullish", "bearish", "neutral"):
            out["sentiment"] = "neutral"
        out["confidence"] = max(0.0, min(1.0, out["confidence"]))
        return out

    except Exception as e:
        return {
            "mode": mode, "decision": "HOLD", "confidence": 0.0,
            "sentiment": "neutral", "explanation": f"API error: {e}"
        }
