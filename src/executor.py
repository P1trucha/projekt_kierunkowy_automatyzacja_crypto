# src/executor.py
from binance.client import Client
from binance.exceptions import BinanceAPIException
from .utils import IS_TESTNET
from decimal import Decimal, ROUND_DOWN


def make_client(api_key, api_secret):
    client = Client(api_key, api_secret, testnet=IS_TESTNET)
    if IS_TESTNET:
        client.API_URL = "https://testnet.binance.vision"
    return client


def market_order(client, symbol, side, quantity):
    try:
        return client.create_order(symbol=symbol, side=side, type="MARKET", quantity=quantity)
    except BinanceAPIException as e:
        raise RuntimeError(f"API error: {e}")


_symbol_filters_cache = {}


def _get_filters(client, symbol):
    if symbol in _symbol_filters_cache:
        return _symbol_filters_cache[symbol]

    info = client.get_symbol_info(symbol)
    if not info:
        raise RuntimeError(f"Symbol info not found for {symbol}")

    lot = next(f for f in info["filters"] if f["filterType"] == "LOT_SIZE")
    notion = next((f for f in info["filters"] if f["filterType"] in ("MIN_NOTIONAL", "NOTIONAL")), None)
    price_filter = next(f for f in info["filters"] if f["filterType"] == "PRICE_FILTER")

    step = Decimal(lot["stepSize"])
    min_qty = Decimal(lot["minQty"])
    min_notional = Decimal(notion["minNotional"]) if notion and "minNotional" in notion else Decimal("0")
    tick_size = Decimal(price_filter["tickSize"])

    _symbol_filters_cache[symbol] = (step, min_qty, min_notional, tick_size)
    return _symbol_filters_cache[symbol]


def quantize_qty(client, symbol, qty: float) -> float:
    step, min_qty, _, _ = _get_filters(client, symbol)
    q = Decimal(str(qty))
    q = (q / step).to_integral_value(rounding=ROUND_DOWN) * step
    if q < min_qty:
        return 0.0
    return float(q)


def quantize_price(client, symbol, price: float) -> float:
    _, _, _, tick_size = _get_filters(client, symbol)
    p = Decimal(str(price))
    p = (p / tick_size).to_integral_value(rounding=ROUND_DOWN) * tick_size
    return float(p)


def check_min_notional(client, symbol, qty: float, price: float) -> bool:
    _, _, min_notional, _ = _get_filters(client, symbol)
    return Decimal(str(qty)) * Decimal(str(price)) >= min_notional


def open_orders(client, symbol):
    return client.get_open_orders(symbol=symbol)


def cancel_all_orders(client, symbol):
    try:
        open_orders = client.get_open_orders(symbol=symbol)
    except BinanceAPIException as e:
        raise RuntimeError(f"Get open orders error: {e}")

    for o in open_orders:
        try:
            client.cancel_order(symbol=symbol, orderId=o["orderId"])
        except BinanceAPIException as e:
            # order już nie istnieje / został wykonany / anulowany
            if getattr(e, "code", None) == -2011 or "Unknown order sent" in str(e):
                print(f"[WARN] order already gone: {o.get('orderId')}", flush=True)
                continue
            raise RuntimeError(f"Cancel error: {e}")


def place_oco_takeprofit_stop(client, symbol, quantity, take_profit_price, stop_price, stop_limit_price):
    """
    OCO na SPOT: SELL z TP + SL.
    Dla SELL musi być:
        TP > current_price > stopPrice > stopLimitPrice
    """
    try:
        last_price = float(client.get_symbol_ticker(symbol=symbol)["price"])

        qty = quantize_qty(client, symbol, quantity)
        tp = quantize_price(client, symbol, take_profit_price)
        stop = quantize_price(client, symbol, stop_price)
        stop_limit = quantize_price(client, symbol, stop_limit_price)

        # Bezpieczne poprawki relacji cen dla SELL OCO
        # 1) TP musi być nad rynkiem
        if tp <= last_price:
            tp = quantize_price(client, symbol, last_price * 1.002)

        # 2) stop musi być pod rynkiem
        if stop >= last_price:
            stop = quantize_price(client, symbol, last_price * 0.995)

        # 3) stop_limit musi być poniżej stop
        if stop_limit >= stop:
            stop_limit = quantize_price(client, symbol, stop * 0.999)

        # 4) finalna walidacja
        if not (tp > last_price > stop > stop_limit):
            raise RuntimeError(
                f"OCO invalid after normalization: tp={tp}, last={last_price}, stop={stop}, stop_limit={stop_limit}"
            )

        return client.create_oco_order(
            symbol=symbol,
            side="SELL",
            quantity=qty,
            price=f"{tp:.8f}",
            stopPrice=f"{stop:.8f}",
            stopLimitPrice=f"{stop_limit:.8f}",
            stopLimitTimeInForce="GTC"
        )

    except BinanceAPIException as e:
        raise RuntimeError(f"OCO error: {e}")