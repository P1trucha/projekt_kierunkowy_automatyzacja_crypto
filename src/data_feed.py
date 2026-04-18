from binance.client import Client
from binance import ThreadedWebsocketManager
from .utils import IS_TESTNET

def history_client(api_key, api_secret):
    c = Client(api_key, api_secret, testnet=IS_TESTNET)
    if IS_TESTNET:
        c.API_URL = "https://testnet.binance.vision"
    return c

def get_klines(client, symbol, interval="1h", limit=500):
    return client.get_klines(symbol=symbol, interval=interval, limit=limit)

def start_ticker_ws(api_key, api_secret, symbol, on_msg):
    twm = ThreadedWebsocketManager(api_key=api_key, api_secret=api_secret, testnet=IS_TESTNET)
    twm.start()
    twm.start_symbol_ticker_socket(callback=on_msg, symbol=symbol)
    return twm
