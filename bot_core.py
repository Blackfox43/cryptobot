# bot_core.py
"""
Core non-UI logic for CryptoBot.
Contains: Asset registry, price fetchers (Bybit + CoinGecko fallback), indicators,
poller thread, trade execution (paper/live stub), persistence.
No Streamlit imports here so it can be unit tested and imported by main.py.
"""

import os
import time
import json
import hmac
import hashlib
import threading
import queue
from datetime import datetime
import requests
import numpy as np

# Configuration
POLL_INTERVAL = 3
HISTORY_POINTS = 300
STATE_FILE = "crypto_state.json"
BYBIT_TICKER_URL = "https://api.bybit.com/v5/market/tickers"
COINGECKO_URL = "https://api.coingecko.com/api/v3/simple/price"
BYBIT_ORDER_URL = "https://api.bybit.com/v5/order/create"

BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")

# Asset class and registry
class Asset:
    def __init__(self, key, name, bybit_symbol, cg_id=None):
        self.key = key
        self.name = name
        self.bybit_symbol = bybit_symbol
        self.cg_id = cg_id

ASSETS = {
    "btcusdt": Asset("btcusdt", "Bitcoin", "BTCUSDT", cg_id="bitcoin"),
    "ethusdt": Asset("ethusdt", "Ethereum", "ETHUSDT", cg_id="ethereum"),
    "solusdt": Asset("solusdt", "Solana", "SOLUSDT", cg_id="solana"),
}

# Persistence
def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def save_state(state):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print("save_state error:", e)

# Indicators
def sma(values, period):
    if len(values) < period:
        return None
    return float(np.mean(values[-period:]))

def ema_series(values, period):
    if len(values) == 0:
        return np.array([])
    values = np.array(values, dtype=float)
    alpha = 2 / (period + 1)
    out = np.empty(len(values), dtype=float)
    out[:] = np.nan
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out

def macd(values, fast=12, slow=26, signal=9):
    if len(values) < slow + signal:
        return None, None, None
    arr = np.array(values, dtype=float)
    fast_e = ema_series(arr, fast)
    slow_e = ema_series(arr, slow)
    macd_line = fast_e - slow_e
    valid = ~np.isnan(macd_line)
    if valid.sum() < signal:
        return None, None, None
    macd_valid = macd_line[valid]
    sig_vals = ema_series(macd_valid, signal)
    sig_full = np.full_like(macd_line, np.nan)
    sig_full[np.where(valid)[0]] = sig_vals
    hist = macd_line - sig_full
    return macd_line.tolist(), sig_full.tolist(), hist.tolist()

def rsi(values, period=14):
    if len(values) < period + 1:
        return None
    arr = np.array(values, dtype=float)
    deltas = np.diff(arr)
    seed = deltas[:period]
    up = seed[seed > 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else np.inf
    rsi_series = np.empty(len(arr), dtype=float)
    rsi_series[:] = np.nan
    rsi_series[period] = 100 - 100 / (1 + rs)
    up_avg = up
    down_avg = down
    for i in range(period + 1, len(arr)):
        delta = deltas[i - 1]
        up_val = max(delta, 0)
        down_val = max(-delta, 0)
        up_avg = (up_avg * (period - 1) + up_val) / period
        down_avg = (down_avg * (period - 1) + down_val) / period
        rs = up_avg / down_avg if down_avg != 0 else np.inf
        rsi_series[i] = 100 - 100 / (1 + rs)
    valid = rsi_series[~np.isnan(rsi_series)]
    return float(valid[-1]) if len(valid) else None

# Price fetchers
def fetch_bybit_price(asset: Asset):
    try:
        r = requests.get(BYBIT_TICKER_URL, params={"category":"spot","symbol":asset.bybit_symbol}, timeout=6)
        if r.status_code == 200:
            payload = r.json()
            lst = payload.get("result", {}).get("list", [])
            if lst and isinstance(lst, list):
                p = lst[0].get("lastPrice")
                if p:
                    return float(p)
    except Exception:
        return None
    return None

def fetch_coingecko_price(asset: Asset):
    if not asset.cg_id:
        return None
    try:
        r = requests.get(COINGECKO_URL, params={"ids": asset.cg_id, "vs_currencies":"usd"}, timeout=6)
        if r.status_code == 200:
            d = r.json()
            p = d.get(asset.cg_id, {}).get("usd")
            if p is not None:
                return float(p)
    except Exception:
        return None
    return None

def fetch_price_with_fallback(asset: Asset):
    p = fetch_bybit_price(asset)
    if p is not None:
        return p, "bybit"
    p = fetch_coingecko_price(asset)
    if p is not None:
        return p, "coingecko"
    return None, None

# Order placement (minimal). Use testnet first.
def place_bybit_order(symbol, side, qty):
    if not (BYBIT_API_KEY and BYBIT_API_SECRET):
        return {"error": "no_api_keys"}
    body = {"category":"spot","symbol":symbol,"side":side,"orderType":"Market","qty":str(qty),"timeInForce":"GTC"}
    ts = str(int(time.time() * 1000))
    recv = "5000"
    body_str = json.dumps(body, separators=(",", ":"))
    pre = ts + BYBIT_API_KEY + recv + body_str
    sig = hmac.new(BYBIT_API_SECRET.encode(), pre.encode(), hashlib.sha256).hexdigest()
    headers = {"Content-Type":"application/json","X-BAPI-APIKEY":BYBIT_API_KEY,"X-BAPI-SIGN":sig,"X-BAPI-TIMESTAMP":ts,"X-BAPI-RECV-WINDOW":recv}
    try:
        r = requests.post(BYBIT_ORDER_URL, headers=headers, data=body_str, timeout=8)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

# Poller thread

def poller(asset_key, asset_obj, q, stop_event, poll_interval=POLL_INTERVAL):
    fail_count = 0
    while not stop_event.is_set():
        try:
            price, provider = fetch_price_with_fallback(asset_obj)
            if price is not None:
                q.put((asset_key, datetime.now().strftime("%H:%M:%S"), price, provider))
                fail_count = 0
            else:
                fail_count += 1
                if fail_count > 8:
                    time.sleep(5)
        except Exception as e:
            print("poller error:", e)
        time.sleep(poll_interval)

# Initialize default state
DEFAULT_PERSIST = {"balances": {}, "positions": {}, "trade_history": []}

if __name__ == "__main__":
    print("bot_core module loaded")
