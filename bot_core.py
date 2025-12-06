# bot_core.py
"""
Stable core engine for CryptoBot (single-asset mode).
- Compatible with Python 3.11 and Streamlit 1.36
- Uses Bybit (primary) -> CoinGecko (fallback)
- Indicators implemented with NumPy (no pandas)
- Threaded poller + safe queue
Exports:
init_session_state, start_bot, stop_bot, process_price_updates,
get_equity_and_profit, get_connection_status, reset_state,
get_asset_config_and_current_asset, POLL_INTERVAL
"""

import os
import time
import json
import threading
import queue
from datetime import datetime
import requests
import numpy as np

# --- config ---
POLL_INTERVAL = 6            # seconds between price polls
HISTORY_POINTS = 300
STATE_FILE = "crypto_state.json"
BYBIT_TICKER_URL = "https://api.bybit.com/v5/market/tickers"
COINGECKO_URL = "https://api.coingecko.com/api/v3/simple/price"
INITIAL_BALANCE = 10000.0

# Bybit creds optional (for real trading later)
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")

# --- asset registry ---
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
    "dogeusdt": Asset("dogeusdt", "Dogecoin", "DOGEUSDT", cg_id="dogecoin"),
    "adausdt": Asset("adausdt", "Cardano", "ADAUSDT", cg_id="cardano"),
}

# --- persistence ---
def load_state():
    if not os.path.exists(STATE_FILE):
        return None
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return None

def save_state(data):
    try:
        safe = {
            "balance": data.get("balance", INITIAL_BALANCE),
            "shares": data.get("shares", 0.0),
            "avg_entry": data.get("avg_entry", 0.0),
            "trades": data.get("trades", []),
            "symbol": data.get("symbol", "btcusdt"),
            "symbol_name": data.get("symbol_name", ASSETS["btcusdt"].name)
        }
        with open(STATE_FILE, "w") as f:
            json.dump(safe, f, indent=2)
    except Exception as e:
        print("[save_state]", e)

# --- price providers ---
def fetch_price_bybit(asset: Asset):
    try:
        params = {"category": "linear", "symbol": asset.bybit_symbol}
        r = requests.get(BYBIT_TICKER_URL, params=params, timeout=6)
        r.raise_for_status()
        d = r.json()
        # Bybit returns result.list[0].lastPrice typically
        lst = d.get("result", {}).get("list", [])
        if lst and isinstance(lst, list):
            last = lst[0].get("lastPrice") or lst[0].get("last_tick") or None
            if last is not None:
                return float(last)
    except Exception:
        pass
    return None

def fetch_price_coingecko(asset: Asset):
    try:
        if not asset.cg_id:
            return None
        r = requests.get(COINGECKO_URL, params={"ids": asset.cg_id, "vs_currencies": "usd"}, timeout=6)
        r.raise_for_status()
        d = r.json()
        p = d.get(asset.cg_id, {}).get("usd")
        if p is not None:
            return float(p)
    except Exception:
        pass
    return None

def fetch_price_with_fallback(asset: Asset):
    p = fetch_price_bybit(asset)
    if p is not None:
        return p, "Bybit"
    p = fetch_price_coingecko(asset)
    if p is not None:
        return p, "CoinGecko"
    return None, None

# --- numeric indicators (NumPy only) ---
def sma(values, period):
    if len(values) < period:
        return None
    return float(np.mean(values[-period:]))

def ema_numpy(values, period):
    # returns last EMA value
    if len(values) < 1:
        return None
    vals = np.array(values, dtype=float)
    alpha = 2.0 / (period + 1)
    ema = vals[0]
    for v in vals[1:]:
        ema = alpha * v + (1 - alpha) * ema
    return float(ema)

def macd_numpy(values, fast=12, slow=26, signal=9):
    if len(values) < slow + signal:
        return None, None, None
    fast_ema = ema_numpy(values[-(slow+10):], fast)  # approximate using tail
    slow_ema = ema_numpy(values[-(slow+10):], slow)
    if fast_ema is None or slow_ema is None:
        return None, None, None
    # build MACD series approximate
    vals = np.array(values, dtype=float)
    ema_fast_series = []
    ema_slow_series = []
    alpha_fast = 2.0 / (fast + 1)
    alpha_slow = 2.0 / (slow + 1)
    ef = vals[0]
    es = vals[0]
    for v in vals:
        ef = alpha_fast * v + (1 - alpha_fast) * ef
        es = alpha_slow * v + (1 - alpha_slow) * es
        ema_fast_series.append(ef)
        ema_slow_series.append(es)
    macd_line = np.array(ema_fast_series) - np.array(ema_slow_series)
    # signal line
    sig = macd_line[0]
    alpha_sig = 2.0 / (signal + 1)
    for m in macd_line[1:]:
        sig = alpha_sig * m + (1 - alpha_sig) * sig
    hist = macd_line[-1] - sig
    return float(macd_line[-1]), float(sig), float(hist)

def rsi_numpy(values, period=14):
    if len(values) < period + 1:
        return None
    deltas = np.diff(np.array(values, dtype=float))
    ups = np.where(deltas > 0, deltas, 0.0)
    downs = np.where(deltas < 0, -deltas, 0.0)
    # Wilder's smoothing (simple average here)
    avg_gain = np.mean(ups[-period:]) if len(ups[-period:]) > 0 else 0.0
    avg_loss = np.mean(downs[-period:]) if len(downs[-period:]) > 0 else 0.0
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

# --- poller thread ---
def poller(asset_key, asset_obj, q, stop_event, poll_interval=POLL_INTERVAL):
    fail = 0
    while not stop_event.is_set():
        price, provider = fetch_price_with_fallback(asset_obj)
        if price is not None:
            ts = datetime.utcnow().strftime("%H:%M:%S")
            q.put((ts, price, provider))
            fail = 0
        else:
            fail += 1
            if fail >= 6:
                print(f"[poller] {asset_key} failed to fetch price repeatedly")
                fail = 0
        # sleep chunked so stop_event can interrupt
        slept = 0.0
        step = 0.5
        while slept < poll_interval:
            if stop_event.is_set():
                break
            time.sleep(step)
            slept += step

# --- session state interface ---
def init_session_state(st_session_state):
    # single top-level state dict for simplicity
    if "state" not in st_session_state:
        saved = load_state()
        default_symbol = "btcusdt"
        asset_obj = ASSETS[default_symbol]
        if saved:
            st_session_state.state = {
                **saved,
                "prices": [], "times": [], "sma": [],
                "current_price": saved.get("current_price", 0.0),
                "provider": saved.get("provider", "---"),
                "active": False,
                "symbol": saved.get("symbol", default_symbol),
                "symbol_name": saved.get("symbol_name", asset_obj.name)
            }
        else:
            st_session_state.state = {
                "prices": [], "times": [], "sma": [],
                "balance": INITIAL_BALANCE,
                "shares": 0.0,
                "avg_entry": 0.0,
                "trades": [],
                "active": False,
                "current_price": 0.0,
                "provider": "---",
                "symbol": default_symbol,
                "symbol_name": asset_obj.name
            }
    if "price_queue" not in st_session_state:
        st_session_state.price_queue = queue.Queue()
    if "stop_event" not in st_session_state:
        st_session_state.stop_event = threading.Event()
    if "poll_thread" not in st_session_state:
        st_session_state.poll_thread = None
    if "config" not in st_session_state:
        st_session_state.config = {
            "SMA_PERIOD": 20,
            "BUY_DIP": 0.005,
            "TP": 0.01,
            "SL": 0.02,
            "INVEST_PCT": 0.90,
            "RSI_PERIOD": 14,
            "MACD_FAST": 12,
            "MACD_SLOW": 26,
            "MACD_SIGNAL": 9
        }

def reset_state(st_session_state):
    # Stop poller then reset values and remove persisted file
    try:
        stop_bot(st_session_state)
    except Exception:
        pass
    if os.path.exists(STATE_FILE):
        try:
            os.remove(STATE_FILE)
        except Exception:
            pass
    init_session_state(st_session_state)
    print("[reset_state] reset complete")

def start_bot(st_session_state):
    state = st_session_state.state
    if state.get("active"):
        return
    # stop any existing thread
    stop_bot(st_session_state)
    symbol = state.get("symbol", "btcusdt")
    asset = ASSETS.get(symbol, list(ASSETS.values())[0])
    # clear old queue to avoid large backlog
    while not st_session_state.price_queue.empty():
        try:
            st_session_state.price_queue.get_nowait()
        except Exception:
            break
    # start thread
    st_session_state.stop_event.clear()
    t = threading.Thread(target=poller, args=(symbol, asset, st_session_state.price_queue, st_session_state.stop_event), daemon=True)
    t.start()
    st_session_state.poll_thread = t
    state["active"] = True
    print(f"[start_bot] started for {symbol}")

def stop_bot(st_session_state):
    state = st_session_state.state
    if not state.get("active"):
        return
    try:
        st_session_state.stop_event.set()
        thr = st_session_state.poll_thread
        if thr is not None:
            thr.join(timeout=POLL_INTERVAL + 3)
    except Exception:
        pass
    st_session_state.poll_thread = None
    state["active"] = False
    print("[stop_bot] stopped")

def process_price_updates(st_session_state):
    state = st_session_state.state
    cfg = st_session_state.config
    q = st_session_state.price_queue
    updated = False
    while not q.empty():
        try:
            ts, price, provider = q.get_nowait()
        except Exception:
            break
        state["current_price"] = price
        state["provider"] = provider
        state["prices"].append(price)
        state["times"].append(ts)
        if len(state["prices"]) > HISTORY_POINTS:
            state["prices"].pop(0)
            state["times"].pop(0)
            if state["sma"]:
                state["sma"].pop(0)
        # SMA
        if len(state["prices"]) >= cfg["SMA_PERIOD"]:
            state["sma"].append(sma(state["prices"], cfg["SMA_PERIOD"]))
        else:
            state["sma"].append(None)
        # Trading logic (paper trading)
        if state.get("active") and state["sma"] and state["sma"][-1] is not None:
            cur_sma = state["sma"][-1]
            # buy condition
            if state.get("shares", 0.0) == 0 and price < cur_sma * (1 - cfg["BUY_DIP"]):
                invest = state.get("balance", INITIAL_BALANCE) * cfg["INVEST_PCT"]
                if invest > 0:
                    state["shares"] = invest / price
                    state["balance"] = state.get("balance", INITIAL_BALANCE) - invest
                    state["avg_entry"] = price
                    state["trades"].insert(0, {"time": ts, "type": "BUY", "price": price, "amount": state["shares"], "pnl": 0})
                    save_state(state)
                    updated = True
            # sell condition
            elif state.get("shares", 0.0) > 0:
                pct = (price - state.get("avg_entry", price)) / max(state.get("avg_entry", price), 1e-9)
                if pct >= cfg["TP"] or pct <= -cfg["SL"]:
                    revenue = state["shares"] * price
                    pnl = revenue - (state["shares"] * state.get("avg_entry", price))
                    state["balance"] = state.get("balance", INITIAL_BALANCE) + revenue
                    state["trades"].insert(0, {"time": ts, "type": "SELL", "price": price, "amount": state["shares"], "pnl": pnl})
                    state["shares"] = 0.0
                    state["avg_entry"] = 0.0
                    save_state(state)
                    updated = True
    if updated:
        save_state(state)

def get_connection_status(st_session_state):
    st = st_session_state.state
    if len(st["times"]) > 0:
        try:
            last_ts = st["times"][-1]
            last_dt = datetime.strptime(last_ts, "%H:%M:%S")
            delay = (datetime.utcnow() - last_dt).total_seconds()
        except Exception:
            delay = POLL_INTERVAL * 2
        if delay < POLL_INTERVAL + 5:
            return "🟢 Live"
        elif delay < 60:
            return "🟡 Slow"
        else:
            return "🔴 Disconnected"
    return "🔴 Waiting"

def get_equity_and_profit(st_session_state):
    st = st_session_state.state
    equity = st.get("balance", INITIAL_BALANCE) + st.get("shares", 0.0) * st.get("current_price", 0.0)
    profit = equity - INITIAL_BALANCE
    return equity, profit

def get_asset_config_and_current_asset(st_session_state):
    symbol = st_session_state.state.get("symbol", "btcusdt")
    return ASSETS, ASSETS.get(symbol, list(ASSETS.values())[0])
