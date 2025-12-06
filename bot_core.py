# bot_core.py
"""
Upgraded core for CryptoBot with RSI + MACD indicator integration.

Provides:
- init_session_state(...)
- start_bot(...)
- stop_bot(...)
- process_price_updates(...)
- get_connection_status(...)
- get_equity_and_profit(...)
- reset_state(...)
- get_asset_config_and_current_asset(...)

Notes:
- Uses only standard libs + requests + numpy.
- Indicator computations are safe (recompute from available history each update).
- Designed to plug into the Main.py and ui.py previously provided.
"""

import os
import time
import json
import threading
import queue
from datetime import datetime
import requests
import numpy as np

# -------------------------
# CONFIG
# -------------------------
POLL_INTERVAL = 5                  # seconds between polls
HISTORY_POINTS = 500               # keep this many historical prices
STATE_FILE = "crypto_state.json"
BYBIT_TICKER_URL = "https://api.bybit.com/v5/market/tickers"
COINGECKO_URL = "https://api.coingecko.com/api/v3/simple/price"
INITIAL_BALANCE = 10000.0

# -------------------------
# ASSET DEFINITIONS
# -------------------------
class Asset:
    def __init__(self, key, name, bybit_symbol, cg_id=None):
        self.key = key
        self.name = name
        self.bybit_symbol = bybit_symbol
        self.cg_id = cg_id

ASSETS = {
    "btcusdt": Asset("btcusdt", "Bitcoin", "BTCUSDT", cg_id="bitcoin"),
    "ethusdt": Asset("ethusdt", "Ethereum", "ETHUSDT", cg_id="ethereum"),
    "bnbusdt": Asset("bnbusdt", "BNB", "BNBUSDT", cg_id="binancecoin"),
    "solusdt": Asset("solusdt", "Solana", "SOLUSDT", cg_id="solana"),
    "xrpusdt": Asset("xrpusdt", "XRP", "XRPUSDT", cg_id="ripple"),
    "dogeusdt": Asset("dogeusdt", "Dogecoin", "DOGEUSDT", cg_id="dogecoin"),
    "adausdt": Asset("adausdt", "Cardano", "ADAUSDT", cg_id="cardano"),
    "avaxusdt": Asset("avaxusdt", "Avalanche", "AVAXUSDT", cg_id="avalanche-2"),
    "dotusdt": Asset("dotusdt", "Polkadot", "DOTUSDT", cg_id="polkadot"),
    "linkusdt": Asset("linkusdt", "Chainlink", "LINKUSDT", cg_id="chainlink"),
    "maticusdt": Asset("maticusdt", "Polygon", "MATICUSDT", cg_id="matic-network"),
    "shibusdt": Asset("shibusdt", "Shiba Inu", "SHIBUSDT", cg_id="shiba-inu"),
}

# -------------------------
# PERSISTENCE
# -------------------------
def load_state():
    if not os.path.exists(STATE_FILE):
        return None
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return None

def save_state(state):
    # Save critical, small subset
    safe = {
        "balance": state.get("balance", INITIAL_BALANCE),
        "shares": state.get("shares", 0.0),
        "avg_entry": state.get("avg_entry", 0.0),
        "trades": state.get("trades", []),
        "symbol": state.get("symbol", "btcusdt"),
        "symbol_name": state.get("symbol_name", ASSETS["btcusdt"].name)
    }
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(safe, f, indent=4)
    except Exception:
        pass

# -------------------------
# PRICE PROVIDERS (Bybit -> CoinGecko fallback)
# -------------------------
def fetch_price_bybit(asset: Asset):
    try:
        params = {"category": "linear", "symbol": asset.bybit_symbol}
        r = requests.get(BYBIT_TICKER_URL, params=params, timeout=6)
        r.raise_for_status()
        data = r.json()
        # Bybit V5 returns retCode == 0 for success
        if data.get("retCode") == 0:
            lst = data.get("result", {}).get("list", [])
            if lst:
                price_s = lst[0].get("lastPrice")
                if price_s is not None:
                    return float(price_s)
    except Exception:
        pass
    return None

def fetch_price_coingecko(asset: Asset):
    if not asset.cg_id:
        return None
    try:
        r = requests.get(COINGECKO_URL, params={"ids": asset.cg_id, "vs_currencies": "usd"}, timeout=6)
        r.raise_for_status()
        data = r.json()
        price = data.get(asset.cg_id, {}).get("usd")
        if price is not None:
            return float(price)
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

# -------------------------
# INDICATOR HELPERS
# -------------------------
def simple_moving_average(prices, period):
    if len(prices) < period:
        return None
    return float(np.mean(prices[-period:]))

def ema(series, period):
    """
    Exponential Moving Average computed on a 1D numeric iterable.
    Returns last EMA value computed over full series.
    """
    if len(series) < period or period <= 0:
        return None
    arr = np.array(series, dtype=float)
    alpha = 2.0 / (period + 1.0)
    ema_val = arr[0]
    for v in arr[1:]:
        ema_val = alpha * v + (1 - alpha) * ema_val
    return float(ema_val)

def compute_rsi(prices, period=14):
    """
    Classic RSI implementation:
      1) compute deltas
      2) average gains & losses (Wilder's smoothing via exponential smoothing)
      3) calculate RSI last value
    Returns float RSI (0..100) or None if not enough data.
    """
    if len(prices) < period + 1:
        return None
    deltas = np.diff(np.array(prices, dtype=float))
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    # Wilder smoothing: first average is simple mean, then exponential smoothing
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    if avg_loss == 0 and avg_gain == 0:
        return 50.0  # neutral RSI
    # Continue smoothing through remainder
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    rs = avg_gain / avg_loss if avg_loss != 0 else np.inf
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return float(rsi)

def compute_macd(prices, fast=12, slow=26, signal=9):
    """
    Compute MACD line, signal line, and histogram.
    Returns tuple (macd, macd_signal, macd_hist) using the end of series.
    If not enough data, returns (None, None, None).
    """
    if len(prices) < slow + signal:
        return None, None, None
    # Use EMA computed on the full available prices
    fast_ema = ema(prices[-(slow + 50):], fast)  # use a slice with enough points
    slow_ema = ema(prices[-(slow + 50):], slow)
    if fast_ema is None or slow_ema is None:
        return None, None, None
    macd_line = fast_ema - slow_ema

    # For signal line, compute MACD series then EMA of it.
    # Build MACD series by computing fast/slow EMA on progressively growing window:
    macd_series = []
    window = max(slow + signal, len(prices))
    # However, building a full macd_series every call can be heavy; we approximate:
    # Compute past MACD points by computing fast & slow EMA on rolling subsets
    for i in range(len(prices)):
        sub = prices[: i + 1]
        if len(sub) < slow:
            macd_series.append(None)
            continue
        f = ema(sub, fast)
        s = ema(sub, slow)
        macd_series.append(f - s if (f is not None and s is not None) else None)
    # Filter None and take last (signal) EMA over macd_series values
    macd_vals = [v for v in macd_series if v is not None]
    if len(macd_vals) < signal:
        return macd_line, None, None
    # compute signal as EMA of macd_vals with period=signal
    signal_val = ema(macd_vals[-(signal + 50):], signal)
    if signal_val is None:
        return macd_line, None, None
    hist = macd_line - signal_val
    return float(macd_line), float(signal_val), float(hist)

# -------------------------
# POLLER (thread)
# -------------------------
def poller(asset_key, asset_obj, q, stop_event, poll_interval=POLL_INTERVAL):
    fail_count = 0
    while not stop_event.is_set():
        try:
            price, provider = fetch_price_with_fallback(asset_obj)
            if price is not None:
                q.put((asset_key, datetime.now().strftime("%H:%M:%S"), float(price), provider))
                fail_count = 0
            else:
                fail_count += 1
                if fail_count >= 8:
                    # rotate or just reset counter to avoid spam
                    fail_count = 0
        except Exception:
            pass
        # sleep while checking stop_event so thread can be stopped promptly
        for _ in range(max(1, int(poll_interval / 0.5))):
            if stop_event.is_set():
                break
            time.sleep(0.5)

# -------------------------
# STREAMLIT INTERFACE FUNCTIONS
# -------------------------
def init_session_state(st_session_state):
    """Initialize session state variables used by the app."""
    if "state" not in st_session_state:
        saved = load_state()
        default_symbol = "btcusdt"
        default_asset = ASSETS[default_symbol]
        if saved:
            st_session_state.state = {
                **saved,
                "prices": [], "times": [], "sma": [], "providers": [],
                "current_price": 0.0, "active": False, "provider": "---",
                # indicators:
                "rsi": None, "macd": None, "macd_signal": None, "macd_hist": None
            }
        else:
            st_session_state.state = {
                "prices": [], "times": [], "sma": [], "providers": [],
                "balance": INITIAL_BALANCE, "shares": 0.0,
                "avg_entry": 0.0, "trades": [],
                "active": False, "current_price": 0.0, "provider": "---",
                "symbol": default_symbol, "symbol_name": default_asset.name,
                "rsi": None, "macd": None, "macd_signal": None, "macd_hist": None
            }

    if "price_queue" not in st_session_state:
        st_session_state.price_queue = queue.Queue()
    if "stop_event" not in st_session_state:
        st_session_state.stop_event = threading.Event()
    if "poll_thread" not in st_session_state:
        st_session_state.poll_thread = None
    if "config" not in st_session_state:
        # Add RSI & MACD config defaults
        st_session_state.config = {
            "SMA_PERIOD": 20,
            "BUY_DIP": 0.005,
            "TP": 0.01,
            "SL": 0.02,
            "INVEST_PCT": 0.9,
            "RSI_PERIOD": 14,
            "RSI_OVERSOLD": 30.0,
            "RSI_OVERBOUGHT": 70.0,
            "MACD_FAST": 12,
            "MACD_SLOW": 26,
            "MACD_SIGNAL": 9,
            "USE_RSI": True,
            "USE_MACD": True,
            "REQUIRE_ALL_SIGNALS": False  # If True require SMA+RSI+MACD for buy
        }

def reset_state(st_session_state):
    """Stop threads, delete persisted state, reset in-memory state."""
    try:
        if os.path.exists(STATE_FILE):
            os.remove(STATE_FILE)
    except Exception:
        pass
    stop_bot(st_session_state)
    cur = st_session_state.state.get("symbol", "btcusdt")
    st_session_state.state = {
        "prices": [], "times": [], "sma": [], "providers": [],
        "balance": INITIAL_BALANCE, "shares": 0.0, "avg_entry": 0.0,
        "trades": [], "active": False, "current_price": 0.0, "provider": "---",
        "symbol": cur, "symbol_name": ASSETS[cur].name,
        "rsi": None, "macd": None, "macd_signal": None, "macd_hist": None
    }
    st_session_state.price_queue = queue.Queue()
    st_session_state.stop_event = threading.Event()
    st_session_state.poll_thread = None

def start_bot(st_session_state):
    """Start the poller thread for the currently selected asset."""
    state = st_session_state.state
    if state.get("active"):
        return
    # ensure previous threads stopped
    stop_bot(st_session_state)
    key = state.get("symbol", "btcusdt")
    asset = ASSETS[key]
    st_session_state.stop_event.clear()
    t = threading.Thread(target=poller, args=(key, asset, st_session_state.price_queue, st_session_state.stop_event), daemon=True)
    t.start()
    st_session_state.poll_thread = t
    state["active"] = True

def stop_bot(st_session_state):
    """Stop poller thread if running."""
    state = st_session_state.state
    if state.get("active") and st_session_state.poll_thread is not None:
        st_session_state.stop_event.set()
        try:
            st_session_state.poll_thread.join(timeout=POLL_INTERVAL + 5)
        except Exception:
            pass
        st_session_state.poll_thread = None
        state["active"] = False

def process_price_updates(st_session_state):
    """
    Consume queued price samples, update state history, compute indicators,
    and run trading logic (paper).
    """
    state = st_session_state.state
    cfg = st_session_state.config

    while not st_session_state.price_queue.empty():
        asset_key, ts, price, provider = st_session_state.price_queue.get()

        # only process updates for the currently selected asset
        if asset_key != state.get("symbol"):
            continue

        # update basic values
        state["current_price"] = float(price)
        state["provider"] = provider or state.get("provider", "---")
        state["prices"].append(float(price))
        state["times"].append(ts)
        state["providers"].append(provider)

        # trim history
        if len(state["prices"]) > HISTORY_POINTS:
            state["prices"].pop(0)
            state["times"].pop(0)
            state["providers"].pop(0)
            if state["sma"]:
                state["sma"].pop(0)

        # SMA
        sma_val = simple_moving_average(state["prices"], cfg.get("SMA_PERIOD", 20))
        state["sma"].append(sma_val)

        # RSI
        if cfg.get("USE_RSI", True):
            rsi_val = compute_rsi(state["prices"], period=cfg.get("RSI_PERIOD", 14))
            state["rsi"] = rsi_val
        else:
            state["rsi"] = None

        # MACD
        if cfg.get("USE_MACD", True):
            macd_line, macd_signal, macd_hist = compute_macd(
                state["prices"],
                fast=cfg.get("MACD_FAST", 12),
                slow=cfg.get("MACD_SLOW", 26),
                signal=cfg.get("MACD_SIGNAL", 9)
            )
            state["macd"] = macd_line
            state["macd_signal"] = macd_signal
            state["macd_hist"] = macd_hist
        else:
            state["macd"] = state["macd_signal"] = state["macd_hist"] = None

        # --- TRADING LOGIC (paper trading) ---
        # Keep behavior conservative:
        # - Require SMA to exist
        # - Optionally require RSI and MACD affirmation
        try:
            if state.get("active") and state["sma"] and state["sma"][-1] is not None:
                current_price = float(price)
                sma_now = float(state["sma"][-1])

                # buy_condition: price below SMA by BUY_DIP
                buy_by_sma = (current_price < sma_now * (1 - cfg.get("BUY_DIP", 0.005)))

                # rsi condition
                rsi_ok = True
                if cfg.get("USE_RSI", True):
                    rsi_val = state.get("rsi")
                    rsi_ok = (rsi_val is not None and rsi_val <= float(cfg.get("RSI_OVERSOLD", 30.0)))

                # macd condition (histogram bullish)
                macd_ok = True
                if cfg.get("USE_MACD", True):
                    macd_hist = state.get("macd_hist")
                    macd_ok = (macd_hist is not None and macd_hist > 0.0)

                # decide final buy decision
                if state.get("shares", 0.0) == 0:
                    if buy_by_sma:
                        if cfg.get("REQUIRE_ALL_SIGNALS", False):
                            final_buy = (buy_by_sma and rsi_ok and macd_ok)
                        else:
                            # require SMA and at least one of RSI or MACD (if enabled)
                            signals = []
                            if cfg.get("USE_RSI", True):
                                signals.append(rsi_ok)
                            if cfg.get("USE_MACD", True):
                                signals.append(macd_ok)
                            # If both indicators disabled, we accept SMA alone
                            if not signals:
                                final_buy = True
                            else:
                                final_buy = buy_by_sma and any(signals)
                        if final_buy:
                            invest = state.get("balance", INITIAL_BALANCE) * cfg.get("INVEST_PCT", 0.9)
                            if invest > 0 and current_price > 0:
                                amount = invest / current_price
                                state["shares"] = amount
                                state["balance"] -= invest
                                state["avg_entry"] = current_price
                                state["trades"].insert(0, {
                                    "time": ts,
                                    "type": "BUY",
                                    "price": current_price,
                                    "amount": amount,
                                    "pnl": 0.0
                                })
                                save_state(state)

                else:
                    # There's an open position: check TP/SL or indicator-based exit
                    pct = (current_price - state.get("avg_entry", current_price)) / (state.get("avg_entry", current_price) or 1.0)
                    take_profit = pct >= cfg.get("TP", 0.01)
                    stop_loss = pct <= -cfg.get("SL", 0.02)

                    # additional sell signals: RSI overbought, MACD histogram negative
                    rsi_sell = False
                    if cfg.get("USE_RSI", True):
                        rsv = state.get("rsi")
                        if rsv is not None and rsv >= float(cfg.get("RSI_OVERBOUGHT", 70.0)):
                            rsi_sell = True

                    macd_sell = False
                    if cfg.get("USE_MACD", True):
                        mh = state.get("macd_hist")
                        if mh is not None and mh < 0.0:
                            macd_sell = True

                    # decide final sell
                    final_sell = take_profit or stop_loss or rsi_sell or macd_sell
                    if final_sell:
                        revenue = state.get("shares", 0.0) * current_price
                        pnl = revenue - (state.get("shares", 0.0) * state.get("avg_entry", 0.0))
                        state["balance"] += revenue
                        state["trades"].insert(0, {
                            "time": ts,
                            "type": "SELL",
                            "price": current_price,
                            "amount": state.get("shares", 0.0),
                            "pnl": pnl
                        })
                        state["shares"] = 0.0
                        state["avg_entry"] = 0.0
                        save_state(state)
        except Exception:
            # keep processing tolerant; do not crash the main UI
            pass

def get_connection_status(st_session_state):
    state = st_session_state.state
    if state.get("times"):
        last_ts = state["times"][-1]
        try:
            last_dt = datetime.strptime(last_ts, "%H:%M:%S")
            delay = (datetime.now() - last_dt).total_seconds()
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
    s = st_session_state.state
    equity = s.get("balance", INITIAL_BALANCE) + s.get("shares", 0.0) * s.get("current_price", 0.0)
    profit = equity - INITIAL_BALANCE
    return equity, profit

def get_asset_config_and_current_asset(st_session_state):
    key = st_session_state.state.get("symbol", "btcusdt")
    return ASSETS, ASSETS[key]
