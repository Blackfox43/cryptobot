# bot_core.py
"""
Core non-UI logic for CryptoBot.
Contains: Asset registry, price fetchers (Bybit + CoinGecko fallback), indicators,
poller thread, trade execution (paper/live stub), persistence.
Requires the Streamlit session_state object to be passed to session-aware functions.
"""

import os
import time
import json
import threading
import queue
from datetime import datetime, timezone
import requests
import numpy as np

# Configuration
POLL_INTERVAL = 15  # seconds between polls (helps with rate limits)
HISTORY_POINTS = 300
STATE_FILE = "crypto_state.json"
BYBIT_TICKER_URL = "https://api.bybit.com/v5/market/tickers"
COINGECKO_URL = "https://api.coingecko.com/api/v3/simple/price"
INITIAL_BALANCE = 10000.0

# API Keys (Read from environment variables)
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")

# ---------------------------------------------------------
# Asset registry
# ---------------------------------------------------------
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
    "maticusdt": Asset("maticusdt", "Polygon", "MATICUSDT", cg_id="polygon"),
    "shibusdt": Asset("shibusdt", "Shiba Inu", "SHIBUSDT", cg_id="shiba-inu"),
}

# ---------------------------------------------------------
# Persistence
# ---------------------------------------------------------
def load_state():
    if not os.path.exists(STATE_FILE):
        return None
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[state] Error loading state: {e}")
        return None


def save_state(state):
    # Save only essential persistent fields
    try:
        safe = {
            "balance": state.get("balance", INITIAL_BALANCE),
            "shares": state.get("shares", 0.0),
            "avg_entry": state.get("avg_entry", 0.0),
            "trades": state.get("trades", []),
            "symbol": state.get("symbol", "btcusdt"),
            "symbol_name": state.get("symbol_name", "Bitcoin"),
        }
        with open(STATE_FILE, "w") as f:
            json.dump(safe, f, indent=4)
    except Exception as e:
        print(f"[state] Error saving state: {e}")


# ---------------------------------------------------------
# Price fetchers (Bybit public ticker + CoinGecko fallback)
# ---------------------------------------------------------
def fetch_price_bybit(asset: Asset):
    """Fetch latest price from Bybit public tickers (returns float or None)."""
    try:
        params = {"category": "linear", "symbol": asset.bybit_symbol}
        resp = requests.get(BYBIT_TICKER_URL, params=params, timeout=6)
        # don't raise for status here; handle gracefully
        data = resp.json() if resp.status_code == 200 else {}
        # Bybit returns retCode 0 for success (V5)
        if data.get("retCode", 0) != 0:
            return None
        ticker_list = data.get("result", {}).get("list", [])
        if ticker_list and isinstance(ticker_list, list):
            last_price_str = ticker_list[0].get("lastPrice")
            if last_price_str is not None:
                return float(last_price_str)
    except Exception as e:
        # Print a short debug message (helps streamlit logs)
        # print(f"[bybit] fetch error: {e}")
        pass
    return None


def fetch_price_coingecko(asset: Asset):
    """Fetch latest price from CoinGecko (returns float or None)."""
    if not asset.cg_id:
        return None
    try:
        params = {"ids": asset.cg_id, "vs_currencies": "usd"}
        resp = requests.get(COINGECKO_URL, params=params, timeout=6)
        # Handle rate limit explicitly
        if resp.status_code == 429:
            print("[coingecko] HTTP 429: Rate limit.")
            return None
        resp.raise_for_status()
        data = resp.json()
        price = data.get(asset.cg_id, {}).get("usd")
        if price is not None:
            return float(price)
    except requests.exceptions.RequestException:
        pass
    except Exception:
        pass
    return None


def fetch_price_with_fallback(asset: Asset):
    """Try Bybit first, then CoinGecko. Returns (price, provider) or (None, None)."""
    p = fetch_price_bybit(asset)
    if p is not None:
        return p, "Bybit"
    p = fetch_price_coingecko(asset)
    if p is not None:
        return p, "CoinGecko"
    return None, None


# ---------------------------------------------------------
# Poller thread
# ---------------------------------------------------------
def poller(asset_key, asset_obj, q, stop_event, poll_interval=POLL_INTERVAL):
    """
    Poll price in a loop and push (asset_key, iso_ts, price, provider) into the queue.
    Uses UTC ISO timestamps so downstream parsing is robust.
    """
    fail_count = 0
    while not stop_event.is_set():
        try:
            price, provider = fetch_price_with_fallback(asset_obj)
            if price is not None:
                ts_iso = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
                q.put((asset_key, ts_iso, price, provider))
                fail_count = 0
            else:
                fail_count += 1
                if fail_count >= 5:
                    print(f"[{asset_key}] consecutive fetch failures.")
                    fail_count = 0
        except Exception as e:
            print(f"[{asset_key}] poller unhandled error: {e}")

        # Sleep in short increments so stop_event is responsive
        slept = 0.0
        interval = float(poll_interval)
        step = 0.5
        while slept < interval:
            if stop_event.is_set():
                break
            time.sleep(step)
            slept += step


# ---------------------------------------------------------
# Streamlit session state interface
# ---------------------------------------------------------
def init_session_state(st_session_state):
    """Initialize Streamlit session state (safe to call multiple times)."""
    if "state" not in st_session_state:
        saved = load_state()
        default_symbol = "btcusdt"
        default_asset = ASSETS[default_symbol]

        if saved:
            st_session_state.state = {
                **saved,
                "prices": [],
                "times": [],
                "sma": [],
                "providers": [],
                "current_price": 0.0,
                "active": False,
                "provider": "---",
            }
        else:
            st_session_state.state = {
                "prices": [],
                "times": [],
                "sma": [],
                "providers": [],
                "balance": INITIAL_BALANCE,
                "shares": 0.0,
                "avg_entry": 0.0,
                "trades": [],
                "active": False,
                "current_price": 0.0,
                "provider": "---",
                "symbol": default_symbol,
                "symbol_name": default_asset.name,
            }

    # queues and threading primitives
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
        }


def reset_state(st_session_state):
    """Reset the app state and stop any running poller."""
    try:
        stop_bot(st_session_state)
    except Exception:
        pass

    if os.path.exists(STATE_FILE):
        try:
            os.remove(STATE_FILE)
        except Exception:
            pass

    asset_key = st_session_state.state.get("symbol", "btcusdt")
    asset = ASSETS.get(asset_key, ASSETS["btcusdt"])

    st_session_state.state = {
        "prices": [],
        "times": [],
        "sma": [],
        "providers": [],
        "balance": INITIAL_BALANCE,
        "shares": 0.0,
        "avg_entry": 0.0,
        "trades": [],
        "active": False,
        "current_price": 0.0,
        "provider": "---",
        "symbol": asset_key,
        "symbol_name": asset.name,
    }
    st_session_state.price_queue = queue.Queue()
    st_session_state.stop_event = threading.Event()
    st_session_state.poll_thread = None
    print("[state] reset completed.")


def start_bot(st_session_state):
    """Start the poller for the currently-selected asset (no-op if already running)."""
    state = st_session_state.state
    # Ensure primitives exist
    if "stop_event" not in st_session_state:
        st_session_state.stop_event = threading.Event()
    if "price_queue" not in st_session_state:
        st_session_state.price_queue = queue.Queue()

    # If a thread exists, try to stop and restart it to pick up any config changes
    if st_session_state.poll_thread and st_session_state.poll_thread.is_alive():
        try:
            stop_bot(st_session_state)
        except Exception:
            pass

    asset_key = state.get("symbol", "btcusdt")
    asset = ASSETS.get(asset_key, ASSETS["btcusdt"])

    st_session_state.stop_event.clear()
    t = threading.Thread(
        target=poller,
        args=(asset_key, asset, st_session_state.price_queue, st_session_state.stop_event),
        daemon=True,
    )
    t.start()
    st_session_state.poll_thread = t
    state["active"] = True
    print(f"[bot] started for {asset.bybit_symbol}")


def stop_bot(st_session_state):
    """Stop poller thread if running."""
    try:
        st_session_state.stop_event.set()
    except Exception:
        pass

    # join the thread if possible
    thr = getattr(st_session_state, "poll_thread", None)
    if thr:
        try:
            thr.join(timeout=POLL_INTERVAL + 5)
        except Exception:
            pass
    st_session_state.poll_thread = None
    # Clear active flag
    try:
        st_session_state.state["active"] = False
    except Exception:
        pass
    print("[bot] stopped.")


# ---------------------------------------------------------
# Queue handling + trading logic
# ---------------------------------------------------------
def process_price_updates(st_session_state):
    """
    Drain the queue, update state arrays, compute SMA and run simple trading logic.
    Should be called from the main Streamlit thread frequently (e.g. once per rerun).
    """
    cfg = st_session_state.config
    state = st_session_state.state
    q = st_session_state.price_queue

    while not q.empty():
        try:
            asset_key, ts_iso, price, provider = q.get_nowait()
        except Exception:
            break

        # Only process updates for the currently selected asset
        if asset_key != state.get("symbol"):
            continue

        # parse and create a human time for display
        try:
            dt = datetime.fromisoformat(ts_iso)
            display_ts = dt.astimezone(timezone.utc).strftime("%H:%M:%S")
        except Exception:
            display_ts = datetime.utcnow().strftime("%H:%M:%S")

        state["current_price"] = price
        state["provider"] = provider
        state["prices"].append(price)
        state["times"].append(display_ts)
        state["providers"].append(provider)

        # trim history
        if len(state["prices"]) > HISTORY_POINTS:
            state["prices"].pop(0)
            state["times"].pop(0)
            state["providers"].pop(0)
            if state["sma"]:
                state["sma"].pop(0)

        # SMA
        if len(state["prices"]) >= cfg["SMA_PERIOD"]:
            state["sma"].append(np.mean(state["prices"][-cfg["SMA_PERIOD"] :]))
        else:
            state["sma"].append(None)

        # Trading logic (paper trading)
        if state.get("active") and state["sma"] and state["sma"][-1] is not None:
            sma = state["sma"][-1]
            # BUY
            if state.get("shares", 0) == 0 and price < sma * (1 - cfg["BUY_DIP"]):
                invest = state.get("balance", INITIAL_BALANCE) * cfg["INVEST_PCT"]
                if invest > 0:
                    state["shares"] = invest / price
                    state["balance"] = state.get("balance", INITIAL_BALANCE) - invest
                    state["avg_entry"] = price
                    state["trades"].insert(
                        0,
                        {
                            "time": display_ts,
                            "type": "BUY",
                            "price": price,
                            "amount": state["shares"],
                            "pnl": 0,
                        },
                    )
                    save_state(state)
                    print(f"[trade] BUY {state['shares']:.6f} @ {price:.4f}")

            # SELL
            elif state.get("shares", 0) > 0:
                pct = (price - state.get("avg_entry", 0.0)) / max(state.get("avg_entry", 1e-9), 1e-9)
                is_sell = False
                trade_type = "SELL"
                if pct >= cfg["TP"]:
                    trade_type = "SELL (TP)"
                    is_sell = True
                elif pct <= -cfg["SL"]:
                    trade_type = "SELL (SL)"
                    is_sell = True

                if is_sell:
                    revenue = state["shares"] * price
                    pnl = revenue - (state["shares"] * state.get("avg_entry", 0.0))
                    state["balance"] = state.get("balance", INITIAL_BALANCE) + revenue
                    state["trades"].insert(
                        0,
                        {
                            "time": display_ts,
                            "type": trade_type,
                            "price": price,
                            "amount": state["shares"],
                            "pnl": pnl,
                        },
                    )
                    print(f"[trade] {trade_type} {state['shares']:.6f} @ {price:.4f} PNL: {pnl:.2f}")
                    state["shares"] = 0.0
                    state["avg_entry"] = 0.0
                    save_state(state)


# ---------------------------------------------------------
# Utility functions used by UI
# ---------------------------------------------------------
def get_connection_status(st_session_state):
    """Return a friendly connection label based on the last update time."""
    state = st_session_state.state
    if state.get("times"):
        last_display = state["times"][-1]
        try:
            # times are stored as HH:MM:SS (display), so compute using UTC now
            last_dt = datetime.strptime(last_display, "%H:%M:%S").time()
            now_utc = datetime.utcnow().time()
            # compute rough delta in seconds by converting to seconds-from-midnight
            def tod_seconds(t):
                return t.hour * 3600 + t.minute * 60 + t.second

            delay = abs(tod_seconds(now_utc) - tod_seconds(last_dt))
        except Exception:
            # fallback: if any parsing fails, treat as disconnected
            delay = POLL_INTERVAL * 10

        if delay < POLL_INTERVAL + 5:
            return "🟢 Live"
        if delay < 60:
            return "🟡 Slow"
        return "🔴 Disconnected"
    return "🔴 Waiting"


def get_equity_and_profit(st_session_state):
    state = st_session_state.state
    equity = state.get("balance", INITIAL_BALANCE) + state.get("shares", 0.0) * state.get("current_price", 0.0)
    profit = equity - INITIAL_BALANCE
    return equity, profit


def get_asset_config_and_current_asset(st_session_state):
    state = st_session_state.state
    key = state.get("symbol", "btcusdt")
    return ASSETS, ASSETS.get(key, ASSETS["btcusdt"])
