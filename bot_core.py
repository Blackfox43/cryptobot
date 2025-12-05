# bot_core.py
"""
Core non-UI logic for CryptoBot.
Contains: Asset registry, price fetchers (Bybit + CoinGecko fallback), indicators,
poller thread, trade execution (paper/live stub), persistence.
Requires Streamlit's session state to be passed explicitly for thread/state management.
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
POLL_INTERVAL = 15 # Increased to 15s to respect API rate limits
HISTORY_POINTS = 300
STATE_FILE = "crypto_state.json"
BYBIT_TICKER_URL = "https://api.bybit.com/v5/market/tickers"
COINGECKO_URL = "https://api.coingecko.com/api/v3/simple/price"
BYBIT_ORDER_URL = "https://api.bybit.com/v5/order/create"
INITIAL_BALANCE = 10000.0

# API Keys (Read from environment variables)
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")

# ======================================================
#            ASSET CLASS AND REGISTRY
# ======================================================

class Asset:
    """A minimal class to hold asset information."""
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

# ======================================================
#               STATE PERSISTENCE
# ======================================================

def load_state():
    """Loads trading state from a local JSON file."""
    if not os.path.exists(STATE_FILE):
        return None
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading state: {e}")
        return None

def save_state(state):
    """Saves critical trading state to a local JSON file."""
    safe = {
        "balance": state["balance"],
        "shares": state["shares"],
        "avg_entry": state["avg_entry"],
        "trades": state["trades"],
        "symbol": state["symbol"],
        "symbol_name": state["symbol_name"]
    }
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(safe, f, indent=4)
    except Exception as e:
        print(f"Error saving state: {e}")

# ======================================================
#           PRICE PROVIDERS
# ======================================================

def fetch_price_bybit(asset: Asset):
    """Fetches the latest price from Bybit V5 (Linear Perpetual)."""
    try:
        params = {"category": "linear", "symbol": asset.bybit_symbol}
        r = requests.get(BYBIT_TICKER_URL, params=params, timeout=6)
        r.raise_for_status()
        data = r.json()
        if data.get("retCode") != 0:
            # print(f"Bybit API Error ({data.get('retCode')}): {data.get('retMsg')}")
            return None
        ticker_list = data.get("result", {}).get("list", [])
        if ticker_list:
            last_price_str = ticker_list[0].get("lastPrice")
            return float(last_price_str) if last_price_str is not None else None
    except requests.exceptions.RequestException as e:
        # print(f"Error fetching price from Bybit: {e}")
        pass
    except Exception:
        pass
    return None

def fetch_price_coingecko(asset: Asset):
    """Fetches the latest price from CoinGecko (Spot)."""
    if not asset.cg_id: return None
    try:
        params = {"ids": asset.cg_id, "vs_currencies": "usd"}
        r = requests.get(COINGECKO_URL, params=params, timeout=6)
        r.raise_for_status()
        data = r.json()
        
        # Check for rate limit error (429) and print a specific message
        if r.status_code == 429:
             print("[poller] HTTP 429 from CoinGecko: Rate Limit Exceeded.")
             return None

        price = data.get(asset.cg_id, {}).get("usd")
        return float(price) if price is not None else None
    except requests.exceptions.RequestException as e:
        # print(f"Error fetching price from CoinGecko: {e}")
        pass
    except Exception:
        pass
    return None

def fetch_price_with_fallback(asset: Asset):
    """Attempts Bybit, falls back to CoinGecko."""
    price = fetch_price_bybit(asset)
    if price is not None:
        return price, "Bybit"
    
    price = fetch_price_coingecko(asset)
    if price is not None:
        return price, "CoinGecko"
        
    return None, None

# ======================================================
#             POLLER THREAD
# ======================================================

def poller(asset_key, asset_obj, q, stop_event, poll_interval=POLL_INTERVAL):
    """Thread function to continuously poll the price."""
    fail_count = 0
    while not stop_event.is_set():
        try:
            price, provider = fetch_price_with_fallback(asset_obj)
            if price is not None:
                # Put the full context into the queue: (key, timestamp, price, provider)
                q.put((asset_key, datetime.now().strftime("%H:%M:%S"), price, provider))
                fail_count = 0
            else:
                fail_count += 1
                if fail_count > 5:
                    print(f"[{asset_key}] Price fetch failed 5 times in a row.")
                    fail_count = 0
        except Exception as e:
            print(f"[{asset_key}] Unhandled error in poller: {e}")

        # Sleep, but check the stop event status regularly
        for _ in range(int(poll_interval / 0.5)):
            if stop_event.is_set():
                break
            time.sleep(0.5)

# ======================================================
#             STREAMLIT STATE INTERFACE
# ======================================================

def init_session_state(st_session_state):
    """Initializes all Streamlit session state variables."""
    if "state" not in st_session_state:
        saved = load_state()
        default_symbol = "btcusdt"
        default_asset = ASSETS[default_symbol]

        if saved:
            st_session_state.state = {
                **saved,
                "prices": [], "times": [], "sma": [], "providers": [],
                "current_price": 0.0, "active": False, "provider": "---"
            }
        else:
            st_session_state.state = {
                "prices": [], "times": [], "sma": [], "providers": [],
                "balance": INITIAL_BALANCE, "shares": 0.0,
                "avg_entry": 0.0, "trades": [],
                "active": False, "current_price": 0.0, "provider": "---",
                "symbol": default_symbol, "symbol_name": default_asset.name
            }

    if "price_queue" not in st_session_state:
        st_session_state.price_queue = queue.Queue()

    if "stop_event" not in st_session_state:
        st_session_state.stop_event = threading.Event()

    if "poll_thread" not in st_session_state:
        st_session_state.poll_thread = None

    if "config" not in st_session_state:
        # Default strategy parameters
        st_session_state.config = {
            "SMA_PERIOD": 20, "BUY_DIP": 0.005, 
            "TP": 0.01, "SL": 0.02, "INVEST_PCT": 0.90
        }

def reset_state(st_session_state):
    """Resets the state and removes the history file."""
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
    
    # Ensure all threads are stopped before resetting
    stop_bot(st_session_state)
    
    current_asset_key = st_session_state.state["symbol"]
    current_asset = ASSETS[current_asset_key]

    st_session_state.state = {
        "prices": [], "times": [], "sma": [], "providers": [],
        "balance": INITIAL_BALANCE, "shares": 0.0, "avg_entry": 0.0,
        "trades": [], "active": False, "current_price": 0.0, "provider": "---",
        "symbol": current_asset_key, "symbol_name": current_asset.name
    }
    st_session_state.price_queue = queue.Queue()
    st_session_state.stop_event = threading.Event()
    st_session_state.poll_thread = None
    print("State reset completed.")


def start_bot(st_session_state):
    """Starts the price poller thread."""
    state = st_session_state.state
    if not state["active"]:
        # Stop any potentially lingering thread first
        stop_bot(st_session_state) 
        
        asset_key = state["symbol"]
        current_asset = ASSETS[asset_key]
        
        st_session_state.stop_event.clear()
        t = threading.Thread(
            target=poller,
            args=(asset_key, current_asset, st_session_state.price_queue, st_session_state.stop_event),
            daemon=True
        )
        t.start()
        st_session_state.poll_thread = t
        state["active"] = True
        print(f"Bot started for {current_asset.bybit_symbol}.")

def stop_bot(st_session_state):
    """Stops the price poller thread."""
    state = st_session_state.state
    if state["active"] and st_session_state.poll_thread:
        st_session_state.stop_event.set()
        # Wait for the thread to finish cleanly
        st_session_state.poll_thread.join(timeout=POLL_INTERVAL + 5) 
        st_session_state.poll_thread = None
        state["active"] = False
        print("Bot stopped.")

def process_price_updates(st_session_state):
    """Pulls data from the queue and updates the state and runs trading logic."""
    state = st_session_state.state
    config = st_session_state.config
    
    while not st_session_state.price_queue.empty():
        asset_key, ts, price, provider = st_session_state.price_queue.get()
        
        # Only process if it's the currently selected asset
        if asset_key != state["symbol"]:
            continue 

        state["current_price"] = price
        state["provider"] = provider
        state["prices"].append(price)
        state["times"].append(ts)
        state["providers"].append(provider)

        # Maintain a clean history
        if len(state["prices"]) > HISTORY_POINTS:
            state["prices"].pop(0)
            state["times"].pop(0)
            state["providers"].pop(0)
            if state["sma"]:
                state["sma"].pop(0)

        # Calculate Simple Moving Average (SMA)
        if len(state["prices"]) >= config["SMA_PERIOD"]:
            state["sma"].append(np.mean(state["prices"][-config["SMA_PERIOD"]:]))
        else:
            state["sma"].append(None)

        # Trading Logic
        if state["active"] and state["sma"][-1] is not None:
            sma = state["sma"][-1]

            # BUY Signal
            if state["shares"] == 0 and price < sma * (1 - config["BUY_DIP"]):
                invest = state["balance"] * config["INVEST_PCT"]
                if invest > 0:
                    state["shares"] = invest / price
                    state["balance"] -= invest
                    state["avg_entry"] = price
                    state["trades"].insert(0, {
                        "time": ts, "type": "BUY",
                        "price": price, "amount": state["shares"], "pnl": 0
                    })
                    save_state(state)
                    print(f"TRADE: BUY {state['shares']:.4f} @ {price:.4f}")

            # SELL Signal
            elif state["shares"] > 0:
                pct = (price - state["avg_entry"]) / state["avg_entry"]
                
                # Check for Take Profit (TP) or Stop Loss (SL)
                if pct >= config["TP"]:
                    trade_type = "SELL (TP)"
                    is_sell = True
                elif pct <= -config["SL"]:
                    trade_type = "SELL (SL)"
                    is_sell = True
                else:
                    is_sell = False
                
                if is_sell:
                    revenue = state["shares"] * price
                    pnl = revenue - (state["shares"] * state["avg_entry"])

                    state["balance"] += revenue
                    state["trades"].insert(0, {
                        "time": ts, "type": trade_type,
                        "price": price, "amount": state["shares"], "pnl": pnl
                    })
                    print(f"TRADE: {trade_type} {state['shares']:.4f} @ {price:.4f} | PNL: {pnl:.2f}")

                    state["shares"] = 0
                    state["avg_entry"] = 0
                    save_state(state)

def get_connection_status(st_session_state):
    """Calculates and returns the connection status label."""
    state = st_session_state.state
    if len(state["times"]) > 0:
        last_ts = state["times"][-1]
        try:
            last_dt = datetime.strptime(last_ts, "%H:%M:%S")
            delay = (datetime.now() - last_dt).total_seconds()
        except ValueError:
            delay = POLL_INTERVAL * 2 

        if delay < POLL_INTERVAL + 5: 
            conn = "🟢 Live"
        elif delay < 60:
            conn = "🟡 Slow"
        else:
            conn = "🔴 Disconnected"
    else:
        conn = "🔴 Waiting"
    return conn

def get_equity_and_profit(st_session_state):
    """Calculates total equity and profit."""
    state = st_session_state.state
    equity = state["balance"] + state["shares"] * state["current_price"]
    profit = equity - INITIAL_BALANCE
    return equity, profit

def get_asset_config_and_current_asset(st_session_state):
    """Retrieves asset registry and current asset object."""
    state = st_session_state.state
    asset_key = state["symbol"]
    return ASSETS, ASSETS[asset_key]
