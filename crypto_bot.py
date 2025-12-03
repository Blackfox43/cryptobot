# crypto_bot.py
import streamlit as st
import threading
import queue
import requests
import json
import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# -------------------------
# Configuration
# -------------------------
STATE_FILE = "bot_state.json"
DEFAULT_SYMBOL = "BTCUSDT"
MAX_POINTS = 500

# CoinGecko mapping for supported symbols
COINGECKO_IDS = {
    "btcusdt": "bitcoin",
    "ethusdt": "ethereum",
    "solusdt": "solana",
    "dogeusdt": "dogecoin"
}
COINGECKO_PRICE_URL = "https://api.coingecko.com/api/v3/simple/price"

# -------------------------
# Persistence helpers
# -------------------------
def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                data = json.load(f)
                required = {"balance", "shares", "avg_entry", "trades", "symbol", "symbol_name"}
                if required.issubset(set(data.keys())):
                    return data
                else:
                    print("State file missing keys; starting fresh.")
                    return None
        except Exception as e:
            print("Error reading state file:", e)
            return None
    return None

def save_state(data):
    try:
        saveable = {
            "balance": data.get("balance", 10000.0),
            "shares": data.get("shares", 0.0),
            "avg_entry": data.get("avg_entry", 0.0),
            "trades": data.get("trades", []),
            "symbol": data.get("symbol", DEFAULT_SYMBOL),
            "symbol_name": data.get("symbol_name", "Asset")
        }
        with open(STATE_FILE, "w") as f:
            json.dump(saveable, f, indent=2)
    except Exception as e:
        print("Error saving state:", e)

# -------------------------
# Streamlit session init
# -------------------------
st.set_page_config(page_title="CoinGecko Hybrid Trader", layout="wide", page_icon="📈")

if "data" not in st.session_state:
    loaded = load_state()
    if loaded:
        st.session_state.data = {
            **loaded,
            "prices": [],
            "times": [],
            "sma": [],
            "current_price": 0.0,
            "active": False,
            "symbol_name": loaded.get("symbol_name", "Asset")
        }
        st.toast("Loaded persisted state", icon="💾")
    else:
        st.session_state.data = {
            "prices": [], "times": [], "sma": [],
            "balance": 10000.0, "shares": 0.0, "avg_entry": 0.0,
            "trades": [], "active": False, "current_price": 0.0,
            "symbol": DEFAULT_SYMBOL.lower(), "symbol_name": "Bitcoin"
        }

# Thread-safe queue for passing polled prices to main thread
if "price_queue" not in st.session_state:
    st.session_state.price_queue = queue.Queue()

# Thread handle and stop event
if "poll_thread" not in st.session_state:
    st.session_state.poll_thread = None
if "poll_stop_event" not in st.session_state:
    st.session_state.poll_stop_event = None
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

# -------------------------
# CoinGecko poller thread
# -------------------------
def coingecko_poller(q: queue.Queue, stop_event: threading.Event, symbol: str, interval: float):
    """
    Poll CoinGecko REST endpoint every `interval` seconds and push (timestamp, price) to queue.
    The thread does not touch st.session_state or files.
    """
    cg_id = COINGECKO_IDS.get(symbol.lower())
    if not cg_id:
        print("[poller] Unsupported symbol for CoinGecko:", symbol)
        return

    params = {"ids": cg_id, "vs_currencies": "usd"}
    print(f"[poller] started for {symbol} -> {cg_id}, interval={interval}s")
    while not stop_event.is_set():
        try:
            resp = requests.get(COINGECKO_PRICE_URL, params=params, timeout=8)
            if resp.status_code == 200:
                payload = resp.json()
                price = payload.get(cg_id, {}).get("usd")
                if price is not None:
                    ts = datetime.now().strftime("%H:%M:%S")
                    q.put((ts, float(price)))
                else:
                    print("[poller] missing price in response:", payload)
            else:
                # Log non-200; CoinGecko may rate-limit (429) or deny (4xx)
                print(f"[poller] HTTP {resp.status_code} from CoinGecko: {resp.text[:200]}")
            # sleep with a small granularity so stop_event can be checked often
            slept = 0.0
            while slept < interval and not stop_event.is_set():
                time.sleep(0.2)
                slept += 0.2
        except Exception as e:
            print("[poller] exception:", e)
            # backoff briefly after errors
            slept = 0.0
            backoff = max(1.0, interval)
            while slept < backoff and not stop_event.is_set():
                time.sleep(0.2)
                slept += 0.2
    print("[poller] stopping thread for", symbol)

def start_poller(symbol: str, interval: float):
    # stop previous if running
    if st.session_state.get("poll_stop_event"):
        st.session_state.poll_stop_event.set()
        st.session_state.poll_stop_event = None
        st.session_state.poll_thread = None
        # flush queue
        while not st.session_state.price_queue.empty():
            try:
                st.session_state.price_queue.get_nowait()
            except Exception:
                break

    stop_event = threading.Event()
    st.session_state.poll_stop_event = stop_event
    t = threading.Thread(target=coingecko_poller, args=(st.session_state.price_queue, stop_event, symbol, interval), daemon=True)
    t.start()
    st.session_state.poll_thread = t
    print("[main] started poller for", symbol, "interval", interval)

def stop_poller():
    if st.session_state.get("poll_stop_event"):
        st.session_state.poll_stop_event.set()
        print("[main] signaled poller stop")
    st.session_state.poll_thread = None

# -------------------------
# UI: Sidebar controls
# -------------------------
st.sidebar.title("🤖 Control Panel")

symbol_map = [
    ("BTCUSDT", "Bitcoin"),
    ("ETHUSDT", "Ethereum"),
    ("SOLUSDT", "Solana"),
    ("DOGEUSDT", "Dogecoin"),
]

selected = st.sidebar.selectbox("Select Asset", symbol_map, format_func=lambda x: x[1], index=0)
selected_symbol, selected_name = selected

# Poll interval input (CoinGecko rate limits: be conservative; default 5s)
poll_interval = st.sidebar.number_input("Poll interval (seconds)", min_value=2.0, max_value=60.0, value=5.0, step=1.0)

# handle symbol change
if selected_symbol.lower() != st.session_state.data.get("symbol", DEFAULT_SYMBOL).lower():
    st.session_state.data["symbol"] = selected_symbol.lower()
    st.session_state.data["symbol_name"] = selected_name
    st.session_state.data["prices"] = []
    st.session_state.data["times"] = []
    st.session_state.data["sma"] = []
    st.sidebar.warning("Symbol changed. Reconnecting to new price feed...")
    # restart poller automatically
    start_poller(selected_symbol, poll_interval)

# Bot toggle
if st.sidebar.button("🟢 START BOT" if not st.session_state.data["active"] else "🔴 STOP BOT"):
    st.session_state.data["active"] = not st.session_state.data["active"]
    if st.session_state.data["active"]:
        st.toast("Bot Activated! Listening for trade signals...", icon="🚀")
    else:
        st.toast("Bot Stopped. Holding position if any.", icon="🛑")

# Reset session
def reset_session():
    st.session_state.data = {
        "prices": [], "times": [], "sma": [],
        "balance": 10000.0, "shares": 0.0, "avg_entry": 0.0,
        "trades": [], "active": False, "current_price": 0.0,
        "symbol": st.session_state.data.get("symbol", DEFAULT_SYMBOL),
        "symbol_name": st.session_state.data.get("symbol_name", "Asset")
    }
    if os.path.exists(STATE_FILE):
        try:
            os.remove(STATE_FILE)
        except Exception as e:
            print("Error deleting state file:", e)
    while not st.session_state.price_queue.empty():
        try:
            st.session_state.price_queue.get_nowait()
        except Exception:
            break
    st.toast("Session reset and persistence removed.", icon="💥")

st.sidebar.markdown("---")
st.sidebar.button("💥 RESET BALANCE & HISTORY", on_click=reset_session)
st.sidebar.markdown("---")
st.sidebar.subheader("Strategy Parameters")

sma_period = st.sidebar.slider("SMA Period (points)", 5, 100, 20, 1, key="sma_period")
buy_dip_pct_ui = st.sidebar.number_input("Buy below SMA by (%)", min_value=0.01, max_value=50.0, value=0.05, step=0.01, format="%.2f")
buy_dip_pct = buy_dip_pct_ui / 100.0
invest_percent = st.sidebar.slider("Invest % of Balance", 1, 100, 90, 1) / 100.0
take_profit_pct_ui = st.sidebar.number_input("Take Profit at (%)", min_value=0.01, max_value=100.0, value=0.10, step=0.01, format="%.2f")
take_profit_pct = take_profit_pct_ui / 100.0
stop_loss_pct_ui = st.sidebar.number_input("Stop Loss at (%)", min_value=0.01, max_value=100.0, value=0.20, step=0.01, format="%.2f")
stop_loss_pct = stop_loss_pct_ui / 100.0

# -------------------------
# Initialization: load some history via CoinGecko (if empty)
# -------------------------
def fetch_coingecko_history(symbol: str, points: int = 100):
    """
    CoinGecko has OHLC & market_chart endpoints but rate-limits heavy use.
    For a light history we will poll the simple price repeatedly using REST.
    Here we'll try to fetch the market_chart for last 'days' = 1 to get frequent points.
    If it fails, return empty.
    """
    cg_id = COINGECKO_IDS.get(symbol.lower())
    if not cg_id:
        return []
    url = f"https://api.coingecko.com/api/v3/coins/{cg_id}/market_chart"
    params = {"vs_currency": "usd", "days": "1"}  # 1 day of history
    try:
        r = requests.get(url, params=params, timeout=8)
        if r.status_code == 200:
            payload = r.json()
            prices = payload.get("prices", [])  # list of [ms, price]
            # Reduce/align to 'points' number of samples if needed
            if not prices:
                return []
            # take last `points` samples evenly
            slice_step = max(1, len(prices) // points)
            sampled = prices[-points * slice_step::slice_step][:points]
            results = []
            for p in sampled:
                ts = datetime.fromtimestamp(p[0] / 1000.0).strftime("%H:%M:%S")
                results.append((ts, float(p[1])))
            return results
        else:
            print(f"[history] CoinGecko history HTTP {r.status_code}")
            return []
    except Exception as e:
        print("[history] exception:", e)
        return []

if len(st.session_state.data["prices"]) == 0:
    # Attempt a lightweight history load (best-effort)
    hist = fetch_coingecko_history(st.session_state.data.get("symbol", DEFAULT_SYMBOL), points=100)
    if hist:
        times, prices = zip(*hist)
        st.session_state.data["times"] = list(times[-MAX_POINTS:])
        st.session_state.data["prices"] = list(prices[-MAX_POINTS:])
        # compute sma aligned
        st.session_state.data["sma"] = []
        for i in range(len(st.session_state.data["prices"])):
            if i + 1 >= sma_period:
                st.session_state.data["sma"].append(float(np.mean(st.session_state.data["prices"][i + 1 - sma_period : i + 1])))
            else:
                st.session_state.data["sma"].append(None)
        st.session_state.data["current_price"] = st.session_state.data["prices"][-1]
        print("[main] loaded initial history points:", len(st.session_state.data["prices"]))

# ensure poller running
if st.session_state.get("poll_thread") is None or not getattr(st.session_state.poll_thread, "is_alive", lambda: False)():
    start_poller(st.session_state.data.get("symbol", DEFAULT_SYMBOL), poll_interval)

# -------------------------
# Main thread: drain queue and update state
# -------------------------
q = st.session_state.price_queue
drained = False
while not q.empty():
    try:
        ts, price = q.get_nowait()
    except Exception:
        break
    drained = True
    data = st.session_state.data

    # Append price/time
    data["current_price"] = float(price)
    data["prices"].append(float(price))
    data["times"].append(ts)

    # Trim
    if len(data["prices"]) > MAX_POINTS:
        data["prices"].pop(0)
        data["times"].pop(0)
        if len(data["sma"]) > 0:
            data["sma"].pop(0)

    # Compute SMA (append None until enough points)
    if len(data["prices"]) >= sma_period:
        sma_val = float(np.mean(data["prices"][-sma_period:]))
        data["sma"].append(sma_val)
    else:
        data["sma"].append(None)

    # Trading logic (main thread)
    if data.get("active", False) and data["sma"][-1] is not None:
        current_sma = data["sma"][-1]
        price_f = float(price)

        # BUY
        if data.get("shares", 0.0) == 0.0 and price_f < current_sma * (1.0 - buy_dip_pct):
            invest_amount = data["balance"] * invest_percent
            if invest_amount > 0 and price_f > 0:
                shares_bought = invest_amount / price_f
                data["shares"] = shares_bought
                data["balance"] -= invest_amount
                data["avg_entry"] = price_f
                now = ts
                data["trades"].insert(0, {"time": now, "type": "BUY", "price": price_f, "amount": shares_bought, "pnl": 0.0})
                save_state(data)
                print(f"[trade] BUY {shares_bought:.6f} @ {price_f:.2f}")

        # SELL
        elif data.get("shares", 0.0) > 0.0:
            profit_pct = (price_f - data["avg_entry"]) / data["avg_entry"]
            if profit_pct > take_profit_pct or profit_pct < -stop_loss_pct:
                revenue = data["shares"] * price_f
                pnl = revenue - (data["shares"] * data["avg_entry"])
                data["balance"] += revenue
                now = ts
                data["trades"].insert(0, {"time": now, "type": "SELL", "price": price_f, "amount": data["shares"], "pnl": pnl})
                print(f"[trade] SELL {data['shares']:.6f} @ {price_f:.2f} pnl={pnl:.2f}")
                data["shares"] = 0.0
                data["avg_entry"] = 0.0
                save_state(data)

# persist after processing
if drained:
    try:
        save_state(st.session_state.data)
    except Exception as e:
        print("Error saving after processing:", e)

# -------------------------
# UI: Main dashboard
# -------------------------
st.title(f"Live {st.session_state.data.get('symbol_name', 'Asset')} Trader (CoinGecko)")

data = st.session_state.data
current_equity = data["balance"] + data["shares"] * data["current_price"]
profit = current_equity - 10000.0

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Live Price", f"${data['current_price']:.2f}")
with col2:
    st.metric("Total Equity", f"${current_equity:.2f}")
with col3:
    st.metric("Total Profit", f"${profit:.2f}", delta=f"${profit:.2f}", delta_color="normal")
with col4:
    st.metric("Holdings", f"{data.get('shares', 0.0):.6f}")

# Chart
if len(data["prices"]) > 0:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["times"], y=data["prices"], mode="lines", name="Price"))
    valid_sma = [(data["times"][i], data["sma"][i]) for i in range(len(data["sma"])) if data["sma"][i] is not None]
    if valid_sma:
        sma_t, sma_v = zip(*valid_sma)
        fig.add_trace(go.Scatter(x=list(sma_t), y=list(sma_v), mode="lines", name=f"SMA ({sma_period})", line=dict(dash="dash")))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning(f"Waiting for live market data... (need {sma_period} points for SMA)")

# Trade log
st.subheader("Transaction History")
if len(data["trades"]) > 0:
    df = pd.DataFrame(data["trades"])
    df = df.rename(columns={"time": "Time", "type": "Type", "price": "Price", "amount": "Amount", "pnl": "P&L"})
    df["Price"] = df["Price"].apply(lambda x: f"${x:.2f}")
    df["P&L"] = df["P&L"].apply(lambda x: f"${x:.2f}")
    st.dataframe(df, use_container_width=True)
else:
    st.text("No trades yet. Press START BOT to begin paper trading.")

# Connection status & manual controls
st.markdown("---")
colA, colB, colC = st.columns([2, 1, 1])
with colA:
    poll_alive = st.session_state.get("poll_thread") is not None and getattr(st.session_state.poll_thread, "is_alive", lambda: False)()
    st.write(f"Poller alive: {poll_alive}  |  Queue size: {st.session_state.price_queue.qsize()}  |  Poll interval: {poll_interval}s")
with colB:
    if st.button("Reconnect Poller"):
        start_poller(st.session_state.data.get("symbol", DEFAULT_SYMBOL), poll_interval)
with colC:
    if st.button("Force Save State"):
        save_state(st.session_state.data)
        st.toast("State saved", icon="💾")

# -------------------------
# Auto-refresh: re-run every 1s to process incoming queue items
# -------------------------
nowt = time.time()
if nowt - st.session_state.last_refresh > 1.0:
    st.session_state.last_refresh = nowt
    # Use modern API
    st.rerun()
