# app.py
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

# Optional: websocket-client package
# pip install websocket-client
import websocket

# -------------------------
# Configuration / Constants
# -------------------------
STATE_FILE = "bot_state.json"
DEFAULT_SYMBOL = "BTCUSDT"
BINANCE_REST_TRADES = "https://api.binance.com/api/v3/trades"  # recent trades endpoint
BINANCE_WS_TEMPLATE = "wss://stream.binance.com:9443/ws/{}@trade"
MAX_POINTS = 500  # maximum points to keep in memory

# -------------------------
# Persistence helpers
# -------------------------
def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                data = json.load(f)
                # Ensure minimum required keys exist
                required = {"balance", "shares", "avg_entry", "trades", "symbol", "symbol_name"}
                if required.issubset(set(data.keys())):
                    return data
                else:
                    print("State file missing keys, ignoring.")
                    return None
        except Exception as e:
            print("Error loading state file:", e)
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
            "symbol_name": data.get("symbol_name", "Bitcoin")
        }
        with open(STATE_FILE, "w") as f:
            json.dump(saveable, f, indent=2)
    except Exception as e:
        print("Error saving state:", e)

# -------------------------
# Session init
# -------------------------
st.set_page_config(page_title="Hybrid Live Trader (REST + WS)", layout="wide", page_icon="📈")

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
            "symbol_name": loaded.get("symbol_name", "Asset"),
        }
        st.toast("Loaded persisted state", icon="💾")
    else:
        st.session_state.data = {
            "prices": [],
            "times": [],
            "sma": [],
            "balance": 10000.0,
            "shares": 0.0,
            "avg_entry": 0.0,
            "trades": [],
            "active": False,
            "current_price": 0.0,
            "symbol": DEFAULT_SYMBOL.lower(),
            "symbol_name": "Bitcoin"
        }

# queue for passing (timestamp, price) tuples from WS thread -> main thread
if "ws_queue" not in st.session_state:
    st.session_state.ws_queue = queue.Queue()

# ws control items
if "ws_thread" not in st.session_state:
    st.session_state.ws_thread = None
if "ws_stop_event" not in st.session_state:
    st.session_state.ws_stop_event = None
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

# -------------------------
# Binance helpers
# -------------------------
def fetch_recent_trades(symbol: str, limit: int = 200):
    """Fetch recent trades via Binance REST (public). Returns list of (timestamp_str, price)."""
    params = {"symbol": symbol.upper(), "limit": limit}
    try:
        r = requests.get(BINANCE_REST_TRADES, params=params, timeout=6)
        r.raise_for_status()
        trades = r.json()
        results = []
        for t in trades:
            # 'price' is string, 'time' is ms timestamp
            price = float(t.get("price", 0.0))
            ts = datetime.fromtimestamp(t.get("time", 0) / 1000.0).strftime("%H:%M:%S")
            results.append((ts, price))
        return results
    except Exception as e:
        print("REST fetch error:", e)
        return []

# -------------------------
# WebSocket thread (producer)
# -------------------------
def websocket_thread_fn(queue_obj: queue.Queue, stop_event: threading.Event, symbol: str):
    """
    Connects to Binance trade stream and pushes (timestamp, price) tuples into queue_obj.
    The thread exits when stop_event is set.
    """
    ws_url = BINANCE_WS_TEMPLATE.format(symbol.lower())
    print(f"[ws] starting thread for {symbol} -> {ws_url}")

    def _on_open(ws):
        print("[ws] connection opened for", symbol)

    def _on_message(ws, message):
        if stop_event.is_set():
            try:
                ws.close()
            except Exception:
                pass
            return
        try:
            msg = json.loads(message)
            price = float(msg.get("p", 0.0))
            ts = datetime.now().strftime("%H:%M:%S")
            queue_obj.put((ts, price))
        except Exception as e:
            print("[ws] parse error:", e)

    def _on_error(ws, err):
        print("[ws] error:", err)

    def _on_close(ws, code, reason):
        print(f"[ws] closed: {code} / {reason}")

    ws = websocket.WebSocketApp(ws_url, on_open=_on_open, on_message=_on_message, on_error=_on_error, on_close=_on_close)
    # run_forever will block; we wrap in try/except
    try:
        # run_forever supports ping to keep connection alive
        ws.run_forever(ping_interval=20, ping_timeout=10)
    except Exception as e:
        print("[ws] run_forever exception:", e)
    print("[ws] exiting thread for", symbol)

def start_ws(symbol: str):
    """Start WS thread for the given symbol, handling prior thread shutdown if needed."""
    # stop previous if alive
    if st.session_state.get("ws_stop_event"):
        st.session_state.ws_stop_event.set()
        st.session_state.ws_stop_event = None
        st.session_state.ws_thread = None
        # queue may hold old messages; flush it
        while not st.session_state.ws_queue.empty():
            try:
                st.session_state.ws_queue.get_nowait()
            except Exception:
                break

    stop_event = threading.Event()
    st.session_state.ws_stop_event = stop_event
    t = threading.Thread(target=websocket_thread_fn, args=(st.session_state.ws_queue, stop_event, symbol), daemon=True)
    t.start()
    st.session_state.ws_thread = t
    print("[main] started ws thread for", symbol)

def stop_ws():
    if st.session_state.get("ws_stop_event"):
        st.session_state.ws_stop_event.set()
        print("[main] signaled ws stop")
    st.session_state.ws_thread = None

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

# handle symbol change
if selected_symbol.lower() != st.session_state.data.get("symbol", DEFAULT_SYMBOL).lower():
    # update session symbol but keep running/stopped behavior
    st.session_state.data["symbol"] = selected_symbol.lower()
    st.session_state.data["symbol_name"] = selected_name
    st.session_state.data["prices"] = []
    st.session_state.data["times"] = []
    st.session_state.data["sma"] = []
    st.sidebar.warning("Symbol changed. Reconnecting to new stream...")
    # restart websocket automatically
    start_ws(selected_symbol)

# bot on/off
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
    # flush queue
    while not st.session_state.ws_queue.empty():
        try:
            st.session_state.ws_queue.get_nowait()
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
# Initialization: load history via REST if empty
# -------------------------
if len(st.session_state.data["prices"]) == 0:
    rest_results = fetch_recent_trades(st.session_state.data.get("symbol", DEFAULT_SYMBOL), limit=200)
    if rest_results:
        times, prices = zip(*rest_results)
        st.session_state.data["times"] = list(times[-MAX_POINTS:])
        st.session_state.data["prices"] = list(prices[-MAX_POINTS:])
        # Precompute SMA list aligned with times (None until enough points)
        st.session_state.data["sma"] = []
        for i in range(len(st.session_state.data["prices"])):
            if i + 1 >= sma_period:
                st.session_state.data["sma"].append(float(np.mean(st.session_state.data["prices"][i + 1 - sma_period : i + 1])))
            else:
                st.session_state.data["sma"].append(None)
        st.session_state.data["current_price"] = st.session_state.data["prices"][-1]
        print("[main] loaded REST history:", len(st.session_state.data["prices"]))
    else:
        print("[main] no REST history available")

# Ensure websocket thread is running
if st.session_state.get("ws_thread") is None or not getattr(st.session_state.ws_thread, "is_alive", lambda: False)():
    start_ws(st.session_state.data.get("symbol", DEFAULT_SYMBOL))

# -------------------------
# Main loop: consume queue and update UI-safe state
# -------------------------
q = st.session_state.ws_queue
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

    # Compute SMA if possible, otherwise append None to keep aligned
    if len(data["prices"]) >= sma_period:
        sma_val = float(np.mean(data["prices"][-sma_period:]))
        data["sma"].append(sma_val)
    else:
        data["sma"].append(None)

    # TRADING LOGIC - performed in main thread to safely mutate session_state
    if data.get("active", False) and data["sma"][-1] is not None:
        current_sma = data["sma"][-1]
        price_f = float(price)

        # BUY condition (no current shares)
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

        # SELL condition (we hold shares)
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

# Persist after processing batch
if drained:
    try:
        save_state(st.session_state.data)
    except Exception as e:
        print("Error saving after processing queue:", e)

# -------------------------
# UI: Main dashboard
# -------------------------
st.title(f"Live {st.session_state.data.get('symbol_name', 'Asset')} Trader (Hybrid)")

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
    # SMA
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

# Connection status footer & manual controls
st.markdown("---")
colA, colB, colC = st.columns([2, 1, 1])
with colA:
    ws_alive = st.session_state.get("ws_thread") is not None and getattr(st.session_state.ws_thread, "is_alive", lambda: False)()
    st.write(f"WebSocket thread alive: {ws_alive}  |  Queue size: {st.session_state.ws_queue.qsize()}")
with colB:
    if st.button("Reconnect WS"):
        start_ws(st.session_state.data.get("symbol", DEFAULT_SYMBOL))
with colC:
    if st.button("Force Save State"):
        save_state(st.session_state.data)
        st.toast("State saved", icon="💾")

# -------------------------
# Auto-refresh mechanism
# -------------------------
# Light-weight rerun to process incoming queue items; avoid busy sleep loops
nowt = time.time()
if nowt - st.session_state.last_refresh > 1.0:
    st.session_state.last_refresh = nowt
    st.rerun()

