import streamlit as st
import websocket
import json
import threading
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Python AlgoTrader",
    layout="wide",
    page_icon="📈"
)

# --- SESSION STATE INITIALIZATION ---
# This keeps data alive between Streamlit re-runs
if "data" not in st.session_state:
    st.session_state.data = {
        "prices": [],
        "times": [],
        "sma": [],
        "balance": 10000.0,  # Starting Paper Money
        "shares": 0.0,
        "avg_entry": 0.0,
        "trades": [],
        "active": False,
        "current_price": 0.0,
        "symbol": "btcusdt",
        "symbol_name": "Bitcoin"
    }

# --- WEBSOCKET HANDLER (Background Thread) ---
# This runs in the background to fetch live data from Binance
def run_websocket():
    def on_message(ws, message):
        json_message = json.loads(message)
        price = float(json_message['p'])
        
        # Update Session State safely
        if "data" in st.session_state:
            now = datetime.now().strftime("%H:%M:%S")
            data = st.session_state.data
            
            # 1. Update Price Data
            data["current_price"] = price
            data["prices"].append(price)
            data["times"].append(now)
            
            # Keep only last 100 data points
            if len(data["prices"]) > 100:
                data["prices"].pop(0)
                data["times"].pop(0)
                if len(data["sma"]) > 0: data["sma"].pop(0)

            # 2. Calculate SMA (Simple Moving Average - 20 periods)
            if len(data["prices"]) >= 20:
                sma_val = np.mean(data["prices"][-20:])
                data["sma"].append(sma_val)
            else:
                data["sma"].append(None)

            # 3. TRADING LOGIC (If Active)
            if data["active"] and data["sma"][-1] is not None:
                current_sma = data["sma"][-1]
                
                # BUY CONDITION: Price is 0.05% below SMA (Buy the dip)
                if data["shares"] == 0 and price < current_sma * 0.9995:
                    invest_amount = data["balance"] * 0.9
                    data["shares"] = invest_amount / price
                    data["balance"] -= invest_amount
                    data["avg_entry"] = price
                    data["trades"].insert(0, {
                        "time": now, "type": "BUY", "price": price, 
                        "amount": data["shares"], "pnl": 0
                    })
                
                # SELL CONDITION: Take Profit (0.1%) or Stop Loss (-0.2%)
                elif data["shares"] > 0:
                    profit_pct = (price - data["avg_entry"]) / data["avg_entry"]
                    
                    if profit_pct > 0.001 or profit_pct < -0.002:
                        revenue = data["shares"] * price
                        pnl = revenue - (data["shares"] * data["avg_entry"])
                        data["balance"] += revenue
                        
                        data["trades"].insert(0, {
                            "time": now, "type": "SELL", "price": price, 
                            "amount": data["shares"], "pnl": pnl
                        })
                        data["shares"] = 0
                        data["avg_entry"] = 0

    def on_error(ws, error):
        print(f"Error: {error}")

    # Connect to Binance Stream
    # Note: Streamlit re-runs script often, but threads persist in background
    symbol = st.session_state.data['symbol']
    socket = f"wss://stream.binance.com:9443/ws/{symbol}@trade"
    
    ws = websocket.WebSocketApp(socket, on_message=on_message, on_error=on_error)
    ws.run_forever()

# Start WebSocket in a separate thread if not already running
if "ws_thread" not in st.session_state:
    ws_thread = threading.Thread(target=run_websocket, daemon=True)
    ws_thread.start()
    st.session_state.ws_thread = ws_thread

# --- UI LAYOUT ---

# 1. Sidebar Controls
st.sidebar.title("🤖 Control Panel")
selected_coin = st.sidebar.selectbox(
    "Select Asset", 
    [("btcusdt", "Bitcoin"), ("ethusdt", "Ethereum"), ("solusdt", "Solana"), ("dogeusdt", "Dogecoin")],
    format_func=lambda x: x[1]
)

# Handle Coin Change
if selected_coin[0] != st.session_state.data["symbol"]:
    st.session_state.data["symbol"] = selected_coin[0]
    st.session_state.data["symbol_name"] = selected_coin[1]
    st.session_state.data["prices"] = [] # Reset history
    st.session_state.data["sma"] = []
    # Note: To fully switch streams, one would need to restart the thread/socket.
    # For this simple demo, please restart the app to switch data streams effectively.
    st.warning("Please restart the app to fetch data for the new symbol.")
    st.rerun()

# Bot Toggle
if st.sidebar.button("🔴 STOP BOT" if st.session_state.data["active"] else "🟢 START BOT"):
    st.session_state.data["active"] = not st.session_state.data["active"]

status_color = "green" if st.session_state.data["active"] else "red"
status_text = "RUNNING" if st.session_state.data['active'] else "STOPPED"
st.sidebar.markdown(f"Status: **:{status_color}[{status_text}]**")

# Strategy Info
st.sidebar.markdown("---")
st.sidebar.subheader("Strategy Logic")
st.sidebar.info("""
**Mean Reversion:**
- **Buy:** Price drops 0.05% below SMA
- **Sell:** Profit > 0.1% OR Loss > 0.2%
""")

# 2. Main Dashboard
st.title(f"Live {st.session_state.data['symbol_name']} Trader")

# Metrics Row
col1, col2, col3, col4 = st.columns(4)
current_equity = st.session_state.data["balance"] + (st.session_state.data["shares"] * st.session_state.data["current_price"])
profit = current_equity - 10000

with col1:
    st.metric("Live Price", f"${st.session_state.data['current_price']:.2f}")
with col2:
    st.metric("Total Equity", f"${current_equity:.2f}")
with col3:
    st.metric("Session Profit", f"${profit:.2f}", delta_color="normal")
with col4:
    holdings = st.session_state.data["shares"]
    st.metric("Holdings", f"{holdings:.4f}")

# 3. Live Chart (Plotly)
if len(st.session_state.data["prices"]) > 0:
    fig = go.Figure()
    
    # Price Line
    fig.add_trace(go.Scatter(
        x=st.session_state.data["times"], 
        y=st.session_state.data["prices"],
        mode='lines',
        name='Price',
        line=dict(color='#00ff00')
    ))

    # SMA Line
    if len(st.session_state.data["sma"]) > 0:
         # Filter out None values for plotting
        valid_sma = [x for x in st.session_state.data["sma"] if x is not None]
        # Align times
        sma_times = st.session_state.data["times"][-len(valid_sma):]
        
        fig.add_trace(go.Scatter(
            x=sma_times,
            y=valid_sma,
            mode='lines',
            name='SMA (20)',
            line=dict(color='#ffd700', dash='dash')
        ))

    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='#333'),
        font=dict(color='white')
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("Waiting for market data... (If this takes too long, check internet connection)")

# 4. Trade Log
st.subheader("Transaction History")
if len(st.session_state.data["trades"]) > 0:
    df = pd.DataFrame(st.session_state.data["trades"])
    st.dataframe(df, use_container_width=True)
else:
    st.text("No trades executed yet.")

# Auto-refresh mechanism (Streamlit needs this to update the UI)
time.sleep(1) 
st.rerun()