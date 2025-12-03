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
    # Inner function to handle messages received from the socket
    def on_message(ws, message):
        try:
            json_message = json.loads(message)
            # We are interested in the 'p' (price) field from a trade stream
            price = float(json_message['p'])
        except (json.JSONDecodeError, ValueError, KeyError):
            # Handle malformed or unexpected messages
            print("Received malformed or incomplete data.")
            return

        # Update Session State safely (This is read-only in the thread, Streamlit handles the write protection)
        if "data" in st.session_state:
            now = datetime.now().strftime("%H:%M:%S")
            data = st.session_state.data
            
            # Retrieve parameters set by the user in the UI
            SMA_PERIOD = st.session_state.get("sma_period", 20)
            BUY_DIP_PCT = st.session_state.get("buy_dip_pct", 0.0005) # Default 0.05%
            TAKE_PROFIT_PCT = st.session_state.get("take_profit_pct", 0.001) # Default 0.1%
            STOP_LOSS_PCT = st.session_state.get("stop_loss_pct", 0.002) # Default 0.2%
            INVEST_PERCENT = st.session_state.get("invest_percent", 0.9) # Default 90%
            
            # 1. Update Price Data
            data["current_price"] = price
            data["prices"].append(price)
            data["times"].append(now)
            
            # Keep only last 100 data points to prevent memory overflow
            MAX_POINTS = 100
            if len(data["prices"]) > MAX_POINTS:
                data["prices"].pop(0)
                data["times"].pop(0)
                if len(data["sma"]) > 0: data["sma"].pop(0)

            # 2. Calculate SMA (Simple Moving Average - based on user input)
            if len(data["prices"]) >= SMA_PERIOD:
                sma_val = np.mean(data["prices"][-SMA_PERIOD:])
                data["sma"].append(sma_val)
            else:
                data["sma"].append(None) # Append None if not enough data points

            # 3. TRADING LOGIC (If Active)
            if data["active"] and data["sma"][-1] is not None:
                current_sma = data["sma"][-1]
                
                # BUY CONDITION: Price is below the SMA by the user-defined percentage
                # The logic is: price < current_sma * (1 - BUY_DIP_PCT)
                if data["shares"] == 0 and price < current_sma * (1 - BUY_DIP_PCT):
                    invest_amount = data["balance"] * INVEST_PERCENT
                    if invest_amount > 0:
                        data["shares"] = invest_amount / price
                        data["balance"] -= invest_amount
                        data["avg_entry"] = price
                        # Log the trade
                        data["trades"].insert(0, {
                            "time": now, "type": "BUY", "price": price, 
                            "amount": data["shares"], "pnl": 0
                        })
                
                # SELL CONDITION: Take Profit or Stop Loss
                elif data["shares"] > 0:
                    profit_pct = (price - data["avg_entry"]) / data["avg_entry"]
                    
                    # Check for Take Profit (profit_pct > TAKE_PROFIT_PCT) or Stop Loss (profit_pct < -STOP_LOSS_PCT)
                    if profit_pct > TAKE_PROFIT_PCT or profit_pct < -STOP_LOSS_PCT:
                        revenue = data["shares"] * price
                        pnl = revenue - (data["shares"] * data["avg_entry"])
                        data["balance"] += revenue
                        
                        # Log the trade
                        data["trades"].insert(0, {
                            "time": now, "type": "SELL", "price": price, 
                            "amount": data["shares"], "pnl": pnl
                        })
                        
                        # Reset holdings
                        data["shares"] = 0
                        data["avg_entry"] = 0

    def on_error(ws, error):
        # In a production environment, you would log this error properly
        print(f"WebSocket Error: {error}")

    def on_close(ws, close_status_code, close_msg):
        print(f"WebSocket closed: {close_status_code} - {close_msg}")

    # Connect to Binance Stream
    symbol = st.session_state.data['symbol']
    # Binance trade stream endpoint for live price updates
    socket = f"wss://stream.binance.com:9443/ws/{symbol}@trade"
    
    ws = websocket.WebSocketApp(
        socket, 
        on_message=on_message, 
        on_error=on_error,
        on_close=on_close
    )
    # run_forever blocks the thread until manually stopped or connection fails
    ws.run_forever()

# Start WebSocket in a separate thread if not already running
# The 'daemon=True' ensures the thread closes when the main app closes
if "ws_thread" not in st.session_state or not st.session_state.ws_thread.is_alive():
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
    
    # Due to the complexity of closing and restarting the WebSocket thread, 
    # we instruct the user to restart the app for the change to take full effect.
    st.warning("Symbol changed. **Please manually restart the Streamlit application** to connect the WebSocket to the new stream.")
    # Optional: Stop the bot automatically upon symbol change
    st.session_state.data["active"] = False 

# Bot Toggle
if st.sidebar.button("🔴 STOP BOT" if st.session_state.data["active"] else "🟢 START BOT"):
    st.session_state.data["active"] = not st.session_state.data["active"]
    if st.session_state.data["active"]:
        st.toast("Bot Activated! Listening for trade signals...", icon='🚀')
    else:
        st.toast("Bot Stopped. Position held (if any).", icon='🛑')


# Display Status
status_color = "green" if st.session_state.data["active"] else "red"
status_text = "RUNNING" if st.session_state.data['active'] else "STOPPED"
st.sidebar.markdown(f"Status: **:{status_color}[{status_text}]**")

# Strategy Parameters
st.sidebar.markdown("---")
st.sidebar.subheader("Strategy Parameters")

# SMA Period
sma_period = st.sidebar.slider("SMA Period (Data Points)", 5, 50, 20, 5, key="sma_period")

# Buy Condition
st.sidebar.markdown("##### Buy Condition")
buy_dip_pct = st.sidebar.number_input(
    "Buy below SMA by (%)", 
    min_value=0.01, 
    max_value=1.0, 
    value=0.05, 
    step=0.01, 
    format="%.2f",
    key="buy_dip_pct_ui"
) / 100
st.session_state["buy_dip_pct"] = buy_dip_pct # Store as decimal for math

invest_percent = st.sidebar.slider("Invest % of Balance", 10, 100, 90, 5, format="%d%%", key="invest_percent_ui") / 100
st.session_state["invest_percent"] = invest_percent # Store as decimal for math


# Sell Conditions
st.sidebar.markdown("##### Sell Conditions")
take_profit_pct = st.sidebar.number_input(
    "Take Profit at (%)", 
    min_value=0.01, 
    max_value=1.0, 
    value=0.1, 
    step=0.01, 
    format="%.2f",
    key="take_profit_pct_ui"
) / 100
st.session_state["take_profit_pct"] = take_profit_pct # Store as decimal for math

stop_loss_pct = st.sidebar.number_input(
    "Stop Loss at (%)", 
    min_value=0.01, 
    max_value=1.0, 
    value=0.2, 
    step=0.01, 
    format="%.2f",
    key="stop_loss_pct_ui"
) / 100
st.session_state["stop_loss_pct"] = stop_loss_pct # Store as decimal for math


# 2. Main Dashboard
st.title(f"Live {st.session_state.data['symbol_name']} Trader")

# Calculate current financial metrics
current_equity = st.session_state.data["balance"] + (st.session_state.data["shares"] * st.session_state.data["current_price"])
profit = current_equity - 10000

# Metrics Row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Live Price", f"${st.session_state.data['current_price']:.2f}")
with col2:
    st.metric("Total Equity", f"${current_equity:.2f}")
with col3:
    st.metric("Session Profit", f"${profit:.2f}", delta=f"{profit:.2f}", delta_color="normal")
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
    # Get the SMA period for the chart name
    sma_period_current = st.session_state.get("sma_period", 20)
    
    if len(st.session_state.data["sma"]) > 0:
        # Filter out None values and align times
        valid_sma_data = [(st.session_state.data["times"][i], st.session_state.data["sma"][i]) 
                          for i in range(len(st.session_state.data["sma"])) 
                          if st.session_state.data["sma"][i] is not None]
        
        sma_times = [t[0] for t in valid_sma_data]
        sma_values = [t[1] for t in valid_sma_data]
        
        fig.add_trace(go.Scatter(
            x=sma_times,
            y=sma_values,
            mode='lines',
            name=f'SMA ({sma_period_current})',
            line=dict(color='#ffd700', dash='dash')
        ))

    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
        # Ensure plot colors are suitable for dark mode (Streamlit default)
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='#333'),
        font=dict(color='white'),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    # Use the current SMA period in the warning message
    st.warning(f"Waiting for live market data stream... (This may take a moment to collect {st.session_state.get('sma_period', 20)} data points for the SMA)")

# 4. Trade Log
st.subheader("Transaction History")
if len(st.session_state.data["trades"]) > 0:
    df = pd.DataFrame(st.session_state.data["trades"])
    # Format P&L for better readability
    df['pnl'] = df['pnl'].apply(lambda x: f"${x:.2f}")
    df['price'] = df['price'].apply(lambda x: f"${x:.2f}")

    st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.text("No trades executed yet. Press '🟢 START BOT' to begin paper trading.")

# Auto-refresh mechanism (Streamlit needs this to update the UI based on the background thread)
time.sleep(1) 
st.rerun()
