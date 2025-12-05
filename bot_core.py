import streamlit as st
import requests
import json
import time
import threading
import queue
import os
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ======================================================
#            CONFIGURATION (Moved to core)
# ======================================================
POLL_INTERVAL = 15 
STATE_FILE = "state.json"
PROVIDER_NAME = "Bybit (Linear Perpetual)"
BYBIT_URL = "https://api.bybit.com/v5/market/tickers"
INITIAL_BALANCE = 10000.0

# ======================================================
#            ASSET CLASS
# ======================================================
class Asset:
    """A minimal class to hold asset information."""
    def __init__(self, symbol, name):
        # symbol is the Bybit market symbol (e.g., "BTCUSDT")
        self.symbol = symbol
        self.name = name


# ======================================================
#            ASSET REGISTRY
# ======================================================
ASSETS = {
    "btcusdt": Asset("BTCUSDT", "Bitcoin"),
    "ethusdt": Asset("ETHUSDT", "Ethereum"),
    "bnbusdt": Asset("BNBUSDT", "BNB"),
    "solusdt": Asset("SOLUSDT", "Solana"),
    "xrpusdt": Asset("XRPUSDT", "XRP"),
    "dogeusdt": Asset("DOGEUSDT", "Dogecoin"),
    "adausdt": Asset("ADAUSDT", "Cardano"),
    "avaxusdt": Asset("AVAXUSDT", "Avalanche"),
    "dotusdt": Asset("DOTUSDT", "Polkadot"),
    "linkusdt": Asset("LINKUSDT", "Chainlink"),
    "maticusdt": Asset("MATICUSDT", "Polygon"),
    "shibusdt": Asset("SHIBUSDT", "Shiba Inu"),
}


# ======================================================
#               LOAD / SAVE STATE
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
#           PRICE PROVIDER (BYBIT)
# ======================================================
def fetch_price(asset: Asset):
    """
    Fetches the latest price from the Bybit V5 Market Tickers API.
    """
    try:
        # Request parameters for a specific symbol in the linear/perpetual market
        params = {
            "category": "linear",
            "symbol": asset.symbol
        }
        
        r = requests.get(BYBIT_URL, params=params, timeout=6)
        r.raise_for_status() # Raises an HTTPError for bad responses (4xx or 5xx)

        data = r.json()

        # Check for Bybit API specific error codes
        if data.get("retCode") != 0:
            print(f"Bybit API Error ({data.get('retCode')}): {data.get('retMsg')}")
            return None

        # Navigate the response structure: result -> list[0] -> lastPrice
        ticker_list = data.get("result", {}).get("list", [])
        if not ticker_list:
            return None

        last_price_str = ticker_list[0].get("lastPrice")
        if last_price_str is not None:
            return float(last_price_str)

    except requests.exceptions.RequestException as e:
        print(f"Error fetching price from Bybit: {e}")
    except (ValueError, TypeError) as e:
        print(f"Error parsing price data: {e}")

    return None


# ======================================================
#             POLLER THREAD
# ======================================================
def price_poller(asset: Asset, q, stop_event):
    """Thread function to continuously poll the price."""
    while not stop_event.is_set():
        price = fetch_price(asset)
        if price is not None:
            ts = datetime.now().strftime("%H:%M:%S")
            q.put((ts, price))
        time.sleep(POLL_INTERVAL)


# ======================================================
#        INITIALIZE STREAMLIT SESSION STATE
# ======================================================
def initialize_state():
    """Initializes Streamlit session state variables."""
    if "state" not in st.session_state:
        saved = load_state()
        default_symbol = "btcusdt"
        default_asset = ASSETS[default_symbol]

        if saved:
            st.session_state.state = {
                **saved,
                "prices": [], "times": [], "sma": [],
                "current_price": 0.0, "active": False
            }
        else:
            st.session_state.state = {
                "prices": [], "times": [], "sma": [],
                "balance": INITIAL_BALANCE, "shares": 0.0,
                "avg_entry": 0.0, "trades": [],
                "active": False, "current_price": 0.0,
                "symbol": default_symbol, "symbol_name": default_asset.name
            }

    if "price_queue" not in st.session_state:
        st.session_state.price_queue = queue.Queue()

    if "stop_event" not in st.session_state:
        st.session_state.stop_event = threading.Event()

    if "poll_thread" not in st.session_state:
        st.session_state.poll_thread = None

    return st.session_state.state

# ======================================================
#                STREAMLIT UI FUNCTIONS
# ======================================================

def reset_all(state, current_asset):
    """Resets the state and removes the history file."""
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
    
    # Ensure all threads are stopped before resetting
    if st.session_state.poll_thread is not None:
        st.session_state.stop_event.set()
    
    st.session_state.state = {
        "prices": [], "times": [], "sma": [],
        "balance": INITIAL_BALANCE, "shares": 0.0, "avg_entry": 0.0,
        "trades": [], "active": False, "current_price": 0.0,
        "symbol": current_asset.symbol.lower(), "symbol_name": current_asset.name
    }
    st.session_state.price_queue = queue.Queue()
    st.session_state.stop_event = threading.Event()
    st.session_state.poll_thread = None


def setup_sidebar(state):
    """Creates sidebar controls and returns configuration."""
    st.sidebar.title("Control Panel")

    # ------------------ Asset Selection ------------------
    symbol = st.sidebar.selectbox(
        "Select Asset",
        list(ASSETS.keys()),
        format_func=lambda x: ASSETS[x].name
    )
    current_asset = ASSETS[symbol]

    # Asset change logic
    if symbol != state["symbol"]:
        state["symbol"] = symbol
        state["symbol_name"] = current_asset.name
        state["prices"], state["times"], state["sma"] = [], [], []
        state["current_price"] = 0.0

        # Stop the existing thread if an asset change occurs
        st.session_state.stop_event.set()
        st.session_state.stop_event = threading.Event()
        st.session_state.poll_thread = None

    # ------------------ Strategy Parameters ------------------
    SMA_PERIOD = st.sidebar.slider("SMA Period", 5, 50, 20, 5)
    BUY_DIP = st.sidebar.number_input("Buy Below SMA (%)", 0.01, 5.0, 0.5) / 100
    TP = st.sidebar.number_input("Take Profit (%)", 0.1, 5.0, 1.0) / 100
    SL = st.sidebar.number_input("Stop Loss (%)", 0.1, 5.0, 2.0) / 100
    INVEST_PCT = st.sidebar.slider("Invest %", 10, 100, 90) / 100

    # ------------------ Reset Button ------------------
    st.sidebar.button("Reset Balance & History", on_click=reset_all, args=(state, current_asset))

    # ------------------ Start/Stop Bot Button ------------------
    if st.sidebar.button("Start Bot" if not state["active"] else "Stop Bot"):
        state["active"] = not state["active"]

        if state["active"]:
            # Start polling thread
            st.session_state.stop_event.clear()
            t = threading.Thread(
                target=price_poller,
                args=(current_asset, st.session_state.price_queue, st.session_state.stop_event),
                daemon=True
            )
            t.start()
            st.session_state.poll_thread = t
        else:
            # Stop polling thread
            st.session_state.stop_event.set()

    return {
        "SMA_PERIOD": SMA_PERIOD,
        "BUY_DIP": BUY_DIP,
        "TP": TP,
        "SL": SL,
        "INVEST_PCT": INVEST_PCT,
        "current_asset": current_asset
    }


def update_and_trade(state, config):
    """Processes price queue, updates history, calculates SMA, and executes trades."""
    asset = config["current_asset"]
    
    while not st.session_state.price_queue.empty():
        ts, price = st.session_state.price_queue.get()

        state["current_price"] = price
        state["prices"].append(price)
        state["times"].append(ts)

        # Maintain a clean history of the last 100 points
        if len(state["prices"]) > 100:
            state["prices"].pop(0)
            state["times"].pop(0)
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


def display_metrics(state, config):
    """Displays connection status and financial metrics."""
    # ------------------ Connection Status ------------------
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

    st.title(f"{state['symbol_name']} Algo Trader")
    st.markdown(f"### Connection Status: {conn}")
    st.caption(f"Provider: **{PROVIDER_NAME}**")

    # ------------------ Metrics Calculation ------------------
    equity = state["balance"] + state["shares"] * state["current_price"]
    
    unrealized_pnl = 0
    pnl_delta_value = None
    pnl_color_mode = "off"

    if state["shares"] > 0 and state["avg_entry"] > 0:
        unrealized_pnl = state["shares"] * (state["current_price"] - state["avg_entry"])
        pnl_delta_value = unrealized_pnl
        pnl_color_mode = "normal" 
    else:
        pnl_color_mode = "off" 
        pnl_delta_value = None

    # ------------------ Metrics Display ------------------
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(f"{config['current_asset'].symbol} Price", f"${state['current_price']:.4f}")
    col2.metric("Cash Balance", f"${state['balance']:.2f}")
    col3.metric("Shares Held", f"{state['shares']:.4f}")

    # The fixed st.metric call
    col4.metric(
        "Unrealized PNL", 
        f"${unrealized_pnl:.2f}", 
        delta=pnl_delta_value,    
        delta_color=pnl_color_mode
    )

    col5.metric("Total Equity", f"${equity:.2f}")


def display_chart(state, config):
    """Generates and displays the Plotly chart."""
    if len(state["prices"]) > 0:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=state["times"], y=state["prices"],
            mode="lines", name="Price", line=dict(color="#10b981") 
        ))

        # Add SMA line
        sma_clean = [(state["times"][i], state["sma"][i])
                     for i in range(len(state["sma"]))
                     if state["sma"][i] is not None]

        if sma_clean:
            fig.add_trace(go.Scatter(
                x=[t[0] for t in sma_clean],
                y=[t[1] for t in sma_clean],
                mode="lines",
                name=f"SMA {config['SMA_PERIOD']}",
                line=dict(color="#fcd34d", dash="dash")
            ))
            
            # Add entry price line if holding shares
            if state["shares"] > 0 and state["avg_entry"] > 0:
                 fig.add_hline(
                    y=state["avg_entry"], 
                    line_dash="dot", 
                    annotation_text="Avg Entry", 
                    annotation_position="top left",
                    line_color="#3b82f6"
                )

        fig.update_layout(
            height=450,
            margin=dict(l=15, r=15, t=30, b=10),
            template="plotly_dark",
            paper_bgcolor="#1f2937", 
            plot_bgcolor="#1f2937",
            font=dict(color="#f9fafb")
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Waiting for price data from Bybit…")


def display_trade_log(state):
    """Displays the trade history table."""
    st.subheader("Trade History")
    if state["trades"]:
        df = pd.DataFrame(state["trades"])
        
        # Calculate cumulative PNL
        df['cumulative_pnl'] = df['pnl'].cumsum()
        
        # Format columns for display
        df["price"] = df["price"].apply(lambda x: f"${x:.4f}")
        df["amount"] = df["amount"].apply(lambda x: f"{x:.4f}")
        df["pnl"] = df["pnl"].apply(lambda x: f"${x:.4f}")
        df["cumulative_pnl"] = df["cumulative_pnl"].apply(lambda x: f"${x:.2f}")

        # Style PNL column
        def color_pnl(val):
            """Colors PNL cells green/red."""
            try:
                val_float = float(val.replace('$', ''))
                color = 'green' if val_float >= 0 else 'red'
                return f'color: {color}'
            except:
                return ''

        st.dataframe(
            df.style.applymap(color_pnl, subset=['pnl', 'cumulative_pnl']),
            hide_index=True, 
            use_container_width=True
        )
    else:
        st.info("No trades have been executed yet.")


def main_app_loop():
    """Main execution function for the Streamlit app components."""
    # 1. Initialize session state
    state = initialize_state()

    # 2. Setup sidebar controls and get configuration
    config = setup_sidebar(state)

    # 3. Process incoming data and execute trades
    update_and_trade(state, config)

    # 4. Display UI
    display_metrics(state, config)
    display_chart(state, config)
    display_trade_log(state)
    
    # 5. Auto-refresh mechanism
    time.sleep(1)
    st.rerun()

# Export the main loop function
# This is the function Main.py will import and run.
# Note: You need to keep the imports at the top of the file as written.
# Do not run main_app_loop() here.
