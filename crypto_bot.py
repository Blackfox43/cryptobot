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
#                STREAMLIT CONFIG
# ======================================================
st.set_page_config(page_title="Algo Trader", layout="wide")

POLL_INTERVAL = 3  # seconds
STATE_FILE = "state.json"

# ======================================================
#           SUPPORTED ASSETS (12 coins)
# ======================================================

# ---- CoinGecko IDs ----
CG_IDS = {
    "btcusdt": "bitcoin",
    "ethusdt": "ethereum",
    "bnbusdt": "binancecoin",
    "solusdt": "solana",
    "xrpusdt": "ripple",
    "dogeusdt": "dogecoin",
    "adausdt": "cardano",
    "avaxusdt": "avalanche-2",
    "dotusdt": "polkadot",
    "linkusdt": "chainlink",
    "maticusdt": "matic-network",
    "shibusdt": "shiba-inu"
}

# ---- CoinPaprika IDs (Correct full ticker IDs) ----
CP_IDS = {
    "btcusdt": "btc-bitcoin",
    "ethusdt": "eth-ethereum",
    "bnbusdt": "bnb-binance-coin",
    "solusdt": "sol-solana",
    "xrpusdt": "xrp-xrp",
    "dogeusdt": "doge-dogecoin",
    "adausdt": "ada-cardano",
    "avaxusdt": "avax-avalanche",
    "dotusdt": "dot-polkadot",
    "linkusdt": "link-chainlink",
    "maticusdt": "matic-polygon",
    "shibusdt": "shib-shiba-inu"
}

ASSET_NAMES = {
    "btcusdt": "Bitcoin",
    "ethusdt": "Ethereum",
    "bnbusdt": "BNB",
    "solusdt": "Solana",
    "xrpusdt": "XRP",
    "dogeusdt": "Dogecoin",
    "adausdt": "Cardano",
    "avaxusdt": "Avalanche",
    "dotusdt": "Polkadot",
    "linkusdt": "Chainlink",
    "maticusdt": "Polygon",
    "shibusdt": "Shiba Inu"
}

CG_URL = "https://api.coingecko.com/api/v3/simple/price"
CURRENT_PROVIDER = "coingecko"


# ======================================================
#           LOAD / SAVE STATE
# ======================================================
def load_state():
    if not os.path.exists(STATE_FILE):
        return None
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except:
        return None


def save_state(state):
    safe = {
        "balance": state["balance"],
        "shares": state["shares"],
        "avg_entry": state["avg_entry"],
        "trades": state["trades"],
        "symbol": state["symbol"],
        "symbol_name": state["symbol_name"]
    }
    with open(STATE_FILE, "w") as f:
        json.dump(safe, f, indent=4)


# ======================================================
#           PRICE PROVIDERS (CG + CP)
# ======================================================
def get_price(symbol):
    """Return live price using CoinGecko with fallback to CoinPaprika."""
    global CURRENT_PROVIDER

    cg_id = CG_IDS[symbol]
    cp_id = CP_IDS[symbol]

    # -----------------------
    # Primary: CoinGecko
    # -----------------------
    if CURRENT_PROVIDER == "coingecko":
        try:
            r = requests.get(
                CG_URL,
                params={"ids": cg_id, "vs_currencies": "usd"},
                timeout=6
            )
            if r.status_code == 200:
                d = r.json()
                p = d.get(cg_id, {}).get("usd")
                if p is not None:
                    return float(p)

            CURRENT_PROVIDER = "coinpaprika"
        except:
            CURRENT_PROVIDER = "coinpaprika"

    # -----------------------
    # Fallback: CoinPaprika
    # -----------------------
    if CURRENT_PROVIDER == "coinpaprika":
        try:
            url = f"https://api.coinpaprika.com/v1/tickers/{cp_id}"
            r = requests.get(url, timeout=6)
            if r.status_code == 200:
                d = r.json()
                p = d.get("quotes", {}).get("USD", {}).get("price")
                if p is not None:
                    return float(p)

            CURRENT_PROVIDER = "coingecko"
        except:
            CURRENT_PROVIDER = "coingecko"

    return None


# ======================================================
#           POLLER THREAD
# ======================================================
def price_poller(symbol, q, stop_event):
    """Continuously poll prices and push into queue."""
    while not stop_event.is_set():
        price = get_price(symbol)
        if price is not None:
            ts = datetime.now().strftime("%H:%M:%S")
            q.put((ts, price))
        time.sleep(POLL_INTERVAL)


# ======================================================
#       INITIALIZE STATE
# ======================================================
if "state" not in st.session_state:
    saved = load_state()
    if saved:
        st.session_state.state = {
            **saved,
            "prices": [],
            "times": [],
            "sma": [],
            "current_price": 0.0,
            "active": False
        }
    else:
        st.session_state.state = {
            "prices": [],
            "times": [],
            "sma": [],
            "balance": 10000.0,
            "shares": 0.0,
            "avg_entry": 0.0,
            "trades": [],
            "active": False,
            "current_price": 0.0,
            "symbol": "btcusdt",
            "symbol_name": "Bitcoin"
        }

if "price_queue" not in st.session_state:
    st.session_state.price_queue = queue.Queue()

if "stop_event" not in st.session_state:
    st.session_state.stop_event = threading.Event()

if "poll_thread" not in st.session_state:
    st.session_state.poll_thread = None


# ======================================================
#            SIDEBAR UI
# ======================================================
st.sidebar.title("Control Panel")

symbol = st.sidebar.selectbox(
    "Select Asset",
    list(ASSET_NAMES.keys()),
    format_func=lambda x: ASSET_NAMES[x]
)

# Switch symbol
if symbol != st.session_state.state["symbol"]:
    st.session_state.state["symbol"] = symbol
    st.session_state.state["symbol_name"] = ASSET_NAMES[symbol]
    st.session_state.state["prices"] = []
    st.session_state.state["sma"] = []
    st.session_state.state["times"] = []
    st.session_state.state["current_price"] = 0.0

    st.session_state.stop_event.set()
    st.session_state.stop_event = threading.Event()
    st.session_state.poll_thread = None

# Strategy settings
SMA_PERIOD = st.sidebar.slider("SMA Period", 5, 50, 20, 5)
BUY_DIP = st.sidebar.number_input("Buy Below SMA (%)", 0.01, 5.0, 0.5) / 100
TAKE_PROFIT = st.sidebar.number_input("Take Profit (%)", 0.1, 5.0, 1.0) / 100
STOP_LOSS = st.sidebar.number_input("Stop Loss (%)", 0.1, 5.0, 2.0) / 100
INVEST_PERCENT = st.sidebar.slider("Invest %", 10, 100, 90) / 100

def reset_all():
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
    st.session_state.state = {
        "prices": [], "times": [], "sma": [],
        "balance": 10000.0, "shares": 0.0, "avg_entry": 0.0,
        "trades": [], "active": False,
        "current_price": 0.0,
        "symbol": symbol,
        "symbol_name": ASSET_NAMES[symbol]
    }
    st.session_state.price_queue = queue.Queue()

st.sidebar.button("Reset Balance & History", on_click=reset_all)

# Start / Stop
if st.sidebar.button("Start Bot" if not st.session_state.state["active"] else "Stop Bot"):
    st.session_state.state["active"] = not st.session_state.state["active"]

    if st.session_state.state["active"]:
        st.session_state.stop_event.clear()
        t = threading.Thread(
            target=price_poller,
            args=(symbol, st.session_state.price_queue, st.session_state.stop_event),
            daemon=True
        )
        t.start()
        st.session_state.poll_thread = t
    else:
        st.session_state.stop_event.set()


# ======================================================
#       PROCESS PRICE QUEUE
# ======================================================
state = st.session_state.state

while not st.session_state.price_queue.empty():
    ts, price = st.session_state.price_queue.get()

    state["current_price"] = price
    state["prices"].append(price)
    state["times"].append(ts)

    if len(state["prices"]) > 100:
        state["prices"].pop(0)
        state["times"].pop(0)
        if state["sma"]:
            state["sma"].pop(0)

    if len(state["prices"]) >= SMA_PERIOD:
        state["sma"].append(np.mean(state["prices"][-SMA_PERIOD:]))
    else:
        state["sma"].append(None)

    # ------------------------------
    #         TRADING LOGIC
    # ------------------------------
    if state["active"] and state["sma"][-1] is not None:
        sma = state["sma"][-1]

        # BUY
        if state["shares"] == 0 and price < sma * (1 - BUY_DIP):
            invest = state["balance"] * INVEST_PERCENT
            if invest > 0:
                state["shares"] = invest / price
                state["balance"] -= invest
                state["avg_entry"] = price
                state["trades"].insert(0, {
                    "time": ts, "type": "BUY",
                    "price": price, "amount": state["shares"],
                    "pnl": 0
                })
                save_state(state)

        # SELL
        elif state["shares"] > 0:
            pct = (price - state["avg_entry"]) / state["avg_entry"]
            if pct > TAKE_PROFIT or pct < -STOP_LOSS:
                revenue = state["shares"] * price
                pnl = revenue - (state["shares"] * state["avg_entry"])

                state["balance"] += revenue
                state["trades"].insert(0, {
                    "time": ts, "type": "SELL",
                    "price": price, "amount": state["shares"],
                    "pnl": pnl
                })

                state["shares"] = 0
                state["avg_entry"] = 0
                save_state(state)


# ======================================================
#       CONNECTION ICON
# ======================================================
if len(state["times"]) > 0:
    last = state["times"][-1]
    last_dt = datetime.strptime(last, "%H:%M:%S")
    delay = (datetime.now() - last_dt).total_seconds()

    if delay < POLL_INTERVAL + 2:
        conn = "🟢 Live"
    elif delay < 20:
        conn = "🟡 Slow"
    else:
        conn = "🔴 Disconnected"
else:
    conn = "🔴 Waiting"


# ======================================================
#            MAIN UI
# ======================================================
st.title(f"{state['symbol_name']} Algo Trader")
st.markdown(f"### Connection Status: {conn}")
st.caption(f"Provider: **{CURRENT_PROVIDER}**")

equity = state["balance"] + state["shares"] * state["current_price"]
profit = equity - 10000

col1, col2, col3, col4 = st.columns(4)
col1.metric("Price", f"${state['current_price']:.4f}")
col2.metric("Balance", f"${state['balance']:.2f}")
col3.metric("Equity", f"${equity:.2f}")
col4.metric("Profit", f"${profit:.2f}", delta=f"{profit:.2f}")

# Chart
if len(state["prices"]) > 0:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=state["times"], y=state["prices"],
        mode="lines", name="Price",
        line=dict(color="lime")
    ))

    clean_sma = [(state["times"][i], state["sma"][i])
                 for i in range(len(state["sma"]))
                 if state["sma"][i] is not None]

    if clean_sma:
        fig.add_trace(go.Scatter(
            x=[t[0] for t in clean_sma],
            y=[t[1] for t in clean_sma],
            mode="lines",
            name=f"SMA {SMA_PERIOD}",
            line=dict(color="gold", dash="dash")
        ))

    fig.update_layout(
        height=420,
        margin=dict(l=15, r=15, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white")
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Waiting for price data…")

# Trade Log
st.subheader("Trades")
if state["trades"]:
    df = pd.DataFrame(state["trades"])
    df["price"] = df["price"].apply(lambda x: f"${x:.4f}")
    df["pnl"] = df["pnl"].apply(lambda x: f"${x:.4f}")
    st.dataframe(df, hide_index=True, use_container_width=True)
else:
    st.info("No trades executed yet.")

# Auto-refresh
time.sleep(1)
st.rerun()
