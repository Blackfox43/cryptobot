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
#            STREAMLIT APP CONFIG
# ======================================================
st.set_page_config(page_title="Algo Trader", layout="wide")
POLL_INTERVAL = 3
STATE_FILE = "state.json"
CURRENT_PROVIDER = "coingecko"


# ======================================================
#            ASSET CLASS (OPTION C)
# ======================================================
class Asset:
    def __init__(self, symbol, name, cg_id, cp_id):
        self.symbol = symbol
        self.name = name
        self.cg_id = cg_id
        self.cp_id = cp_id


# ======================================================
#            ASSET REGISTRY (12 COINS)
# ======================================================
ASSETS = {
    "btcusdt": Asset("btcusdt", "Bitcoin", "bitcoin", "btc-bitcoin"),
    "ethusdt": Asset("ethusdt", "Ethereum", "ethereum", "eth-ethereum"),
    "bnbusdt": Asset("bnbusdt", "BNB", "binancecoin", "bnb-binance-coin"),
    "solusdt": Asset("solusdt", "Solana", "solana", "sol-solana"),
    "xrpusdt": Asset("xrpusdt", "XRP", "ripple", "xrp-xrp"),
    "dogeusdt": Asset("dogeusdt", "Dogecoin", "dogecoin", "doge-dogecoin"),
    "adausdt": Asset("adausdt", "Cardano", "cardano", "ada-cardano"),
    "avaxusdt": Asset("avaxusdt", "Avalanche", "avalanche-2", "avax-avalanche"),
    "dotusdt": Asset("dotusdt", "Polkadot", "polkadot", "dot-polkadot"),
    "linkusdt": Asset("linkusdt", "Chainlink", "chainlink", "link-chainlink"),
    "maticusdt": Asset("maticusdt", "Polygon", "matic-network", "matic-polygon"),
    "shibusdt": Asset("shibusdt", "Shiba Inu", "shiba-inu", "shib-shiba-inu"),
}

CG_URL = "https://api.coingecko.com/api/v3/simple/price"


# ======================================================
#               LOAD / SAVE STATE
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
#           PRICE PROVIDER (CG + CP)
# ======================================================
def fetch_price(asset: Asset):
    global CURRENT_PROVIDER

    # -----------------------
    # PRIMARY: COINGECKO
    # -----------------------
    if CURRENT_PROVIDER == "coingecko":
        try:
            r = requests.get(
                CG_URL,
                params={"ids": asset.cg_id, "vs_currencies": "usd"},
                timeout=6
            )
            if r.status_code == 200:
                d = r.json()
                p = d.get(asset.cg_id, {}).get("usd")
                if p is not None:
                    return float(p)
            CURRENT_PROVIDER = "coinpaprika"
        except:
            CURRENT_PROVIDER = "coinpaprika"

    # -----------------------
    # FALLBACK: COINPAPRIKA
    # -----------------------
    if CURRENT_PROVIDER == "coinpaprika":
        try:
            url = f"https://api.coinpaprika.com/v1/tickers/{asset.cp_id}"
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
#             POLLER THREAD
# ======================================================
def price_poller(asset: Asset, q, stop_event):
    while not stop_event.is_set():
        price = fetch_price(asset)
        if price is not None:
            ts = datetime.now().strftime("%H:%M:%S")
            q.put((ts, price))
        time.sleep(POLL_INTERVAL)


# ======================================================
#        INITIALIZE STREAMLIT SESSION STATE
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
            "prices": [], "times": [], "sma": [],
            "balance": 10000.0, "shares": 0.0,
            "avg_entry": 0.0, "trades": [],
            "active": False, "current_price": 0.0,
            "symbol": "btcusdt", "symbol_name": ASSETS["btcusdt"].name
        }

if "price_queue" not in st.session_state:
    st.session_state.price_queue = queue.Queue()

if "stop_event" not in st.session_state:
    st.session_state.stop_event = threading.Event()

if "poll_thread" not in st.session_state:
    st.session_state.poll_thread = None


state = st.session_state.state


# ======================================================
#                SIDEBAR CONTROLS
# ======================================================
st.sidebar.title("Control Panel")

symbol = st.sidebar.selectbox(
    "Select Asset",
    list(ASSETS.keys()),
    format_func=lambda x: ASSETS[x].name
)

current_asset = ASSETS[symbol]

# Asset change
if symbol != state["symbol"]:
    state["symbol"] = symbol
    state["symbol_name"] = current_asset.name
    state["prices"], state["times"], state["sma"] = [], [], []
    state["current_price"] = 0.0

    st.session_state.stop_event.set()
    st.session_state.stop_event = threading.Event()
    st.session_state.poll_thread = None


# Strategy params
SMA_PERIOD = st.sidebar.slider("SMA Period", 5, 50, 20, 5)
BUY_DIP = st.sidebar.number_input("Buy Below SMA (%)", 0.01, 5.0, 0.5) / 100
TP = st.sidebar.number_input("Take Profit (%)", 0.1, 5.0, 1.0) / 100
SL = st.sidebar.number_input("Stop Loss (%)", 0.1, 5.0, 2.0) / 100
INVEST_PCT = st.sidebar.slider("Invest %", 10, 100, 90) / 100


# Reset balance + history
def reset_all():
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
    st.session_state.state = {
        "prices": [], "times": [], "sma": [],
        "balance": 10000.0, "shares": 0.0, "avg_entry": 0.0,
        "trades": [], "active": False, "current_price": 0.0,
        "symbol": symbol, "symbol_name": current_asset.name
    }
    st.session_state.price_queue = queue.Queue()


st.sidebar.button("Reset Balance & History", on_click=reset_all)


# Start/stop bot
if st.sidebar.button("Start Bot" if not state["active"] else "Stop Bot"):
    state["active"] = not state["active"]

    if state["active"]:
        st.session_state.stop_event.clear()
        t = threading.Thread(
            target=price_poller,
            args=(current_asset, st.session_state.price_queue, st.session_state.stop_event),
            daemon=True
        )
        t.start()
        st.session_state.poll_thread = t
    else:
        st.session_state.stop_event.set()


# ======================================================
#              PROCESS PRICE QUEUE
# ======================================================
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

    # Trading
    if state["active"] and state["sma"][-1] is not None:
        sma = state["sma"][-1]

        # BUY
        if state["shares"] == 0 and price < sma * (1 - BUY_DIP):
            invest = state["balance"] * INVEST_PCT
            if invest > 0:
                state["shares"] = invest / price
                state["balance"] -= invest
                state["avg_entry"] = price
                state["trades"].insert(0, {
                    "time": ts, "type": "BUY",
                    "price": price, "amount": state["shares"], "pnl": 0
                })
                save_state(state)

        # SELL
        elif state["shares"] > 0:
            pct = (price - state["avg_entry"]) /
