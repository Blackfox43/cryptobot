import os
import time
import json
import queue
import threading
from datetime import datetime
import requests
import numpy as np


POLL_INTERVAL = 3
STATE_FILE = "state.json"
CURRENT_PROVIDER = "coingecko"


class Asset:
    def __init__(self, symbol, name, cg_id, cp_id):
        self.symbol = symbol
        self.name = name
        self.cg_id = cg_id
        self.cp_id = cp_id


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


def init_session_state(st):
    saved = load_state()

    if "state" not in st.session_state:
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


def fetch_price(asset: Asset):
    global CURRENT_PROVIDER

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


def price_poller(asset: Asset, q, stop_event):
    while not stop_event.is_set():
        p = fetch_price(asset)
        if p is not None:
            ts = datetime.now().strftime("%H:%M:%S")
            q.put((ts, p))
        time.sleep(POLL_INTERVAL)


def start_bot(st, asset):
    if st.session_state.state["active"]:
        return

    st.session_state.state["active"] = True
    st.session_state.stop_event.clear()

    t = threading.Thread(
        target=price_poller,
        args=(asset, st.session_state.price_queue, st.session_state.stop_event),
        daemon=True
    )
    t.start()
    st.session_state.poll_thread = t


def stop_bot(st):
    if not st.session_state.state["active"]:
        return

    st.session_state.state["active"] = False
    st.session_state.stop_event.set()


def process_prices(state, SMA_PERIOD, BUY_DIP, TP, SL, INVEST_PCT):
    while not state["price_queue"].empty():
        ts, price = state["price_queue"].get()

        st_state = state["state"]
        st_state["current_price"] = price
        st_state["prices"].append(price)
        st_state["times"].append(ts)

        if len(st_state["prices"]) > 100:
            st_state["prices"].pop(0)
            st_state["times"].pop(0)
            if st_state["sma"]:
                st_state["sma"].pop(0)

        if len(st_state["prices"]) >= SMA_PERIOD:
            st_state["sma"].append(np.mean(st_state["prices"][-SMA_PERIOD:]))
        else:
            st_state["sma"].append(None)

        if st_state["active"] and st_state["sma"][-1] is not None:
            sma = st_state["sma"][-1]

            if st_state["shares"] == 0 and price < sma * (1 - BUY_DIP):
                invest = st_state["balance"] * INVEST_PCT
                if invest > 0:
                    st_state["shares"] = invest / price
                    st_state["balance"] -= invest
                    st_state["avg_entry"] = price

                    st_state["trades"].insert(0, {
                        "time": ts,
                        "type": "BUY",
                        "price": price,
                        "amount": st_state["shares"],
                        "pnl": 0
                    })
                    save_state(st_state)

            elif st_state["shares"] > 0:
                pct = (price - st_state["avg_entry"]) / st_state["avg_entry"]
                if pct > TP or pct < -SL:
                    revenue = st_state["shares"] * price
                    pnl = revenue - (st_state["shares"] * st_state["avg_entry"])

                    st_state["balance"] += revenue
                    st_state["trades"].insert(0, {
                        "time": ts,
                        "type": "SELL",
                        "price": price,
                        "amount": st_state["shares"],
                        "pnl": pnl
                    })

                    st_state["shares"] = 0
                    st_state["avg_entry"] = 0
                    save_state(st_state)


def get_state(st):
    return st.session_state
