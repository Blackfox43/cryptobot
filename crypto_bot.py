# app.py
"""
Multi-Asset Algo Trader (Bybit) with indicators, Telegram alerts, and optional live trading.

Features:
- Option C asset class architecture
- Bybit public polling (per-asset threads)
- Indicators: SMA crossover, RSI(14), MACD(12,26,9)
- Multi-asset simultaneous bots (select multiple from sidebar)
- Telegram alerts for signals/trades (optional, env vars)
- Real trading via Bybit (optional; set trade_mode == 'live' and API keys in env)
- Paper trading is default and safe.

WARNING: If you enable live trading, you are responsible for API keys and fund safety.
Test on Bybit testnet first.
"""

import streamlit as st
import requests, json, os, time, threading, queue, hmac, hashlib
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# -------------------------------
# Config
# -------------------------------
st.set_page_config(page_title="Multi-Asset Algo Trader", layout="wide")
POLL_INTERVAL = 3           # seconds between polls per asset
HISTORY_POINTS = 200        # keep this many points per asset
STATE_FILE = "multi_state.json"
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT = os.getenv("TELEGRAM_CHAT_ID")
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")

# -------------------------------
# Asset class & registry (Option C)
# -------------------------------
class Asset:
    def __init__(self, symbol, name, bybit_symbol):
        self.symbol = symbol
        self.name = name
        self.bybit_symbol = bybit_symbol

ASSETS = {
    "btcusdt": Asset("btcusdt", "Bitcoin", "BTCUSDT"),
    "ethusdt": Asset("ethusdt", "Ethereum", "ETHUSDT"),
    "bnbusdt": Asset("bnbusdt", "BNB", "BNBUSDT"),
    "solusdt": Asset("solusdt", "Solana", "SOLUSDT"),
    "xrpusdt": Asset("xrpusdt", "XRP", "XRPUSDT"),
    "dogeusdt": Asset("dogeusdt", "Dogecoin", "DOGEUSDT"),
    "adausdt": Asset("adausdt", "Cardano", "ADAUSDT"),
    "avaxusdt": Asset("avaxusdt", "Avalanche", "AVAXUSDT"),
    "dotusdt": Asset("dotusdt", "Polkadot", "DOTUSDT"),
    "linkusdt": Asset("linkusdt", "Chainlink", "LINKUSDT"),
    "maticusdt": Asset("maticusdt", "Polygon", "MATICUSDT"),
    "shibusdt": Asset("shibusdt", "Shiba Inu", "SHIBUSDT"),
}

BYBIT_TICKER_URL = "https://api.bybit.com/v5/market/tickers"
BYBIT_ORDER_URL = "https://api.bybit.com/v5/order/create"

# -------------------------------
# Persistence
# -------------------------------
def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return None

def save_state(state):
    try:
        safe = {k: v for k, v in state.items() if k in ("balances", "positions", "trade_history")}
        with open(STATE_FILE, "w") as f:
            json.dump(safe, f, indent=2)
    except:
        pass

# -------------------------------
# Indicators
# -------------------------------
def sma(prices, period):
    if len(prices) < period:
        return None
    return float(np.mean(prices[-period:]))

def ema_series(prices, period):
    if len(prices) == 0:
        return np.array([])
    alpha = 2 / (period + 1)
    prices = np.asarray(prices, float)
    out = np.empty_like(prices)
    out[:] = np.nan
    out[0] = prices[0]
    for i in range(1, len(prices)):
        out[i] = alpha * prices[i] + (1 - alpha) * out[i - 1]
    return out

def macd(prices, fast=12, slow=26, signal=9):
    if len(prices) < slow + signal:
        return None, None, None
    arr = np.asarray(prices, float)
    ema_fast = ema_series(arr, fast)
    ema_slow = ema_series(arr, slow)
    macd_line = ema_fast - ema_slow

    valid = ~np.isnan(macd_line)
    valid_values = macd_line[valid]
    if len(valid_values) < signal:
        return None, None, None

    sig_full = np.full_like(macd_line, np.nan)
    sig_values = ema_series(valid_values, signal)
    sig_full[np.where(valid)[0]] = sig_values

    hist = macd_line - sig_full
    return macd_line.tolist(), sig_full.tolist(), hist.tolist()

def rsi(prices, period=14):
    if len(prices) < period + 1:
        return None
    arr = np.asarray(prices, float)
    deltas = np.diff(arr)
    seed = deltas[:period]
    up = seed[seed > 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else np.inf
    rsi_series = np.empty_like(arr)
    rsi_series[:] = np.nan
    rsi_series[period] = 100 - (100 / (1 + rs))
    up_avg = up
    down_avg = down
    for i in range(period + 1, len(arr)):
        delta = deltas[i - 1]
        up_val = max(delta, 0)
        down_val = max(-delta, 0)
        up_avg = (up_avg * (period - 1) + up_val) / period
        down_avg = (down_avg * (period - 1) + down_val) / period
        rs = up_avg / down_avg if down_avg != 0 else np.inf
        rsi_series[i] = 100 - (100 / (1 + rs))
    valid = rsi_series[~np.isnan(rsi_series)]
    return float(valid[-1]) if len(valid) else None

# -------------------------------
# Bybit price + orders
# -------------------------------
def fetch_bybit_price(asset: Asset):
    try:
        r = requests.get(BYBIT_TICKER_URL,
                         params={"category": "spot", "symbol": asset.bybit_symbol},
                         timeout=6)
        if r.status_code == 200:
            data = r.json()
            lst = data.get("result", {}).get("list", [])
            if lst:
                p = lst[0].get("lastPrice")
                return float(p) if p else None
    except:
        pass
    return None

def place_bybit_order(symbol, side, qty):
    if not BYBIT_API_KEY or not BYBIT_API_SECRET:
        return {"error": "missing_api_keys"}

    body = {
        "category": "spot",
        "symbol": symbol,
        "side": side,
        "orderType": "Market",
        "qty": str(qty),
        "timeInForce": "GTC"
    }
    timestamp = str(int(time.time() * 1000))
    recv = "5000"
    body_str = json.dumps(body, separators=(',', ':'))
    payload = timestamp + BYBIT_API_KEY + recv + body_str
    sign = hmac.new(BYBIT_API_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()

    headers = {
        "Content-Type": "application/json",
        "X-BAPI-APIKEY": BYBIT_API_KEY,
        "X-BAPI-SIGN": sign,
        "X-BAPI-TIMESTAMP": timestamp,
        "X-BAPI-RECV-WINDOW": recv
    }

    try:
        r = requests.post(BYBIT_ORDER_URL, headers=headers, data=body_str, timeout=8)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

# -------------------------------
# Telegram
# -------------------------------
def send_telegram(msg):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": TELEGRAM_CHAT, "text": msg}, timeout=6)
    except:
        pass

# -------------------------------
# Polling thread (per asset)
# -------------------------------
def poller(asset_key, asset_obj, q, stop):
    while not stop.is_set():
        p = fetch_bybit_price(asset_obj)
        if p:
            q.put((asset_key, datetime.now().strftime("%H:%M:%S"), p))
        time.sleep(POLL_INTERVAL)

# -------------------------------
# Init Session State
# -------------------------------
if "machines" not in st.session_state:
    st.session_state.machines = {}

if "trade_state" not in st.session_state:
    loaded = load_state()
    st.session_state.trade_state = loaded if loaded else {
        "balances": {},
        "positions": {},
        "trade_history": []
    }

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("Multi-Asset Trader")

selected = st.sidebar.multiselect(
    "Choose assets to run",
    list(ASSETS.keys()),
    format_func=lambda k: ASSETS[k].name,
    default=["btcusdt"]
)

st.sidebar.subheader("Indicators")
fast_sma = st.sidebar.number_input("Fast SMA", 5, 100, 10)
slow_sma = st.sidebar.number_input("Slow SMA", fast_sma+1, 200, 30)
rsi_period = st.sidebar.number_input("RSI", 5, 30, 14)
macd_fast = st.sidebar.number_input("MACD Fast", 6, 26, 12)
macd_slow = st.sidebar.number_input("MACD Slow", 13, 52, 26)
macd_signal = st.sidebar.number_input("MACD Signal", 5, 20, 9)

trade_mode = st.sidebar.selectbox("Trade mode", ["paper", "live"], index=0)
paper_per_asset = st.sidebar.number_input("Paper USD per asset", 10.0, 100000.0, 1000.0, 10.0)
risk_pct = st.sidebar.slider("Risk % per trade", 1, 100, 10) / 100.0

enable_telegram = st.sidebar.checkbox("Telegram Alerts")
enable_live = st.sidebar.checkbox("Enable Live Trading")

if st.sidebar.button("Start Bots"):
    for key in selected:
        if key not in st.session_state.machines:
            q = queue.Queue()
            stop = threading.Event()
            t = threading.Thread(target=poller, args=(key, ASSETS[key], q, stop), daemon=True)
            t.start()
            st.session_state.machines[key] = {
                "thread": t,
                "stop": stop,
                "queue": q,
                "prices": [],
                "times": [],
                "sma_fast": [],
                "sma_slow": [],
                "rsi": None,
                "macd": None
            }
            if key not in st.session_state.trade_state["balances"]:
                st.session_state.trade_state["balances"][key] = paper_per_asset
                st.session_state.trade_state["positions"][key] = 0.0

if st.sidebar.button("Stop Bots"):
    for key, m in st.session_state.machines.items():
        try: m["stop"].set()
        except: pass
    st.session_state.machines = {}

# -------------------------------
# UI Cards
# -------------------------------
cols = st.columns(3)
ci = 0

for key, asset_obj in ASSETS.items():
    col = cols[ci % 3]
    ci += 1

    with col:
        st.header(asset_obj.name)
        running = key in st.session_state.machines
        st.write("Status:", "🟢 Running" if running else "🔴 Stopped")

        if running:
            m = st.session_state.machines[key]
            q = m["queue"]

            while not q.empty():
                akey, ts, price = q.get()
                m["prices"].append(price)
                m["times"].append(ts)

                if len(m["prices"]) > HISTORY_POINTS:
                    m["prices"].pop(0)
                    m["times"].pop(0)

                # Indicators
                m["sma_fast"].append(sma(m["prices"], fast_sma))
                m["sma_slow"].append(sma(m["prices"], slow_sma))
                m["rsi"] = rsi(m["prices"], rsi_period)
                mc, sig, hist = macd(m["prices"], macd_fast, macd_slow, macd_signal)
                m["macd"] = {"macd": mc, "sig": sig, "hist": hist}

                # Signal logic (simple)
                sigs = []
                # SMA crossover
                if len(m["sma_fast"]) >= 2 and len(m["sma_slow"]) >= 2:
                    f_prev, f_now = m["sma_fast"][-2], m["sma_fast"][-1]
                    s_prev, s_now = m["sma_slow"][-2], m["sma_slow"][-1]
                    if f_prev and s_prev and f_now and s_now:
                        if f_prev <= s_prev and f_now > s_now:
                            sigs.append("BUY")
                        if f_prev >= s_prev and f_now < s_now:
                            sigs.append("SELL")

                # RSI
                if m["rsi"]:
                    if m["rsi"] < 30: sigs.append("BUY")
                    if m["rsi"] > 70: sigs.append("SELL")

                # MACD
                if mc and sig:
                    def last2(arr):
                        idx = [i for i, v in enumerate(arr) if v]
                        return idx[-2:] if len(idx) > 2 else []
                    idxs = last2(mc)
                    if idxs:
                        i0, i1 = idxs
                        if mc[i0] <= sig[i0] and mc[i1] > sig[i1]:
                            sigs.append("BUY")
                        if mc[i0] >= sig[i0] and mc[i1] < sig[i1]:
                            sigs.append("SELL")

                # vote
                buy_votes = sigs.count("BUY")
                sell_votes = sigs.count("SELL")
                act = None
                if buy_votes >= 2: act = "BUY"
                if sell_votes >= 2: act = "SELL"

                if act:
                    bal = st.session_state.trade_state["balances"].get(key, 0)
                    pos = st.session_state.trade_state["positions"].get(key, 0)
                    qty = (bal * risk_pct) / price if act == "BUY" else pos

                    if qty > 0:
                        if trade_mode == "paper" or not enable_live:
                            if act == "BUY":
                                usd = bal * risk_pct
                                st.session_state.trade_state["balances"][key] = bal - usd
                                st.session_state.trade_state["positions"][key] = pos + qty
                                rec = {"time": ts, "asset": key, "type": "BUY", "price": price, "qty": qty}
                                st.session_state.trade_state["trade_history"].insert(0, rec)
                                if enable_telegram: send_telegram(f"PAPER BUY {asset_obj.name} @ {price}")
                            else:
                                revenue = pos * price
                                st.session_state.trade_state["balances"][key] = bal + revenue
                                st.session_state.trade_state["positions"][key] = 0
                                rec = {"time": ts, "asset": key, "type": "SELL", "price": price, "qty": qty}
                                st.session_state.trade_state["trade_history"].insert(0, rec)
                                if enable_telegram: send_telegram(f"PAPER SELL {asset_obj.name} @ {price}")

                        else:
                            res = place_bybit_order(asset_obj.bybit_symbol, act, qty)
                            rec = {"time": ts, "asset": key, "type": act+"_LIVE", "price": price, "qty": qty, "resp": res}
                            st.session_state.trade_state["trade_history"].insert(0, rec)
                            if enable_telegram: send_telegram(f"LIVE {act} {asset_obj.name} @ {price}")

                        save_state(st.session_state.trade_state)

            # Display last price
            if m["prices"]:
                st.metric("Last Price", f"${m['prices'][-1]:.4f}")

            # small chart
            if m["prices"]:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=m["times"], y=m["prices"], mode="lines", name="Price"))
                if len(m["sma_fast"]) == len(m["prices"]):
                    fig.add_trace(go.Scatter(x=m["times"], y=m["sma_fast"], mode="lines", name="Fast SMA"))
                if len(m["sma_slow"]) == len(m["prices"]):
                    fig.add_trace(go.Scatter(x=m["times"], y=m["sma_slow"], mode="lines", name="Slow SMA"))
                fig.update_layout(height=200)
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.write("Start this asset from the sidebar.")

# -------------------------------
# Trade history
# -------------------------------
st.subheader("Trade History")
if st.session_state.trade_state["trade_history"]:
    df = pd.DataFrame(st.session_state.trade_state["trade_history"])
    st.dataframe(df, use_container_width=True)
else:
    st.write("No trades yet.")

# -------------------------------
# AUTO-RERUN (Stable replacement)
# -------------------------------
time.sleep(1)
st.rerun()
