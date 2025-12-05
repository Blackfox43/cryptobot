# crypto_bot.py
"""
CryptoBot - Multi-Asset Algo Trader (Bybit primary, CoinGecko fallback)
Features:
- Bybit public tickers for price (works in Nigeria)
- CoinGecko fallback
- Multi-asset simultaneous pollers
- Indicators: SMA crossover (fast/slow), RSI(14), MACD(12,26,9)
- Telegram alerts (optional)
- Optional live trading via Bybit (use testnet keys first)
- Persistence to JSON
- No pandas dependency
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
import streamlit as st
import plotly.graph_objects as go

# -------------------------
# Basic Config
# -------------------------
st.set_page_config(page_title="CryptoBot (Bybit)", layout="wide")
POLL_INTERVAL = 3  # seconds
HISTORY_POINTS = 300
STATE_FILE = "crypto_state.json"
BYBIT_TICKER_URL = "https://api.bybit.com/v5/market/tickers"
COINGECKO_URL = "https://api.coingecko.com/api/v3/simple/price"

# Env-configurable
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT = os.getenv("TELEGRAM_CHAT_ID")
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")

# -------------------------
# Asset registry
# -------------------------
class Asset:
    def __init__(self, key, name, bybit_symbol, cg_id=None):
        self.key = key
        self.name = name
        self.bybit_symbol = bybit_symbol  # e.g. BTCUSDT
        self.cg_id = cg_id                # e.g. bitcoin (optional)

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
    "maticusdt": Asset("maticusdt", "Polygon", "MATICUSDT", cg_id="matic-network"),
    "shibusdt": Asset("shibusdt", "Shiba Inu", "SHIBUSDT", cg_id="shiba-inu"),
}

# -------------------------
# Persistence helpers
# -------------------------
def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def save_state(state):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print("save_state error:", e)

# Default persisted structure
initial_persist = {
    "balances": {},       # USD per-asset (paper)
    "positions": {},      # units held per asset
    "trade_history": []   # list of trades
}

# -------------------------
# Indicator functions
# -------------------------
def sma(values, period):
    if len(values) < period:
        return None
    return float(np.mean(values[-period:]))

def ema_series(values, period):
    if len(values) == 0:
        return np.array([])
    values = np.array(values, dtype=float)
    alpha = 2 / (period + 1)
    out = np.empty(len(values), dtype=float)
    out[:] = np.nan
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out

def macd(values, fast=12, slow=26, signal=9):
    if len(values) < slow + signal:
        return None, None, None
    arr = np.array(values, dtype=float)
    fast_e = ema_series(arr, fast)
    slow_e = ema_series(arr, slow)
    macd_line = fast_e - slow_e
    valid = ~np.isnan(macd_line)
    if valid.sum() < signal:
        return None, None, None
    macd_valid = macd_line[valid]
    sig_vals = ema_series(macd_valid, signal)
    sig_full = np.full_like(macd_line, np.nan)
    sig_full[np.where(valid)[0]] = sig_vals
    hist = macd_line - sig_full
    return macd_line.tolist(), sig_full.tolist(), hist.tolist()

def rsi(values, period=14):
    if len(values) < period + 1:
        return None
    arr = np.array(values, dtype=float)
    deltas = np.diff(arr)
    seed = deltas[:period]
    up = seed[seed > 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else np.inf
    rsi_series = np.empty(len(arr), dtype=float)
    rsi_series[:] = np.nan
    rsi_series[period] = 100 - 100 / (1 + rs)
    up_avg = up
    down_avg = down
    for i in range(period + 1, len(arr)):
        delta = deltas[i - 1]
        up_val = max(delta, 0)
        down_val = max(-delta, 0)
        up_avg = (up_avg * (period - 1) + up_val) / period
        down_avg = (down_avg * (period - 1) + down_val) / period
        rs = up_avg / down_avg if down_avg != 0 else np.inf
        rsi_series[i] = 100 - 100 / (1 + rs)
    valid = rsi_series[~np.isnan(rsi_series)]
    return float(valid[-1]) if len(valid) else None

# -------------------------
# Bybit + CoinGecko price fetchers
# -------------------------
def fetch_bybit_price(asset: Asset):
    try:
        res = requests.get(BYBIT_TICKER_URL, params={"category":"spot","symbol":asset.bybit_symbol}, timeout=6)
        if res.status_code == 200:
            payload = res.json()
            lst = payload.get("result", {}).get("list", [])
            if lst and isinstance(lst, list):
                p = lst[0].get("lastPrice")
                if p:
                    return float(p)
    except Exception as e:
        #print("Bybit fetch error:", e)
        return None
    return None

def fetch_coingecko_price(asset: Asset):
    if not asset.cg_id:
        return None
    try:
        r = requests.get(COINGECKO_URL, params={"ids": asset.cg_id, "vs_currencies":"usd"}, timeout=6)
        if r.status_code == 200:
            d = r.json()
            p = d.get(asset.cg_id, {}).get("usd")
            if p is not None:
                return float(p)
    except Exception:
        return None
    return None

# High level fetch with fallback
def fetch_price_with_fallback(asset: Asset):
    # try Bybit first
    price = fetch_bybit_price(asset)
    if price is not None:
        return price, "bybit"
    # small wait to avoid hammering fallback provider
    price = fetch_coingecko_price(asset)
    if price is not None:
        return price, "coingecko"
    return None, None

# -------------------------
# Bybit order (minimal) - use testnet first
# -------------------------
BYBIT_ORDER_URL = "https://api.bybit.com/v5/order/create"
def place_bybit_order(symbol, side, qty):
    if not (BYBIT_API_KEY and BYBIT_API_SECRET):
        return {"error": "no_api_keys"}
    body = {
        "category":"spot",
        "symbol": symbol,
        "side": side,
        "orderType": "Market",
        "qty": str(qty),
        "timeInForce": "GTC"
    }
    ts = str(int(time.time() * 1000))
    recv = "5000"
    body_str = json.dumps(body, separators=(",", ":"))
    pre_sign = ts + BYBIT_API_KEY + recv + body_str
    sig = hmac.new(BYBIT_API_SECRET.encode(), pre_sign.encode(), hashlib.sha256).hexdigest()
    headers = {
        "Content-Type":"application/json",
        "X-BAPI-APIKEY": BYBIT_API_KEY,
        "X-BAPI-SIGN": sig,
        "X-BAPI-TIMESTAMP": ts,
        "X-BAPI-RECV-WINDOW": recv
    }
    try:
        r = requests.post(BYBIT_ORDER_URL, headers=headers, data=body_str, timeout=8)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

# -------------------------
# Telegram helper
# -------------------------
def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        resp = requests.post(url, json={"chat_id": TELEGRAM_CHAT, "text": message}, timeout=6)
        return resp.status_code == 200
    except Exception:
        return False

# -------------------------
# Poller thread function (per asset)
# -------------------------
def poller(asset_key, asset_obj, q, stop_event):
    fail_count = 0
    while not stop_event.is_set():
        try:
            price, provider = fetch_price_with_fallback(asset_obj)
            if price is not None:
                q.put((asset_key, datetime.now().strftime("%H:%M:%S"), price, provider))
                fail_count = 0
            else:
                fail_count += 1
                # If repeated fails, sleep a bit longer
                if fail_count > 8:
                    time.sleep(5)
        except Exception as e:
            # keep thread alive on unexpected errors
            print(f"Poller error for {asset_key}: {e}")
        time.sleep(POLL_INTERVAL)

# -------------------------
# Streamlit session state init
# -------------------------
if "machines" not in st.session_state:
    st.session_state.machines = {}  # asset_key -> { thread, stop_event, queue, prices, times, indicators }

if "trade_state" not in st.session_state:
    persisted = load_state()
    st.session_state.trade_state = persisted if persisted else initial_persist.copy()

# Sidebar UI controls
st.sidebar.title("Control Panel")
selected = st.sidebar.multiselect("Select assets", options=list(ASSETS.keys()),
                                  format_func=lambda k: ASSETS[k].name, default=["btcusdt"])
st.sidebar.markdown("---")

st.sidebar.subheader("Strategy")
fast_sma = st.sidebar.number_input("Fast SMA", min_value=3, max_value=200, value=10)
slow_sma = st.sidebar.number_input("Slow SMA", min_value=fast_sma+1, max_value=400, value=30)
rsi_period = st.sidebar.number_input("RSI period", min_value=5, max_value=30, value=14)
macd_fast = st.sidebar.number_input("MACD fast", min_value=6, max_value=30, value=12)
macd_slow = st.sidebar.number_input("MACD slow", min_value=macd_fast+1, max_value=60, value=26)
macd_signal = st.sidebar.number_input("MACD signal", min_value=5, max_value=20, value=9)

st.sidebar.markdown("---")
trade_mode = st.sidebar.selectbox("Trade mode", options=["paper", "live"], index=0)
paper_usd = st.sidebar.number_input("Paper USD per asset", min_value=10.0, value=1000.0, step=10.0)
risk_pct = st.sidebar.slider("Risk % per trade", 1, 100, 10) / 100.0
enable_telegram = st.sidebar.checkbox("Enable Telegram alerts", value=bool(TELEGRAM_TOKEN and TELEGRAM_CHAT))
enable_live = st.sidebar.checkbox("Enable Bybit live orders", value=False)

st.sidebar.markdown("---")
if st.sidebar.button("Start selected bots"):
    for key in selected:
        if key in st.session_state.machines:
            continue
        q = queue.Queue()
        stop_ev = threading.Event()
        t = threading.Thread(target=poller, args=(key, ASSETS[key], q, stop_ev), daemon=True)
        t.start()
        st.session_state.machines[key] = {
            "thread": t,
            "stop": stop_ev,
            "queue": q,
            "prices": [],
            "times": [],
            "sma_fast": [],
            "sma_slow": [],
            "rsi": None,
            "macd": None,
            "provider": None
        }
        # initialize paper balance if missing
        if key not in st.session_state.trade_state["balances"]:
            st.session_state.trade_state["balances"][key] = paper_usd
            st.session_state.trade_state["positions"][key] = 0.0

if st.sidebar.button("Stop all bots"):
    for key, m in list(st.session_state.machines.items()):
        try:
            m["stop"].set()
        except:
            pass
    st.session_state.machines = {}

if st.sidebar.button("Force save state"):
    save_state(st.session_state.trade_state)
    st.sidebar.success("State saved.")

if st.sidebar.button("Reset persisted state"):
    st.session_state.trade_state = initial_persist.copy()
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
    st.sidebar.success("Persistence reset.")

st.sidebar.markdown("---")
st.sidebar.write("Telegram:", "configured" if (TELEGRAM_TOKEN and TELEGRAM_CHAT) else "not configured")
st.sidebar.write("Bybit keys:", "configured" if (BYBIT_API_KEY and BYBIT_API_SECRET) else "not configured")

# -------------------------
# Main UI: Cards for each asset
# -------------------------
st.title("CryptoBot — Multi-Asset (Bybit primary)")

cols = st.columns(3)
ci = 0

for asset_key, asset in ASSETS.items():
    col = cols[ci % len(cols)]
    ci += 1
    with col:
        st.subheader(asset.name)
        running = asset_key in st.session_state.machines
        status_icon = "🟢" if running else "🔴"
        st.write("Status:", f"{status_icon} {'Running' if running else 'Stopped'}")

        # Show last price + connection indicator
        last_price = None
        last_time = None
        provider = None
        if running:
            machine = st.session_state.machines[asset_key]
            q = machine["queue"]
            drained = False
            while not q.empty():
                try:
                    k, ts, price, prov = q.get_nowait()
                except Exception:
                    break
                drained = True
                machine["prices"].append(price)
                machine["times"].append(ts)
                machine["provider"] = prov
                # trim
                if len(machine["prices"]) > HISTORY_POINTS:
                    machine["prices"].pop(0)
                    machine["times"].pop(0)
                    if machine["sma_fast"]: machine["sma_fast"].pop(0)
                    if machine["sma_slow"]: machine["sma_slow"].pop(0)
                # indicators
                machine["sma_fast"].append(sma(machine["prices"], fast_sma))
                machine["sma_slow"].append(sma(machine["prices"], slow_sma))
                machine["rsi"] = rsi(machine["prices"], rsi_period)
                mc, sig, hist = macd(machine["prices"], fast=macd_fast, slow=macd_slow, signal=macd_signal)
                machine["macd"] = {"macd": mc, "sig": sig, "hist": hist}
                last_price = price
                last_time = ts
                provider = prov

                # --- Strategy logic & signals ---
                # Build simple voting from SMA crossover, RSI extremes, MACD crossover
                signals = []
                # SMA crossover
                if len(machine["sma_fast"]) >= 2 and len(machine["sma_slow"]) >= 2:
                    f_prev, f_now = machine["sma_fast"][-2], machine["sma_fast"][-1]
                    s_prev, s_now = machine["sma_slow"][-2], machine["sma_slow"][-1]
                    if f_prev is not None and s_prev is not None and f_now is not None and s_now is not None:
                        if f_prev <= s_prev and f_now > s_now:
                            signals.append("BUY_SMA")
                        elif f_prev >= s_prev and f_now < s_now:
                            signals.append("SELL_SMA")
                # RSI
                if machine["rsi"] is not None:
                    if machine["rsi"] < 30:
                        signals.append("BUY_RSI")
                    elif machine["rsi"] > 70:
                        signals.append("SELL_RSI")
                # MACD
                macd_struct = machine["macd"]
                if macd_struct and macd_struct["macd"] and macd_struct["sig"]:
                    macd_list = macd_struct["macd"]
                    sig_list = macd_struct["sig"]
                    # find last two non-nan indices
                    valid_idx = [i for i, v in enumerate(macd_list) if v is not None and not (np.isnan(v))]
                    if len(valid_idx) >= 2:
                        i0, i1 = valid_idx[-2], valid_idx[-1]
                        if macd_list[i0] <= sig_list[i0] and macd_list[i1] > sig_list[i1]:
                            signals.append("BUY_MACD")
                        elif macd_list[i0] >= sig_list[i0] and macd_list[i1] < sig_list[i1]:
                            signals.append("SELL_MACD")

                # Voting: need >=2 buy signals to buy, >=2 sell signals to sell
                buy_votes = sum(1 for s in signals if s.startswith("BUY"))
                sell_votes = sum(1 for s in signals if s.startswith("SELL"))
                action = None
                if buy_votes >= 2:
                    action = "BUY"
                elif sell_votes >= 2:
                    action = "SELL"

                # Execute action: paper or live depending on settings
                if action:
                    now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    balances = st.session_state.trade_state["balances"]
                    positions = st.session_state.trade_state["positions"]
                    balances.setdefault(asset_key, paper_usd)
                    positions.setdefault(asset_key, 0.0)
                    bal = balances[asset_key]
                    pos = positions[asset_key]

                    if action == "BUY":
                        usd_to_use = bal * risk_pct
                        qty = usd_to_use / price if price > 0 else 0
                        if qty <= 0:
                            continue
                        if trade_mode == "paper" or not enable_live or not (BYBIT_API_KEY and BYBIT_API_SECRET):
                            # paper buy
                            balances[asset_key] = max(0.0, bal - usd_to_use)
                            positions[asset_key] = pos + qty
                            rec = {"time": now_ts, "asset": asset_key, "type": "PAPER_BUY", "price": price, "qty": qty}
                            st.session_state.trade_state["trade_history"].insert(0, rec)
                            if enable_telegram:
                                send_telegram(f"[PAPER BUY] {asset.name}: {qty:.6f} @ {price:.4f}")
                        else:
                            # live buy
                            res = place_bybit_order(asset.bybit_symbol, "Buy", qty)
                            rec = {"time": now_ts, "asset": asset_key, "type": "LIVE_BUY", "price": price, "qty": qty, "resp": res}
                            st.session_state.trade_state["trade_history"].insert(0, rec)
                            if enable_telegram:
                                send_telegram(f"[LIVE BUY] {asset.name}: resp {res}")
                    elif action == "SELL":
                        qty = pos
                        if qty <= 0:
                            continue
                        if trade_mode == "paper" or not enable_live or not (BYBIT_API_KEY and BYBIT_API_SECRET):
                            revenue = qty * price
                            balances[asset_key] = bal + revenue
                            positions[asset_key] = 0.0
                            rec = {"time": now_ts, "asset": asset_key, "type": "PAPER_SELL", "price": price, "qty": qty}
                            st.session_state.trade_state["trade_history"].insert(0, rec)
                            if enable_telegram:
                                send_telegram(f"[PAPER SELL] {asset.name}: {qty:.6f} @ {price:.4f}")
                        else:
                            res = place_bybit_order(asset.bybit_symbol, "Sell", qty)
                            rec = {"time": now_ts, "asset": asset_key, "type": "LIVE_SELL", "price": price, "qty": qty, "resp": res}
                            st.session_state.trade_state["trade_history"].insert(0, rec)
                            if enable_telegram:
                                send_telegram(f"[LIVE SELL] {asset.name}: resp {res}")

                    # persist after trade
                    save_state(st.session_state.trade_state)

            # if drained update UI fields
            if last_price is None and machine["prices"]:
                last_price = machine["prices"][-1]
                last_time = machine["times"][-1]
                provider = machine.get("provider")

            # Display metrics
            if last_price is not None:
                st.metric("Last Price", f"${last_price:.6f}", delta=None)
                st.write("Provider:", provider or "—")
            else:
                st.write("Waiting for price...")

            st.write("Position (units):", f"{st.session_state.trade_state['positions'].get(asset_key, 0.0):.6f}")
            st.write("Balance (USD):", f"${st.session_state.trade_state['balances'].get(asset_key, 0.0):.2f}")

            # small chart
            if machine["prices"]:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=machine["times"], y=machine["prices"], mode="lines", name="Price"))
                if len(machine["sma_fast"]) == len(machine["prices"]):
                    fig.add_trace(go.Scatter(x=machine["times"], y=machine["sma_fast"], mode="lines", name=f"SMA{fast_sma}", line=dict(dash="dash")))
                if len(machine["sma_slow"]) == len(machine["prices"]):
                    fig.add_trace(go.Scatter(x=machine["times"], y=machine["sma_slow"], mode="lines", name=f"SMA{slow_sma}", line=dict(dash="dot")))
                fig.update_layout(height=260, margin=dict(t=5,b=5))
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.write("Not running. Start this asset from the sidebar to begin polling.")

# -------------------------
# Global trade history & controls
# -------------------------
st.markdown("---")
st.subheader("Global Trade History (latest first)")
th = st.session_state.trade_state.get("trade_history", [])
if th:
    # Keep history small in UI
    display = th[:200]
    # render simple table using metric-like display via loop (no pandas)
    for rec in display:
        st.write(f"{rec.get('time')} • {rec.get('asset')} • {rec.get('type')} • {rec.get('qty'):.6f} @ {rec.get('price'):.6f}")
else:
    st.write("No trades yet.")

# Sidebar small status for running machines
st.sidebar.markdown("---")
st.sidebar.subheader("Running machines")
for k, m in st.session_state.machines.items():
    alive = m["thread"].is_alive() if m.get("thread") else False
    st.sidebar.write(f"{ASSETS[k].name}: {'🟢' if alive else '🔴'}")

# Provide manual reconnect for each machine
if st.sidebar.button("Restart all pollers"):
    for k, m in st.session_state.machines.items():
        try:
            m["stop"].set()
        except:
            pass
    st.session_state.machines = {}
    st.experimental_rerun()

# Persist periodically
if int(time.time()) % 30 == 0:
    save_state(st.session_state.trade_state)

# Auto refresh UI
time.sleep(1)
st.rerun()
