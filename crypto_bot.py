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
import requests, json, os, time, threading, queue, hmac, hashlib, base64
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
        except Exception as e:
            print("load_state error:", e)
    return None

def save_state(state):
    try:
        # Only save non-transient fields
        safe = {k: v for k, v in state.items() if k in ("balances", "positions", "trade_history")}
        with open(STATE_FILE, "w") as f:
            json.dump(safe, f, indent=2)
    except Exception as e:
        print("save_state error:", e)

# -------------------------------
# Utilities: indicators
# -------------------------------
def sma(prices, period):
    prices = np.asarray(prices, dtype=float)
    if len(prices) < period:
        return None
    return float(np.mean(prices[-period:]))

def ema_series(prices, period):
    # returns numpy array same length as prices (first values are NaN until enough data)
    prices = np.asarray(prices, dtype=float)
    if len(prices) == 0:
        return np.array([])
    alpha = 2 / (period + 1)
    out = np.empty_like(prices)
    out[:] = np.nan
    out[0] = prices[0]
    for i in range(1, len(prices)):
        out[i] = alpha * prices[i] + (1 - alpha) * out[i-1]
    return out

def macd(prices, fast=12, slow=26, signal=9):
    if len(prices) < slow + signal:
        return None, None, None
    fast_ema = ema_series(prices, fast)
    slow_ema = ema_series(prices, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema_series(macd_line[~np.isnan(macd_line)], signal)  # compute on valid part
    # align lengths: easiest is to compute full macd and pad signal with NaN to match macd_line length
    # compute signal properly:
    valid_macd = macd_line.copy()
    valid_idx = ~np.isnan(valid_macd)
    if sum(valid_idx) < signal:
        return None, None, None
    sig_full = np.full_like(macd_line, np.nan)
    sig_values = ema_series(valid_macd[valid_idx], signal)
    sig_full[np.where(valid_idx)[0]] = sig_values
    hist = macd_line - sig_full
    return macd_line.tolist(), sig_full.tolist(), hist.tolist()

def rsi(prices, period=14):
    if len(prices) < period + 1:
        return None
    arr = np.asarray(prices, dtype=float)
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
    # return last rsi value
    last_valid = rsi_series[~np.isnan(rsi_series)]
    return float(last_valid[-1]) if len(last_valid) > 0 else None

# -------------------------------
# Bybit helpers (price + optional order)
# -------------------------------
def fetch_bybit_price(asset: Asset):
    try:
        r = requests.get(BYBIT_TICKER_URL, params={"category":"spot","symbol":asset.bybit_symbol}, timeout=6)
        if r.status_code == 200:
            payload = r.json()
            lst = payload.get("result", {}).get("list", [])
            if lst:
                price = lst[0].get("lastPrice")
                return float(price) if price is not None else None
    except Exception as e:
        print("fetch_bybit_price error:", e)
    return None

# Minimal Bybit order function (experimental). Requires BYBIT_API_KEY and BYBIT_API_SECRET as env vars.
def place_bybit_order(asset_symbol, side, qty, reduce_only=False):
    """
    Attempts to create a market order on Bybit.
    This implementation uses Bybit v5 REST create order endpoint and HMAC SHA256 signature.
    PLEASE test with testnet keys before enabling 'live' mode.
    """
    if not BYBIT_API_KEY or not BYBIT_API_SECRET:
        print("Bybit API keys not provided; skipping live order.")
        return {"error":"no_api_keys"}

    # Build body
    body = {
        "category":"spot",
        "symbol": asset_symbol,
        "side": side.upper(),   # BUY or SELL
        "orderType": "Market",
        "qty": str(qty),        # quantity in base asset units
        "timeInForce": "GTC",
        "reduceOnly": reduce_only
    }
    # Bybit v5 signing requires timestamp + recv_window + apiKey + body (stringified)
    timestamp = str(int(time.time() * 1000))
    recv_window = "5000"
    body_str = json.dumps(body, separators=(',', ':'))
    pre_sign = timestamp + BYBIT_API_KEY + recv_window + body_str
    signature = hmac.new(BYBIT_API_SECRET.encode(), pre_sign.encode(), hashlib.sha256).hexdigest()
    headers = {
        "Content-Type": "application/json",
        "X-BAPI-APIKEY": BYBIT_API_KEY,
        "X-BAPI-SIGN": signature,
        "X-BAPI-TIMESTAMP": timestamp,
        "X-BAPI-RECV-WINDOW": recv_window
    }
    try:
        r = requests.post(BYBIT_ORDER_URL, headers=headers, data=body_str, timeout=8)
        return r.json()
    except Exception as e:
        print("place_bybit_order error:", e)
        return {"error": str(e)}

# -------------------------------
# Telegram helper
# -------------------------------
def send_telegram(msg):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        print("Telegram not configured; message:", msg)
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        resp = requests.post(url, json={"chat_id": TELEGRAM_CHAT, "text": msg}, timeout=6)
        return resp.status_code == 200
    except Exception as e:
        print("Telegram send error:", e)
        return False

# -------------------------------
# Poller thread per asset
# -------------------------------
def poller_thread(asset_key, asset_obj, q, stop_event):
    while not stop_event.is_set():
        price = fetch_bybit_price(asset_obj)
        if price is not None:
            q.put((asset_key, datetime.now().strftime("%H:%M:%S"), price))
        time.sleep(POLL_INTERVAL)

# -------------------------------
# Session init
# -------------------------------
if "machines" not in st.session_state:
    # structure:
    # machines: {
    #   asset_key: { "thread": Thread, "stop_event": Event, "queue": Queue }
    # }
    st.session_state.machines = {}

# Global state for trading info & persistence
if "trade_state" not in st.session_state:
    loaded = load_state()
    st.session_state.trade_state = loaded if loaded else {
        "balances": {},            # per-asset paper balances (in USD)
        "positions": {},           # per-asset position (amount held)
        "trade_history": []        # global trade log
    }

# UI controls
st.sidebar.title("Multi-Asset Algo Trader")
selected = st.sidebar.multiselect("Choose assets to run", options=list(ASSETS.keys()),
                                  format_func=lambda k: ASSETS[k].name, default=["btcusdt"])

# Strategy settings
st.sidebar.subheader("Strategy / Indicator settings")
fast_sma = st.sidebar.number_input("Fast SMA (crossover)", min_value=5, max_value=100, value=10)
slow_sma = st.sidebar.number_input("Slow SMA (crossover)", min_value=fast_sma+1, max_value=200, value=30)
rsi_period = st.sidebar.number_input("RSI period", min_value=5, max_value=30, value=14)
macd_fast = st.sidebar.number_input("MACD fast", min_value=6, max_value=26, value=12)
macd_slow = st.sidebar.number_input("MACD slow", min_value=13, max_value=52, value=26)
macd_signal = st.sidebar.number_input("MACD signal", min_value=5, max_value=20, value=9)

trade_mode = st.sidebar.selectbox("Trade mode", options=["paper", "live"], index=0)
paper_usd_per_asset = st.sidebar.number_input("Paper USD per asset", min_value=10.0, value=1000.0, step=10.0)
risk_pct = st.sidebar.slider("Risk % per trade (paper)", min_value=1, max_value=100, value=10) / 100.0

# Alerts & trading toggles
st.sidebar.subheader("Alerts & Trading")
enable_telegram = st.sidebar.checkbox("Enable Telegram alerts", value=bool(TELEGRAM_TOKEN and TELEGRAM_CHAT))
enable_live_trading = st.sidebar.checkbox("Enable live trading (Bybit)", value=False)
if enable_live_trading and trade_mode != "live":
    st.sidebar.error("To use live trading, set Trade mode = live")

# Start/Stop controls
if st.sidebar.button("Start selected bots"):
    # start pollers & machine entries for selected assets
    for key in selected:
        if key in st.session_state.machines:
            continue
        q = queue.Queue()
        stop_ev = threading.Event()
        t = threading.Thread(target=poller_thread, args=(key, ASSETS[key], q, stop_ev), daemon=True)
        t.start()
        st.session_state.machines[key] = {"thread": t, "stop": stop_ev, "queue": q, "prices": [], "times": [], "sma_fast": [], "sma_slow": [], "macd": None, "rsi": None}
        # initialize paper balance/position if not present
        if key not in st.session_state.trade_state["balances"]:
            st.session_state.trade_state["balances"][key] = paper_usd_per_asset
            st.session_state.trade_state["positions"][key] = 0.0

if st.sidebar.button("Stop all bots"):
    for key, m in st.session_state.machines.items():
        try:
            m["stop"].set()
        except:
            pass
    st.session_state.machines = {}

# Manual reconnect single asset
if st.sidebar.button("Restart queues"):
    for key, m in st.session_state.machines.items():
        try:
            m["stop"].set()
        except:
            pass
    st.session_state.machines = {}

# -------------------------------
# Main processing: drain all machines' queues
# -------------------------------
# We'll gather per-asset UI cards
cols = st.columns(3)
col_idx = 0

for key, asset_obj in ASSETS.items():
    # prepare a card for each asset whether running or not
    col = cols[col_idx % len(cols)]
    col_idx += 1

    with col:
        st.header(asset_obj.name)
        running = key in st.session_state.machines
        status = "Running" if running else "Stopped"
        st.write(f"Status: **{status}**")
        # show last price if we have it
        last_price = None
        if running:
            m = st.session_state.machines[key]
            # drain queue
            q = m["queue"]
            drained = False
            while not q.empty():
                try:
                    akey, ts, price = q.get_nowait()
                except Exception:
                    break
                drained = True
                m["prices"].append(price)
                m["times"].append(ts)
                # trim
                if len(m["prices"]) > HISTORY_POINTS:
                    m["prices"].pop(0); m["times"].pop(0)
                # compute indicators on the fly
                # SMA fast/slow
                if len(m["prices"]) >= fast_sma:
                    m["sma_fast"].append(float(np.mean(m["prices"][-fast_sma:])))
                else:
                    m["sma_fast"].append(None)
                if len(m["prices"]) >= slow_sma:
                    m["sma_slow"].append(float(np.mean(m["prices"][-slow_sma:])))
                else:
                    m["sma_slow"].append(None)
                # RSI
                m["rsi"] = rsi(m["prices"], period=rsi_period)
                # MACD
                macd_line, macd_sig, macd_hist = macd(m["prices"], fast=macd_fast, slow=macd_slow, signal=macd_signal)
                m["macd"] = {"macd": macd_line, "sig": macd_sig, "hist": macd_hist}
                last_price = price

                # Strategy signals (simple interpretation)
                # SMA crossover signal: when fast SMA crosses above slow SMA -> BUY; below -> SELL
                sig = None
                if len(m["sma_fast"]) >= 2 and len(m["sma_slow"]) >= 2:
                    f_prev = m["sma_fast"][-2]; f_now = m["sma_fast"][-1]
                    s_prev = m["sma_slow"][-2]; s_now = m["sma_slow"][-1]
                    if f_prev is not None and s_prev is not None and f_now is not None and s_now is not None:
                        if f_prev <= s_prev and f_now > s_now:
                            sig = "SMA_BUY"
                        elif f_prev >= s_prev and f_now < s_now:
                            sig = "SMA_SELL"

                # RSI signal: overbought (>70) -> SELL, oversold (<30) -> BUY
                rsi_signal = None
                if m["rsi"] is not None:
                    if m["rsi"] > 70:
                        rsi_signal = "RSI_SELL"
                    elif m["rsi"] < 30:
                        rsi_signal = "RSI_BUY"

                # MACD signal: cross of macd and signal line
                macd_signal_flag = None
                if m["macd"] and m["macd"]["macd"] and m["macd"]["sig"]:
                    macd_macd = m["macd"]["macd"]
                    macd_sig_v = m["macd"]["sig"]
                    # find last two valid indices
                    def last_two_valid(arr):
                        idxs = [i for i, v in enumerate(arr) if v is not None and not (np.isnan(v))]
                        return idxs[-2:] if len(idxs) >= 2 else []
                    idxs = last_two_valid(macd_macd)
                    if len(idxs) == 2:
                        i0, i1 = idxs
                        m0, m1 = macd_macd[i0], macd_macd[i1]
                        s0, s1 = macd_sig_v[i0], macd_sig_v[i1]
                        if m0 <= s0 and m1 > s1:
                            macd_signal_flag = "MACD_BUY"
                        elif m0 >= s0 and m1 < s1:
                            macd_signal_flag = "MACD_SELL"

                # Compose overall signal (simple voting)
                signals = [s for s in (sig, rsi_signal, macd_signal_flag) if s]
                # vote BUY if at least two BUY signals, SELL if at least two SELL signals
                buy_votes = sum(1 for s in signals if s.endswith("_BUY"))
                sell_votes = sum(1 for s in signals if s.endswith("_SELL"))
                action = None
                if buy_votes >= 2:
                    action = "BUY"
                elif sell_votes >= 2:
                    action = "SELL"

                # If action triggered, create paper trade or live order
                if action:
                    entry_price = price
                    # determine qty: for paper, use risk_pct of balance converted to units
                    bal = st.session_state.trade_state["balances"].get(key, paper_usd_per_asset if 'paper_usd_per_asset' in locals() else 1000.0)
                    if action == "BUY":
                        usd_to_use = bal * risk_pct
                        qty = usd_to_use / entry_price
                        if trade_mode == "paper" or not BYBIT_API_KEY:
                            # paper: update position & balance
                            st.session_state.trade_state["balances"][key] = max(0.0, bal - usd_to_use)
                            st.session_state.trade_state["positions"][key] = st.session_state.trade_state["positions"].get(key, 0.0) + qty
                            rec = {"time": ts, "asset": key, "type": "BUY", "price": entry_price, "qty": qty, "mode": trade_mode}
                            st.session_state.trade_state["trade_history"].insert(0, rec)
                            msg = f"[PAPER BUY] {asset_obj.name}: {qty:.6f} @ {entry_price:.4f}"
                            print(msg)
                            if enable_telegram: send_telegram(msg)
                        else:
                            # live
                            res = place_bybit_order(asset_obj.bybit_symbol, "Buy", qty, reduce_only=False)
                            rec = {"time": ts, "asset": key, "type": "BUY_LIVE", "price": entry_price, "qty": qty, "resp": res}
                            st.session_state.trade_state["trade_history"].insert(0, rec)
                            if enable_telegram: send_telegram(f"[LIVE BUY] {asset_obj.name} resp: {res}")
                    else: # SELL
                        pos = st.session_state.trade_state["positions"].get(key, 0.0)
                        if pos > 0:
                            qty = pos
                            if trade_mode == "paper" or not BYBIT_API_KEY:
                                revenue = qty * entry_price
                                st.session_state.trade_state["balances"][key] = st.session_state.trade_state["balances"].get(key, 0.0) + revenue
                                st.session_state.trade_state["positions"][key] = 0.0
                                rec = {"time": ts, "asset": key, "type": "SELL", "price": entry_price, "qty": qty, "mode": trade_mode}
                                st.session_state.trade_state["trade_history"].insert(0, rec)
                                msg = f"[PAPER SELL] {asset_obj.name}: {qty:.6f} @ {entry_price:.4f}"
                                print(msg)
                                if enable_telegram: send_telegram(msg)
                            else:
                                res = place_bybit_order(asset_obj.bybit_symbol, "Sell", qty, reduce_only=False)
                                rec = {"time": ts, "asset": key, "type": "SELL_LIVE", "price": entry_price, "qty": qty, "resp": res}
                                st.session_state.trade_state["trade_history"].insert(0, rec)
                                if enable_telegram: send_telegram(f"[LIVE SELL] {asset_obj.name} resp: {res}")

            # end draining
            if drained:
                # persist trade_state
                save_state(st.session_state.trade_state)

            # display last metrics
            last_price = m["prices"][-1] if m["prices"] else None
            st.write("Last:", f"${last_price:.4f}" if last_price else "—")
            st.write("Position (units):", f"{st.session_state.trade_state['positions'].get(key, 0.0):.6f}")
            st.write("Balance (USD):", f"{st.session_state.trade_state['balances'].get(key, 0.0):.2f}")
        else:
            st.write("Not running. Click 'Start selected bots' in sidebar.")

        # small chart preview for that asset (if prices exist)
        m_prices = st.session_state.machines.get(key, {}).get("prices", [])
        m_times = st.session_state.machines.get(key, {}).get("times", [])
        if m_prices:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=m_times, y=m_prices, mode="lines", name="Price"))
            # add last SMA values if exist
            smf = st.session_state.machines[key].get("sma_fast", [])
            sms = st.session_state.machines[key].get("sma_slow", [])
            # plot aligned if lengths match
            try:
                if len(smf) == len(m_times):
                    fig.add_trace(go.Scatter(x=m_times, y=smf, mode="lines", name=f"SMA{fast_sma}", line=dict(dash="dash")))
                if len(sms) == len(m_times):
                    fig.add_trace(go.Scatter(x=m_times, y=sms, mode="lines", name=f"SMA{slow_sma}", line=dict(dash="dot")))
            except Exception:
                pass
            fig.update_layout(height=240, margin=dict(t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Global trade history display
# -------------------------------
st.markdown("---")
st.subheader("Global Trade History (most recent first)")
th = st.session_state.trade_state.get("trade_history", [])
if th:
    df = pd.DataFrame(th[:200])
    st.dataframe(df, use_container_width=True)
else:
    st.write("No trades executed yet.")

# -------------------------------
# Connection status & housekeeping
# -------------------------------
alive = {k: (v["thread"].is_alive() if v.get("thread") else False) for k, v in st.session_state.machines.items()}
st.sidebar.markdown("### Running machines")
for k, v in alive.items():
    st.sidebar.write(f"{ASSETS[k].name}: {'🟢' if v else '🔴'}")

# Auto-refresh
time.sleep(1)
st.rerun()

