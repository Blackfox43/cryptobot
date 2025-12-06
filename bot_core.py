# bot_core.py
"""
Enhanced core: multi-asset pollers, SMA crossover, RSI, MACD, Telegram alerts,
paper trading engine per-asset, persistence.
Designed for Python 3.11 + Streamlit 1.36.
"""

import os
import time
import json
import threading
import queue
from datetime import datetime, timezone
import requests
import numpy as np

# ---------- Config ----------
POLL_INTERVAL = 10
HISTORY_POINTS = 500
STATE_FILE = "crypto_state_multi.json"
BYBIT_TICKER_URL = "https://api.bybit.com/v5/market/tickers"
COINGECKO_URL = "https://api.coingecko.com/api/v3/simple/price"
INITIAL_BALANCE = 10000.0

BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


# ---------- Assets ----------
class Asset:
    def __init__(self, key, name, bybit_symbol, cg_id=None):
        self.key = key
        self.name = name
        self.bybit_symbol = bybit_symbol
        self.cg_id = cg_id

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
    "maticusdt": Asset("maticusdt", "Polygon", "MATICUSDT", cg_id="polygon"),
    "shibusdt": Asset("shibusdt", "Shiba Inu", "SHIBUSDT", cg_id="shiba-inu"),
}


# ---------- persistence ----------
def load_state():
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def save_state(all_state):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(all_state, f, indent=2)
    except Exception as e:
        print("[save_state] error", e)


# ---------- price providers ----------
def fetch_price_bybit(asset: Asset):
    try:
        params = {"category": "linear", "symbol": asset.bybit_symbol}
        resp = requests.get(BYBIT_TICKER_URL, params=params, timeout=6)
        if resp.status_code != 200:
            return None
        d = resp.json()
        if d.get("retCode", 0) != 0:
            return None
        lst = d.get("result", {}).get("list", [])
        if lst:
            last = lst[0].get("lastPrice")
            return float(last) if last is not None else None
    except Exception:
        pass
    return None

def fetch_price_coingecko(asset: Asset):
    if not asset.cg_id:
        return None
    try:
        resp = requests.get(COINGECKO_URL, params={"ids": asset.cg_id, "vs_currencies": "usd"}, timeout=6)
        if resp.status_code != 200:
            return None
        data = resp.json()
        p = data.get(asset.cg_id, {}).get("usd")
        return float(p) if p is not None else None
    except Exception:
        return None

def fetch_price_with_fallback(asset: Asset):
    p = fetch_price_bybit(asset)
    if p is not None:
        return p, "Bybit"
    p = fetch_price_coingecko(asset)
    if p is not None:
        return p, "CoinGecko"
    return None, None


# ---------- indicators ----------
def sma(values, period):
    if len(values) < period:
        return None
    return float(np.mean(values[-period:]))

def rsi(values, period=14):
    if len(values) < period + 1:
        return None
    deltas = np.diff(np.array(values))
    ups = deltas.copy()
    downs = deltas.copy()
    ups[ups < 0] = 0
    downs[downs > 0] = 0
    # first average
    roll_up = np.mean(ups[-period:])
    roll_down = -np.mean(downs[-period:])
    if roll_down == 0:
        return 100.0
    rs = roll_up / roll_down
    return 100.0 - (100.0 / (1.0 + rs))

def macd(values, fast=12, slow=26, signal=9):
    # simple EMA based MACD (approx using pandas would be nicer, but avoid heavy deps during op)
    import pandas as pd
    s = pd.Series(values)
    if len(s) < slow + signal:
        return None, None, None
    ema_fast = s.ewm(span=fast, adjust=False).mean()
    ema_slow = s.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return float(macd_line.iloc[-1]), float(signal_line.iloc[-1]), float(hist.iloc[-1])


# ---------- telegram ----------
def send_telegram(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        resp = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text})
        return resp.status_code == 200
    except Exception:
        return False


# ---------- multi-asset engine ----------
# session-level structures are stored inside Streamlit session_state,
# but we also persist trades/balances per-asset in STATE_FILE.

def make_default_asset_state(asset_key, asset_name):
    return {
        "prices": [],
        "times": [],
        "sma": [],
        "provider": "---",
        "balance": INITIAL_BALANCE,
        "shares": 0.0,
        "avg_entry": 0.0,
        "trades": [],
        "active": False,
        "current_price": 0.0,
        "symbol": asset_key,
        "symbol_name": asset_name,
    }

# Poller thread for a single asset
def poller(asset_key, asset_obj, q, stop_event, poll_interval=POLL_INTERVAL):
    fail_count = 0
    while not stop_event.is_set():
        try:
            price, provider = fetch_price_with_fallback(asset_obj)
            if price is not None:
                ts_iso = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
                q.put((asset_key, ts_iso, price, provider))
                fail_count = 0
            else:
                fail_count += 1
                if fail_count >= 6:
                    print(f"[poller] {asset_key} fetch failed {fail_count} times")
                    fail_count = 0
        except Exception as e:
            print("[poller] unexpected:", e)

        slept = 0.0
        step = 0.5
        while slept < float(poll_interval):
            if stop_event.is_set():
                break
            time.sleep(step)
            slept += step


# ---------- session helpers (to be called from main) ----------
def init_session_state(st_session_state):
    # multi_state: mapping asset_key -> asset_state
    if "multi_state" not in st_session_state:
        # load persisted
        persisted = load_state()
        multi = {}
        for key, asset in ASSETS.items():
            if key in persisted:
                base = persisted[key]
                # ensure keys exist
                s = make_default_asset_state(key, asset.name)
                s.update(base)
            else:
                s = make_default_asset_state(key, asset.name)
            multi[key] = s
        st_session_state.multi_state = multi

    if "poll_threads" not in st_session_state:
        st_session_state.poll_threads = {}  # asset_key -> thread
    if "stop_events" not in st_session_state:
        st_session_state.stop_events = {}  # asset_key -> Event
    if "price_queue" not in st_session_state:
        st_session_state.price_queue = queue.Queue()
    if "strategy" not in st_session_state:
        # default strategy parameters (can be changed from UI)
        st_session_state.strategy = {
            "mode": "sma_dip",  # options: sma_dip, sma_crossover, rsi, macd, combo
            "SMA_PERIOD": 20,
            "BUY_DIP": 0.005,
            "TP": 0.01,
            "SL": 0.02,
            "INVEST_PCT": 0.9,
            # SMA crossover params
            "SMA_SHORT": 5,
            "SMA_LONG": 20,
            # RSI params
            "RSI_PERIOD": 14,
            "RSI_BUY": 30,
            "RSI_SELL": 70,
            # MACD params
            "MACD_FAST": 12,
            "MACD_SLOW": 26,
            "MACD_SIGNAL": 9,
            # misc
            "multi_assets": [],  # list of asset keys when multi mode selected
            "telegram_alerts": False,
        }

    if "lock" not in st_session_state:
        st_session_state.lock = threading.Lock()

def persist_all(st_session_state):
    """Persist balances/trades per-asset to disk (for quick recovery)."""
    try:
        out = {}
        for k, s in st_session_state.multi_state.items():
            # keep only necessary fields
            out[k] = {
                "balance": s.get("balance", INITIAL_BALANCE),
                "shares": s.get("shares", 0.0),
                "avg_entry": s.get("avg_entry", 0.0),
                "trades": s.get("trades", []),
                "symbol": s.get("symbol"),
                "symbol_name": s.get("symbol_name"),
            }
        save_state(out)
    except Exception as e:
        print("[persist_all]", e)


# --- start monitoring one or many assets ---
def start_asset_poll(st_session_state, asset_key):
    if asset_key in st_session_state.poll_threads and st_session_state.poll_threads[asset_key] is not None:
        return
    asset = ASSETS[asset_key]
    stop_ev = threading.Event()
    st_session_state.stop_events[asset_key] = stop_ev
    t = threading.Thread(target=poller, args=(asset_key, asset, st_session_state.price_queue, stop_ev), daemon=True)
    t.start()
    st_session_state.poll_threads[asset_key] = t
    # mark active flag in multi_state only when trading started
    st_session_state.multi_state[asset_key]["active"] = True
    print("[poller] started", asset_key)

def stop_asset_poll(st_session_state, asset_key):
    ev = st_session_state.stop_events.get(asset_key)
    if ev:
        ev.set()
    thr = st_session_state.poll_threads.get(asset_key)
    if thr:
        try:
            thr.join(timeout=5)
        except Exception:
            pass
    st_session_state.poll_threads[asset_key] = None
    st_session_state.stop_events[asset_key] = None
    st_session_state.multi_state[asset_key]["active"] = False
    print("[poller] stopped", asset_key)


def start_multi(st_session_state, asset_keys):
    # start pollers for each requested asset
    for k in asset_keys:
        start_asset_poll(st_session_state, k)

def stop_multi(st_session_state, asset_keys=None):
    # stop either specified or all
    keys = asset_keys if asset_keys is not None else list(st_session_state.poll_threads.keys())
    for k in keys:
        if k in st_session_state.poll_threads and st_session_state.poll_threads[k] is not None:
            stop_asset_poll(st_session_state, k)


# ---------- strategy & trading logic ----------
def process_price_updates(st_session_state):
    """Consume queue and update per-asset state, run indicators and trade decisions."""
    q = st_session_state.price_queue
    strat = st_session_state.strategy
    multi = st_session_state.multi_state
    lock = st_session_state.lock

    updated = False
    while not q.empty():
        try:
            asset_key, ts_iso, price, provider = q.get_nowait()
        except Exception:
            break

        if asset_key not in multi:
            continue

        s = multi[asset_key]
        # convert to HH:MM:SS for UI
        try:
            dt = datetime.fromisoformat(ts_iso)
            display_ts = dt.astimezone(timezone.utc).strftime("%H:%M:%S")
        except Exception:
            display_ts = datetime.utcnow().strftime("%H:%M:%S")

        s["current_price"] = price
        s["provider"] = provider
        s["prices"].append(price)
        s["times"].append(display_ts)
        if len(s["prices"]) > HISTORY_POINTS:
            s["prices"].pop(0)
            s["times"].pop(0)
            if s["sma"]:
                s["sma"].pop(0)

        # compute indicators
        # SMA (generic)
        sma_val = sma(s["prices"], strat["SMA_PERIOD"]) if len(s["prices"]) >= strat["SMA_PERIOD"] else None
        s["sma"].append(sma_val)

        # SMA crossover (short/long)
        sma_short = sma(s["prices"], strat["SMA_SHORT"]) if len(s["prices"]) >= strat["SMA_SHORT"] else None
        sma_long = sma(s["prices"], strat["SMA_LONG"]) if len(s["prices"]) >= strat["SMA_LONG"] else None

        # RSI
        rsi_val = rsi(s["prices"], strat["RSI_PERIOD"]) if len(s["prices"]) >= (strat["RSI_PERIOD"] + 1) else None

        # MACD
        macd_line, macd_signal, macd_hist = macd(s["prices"], strat["MACD_FAST"], strat["MACD_SLOW"], strat["MACD_SIGNAL"]) if len(s["prices"]) >= (strat["MACD_SLOW"] + strat["MACD_SIGNAL"]) else (None, None, None)

        # Decide signals according to selected strategy or combo
        buy_signal = False
        sell_signal = False
        signal_reasons = []

        mode = strat.get("mode", "sma_dip")

        if mode == "sma_dip":
            # old behavior: buy when price < sma * (1 - BUY_DIP)
            if s.get("sma") and s["sma"][-1] is not None and s.get("shares", 0) == 0:
                if price < s["sma"][-1] * (1 - strat["BUY_DIP"]):
                    buy_signal = True
                    signal_reasons.append("SMA dip")
            if s.get("shares", 0) > 0:
                pct = (price - s.get("avg_entry", 0)) / max(s.get("avg_entry", 1e-9), 1e-9)
                if pct >= strat["TP"]:
                    sell_signal = True
                    signal_reasons.append("Take Profit")
                elif pct <= -strat["SL"]:
                    sell_signal = True
                    signal_reasons.append("Stop Loss")

        if mode == "sma_crossover":
            # detect cross: previous values needed
            if sma_short is not None and sma_long is not None:
                # check previous points if available
                prev_short = sma(s["prices"][:-1], strat["SMA_SHORT"]) if len(s["prices"]) - 1 >= strat["SMA_SHORT"] else None
                prev_long = sma(s["prices"][:-1], strat["SMA_LONG"]) if len(s["prices"]) - 1 >= strat["SMA_LONG"] else None
                if prev_short is not None and prev_long is not None:
                    # upward cross
                    if prev_short <= prev_long and sma_short > sma_long and s.get("shares", 0) == 0:
                        buy_signal = True
                        signal_reasons.append("SMA crossover (golden)")
                    # downward cross
                    if prev_short >= prev_long and sma_short < sma_long and s.get("shares", 0) > 0:
                        sell_signal = True
                        signal_reasons.append("SMA crossover (death)")

        if mode == "rsi":
            if rsi_val is not None:
                if rsi_val < strat["RSI_BUY"] and s.get("shares", 0) == 0:
                    buy_signal = True
                    signal_reasons.append(f"RSI {rsi_val:.1f} < {strat['RSI_BUY']}")
                if rsi_val > strat["RSI_SELL"] and s.get("shares", 0) > 0:
                    sell_signal = True
                    signal_reasons.append(f"RSI {rsi_val:.1f} > {strat['RSI_SELL']}")

        if mode == "macd":
            if macd_line is not None and macd_signal is not None:
                # detect cross using last two values via recompute
                import pandas as pd
                s_series = pd.Series(s["prices"])
                ema_fast = s_series.ewm(span=strat["MACD_FAST"], adjust=False).mean()
                ema_slow = s_series.ewm(span=strat["MACD_SLOW"], adjust=False).mean()
                macd_full = ema_fast - ema_slow
                sig_full = macd_full.ewm(span=strat["MACD_SIGNAL"], adjust=False).mean()
                if len(macd_full) >= 2:
                    prev_macd = macd_full.iloc[-2]
                    prev_sig = sig_full.iloc[-2]
                    cur_macd = macd_full.iloc[-1]
                    cur_sig = sig_full.iloc[-1]
                    if prev_macd <= prev_sig and cur_macd > cur_sig and s.get("shares", 0) == 0:
                        buy_signal = True
                        signal_reasons.append("MACD cross up")
                    if prev_macd >= prev_sig and cur_macd < cur_sig and s.get("shares", 0) > 0:
                        sell_signal = True
                        signal_reasons.append("MACD cross down")

        if mode == "combo":
            # if any indicator gives a buy -> buy; sell if any sell
            if s.get("sma") and s["sma"][-1] is not None and s.get("shares",0) == 0:
                if price < s["sma"][-1] * (1 - strat["BUY_DIP"]):
                    buy_signal = True
                    signal_reasons.append("SMA dip")
            if rsi_val is not None and s.get("shares",0) == 0 and rsi_val < strat["RSI_BUY"]:
                buy_signal = True
                signal_reasons.append("RSI")
            if macd_line is not None and macd_signal is not None and s.get("shares",0) == 0:
                # simple macd check last>signal
                if macd_line > macd_signal:
                    buy_signal = True
                    signal_reasons.append("MACD")
            # sells: TP/SL or indicators
            if s.get("shares",0) > 0:
                pct = (price - s.get("avg_entry",0)) / max(s.get("avg_entry",1e-9),1e-9)
                if pct >= strat["TP"] or pct <= -strat["SL"]:
                    sell_signal = True
                    signal_reasons.append("TP/SL")

        # execute paper trades if signals true
        if buy_signal:
            # buy with INVEST_PCT of balance
            invest = s.get("balance", INITIAL_BALANCE) * strat["INVEST_PCT"]
            if invest > 0:
                s["shares"] = invest / price
                s["balance"] = s.get("balance", INITIAL_BALANCE) - invest
                s["avg_entry"] = price
                s["trades"].insert(0, {"time": display_ts, "type": "BUY", "price": price, "amount": s["shares"], "pnl": 0, "reason": ", ".join(signal_reasons)})
                persist_all(st_session_state)
                updated = True
                if strat.get("telegram_alerts"):
                    send_telegram(f"[TRADE BUY] {s['symbol_name']} {s['shares']:.6f} @ ${price:.4f} | reason: {', '.join(signal_reasons)}")

        if sell_signal and s.get("shares", 0) > 0:
            revenue = s["shares"] * price
            pnl = revenue - (s["shares"] * s.get("avg_entry", 0.0))
            s["balance"] = s.get("balance", INITIAL_BALANCE) + revenue
            s["trades"].insert(0, {"time": display_ts, "type": "SELL", "price": price, "amount": s["shares"], "pnl": pnl, "reason": ", ".join(signal_reasons)})
            s["shares"] = 0.0
            s["avg_entry"] = 0.0
            persist_all(st_session_state)
            updated = True
            if strat.get("telegram_alerts"):
                send_telegram(f"[TRADE SELL] {s['symbol_name']} PNL ${pnl:.2f} @ ${price:.4f} | reason: {', '.join(signal_reasons)}")

        # alerts for signals (without trade) if enabled
        if strat.get("telegram_alerts") and not (buy_signal or sell_signal) and len(signal_reasons) > 0:
            # send occasional signal alert (rate-limited by not spamming: only if last trade older than X or every N)
            # Simple gating: only send alert if last trade time not equal display_ts
            last_trade_time = s["trades"][0]["time"] if s["trades"] else None
            if last_trade_time != display_ts:
                send_telegram(f"[SIGNAL] {s['symbol_name']} price ${price:.4f} signals: {', '.join(signal_reasons)}")

        # end processing for this tick
    if updated:
        persist_all(st_session_state)


# ---------- helpers ----------
def get_multi_state(st_session_state):
    return st_session_state.multi_state

def get_asset_config_and_current_asset(st_session_state, key=None):
    k = key if key else st_session_state.multi_state and list(st_session_state.multi_state.keys())[0]
    return ASSETS, ASSETS.get(k, list(ASSETS.values())[0])
