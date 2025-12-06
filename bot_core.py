# bot_core.py
import os
import time
import json
import threading
import queue
from datetime import datetime, timezone
import requests
import numpy as np

POLL_INTERVAL = 10  # seconds (adjustable)
HISTORY_POINTS = 300
STATE_FILE = "crypto_state.json"
BYBIT_TICKER_URL = "https://api.bybit.com/v5/market/tickers"
COINGECKO_URL = "https://api.coingecko.com/api/v3/simple/price"
INITIAL_BALANCE = 10000.0

BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")


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


def load_state():
    if not os.path.exists(STATE_FILE):
        return None
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return None


def save_state(state):
    try:
        safe = {
            "balance": state.get("balance", INITIAL_BALANCE),
            "shares": state.get("shares", 0.0),
            "avg_entry": state.get("avg_entry", 0.0),
            "trades": state.get("trades", []),
            "symbol": state.get("symbol", "btcusdt"),
            "symbol_name": state.get("symbol_name", "Bitcoin"),
        }
        with open(STATE_FILE, "w") as f:
            json.dump(safe, f, indent=4)
    except Exception as e:
        print("[save_state] error:", e)


def fetch_price_bybit(asset: Asset):
    try:
        params = {"category": "linear", "symbol": asset.bybit_symbol}
        resp = requests.get(BYBIT_TICKER_URL, params=params, timeout=6)
        if resp.status_code != 200:
            return None
        data = resp.json()
        if data.get("retCode", 0) != 0:
            return None
        lst = data.get("result", {}).get("list", [])
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
        if resp.status_code == 429:
            return None
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


def init_session_state(st_session_state):
    if "state" not in st_session_state:
        saved = load_state()
        default = "btcusdt"
        if saved:
            st_session_state.state = {
                **saved,
                "prices": [],
                "times": [],
                "sma": [],
                "providers": [],
                "current_price": 0.0,
                "provider": "---",
                "active": False,
            }
        else:
            st_session_state.state = {
                "prices": [],
                "times": [],
                "sma": [],
                "providers": [],
                "balance": INITIAL_BALANCE,
                "shares": 0.0,
                "avg_entry": 0.0,
                "trades": [],
                "active": False,
                "current_price": 0.0,
                "provider": "---",
                "symbol": default,
                "symbol_name": ASSETS[default].name,
            }

    if "price_queue" not in st_session_state:
        st_session_state.price_queue = queue.Queue()
    if "stop_event" not in st_session_state:
        st_session_state.stop_event = threading.Event()
    if "poll_thread" not in st_session_state:
        st_session_state.poll_thread = None
    if "config" not in st_session_state:
        st_session_state.config = {
            "SMA_PERIOD": 20,
            "BUY_DIP": 0.005,
            "TP": 0.01,
            "SL": 0.02,
            "INVEST_PCT": 0.9,
        }


def reset_state(st_session_state):
    try:
        stop_bot(st_session_state)
    except Exception:
        pass
    if os.path.exists(STATE_FILE):
        try:
            os.remove(STATE_FILE)
        except Exception:
            pass

    key = st_session_state.state.get("symbol", "btcusdt")
    st_session_state.state = {
        "prices": [],
        "times": [],
        "sma": [],
        "providers": [],
        "balance": INITIAL_BALANCE,
        "shares": 0.0,
        "avg_entry": 0.0,
        "trades": [],
        "active": False,
        "current_price": 0.0,
        "provider": "---",
        "symbol": key,
        "symbol_name": ASSETS.get(key, ASSETS["btcusdt"]).name,
    }
    st_session_state.price_queue = queue.Queue()
    st_session_state.stop_event = threading.Event()
    st_session_state.poll_thread = None
    print("[state] reset")


def start_bot(st_session_state):
    state = st_session_state.state
    if state.get("active"):
        return
    # ensure primitives
    if "stop_event" not in st_session_state:
        st_session_state.stop_event = threading.Event()
    if "price_queue" not in st_session_state:
        st_session_state.price_queue = queue.Queue()

    # stop any existing
    if getattr(st_session_state, "poll_thread", None):
        try:
            stop_bot(st_session_state)
        except Exception:
            pass

    key = state.get("symbol", "btcusdt")
    asset = ASSETS.get(key, ASSETS["btcusdt"])
    st_session_state.stop_event.clear()
    t = threading.Thread(target=poller, args=(key, asset, st_session_state.price_queue, st_session_state.stop_event), daemon=True)
    t.start()
    st_session_state.poll_thread = t
    state["active"] = True
    print("[bot] started", key)


def stop_bot(st_session_state):
    try:
        st_session_state.stop_event.set()
    except Exception:
        pass
    thr = getattr(st_session_state, "poll_thread", None)
    if thr:
        try:
            thr.join(timeout=POLL_INTERVAL + 5)
        except Exception:
            pass
    st_session_state.poll_thread = None
    st_session_state.state["active"] = False
    print("[bot] stopped")


def process_price_updates(st_session_state):
    cfg = st_session_state.config
    state = st_session_state.state
    q = st_session_state.price_queue
    while not q.empty():
        try:
            asset_key, ts_iso, price, provider = q.get_nowait()
        except Exception:
            break
        if asset_key != state.get("symbol"):
            continue
        # human display time
        try:
            dt = datetime.fromisoformat(ts_iso)
            display_ts = dt.astimezone(timezone.utc).strftime("%H:%M:%S")
        except Exception:
            display_ts = datetime.utcnow().strftime("%H:%M:%S")

        state["current_price"] = price
        state["provider"] = provider
        state["prices"].append(price)
        state["times"].append(display_ts)
        state["providers"].append(provider)

        if len(state["prices"]) > HISTORY_POINTS:
            state["prices"].pop(0)
            state["times"].pop(0)
            state["providers"].pop(0)
            if state["sma"]:
                state["sma"].pop(0)

        if len(state["prices"]) >= cfg["SMA_PERIOD"]:
            state["sma"].append(np.mean(state["prices"][-cfg["SMA_PERIOD"] :]))
        else:
            state["sma"].append(None)

        # Trading logic
        if state.get("active") and state["sma"] and state["sma"][-1] is not None:
            sma = state["sma"][-1]
            if state.get("shares", 0) == 0 and price < sma * (1 - cfg["BUY_DIP"]):
                invest = state.get("balance", INITIAL_BALANCE) * cfg["INVEST_PCT"]
                if invest > 0:
                    state["shares"] = invest / price
                    state["balance"] = state.get("balance", INITIAL_BALANCE) - invest
                    state["avg_entry"] = price
                    state["trades"].insert(0, {"time": display_ts, "type": "BUY", "price": price, "amount": state["shares"], "pnl": 0})
                    save_state(state)
                    print("[trade] BUY", state["shares"], "@", price)
            elif state.get("shares", 0) > 0:
                pct = (price - state.get("avg_entry", 0.0)) / max(state.get("avg_entry", 1e-9), 1e-9)
                is_sell = False
                ttype = "SELL"
                if pct >= cfg["TP"]:
                    ttype = "SELL (TP)"
                    is_sell = True
                elif pct <= -cfg["SL"]:
                    ttype = "SELL (SL)"
                    is_sell = True
                if is_sell:
                    revenue = state["shares"] * price
                    pnl = revenue - (state["shares"] * state.get("avg_entry", 0.0))
                    state["balance"] = state.get("balance", INITIAL_BALANCE) + revenue
                    state["trades"].insert(0, {"time": display_ts, "type": ttype, "price": price, "amount": state["shares"], "pnl": pnl})
                    state["shares"] = 0.0
                    state["avg_entry"] = 0.0
                    save_state(state)
                    print("[trade]", ttype, "PNL:", pnl)


def get_connection_status(st_session_state):
    state = st_session_state.state
    if state.get("times"):
        last = state["times"][-1]
        try:
            last_dt = datetime.strptime(last, "%H:%M:%S").time()
            now = datetime.utcnow().time()
            def tod(t): return t.hour * 3600 + t.minute * 60 + t.second
            delay = abs(tod(now) - tod(last_dt))
        except Exception:
            delay = POLL_INTERVAL * 10
        if delay < POLL_INTERVAL + 5:
            return "🟢 Live"
        if delay < 60:
            return "🟡 Slow"
        return "🔴 Disconnected"
    return "🔴 Waiting"


def get_equity_and_profit(st_session_state):
    s = st_session_state.state
    equity = s.get("balance", INITIAL_BALANCE) + s.get("shares", 0.0) * s.get("current_price", 0.0)
    profit = equity - INITIAL_BALANCE
    return equity, profit


def get_asset_config_and_current_asset(st_session_state):
    state = st_session_state.state
    key = state.get("symbol", "btcusdt")
    return ASSETS, ASSETS.get(key, ASSETS["btcusdt"])
