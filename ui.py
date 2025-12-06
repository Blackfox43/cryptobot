# ui.py
"""
UI for CryptoBot (mobile-friendly).
Exposes:
- set_mobile_theme()
- get_active_page(), set_active_page()
- render_app_shell(...)
Designed to work with bot_core.py above and Main.py provided earlier.
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np

# ---------- Theme + CSS ----------
def set_mobile_theme():
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "home"
    st.markdown("""
    <style>
      .stApp { background: linear-gradient(180deg,#061022,#071226); color: #e6eef8; }
      .block-container { padding: 12px 14px 80px 14px; }
      .header { display:flex; justify-content:space-between; align-items:center; }
      .bottom-nav { position: fixed; bottom: 8px; left: 8px; right: 8px; display:flex; justify-content:space-around; gap:8px; z-index:9999; }
      .nav-btn { background:#0b1220; color:#9fb0c8; padding:8px 12px; border-radius:12px; border:1px solid #123; width:100%; }
      .nav-btn-active { background:#073544; color:#b6f0ff; border:1px solid #1f9; }
      footer { visibility:hidden; }
      .metric { background:#0f1720; padding:10px; border-radius:10px; }
    </style>
    """, unsafe_allow_html=True)

def get_active_page():
    return st.session_state.get("current_page", "home")

def set_active_page(page):
    st.session_state["current_page"] = page

# ---------- small helpers ----------
def _format_price(p):
    try:
        return f"${p:,.4f}"
    except Exception:
        return "$0.00"

# ---------- page pieces ----------
def render_header(symbol_name, connection_label):
    st.markdown(f"### Algo Trader — {symbol_name}  <small style='color:#9fb0c8'>{connection_label}</small>", unsafe_allow_html=True)

def render_metrics(state, equity, profit):
    col1, col2, col3, col4 = st.columns(4, gap="small")
    col1.metric("Price", _format_price(state.get("current_price", 0.0)))
    col2.metric("Balance", f"${state.get('balance',0):,.2f}")
    col3.metric("Equity", f"${equity:,.2f}")
    col4.metric("Profit", f"${profit:,.2f}")

def render_chart(state, config):
    st.subheader("Price Chart")
    if not state.get("prices"):
        st.info("Waiting for price data...")
        return
    fig = go.Figure()
    times = state.get("times", [])
    prices = state.get("prices", [])
    fig.add_trace(go.Scatter(x=times, y=prices, mode="lines", name="Price", line=dict(color="#10b981")))
    # SMA if present
    if state.get("sma"):
        sma_pts = [(times[i], state["sma"][i]) for i in range(len(state["sma"])) if state["sma"][i] is not None]
        if sma_pts:
            fig.add_trace(go.Scatter(x=[p[0] for p in sma_pts], y=[p[1] for p in sma_pts], mode="lines", name=f"SMA ({config.get('SMA_PERIOD',20)})", line=dict(color="#f59e0b", dash="dash")))
    fig.update_layout(template="plotly_dark", height=420, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)

def render_trades(state):
    st.subheader("Trade History")
    trades = state.get("trades", [])
    if not trades:
        st.info("No trades yet.")
        return
    # simple table
    st.table(trades[:20])

def render_bot_controls(state, config, ASSETS, start_bot, stop_bot, reset_state, poll_interval):
    st.subheader("Bot Controls & Strategy")
    col1, col2 = st.columns(2)
    with col1:
        symbol = st.selectbox("Asset", list(ASSETS.keys()), index=list(ASSETS.keys()).index(state.get("symbol", "btcusdt")), format_func=lambda k: ASSETS[k].name)
    with col2:
        if symbol != state.get("symbol"):
            # update selected symbol live — user will need to restart the bot to change polling
            st.session_state.state["symbol"] = symbol
            st.session_state.state["symbol_name"] = ASSETS[symbol].name
            st.session_state.state["prices"] = []
            st.session_state.state["times"] = []
            st.session_state.state["sma"] = []
            st.session_state.state["current_price"] = 0.0
            stop_bot(st.session_state)

    st.divider()
    # strategy params (convert fraction <-> percent)
    sma_p = st.number_input("SMA period", min_value=3, max_value=200, value=int(config.get("SMA_PERIOD", 20)))
    st.session_state.config["SMA_PERIOD"] = int(sma_p)

    buy_dip_pct = st.number_input("Buy below SMA (%)", min_value=0.01, max_value=50.0, value=float(config.get("BUY_DIP",0.005))*100.0)
    st.session_state.config["BUY_DIP"] = float(buy_dip_pct)/100.0

    tp_pct = st.number_input("Take profit (%)", min_value=0.01, max_value=50.0, value=float(config.get("TP", 0.01))*100.0)
    st.session_state.config["TP"] = float(tp_pct)/100.0

    sl_pct = st.number_input("Stop loss (%)", min_value=0.01, max_value=50.0, value=float(config.get("SL", 0.02))*100.0)
    st.session_state.config["SL"] = float(sl_pct)/100.0

    invest_pct = st.slider("Invest % of balance", min_value=1, max_value=100, value=int(config.get("INVEST_PCT",0.9)*100))
    st.session_state.config["INVEST_PCT"] = float(invest_pct)/100.0

    st.divider()
    col_a, col_b, col_c = st.columns([1,1,1])
    with col_a:
        if not state.get("active"):
            if st.button("▶ Start Bot", use_container_width=True):
                start_bot(st.session_state)
        else:
            if st.button("⏹ Stop Bot", use_container_width=True):
                stop_bot(st.session_state)
    with col_b:
        if st.button("Reset State", use_container_width=True):
            reset_state(st.session_state)
    with col_c:
        st.caption(f"Poll interval: {poll_interval}s")

# ---------- main renderer ----------
def render_app_shell(page, state, config, ASSETS, current_asset, connection_label, equity, profit, start_bot, stop_bot, reset_state, poll_interval):
    # Header + metrics
    render_header(state.get("symbol_name", "Asset"), connection_label)
    render_metrics(state, equity, profit)
    st.markdown("---")
    # Pages
    if page == "home":
        render_chart(state, config)
        st.divider()
        render_trades(state)
    elif page == "chart":
        render_chart(state, config)
    elif page == "trades":
        render_trades(state)
    elif page == "bot":
        render_bot_controls(state, config, ASSETS, start_bot, stop_bot, reset_state, poll_interval)
    else:
        render_chart(state, config)
    # bottom nav
    st.markdown("<div style='height:70px'></div>", unsafe_allow_html=True)
    cols = st.columns(4)
    labels = [("home","🏠"),("chart","📈"),("trades","📜"),("bot","⚙️")]
    for i,(p,icon) in enumerate(labels):
        btn = cols[i].button(f"{icon} {p.title()}", key=f"nav_{p}")
        if btn:
            set_active_page(p)
