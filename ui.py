# ui.py (mobile-friendly UI, no pandas)
import streamlit as st
import plotly.graph_objects as go

def set_mobile_theme():
    if "current_page" not in st.session_state:
        st.session_state.current_page = "home"
    st.markdown(
        """
    <style>
      .stApp { background-color: #0d0f12 !important; color: white !important; }
      div[data-testid="metric-container"]{ background:#15181d; border-radius:12px; padding:10px; }
      .app-header{ padding:8px 0; text-align:center; color:#f0f0f0; border-bottom:1px solid #222; margin-bottom:8px;}
      .app-header h1{ margin:0; font-size:1.4rem;}
      .app-header span{ font-size:0.8rem; color:#aaa;}
      .block-container{ padding-bottom:90px !important; } /* room for bottom nav */
      .bottom-buttons { position: fixed; left: 0; bottom: 0; width: 100%; background:#0b0d10; padding:6px 8px; border-top:1px solid #222; z-index:9999; display:flex; gap:6px; justify-content:space-around; }
      .nav-btn{ color:#bfc9d9; font-weight:600; }
    </style>
    """,
        unsafe_allow_html=True,
    )

def get_active_page():
    return st.session_state.get("current_page", "home")

def set_active_page(page_name):
    st.session_state.current_page = page_name

def render_header(symbol_name, connection_label):
    st.markdown(f'<div class="app-header"><h1>Algo Trader</h1><span>{symbol_name} · {connection_label}</span></div>', unsafe_allow_html=True)

def render_bottom_nav(active_page):
    cols = st.columns(4)
    labels = [("home","Metrics"), ("chart","Chart"), ("trades","Trades"), ("bot","Bot")]
    for i, (key, label) in enumerate(labels):
        with cols[i]:
            clicked = st.button(label, key=f"nav_{key}")
            if clicked:
                set_active_page(key)

def render_home_page(state, config, ASSETS, current_asset, equity, profit):
    st.subheader(f"{current_asset.name} Price & Equity")
    unrealized = 0.0
    if state.get("shares", 0) > 0 and state.get("avg_entry", 0) > 0:
        unrealized = state["shares"] * (state["current_price"] - state["avg_entry"])
    col1, col2 = st.columns(2)
    col1.metric("Price", f"${state['current_price']:.4f}")
    col2.metric("Equity", f"${equity:.2f}")
    col3, col4 = st.columns(2)
    col3.metric("Balance", f"${state['balance']:.2f}")
    col4.metric("Unrealized PNL", f"${unrealized:.2f}")
    st.caption(f"Source: **{state.get('provider','---')}**")
    st.divider()
    st.subheader("Recent trades")
    if state.get("trades"):
        st.table(state["trades"][:10])
    else:
        st.info("No trades yet.")

def render_chart_page(state, config, current_asset):
    st.subheader(f"{current_asset.bybit_symbol} Chart")
    if not state.get("prices"):
        st.info("Waiting for price data...")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=state["times"], y=state["prices"], mode="lines", name="Price"))
    sma = state.get("sma", [])
    sma_points = [(state["times"][i], sma[i]) for i in range(len(sma)) if sma[i] is not None]
    if sma_points:
        fig.add_trace(go.Scatter(x=[p[0] for p in sma_points], y=[p[1] for p in sma_points], mode="lines", name=f"SMA {config['SMA_PERIOD']}", line=dict(dash="dash")))
    fig.update_layout(template="plotly_dark", height=420, margin=dict(l=8,r=8,t=8,b=8))
    st.plotly_chart(fig, use_container_width=True)

def render_trades_page(state):
    st.subheader("Trades")
    if not state.get("trades"):
        st.info("No trades yet.")
        return
    st.table(state["trades"])

def render_bot_page(state, config, ASSETS, current_asset, start_bot, stop_bot, reset_state, poll_interval):
    st.subheader("Bot & Strategy")
    cols = st.columns([3,1])
    with cols[0]:
        keys = list(ASSETS.keys())
        idx = keys.index(state.get("symbol","btcusdt"))
        sel = st.selectbox("Asset", keys, index=idx, format_func=lambda k: ASSETS[k].name, key="asset_select")
        if sel != state.get("symbol"):
            st.session_state.state["symbol"] = sel
            st.session_state.state["symbol_name"] = ASSETS[sel].name
    with cols[1]:
        if st.button("Reset", use_container_width=True):
            reset_state()
    st.divider()
    st.subheader("Strategy Parameters")
    sma = st.slider("SMA Period", 5, 50, config["SMA_PERIOD"], key="ui_sma")
    st.session_state.config["SMA_PERIOD"] = sma
    invest = st.slider("Invest %", 10, 100, int(config["INVEST_PCT"]*100), key="ui_invest")
    st.session_state.config["INVEST_PCT"] = invest/100.0
    st.caption(f"Polling interval: {poll_interval} s")
    st.divider()
    status = "Running" if state.get("active") else "Stopped"
    st.write(f"**Status:** {status}")
    if not state.get("active"):
        if st.button("Start Bot"):
            start_bot()
    else:
        if st.button("Stop Bot"):
            stop_bot()

def render_app_shell(page, state, config, ASSETS, current_asset, connection_label, equity, profit, start_bot, stop_bot, reset_state, poll_interval):
    render_header(state.get("symbol_name",""), connection_label)
    if page == "home":
        render_home_page(state, config, ASSETS, current_asset, equity, profit)
    elif page == "chart":
        render_chart_page(state, config, current_asset)
    elif page == "trades":
        render_trades_page(state)
    elif page == "bot":
        render_bot_page(state, config, ASSETS, current_asset, start_bot, stop_bot, reset_state, poll_interval)
    # bottom nav
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    render_bottom_nav(get_active_page())
