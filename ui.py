# ui.py
"""
UI layer for CryptoBot multi-asset app.
- Fixed widget default values and conversions (percent <-> fraction)
- Clean sidebar controls and safe callbacks
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def set_mobile_theme():
    if "current_page" not in st.session_state:
        st.session_state.current_page = "dashboard"
    st.markdown(
        """
        <style>
          .stApp { background-color: #071227; color: #e6eef8; }
          .block-container { padding: 1rem 1.5rem 5.5rem 1.5rem; }
          .sidebar .sidebar-content { background: linear-gradient(180deg,#041021,#071227); }
          .header-title { font-size: 1.4rem; font-weight:700; margin:0; color:#e6eef8; }
          .header-sub { color:#9fb0c8; font-size:0.85rem; }
          footer { visibility:hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )

def get_active_page():
    return st.session_state.get("current_page", "dashboard")

def set_active_page(page):
    st.session_state.current_page = page

def render_header(symbol_name, connection_label, equity, profit):
    col1, col2 = st.columns([3, 1], gap="large")
    with col1:
        st.markdown(f"<div style='font-weight:700;font-size:1.2rem'>{symbol_name} · {connection_label}</div>", unsafe_allow_html=True)
    with col2:
        st.metric("Equity", f"${equity:,.2f}", delta=f"${profit:,.2f}")

def render_sidebar_ui(ASSETS, st_state, start_multi, stop_multi, persist_and_reset):
    st.sidebar.title("Controls")
    st.sidebar.markdown("### Asset Selection")
    keys = list(ASSETS.keys())
    default_selected = st_state.strategy.get("multi_assets") or [keys[0]]
    multi_selected = st.sidebar.multiselect("Monitor assets (multi-select)", keys, default=default_selected, format_func=lambda k: ASSETS[k].name)
    st_state.strategy["multi_assets"] = multi_selected

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Strategy Mode")
    modes = ["sma_dip", "sma_crossover", "rsi", "macd", "combo"]
    mode_index = modes.index(st_state.strategy.get("mode", "sma_dip")) if st_state.strategy.get("mode", "sma_dip") in modes else 0
    mode = st.sidebar.selectbox("Mode", modes, index=mode_index)
    st_state.strategy["mode"] = mode

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Indicators / Params")

    # Convert internal fractions to display percentages where appropriate
    sma_period = st.sidebar.slider("SMA period", 5, 100, int(st_state.strategy.get("SMA_PERIOD", 20)))
    st_state.strategy["SMA_PERIOD"] = sma_period

    buy_dip_pct = st.sidebar.number_input("Buy dip (%)", min_value=0.01, max_value=10.0, value=float(st_state.strategy.get("BUY_DIP", 0.005)) * 100.0)
    st_state.strategy["BUY_DIP"] = float(buy_dip_pct) / 100.0

    sma_short = st.sidebar.slider("SMA short (crossover)", 3, 50, int(st_state.strategy.get("SMA_SHORT", 5)))
    sma_long = st.sidebar.slider("SMA long (crossover)", 5, 200, int(st_state.strategy.get("SMA_LONG", 20)))
    st_state.strategy["SMA_SHORT"] = sma_short
    st_state.strategy["SMA_LONG"] = sma_long

    rsi_period = st.sidebar.number_input("RSI period", 7, 30, int(st_state.strategy.get("RSI_PERIOD", 14)))
    st_state.strategy["RSI_PERIOD"] = rsi_period

    rsi_buy = st.sidebar.number_input("RSI buy threshold", 5, 50, int(st_state.strategy.get("RSI_BUY", 30)))
    rsi_sell = st.sidebar.number_input("RSI sell threshold", 50, 95, int(st_state.strategy.get("RSI_SELL", 70)))
    st_state.strategy["RSI_BUY"] = int(rsi_buy)
    st_state.strategy["RSI_SELL"] = int(rsi_sell)

    macd_fast = st.sidebar.number_input("MACD fast", 6, 26, int(st_state.strategy.get("MACD_FAST", 12)))
    macd_slow = st.sidebar.number_input("MACD slow", 13, 52, int(st_state.strategy.get("MACD_SLOW", 26)))
    macd_signal = st.sidebar.number_input("MACD signal", 6, 20, int(st_state.strategy.get("MACD_SIGNAL", 9)))
    st_state.strategy["MACD_FAST"] = int(macd_fast)
    st_state.strategy["MACD_SLOW"] = int(macd_slow)
    st_state.strategy["MACD_SIGNAL"] = int(macd_signal)

    tp_pct = st.sidebar.number_input("Take profit (%)", min_value=0.01, max_value=50.0, value=float(st_state.strategy.get("TP", 0.01)) * 100.0)
    st_state.strategy["TP"] = float(tp_pct) / 100.0

    sl_pct = st.sidebar.number_input("Stop loss (%)", min_value=0.01, max_value=50.0, value=float(st_state.strategy.get("SL", 0.02)) * 100.0)
    st_state.strategy["SL"] = float(sl_pct) / 100.0

    invest_pct = st.sidebar.slider("Invest %", 1, 100, int(st_state.strategy.get("INVEST_PCT", 0.9) * 100))
    st_state.strategy["INVEST_PCT"] = float(invest_pct) / 100.0

    st_state.strategy["telegram_alerts"] = st.sidebar.checkbox("Telegram alerts", value=bool(st_state.strategy.get("telegram_alerts", False)))

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Bot Control")
    if st.sidebar.button("Start Monitoring Selected"):
        start_multi(st_state, multi_selected)
    if st.sidebar.button("Stop Monitoring All"):
        stop_multi(st_state)
    if st.sidebar.button("Save (persist) state to disk"):
        persist_and_reset(st_state)
    st.sidebar.caption("Pro UI — Multi-asset & indicators")

def render_asset_cards(st_state, ASSETS):
    keys = st_state.strategy.get("multi_assets", [])
    if not keys:
        st.info("No assets selected — pick assets in the sidebar to monitor them.")
        return
    cols = st.columns(3)
    for i, k in enumerate(keys):
        col = cols[i % 3]
        s = st_state.multi_state.get(k, {})
        col.markdown(f"#### {ASSETS[k].name}")
        price = s.get("current_price", 0.0)
        bal = s.get("balance", 0.0)
        shares = s.get("shares", 0.0)
        col.metric("Price", f"${price:.4f}")
        col.metric("Balance", f"${bal:.2f}")
        col.metric("Holdings", f"{shares:.6f}")

def render_dashboard_page(st_state, ASSETS, connection_label, equity, profit):
    render_header("Multi Asset Dashboard", connection_label, equity, profit)
    render_asset_cards(st_state, ASSETS)
    st.divider()
    st.markdown("### Recent Trades Across Assets")
    rows = []
    for k, s in st_state.multi_state.items():
        for t in s.get("trades", [])[:5]:
            rows.append({"asset": s.get("symbol_name"), **t})
    if rows:
        df = pd.DataFrame(rows)
        st.table(df.head(20))
    else:
        st.info("No trades recorded yet.")

def render_asset_detail_page(st_state, asset_key):
    s = st_state.multi_state.get(asset_key, {})
    render_header(s.get("symbol_name",""), s.get("provider","---"), s.get("balance",0), s.get("balance",0)-10000.0)
    st.markdown("### Live Chart")
    if not s.get("prices"):
        st.info("Waiting for price data...")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s["times"], y=s["prices"], mode="lines", name="Price"))
    if s.get("sma"):
        points = [(s["times"][i], s["sma"][i]) for i in range(len(s["sma"])) if s["sma"][i] is not None]
        if points:
            fig.add_trace(go.Scatter(x=[p[0] for p in points], y=[p[1] for p in points], mode="lines", name="SMA"))
    fig.update_layout(template="plotly_dark", height=420)
    st.plotly_chart(fig, use_container_width=True)
    st.divider()
    st.markdown("### Trades")
    if s.get("trades"):
        df = pd.DataFrame(s["trades"])
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No trades for this asset.")

def render_settings_page():
    st.markdown("### Settings")
    st.info("Configure API keys and secrets in Streamlit app secrets for safety.")

def render_main_layout(page, st_state, ASSETS, current_asset, connection_label, equity, profit, start_multi, stop_multi, persist_and_reset):
    render_sidebar_ui(ASSETS, st_state, start_multi, stop_multi, persist_and_reset)
    if page == "dashboard":
        render_dashboard_page(st_state, ASSETS, connection_label, equity, profit)
    else:
        if page in st_state.multi_state:
            render_asset_detail_page(st_state, page)
        elif page == "settings":
            render_settings_page()
        else:
            render_dashboard_page(st_state, ASSETS, connection_label, equity, profit)
    st.markdown("<div style='height:60px'></div>", unsafe_allow_html=True)
