# ui.py
"""
Professional UI — sidebar navigation, multi-asset controls, indicators panel, charts and trade logs.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

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

# Header
def render_header(symbol_name, connection_label, equity, profit):
    col1, col2 = st.columns([3, 1], gap="large")
    with col1:
        st.markdown(f"<div style='font-weight:700;font-size:1.2rem'>{symbol_name} · {connection_label}</div>", unsafe_allow_html=True)
    with col2:
        st.metric("Equity", f"${equity:,.2f}", delta=f"${profit:,.2f}")

# Sidebar
def render_sidebar_ui(ASSETS, st_state, start_multi, stop_multi, persist_and_reset):
    st.sidebar.title("Controls")
    st.sidebar.markdown("### Asset Selection")
    keys = list(ASSETS.keys())
    # Multi-select
    multi_selected = st.sidebar.multiselect("Monitor assets (multi-select)", keys, default=st_state.strategy.get("multi_assets", [list(ASSETS.keys())[0]]), format_func=lambda k: ASSETS[k].name)
    st_state.strategy["multi_assets"] = multi_selected

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Strategy Mode")
    mode = st.sidebar.selectbox("Mode", ["sma_dip", "sma_crossover", "rsi", "macd", "combo"], index=["sma_dip","sma_crossover","rsi","macd","combo"].index(st_state.strategy.get("mode","sma_dip")))
    st_state.strategy["mode"] = mode

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Indicators / Params")
    st_state.strategy["SMA_PERIOD"] = st.sidebar.slider("SMA period", 5, 100, int(st_state.strategy.get("SMA_PERIOD",20)))
    st_state.strategy["BUY_DIP"] = float(st.sidebar.number_input("Buy dip (%)", 0.01, 10.0, float(st_state.strategy.get("BUY_DIP",0.5))*100)/100.0)
    st_state.strategy["SMA_SHORT"] = st.sidebar.slider("SMA short (crossover)", 3, 50, int(st_state.strategy.get("SMA_SHORT",5)))
    st_state.strategy["SMA_LONG"] = st.sidebar.slider("SMA long (crossover)", 5, 200, int(st_state.strategy.get("SMA_LONG",20)))
    st_state.strategy["RSI_PERIOD"] = st.sidebar.number_input("RSI period", 7, 30, int(st_state.strategy.get("RSI_PERIOD",14)))
    st_state.strategy["RSI_BUY"] = st.sidebar.number_input("RSI buy threshold", 5, 50, int(st_state.strategy.get("RSI_BUY",30)))
    st_state.strategy["RSI_SELL"] = st.sidebar.number_input("RSI sell threshold", 50, 95, int(st_state.strategy.get("RSI_SELL",70)))
    st_state.strategy["MACD_FAST"] = st.sidebar.number_input("MACD fast", 6, 26, int(st_state.strategy.get("MACD_FAST",12)))
    st_state.strategy["MACD_SLOW"] = st.sidebar.number_input("MACD slow", 13, 52, int(st_state.strategy.get("MACD_SLOW",26)))
    st_state.strategy["MACD_SIGNAL"] = st.sidebar.number_input("MACD signal", 6, 20, int(st_state.strategy.get("MACD_SIGNAL",9)))
    st_state.strategy["TP"] = float(st.sidebar.number_input("Take profit (%)", 0.01, 50.0, float(st_state.strategy.get("TP",1.0)))/100.0)
    st_state.strategy["SL"] = float(st.sidebar.number_input("Stop loss (%)", 0.01, 50.0, float(st_state.strategy.get("SL",2.0)))/100.0)
    st_state.strategy["INVEST_PCT"] = float(st.sidebar.slider("Invest %", 1, 100, int(st_state.strategy.get("INVEST_PCT",0.9)*100))/100.0)
    st_state.strategy["telegram_alerts"] = st.sidebar.checkbox("Telegram alerts", value=st_state.strategy.get("telegram_alerts", False))
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Bot Control")
    if st.sidebar.button("Start Monitoring Selected"):
        start_multi(st_state, multi_selected)
    if st.sidebar.button("Stop Monitoring All"):
        stop_multi(st_state)
    if st.sidebar.button("Save & Reset balances (persist)"):
        persist_and_reset(st_state)
    st.sidebar.caption("Pro UI — Multi-asset & indicators")

# Small asset cards
def render_asset_cards(st_state, ASSETS):
    cols = st.columns(3)
    keys = st_state.strategy.get("multi_assets", [])
    if not keys:
        st.info("No assets selected — pick assets in the sidebar to monitor them.")
        return
    for i, k in enumerate(keys):
        if i % 3 == 0:
            cols = st.columns(3)
        s = st_state.multi_state.get(k, {})
        col = cols[i % 3]
        col.markdown(f"#### {ASSETS[k].name}")
        price = s.get("current_price", 0.0)
        bal = s.get("balance", 0.0)
        shares = s.get("shares", 0.0)
        col.metric("Price", f"${price:.4f}")
        col.metric("Balance", f"${bal:.2f}")
        col.metric("Holdings", f"{shares:.6f}")

# Pages
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
    render_header(s.get("symbol_name",""), s.get("provider","---"), s.get("balance",0), s.get("balance",0)-INITIAL_BALANCE)
    st.markdown("### Live Chart")
    if not s.get("prices"):
        st.info("Waiting for price data...")
        return
    import plotly.graph_objects as go
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

# App shell
def render_main_layout(page, st_state, ASSETS, current_asset, connection_label, equity, profit, start_multi, stop_multi, persist_and_reset):
    # Sidebar
    render_sidebar_ui(ASSETS, st_state, start_multi, stop_multi, persist_and_reset)

    # Main content
    if page == "dashboard":
        render_dashboard_page(st_state, ASSETS, connection_label, equity, profit)
    else:
        # asset detail pages use page to select a key
        if page in st_state.multi_state:
            render_asset_detail_page(st_state, page)
        elif page == "settings":
            render_settings_page()
        else:
            render_dashboard_page(st_state, ASSETS, connection_label, equity, profit)

    # Small footer nav (desktop)
    st.markdown("<div style='height:60px'></div>", unsafe_allow_html=True)
