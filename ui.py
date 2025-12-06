import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd

# ===========================================================
# THEME ENGINE (MOBILE FIRST)
# ===========================================================
def set_mobile_theme():
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = 'home'
        
    st.markdown("""
        <style>
            .stApp {
                background-color: #0d0f12 !important;
                color: white !important;
            }

            div[data-testid="metric-container"] {
                background: #15181d;
                border-radius: 16px;
                padding: 12px;
                margin: 6px 0;
            }

            /* FIXED bottom nav — now safe */
            .bottom-nav {
                position: fixed;
                bottom: 0;
                left: 0;
                width: 100%;
                background: #111318;
                padding: 8px 0 6px 0;
                display: flex;
                justify-content: space-around;
                border-top: 1px solid #222;
                z-index: 999999;
            }

            .nav-link {
                text-align: center;
                font-size: 12px;
                color: #aaa;
                cursor: pointer;
                padding: 4px 6px;
            }

            .nav-link.active {
                color: #4fc3f7;
                font-weight: 700;
            }

            .app-header {
                padding: 8px 0 4px 0;
                text-align: center;
                color: #f0f0f0;
                border-bottom: 1px solid #222;
                margin-bottom: 10px;
            }
            .app-header h1 {
                font-size: 1.5rem;
                margin: 0;
            }
            .app-header span {
                font-size: 0.8rem;
                color: #aaa;
            }
            
            #MainMenu, footer {
                visibility: hidden;
            }

            /* SAFER content padding */
            .block-container {
                padding-bottom: 80px !important;
            }
        </style>
    """, unsafe_allow_html=True)

# ===========================================================
# NAV LOGIC
# ===========================================================
def get_active_page():
    return st.session_state.get("current_page", "home")

def set_active_page(page_name):
    st.session_state["current_page"] = page_name

# ===========================================================
# HEADER
# ===========================================================
def render_header(symbol_name, connection_label):
    st.markdown(f"""
        <div class="app-header">
            <h1>Algo Trader</h1>
            <span>{symbol_name} | {connection_label}</span>
        </div>
    """, unsafe_allow_html=True)

# ===========================================================
# FIXED BOTTOM NAVIGATION BAR (HTML + JS)
# ===========================================================
def render_bottom_nav(active_page):

    nav_items = [
        ("home", "📊", "Metrics"),
        ("chart", "📈", "Chart"),
        ("trades", "📜", "Trades"),
        ("bot", "⚙️", "Bot"),
    ]

    html = '<div class="bottom-nav">'

    for page, icon, label in nav_items:
        cls = "nav-link active" if page == active_page else "nav-link"
        html += f"""
            <div class="{cls}" onclick="fetch('/_stcore/trigger?nav={page}')">
                {icon}<br>{label}
            </div>
        """

    html += "</div>"

    st.markdown(html, unsafe_allow_html=True)

    # JS → triggers Streamlit event
    st.markdown("""
        <script>
            document.addEventListener("click", function(e) {
                if(e.target.closest(".nav-link")) {
                    const page = e.target.closest(".nav-link").innerText.split("\\n")[0];
                }
            });
        </script>
    """, unsafe_allow_html=True)

    # hidden form that updates session_state
    # The query param check is handled in main.py using st.session_state
    for page, _, _ in nav_items:
        if st.query_params.get("nav") == page:
            st.session_state["current_page"] = page

# ===========================================================
# HOME PAGE
# ===========================================================
def render_home_page(state, config, ASSETS, current_asset, equity, profit):

    st.subheader(f"{current_asset.name} Price & Equity")

    unrealized_pnl = 0
    pnl_delta_value = None
    pnl_color_mode = "off"

    if state["shares"] > 0 and state["avg_entry"] > 0:
        unrealized_pnl = state["shares"] * (state["current_price"] - state["avg_entry"])
        pnl_delta_value = unrealized_pnl
        pnl_color_mode = "normal"

    col1, col2 = st.columns(2)
    col1.metric("Current Price", f"${state['current_price']:.2f}")
    col2.metric("Total Equity", f"${equity:.2f}")

    col3, col4 = st.columns(2)
    col3.metric("Cash Balance", f"${state['balance']:.2f}")
    col4.metric("Unrealized PNL", f"${unrealized_pnl:.2f}", delta=pnl_delta_value, delta_color=pnl_color_mode)

    col5, col6 = st.columns(2)
    col5.metric("Shares Held", f"{state['shares']:.4f}")
    col6.metric("Total Profit", f"${profit:.2f}", delta=profit, delta_color="normal")

    st.caption(f"Price source: **{state['provider']}**")

# ===========================================================
# CHART PAGE
# ===========================================================
def render_chart_page(state, config, current_asset):
    st.subheader(f"{current_asset.bybit_symbol} Price Chart")

    if len(state["prices"]) == 0:
        st.info("Waiting for price data…")
        return

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=state["times"],
        y=state["prices"],
        mode="lines",
        name="Price",
        line=dict(color="#10b981")
    ))

    sma = state["sma"]
    sma_clean = [(state["times"][i], sma[i]) for i in range(len(sma)) if sma[i] is not None]

    if sma_clean:
        fig.add_trace(go.Scatter(
            x=[t for t, _ in sma_clean],
            y=[v for _, v in sma_clean],
            mode="lines",
            name="SMA",
            line=dict(color="#fcd34d", dash="dash")
        ))

    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=10, b=10)
    )

    st.plotly_chart(fig, use_container_width=True)

# ===========================================================
# TRADES PAGE
# ===========================================================
def render_trades_page(state):
    st.subheader("Trade History")

    if not state["trades"]:
        st.info("No trades executed yet.")
        return

    df = pd.DataFrame(state["trades"])
    df['cumulative_pnl'] = df['pnl'].cumsum()

    st.dataframe(df, hide_index=True, use_container_width=True)

# ===========================================================
# BOT PAGE
# ===========================================================
def render_bot_page(state, config, ASSETS, current_asset, start_bot, stop_bot, reset_state, poll_interval):

    st.subheader("Bot Settings")

    colA, colB = st.columns([2, 1])
    with colA:
        keys = list(ASSETS.keys())
        idx = keys.index(state["symbol"])
        selected = st.selectbox(
            "Asset",
            keys,
            index=idx,
            format_func=lambda x: ASSETS[x].name
        )
    with colB:
        if st.button("Reset State"):
            reset_state()

    if selected != state["symbol"]:
        state["symbol"] = selected
        state["symbol_name"] = ASSETS[selected].name

    st.divider()

    st.slider("SMA Period", 5, 50, config["SMA_PERIOD"], key="SMA_PERIOD")
    st.slider("Invest %", 10, 100, int(config["INVEST_PCT"] * 100), key="INVEST_PCT")

    st.divider()

    status = "Running" if state["active"] else "Inactive"
    st.write(f"**Bot Status:** {status}")

    if not state["active"]:
        if st.button("Start Bot"):
            start_bot()
    else:
        if st.button("Stop Bot"):
            stop_bot()

# ===========================================================
# MAIN APP SHELL
# ===========================================================
def render_app_shell(page, state, config, ASSETS, current_asset, connection_label, equity, profit, start_bot, stop_bot, reset_state, poll_interval):

    render_header(state["symbol_name"], connection_label)

    if page == "home":
        render_home_page(state, config, ASSETS, current_asset, equity, profit)
    elif page == "chart":
        render_chart_page(state, config, current_asset)
    elif page == "trades":
        render_trades_page(state)
    elif page == "bot":
        render_bot_page(state, config, ASSETS, current_asset, start_bot, stop_bot, reset_state, poll_interval)

    render_bottom_nav(page)
