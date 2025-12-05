import streamlit as st
import plotly.graph_objects as go


# ============================================================
# THEME ENGINE (MOBILE FIRST)
# ============================================================
def set_mobile_theme():
    st.markdown("""
        <style>
            /* App background */
            .stApp {
                background-color: #0d0f12 !important;
                color: white !important;
            }

            /* Metrics: mobile friendly */
            div[data-testid="metric-container"] {
                background: #15181d;
                border-radius: 16px;
                padding: 12px;
                margin: 6px 0;
            }

            /* Bottom navigation bar */
            .bottom-nav {
                position: fixed;
                bottom: 0;
                left: 0;
                width: 100%;
                background: #111318;
                padding: 10px 0;
                display: flex;
                justify-content: space-around;
                border-top: 1px solid #222;
                z-index: 9999;
            }

            .nav-item {
                text-align: center;
                font-size: 13px;
                color: #aaa;
                cursor: pointer;
            }

            .nav-item-active {
                color: #4fc3f7;
                font-weight: 700;
            }

            /* Header */
            .app-header {
                padding: 8px 0 4px 0;
                text-align: center;
                color: #e0e0e0;
                font-size: 22px;
                font-weight: 600;
            }

            /* Page container spacing fix */
            .block-container {
                padding-top: 1.2rem;
                padding-bottom: 6rem; /* room for bottom nav */
            }
        </style>
    """, unsafe_allow_html=True)


# ============================================================
# PAGE ROUTER
# ============================================================
def get_active_page():
    if "page" not in st.session_state:
        st.session_state.page = "home"
    return st.session_state.page


def switch_page(name):
    st.session_state.page = name


# ============================================================
# UI PARTIALS
# ============================================================

def render_header(symbol_name, connection_label):
    st.markdown(f"""
        <div class='app-header'>
            {symbol_name} Algo Trader
            <div style='font-size:13px; color:#4fc3f7'>{connection_label}</div>
        </div>
    """, unsafe_allow_html=True)


def render_bottom_nav(active="home"):
    nav_items = [
        ("home", "🏠", "Home"),
        ("chart", "📈", "Chart"),
        ("trades", "📑", "Trades"),
        ("bot", "🤖", "Bot"),
        ("settings", "⚙️", "Settings"),
    ]

    html = "<div class='bottom-nav'>"

    for key, icon, label in nav_items:
        klass = "nav-item-active" if active == key else "nav-item"
        html += f"""
            <div class="{klass}" onclick="window.location.href='?page={key}'">
                {icon}<br>{label}
            </div>
        """

    html += "</div>"

    st.markdown(html, unsafe_allow_html=True)


# ============================================================
# HOME PAGE
# ============================================================
def render_home_page(state, equity, profit):
    price = state["current_price"]

    col1, col2 = st.columns(2)
    col1.metric("Price", f"${price:,.4f}")
    col2.metric("Balance", f"${state['balance']:,.2f}")

    col3, col4 = st.columns(2)
    col3.metric("Equity", f"${equity:,.2f}")
    col4.metric("Profit", f"${profit:,.2f}")

    st.divider()
    st.subheader("Latest Trades")

    if state["trades"]:
        import pandas as pd
        df = pd.DataFrame(state["trades"])
        df["price"] = df["price"].apply(lambda x: f"${x:.4f}")
        df["pnl"] = df["pnl"].apply(lambda x: f"${x:.4f}")
        st.dataframe(df, hide_index=True, use_container_width=True)
    else:
        st.info("No trades yet.")


# ============================================================
# CHART PAGE
# ============================================================
def render_chart_page(state):
    if len(state["prices"]) == 0:
        st.warning("Waiting for price data…")
        return

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=state["times"],
        y=state["prices"],
        mode="lines",
        name="Price",
        line=dict(color="lime")
    ))

    sma_clean = [
        (state["times"][i], state["sma"][i])
        for i in range(len(state["sma"]))
        if state["sma"][i] is not None
    ]

    if sma_clean:
        fig.add_trace(go.Scatter(
            x=[t for t, v in sma_clean],
            y=[v for t, v in sma_clean],
            mode="lines",
            name="SMA",
            line=dict(color="gold", dash="dash")
        ))

    fig.update_layout(
        height=450,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white")
    )

    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# TRADES PAGE
# ============================================================
def render_trades_page(state):
    st.subheader("Trade History")

    if len(state["trades"]) == 0:
        st.info("No trades yet.")
        return

    import pandas as pd
    df = pd.DataFrame(state["trades"])
    df["pnl"] = df["pnl"].apply(lambda x: f"${x:.4f}")
    df["price"] = df["price"].apply(lambda x: f"${x:.4f}")

    st.dataframe(df, hide_index=True, use_container_width=True)


# ============================================================
# BOT PAGE
# ============================================================
def render_bot_page(state, start_bot, stop_bot):
    st.subheader("Bot Control")

    if not state["active"]:
        if st.button("▶️ Start Algo Bot", use_container_width=True):
            start_bot()
    else:
        if st.button("⏹ Stop Algo Bot", use_container_width=True):
            stop_bot()

    st.divider()

    st.write("### Status")
    st.write(f"**Current Price:** ${state['current_price']:.4f}")
    st.write(f"**Shares:** {state['shares']:.4f}")
    st.write(f"**Avg Entry:** {state['avg_entry']:.4f}")


# ============================================================
# SETTINGS PAGE
# ============================================================
def render_settings_page():
    st.subheader("Settings")

    st.info("More features coming soon: Notifications, API keys, Security mode, Themes.")


# ============================================================
# MAIN APP SHELL
# ============================================================
def render_app_shell(page, state, connection_label, equity, profit, start_bot, stop_bot):
    render_header(state["symbol_name"], connection_label)

    # PAGES
    if page == "home":
        render_home_page(state, equity, profit)
    elif page == "chart":
        render_chart_page(state)
    elif page == "trades":
        render_trades_page(state)
    elif page == "bot":
        render_bot_page(state, start_bot, stop_bot)
    elif page == "settings":
        render_settings_page()

    # Bottom navigation (always visible)
    render_bottom_nav(active=page)
