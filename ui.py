import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import time 

# ===========================================================
# THEME ENGINE (MOBILE FIRST)
# ===========================================================
def set_mobile_theme():
    # Streamlit requires a placeholder for the URL when setting query params.
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = 'home'
        
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
            
            /* Hide the main menu button and footer in deployed environments */
            #MainMenu, footer {
                visibility: hidden;
            }
            
            /* Content area padding adjustment for bottom nav */
            .main > div {
                padding-bottom: 70px; /* Space for the bottom nav */
            }
            
            /* Trade log table styling */
            .dataframe {
                border-radius: 10px;
                overflow: hidden;
            }
        </style>
    """, unsafe_allow_html=True)

def get_active_page():
    """Manages the page navigation based on query parameters or session state."""
    # We use a button-click system (via set_query_param) to navigate
    # Note: st.query_params is not directly used for routing due to Canvas limitations
    return st.session_state.get('current_page', 'home')

def set_active_page(page_name):
    """Callback for navigation buttons."""
    st.session_state['current_page'] = page_name

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
# NAVIGATION BAR
# ===========================================================
def render_bottom_nav(active_page):
    nav_items = [
        ("home", "📊", "Metrics"),
        ("chart", "📈", "Chart"),
        ("trades", "📜", "Trades"),
        ("bot", "⚙️", "Bot"),
    ]
    
    html = '<div class="bottom-nav">'
    
    for page_name, icon, label in nav_items:
        is_active = page_name == active_page
        class_name = "nav-item-active" if is_active else "nav-item"
        
        # Use an empty st.button inside a column to act as a trigger, using the column's markdown for display
        col = st.columns(1)[0]
        with col:
            # The key must be unique for each button
            st.button(
                f'{icon} {label}', 
                key=f'nav_{page_name}', 
                on_click=set_active_page, 
                args=(page_name,),
                use_container_width=True
            )
    
    st.markdown('</div>', unsafe_allow_html=True)

# ===========================================================
# HOME/METRICS PAGE
# ===========================================================
def render_home_page(state, config, ASSETS, current_asset, equity, profit):
    st.subheader(f"{current_asset.name} Price & Equity")
    
    # Calculate unrealized PNL
    unrealized_pnl = 0
    pnl_delta_value = None
    pnl_color_mode = "off"

    if state["shares"] > 0 and state["avg_entry"] > 0:
        unrealized_pnl = state["shares"] * (state["current_price"] - state["avg_entry"])
        pnl_delta_value = unrealized_pnl
        pnl_color_mode = "normal" 
    else:
        pnl_color_mode = "off" 
        pnl_delta_value = None

    # Row 1: Key Metrics
    col1, col2 = st.columns(2)
    col1.metric("Current Price", f"${state['current_price']:.2f}")
    col2.metric("Total Equity", f"${equity:.2f}")

    # Row 2: Balances
    col3, col4 = st.columns(2)
    col3.metric("Cash Balance", f"${state['balance']:.2f}")
    col4.metric(
        "Unrealized PNL", 
        f"${unrealized_pnl:.2f}",
        delta=pnl_delta_value,    
        delta_color=pnl_color_mode
    )
    
    # Row 3: Holding & Profit
    col5, col6 = st.columns(2)
    col5.metric("Shares Held", f"{state['shares']:.4f}", help=f"Avg. Entry: ${state['avg_entry']:.2f}")
    col6.metric("Total Profit", f"${profit:.2f}", delta=profit, delta_color="normal")
    
    st.caption(f"Price from: **{state['provider']}**")
    
# ===========================================================
# CHART PAGE
# ===========================================================
def render_chart_page(state, config, current_asset):
    st.subheader(f"Price Chart ({current_asset.bybit_symbol})")
    
    if len(state["prices"]) > 0:
        fig = go.Figure()

        # Price Trace
        fig.add_trace(go.Scatter(
            x=state["times"], y=state["prices"],
            mode="lines", name="Price", line=dict(color="#10b981") # Tailwind emerald-500
        ))

        # SMA Trace
        sma_period = config['SMA_PERIOD']
        sma_clean = [(state["times"][i], state["sma"][i])
                     for i in range(len(state["sma"]))
                     if state["sma"][i] is not None]

        if sma_clean:
            fig.add_trace(go.Scatter(
                x=[t[0] for t in sma_clean],
                y=[t[1] for t in sma_clean],
                mode="lines",
                name=f"SMA {sma_period}",
                line=dict(color="#fcd34d", dash="dash") # Tailwind amber-300
            ))
            
            # Avg Entry Line
            if state["shares"] > 0 and state["avg_entry"] > 0:
                 fig.add_hline(
                    y=state["avg_entry"], 
                    line_dash="dot", 
                    annotation_text="Avg Entry", 
                    annotation_position="top left",
                    line_color="#3b82f6" # Tailwind blue-500
                )

        fig.update_layout(
            height=400,
            margin=dict(l=15, r=15, t=10, b=10),
            template="plotly_dark",
            paper_bgcolor="#1f2937", 
            plot_bgcolor="#1f2937",
            font=dict(color="#f9fafb"),
            xaxis_title="Time",
            yaxis_title="Price (USD)"
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Waiting for enough price data to render chart...")

# ===========================================================
# TRADES PAGE
# ===========================================================
def render_trades_page(state):
    st.subheader("Trade History")
    
    if state["trades"]:
        df = pd.DataFrame(state["trades"])
        
        # Calculate cumulative PNL
        df['cumulative_pnl'] = df['pnl'].cumsum()
        
        # Format columns for display
        df["price"] = df["price"].apply(lambda x: f"${x:.4f}")
        df["amount"] = df["amount"].apply(lambda x: f"{x:.4f}")
        df["pnl"] = df["pnl"].apply(lambda x: f"${x:.4f}")
        df["cumulative_pnl"] = df["cumulative_pnl"].apply(lambda x: f"${x:.2f}")

        # Style PNL column
        def color_pnl(val):
            """Colors PNL cells green/red."""
            try:
                # Assuming val is a string like '$123.4567'
                val_float = float(val.replace('$', '').strip())
                color = 'green' if val_float >= 0 else 'red'
                return f'color: {color}'
            except:
                return ''

        st.dataframe(
            df.style.applymap(color_pnl, subset=['pnl', 'cumulative_pnl']),
            hide_index=True, 
            use_container_width=True
        )
    else:
        st.info("No trades have been executed yet.")

# ===========================================================
# BOT/SETTINGS PAGE
# ===========================================================
def render_bot_page(state, config, ASSETS, current_asset, start_bot, stop_bot, reset_state, poll_interval): 
    st.subheader("Asset & Strategy Configuration")
    
    # ------------------ Asset Selection ------------------
    # Use st.columns to prevent the selectbox from affecting the app's height
    col_asset, col_reset = st.columns([2, 1])
    
    with col_asset:
        # Get the list of asset keys
        asset_keys = list(ASSETS.keys())
        
        # Find the current index based on state
        try:
            default_index = asset_keys.index(state["symbol"])
        except ValueError:
            default_index = 0

        # Create the selectbox
        new_symbol_key = st.selectbox(
            "Select Asset",
            asset_keys,
            index=default_index,
            format_func=lambda x: ASSETS[x].name,
            key='asset_selector'
        )
        
    with col_reset:
        st.markdown("<br>", unsafe_allow_html=True) # Vertical spacing
        if st.button("Reset State", use_container_width=True):
            reset_state()
            
    # Check if a new asset was selected
    if new_symbol_key != state["symbol"]:
        # Update the core state variables (symbol, symbol_name) directly in session state
        st.session_state.state["symbol"] = new_symbol_key
        st.session_state.state["symbol_name"] = ASSETS[new_symbol_key].name
        # The main loop will handle stopping the old poller and resetting chart data
        
    st.divider()

    # ------------------ Strategy Parameters ------------------
    st.subheader("Strategy Parameters")
    
    # Use a dictionary of keys to manage the state updates
    config_keys = {
        "SMA_PERIOD": ("SMA Period", 5, 50, 5, 20),
        "BUY_DIP": ("Buy Below SMA (%)", 0.01, 5.0, 0.01, 0.5), 
        "TP": ("Take Profit (%)", 0.1, 5.0, 0.1, 1.0),
        "SL": ("Stop Loss (%)", 0.1, 5.0, 0.1, 2.0),
        "INVEST_PCT": ("Invest %", 10, 100, 10, 90)
    }

    # Map the stored config to the display inputs, ensuring they update st.session_state.config
    col_params_1, col_params_2 = st.columns(2)
    
    # SMA Period
    with col_params_1:
        new_sma = st.slider(config_keys["SMA_PERIOD"][0], config_keys["SMA_PERIOD"][1], config_keys["SMA_PERIOD"][2], config["SMA_PERIOD"], config_keys["SMA_PERIOD"][3])
        st.session_state.config["SMA_PERIOD"] = new_sma
        
        new_tp = st.number_input(config_keys["TP"][0], config_keys["TP"][1], config_keys["TP"][2], config["TP"] * 100, config_keys["TP"][3]) / 100
        st.session_state.config["TP"] = new_tp
        
        new_invest_pct = st.slider(config_keys["INVEST_PCT"][0], config_keys["INVEST_PCT"][1], config_keys["INVEST_PCT"][2], int(config["INVEST_PCT"] * 100), config_keys["INVEST_PCT"][3]) / 100
        st.session_state.config["INVEST_PCT"] = new_invest_pct
        
    # Buy Dip & Stop Loss
    with col_params_2:
        new_buy_dip = st.number_input(config_keys["BUY_DIP"][0], config_keys["BUY_DIP"][1], config_keys["BUY_DIP"][2], config["BUY_DIP"] * 100, config_keys["BUY_DIP"][3]) / 100
        st.session_state.config["BUY_DIP"] = new_buy_dip
        
        new_sl = st.number_input(config_keys["SL"][0], config_keys["SL"][1], config_keys["SL"][2], config["SL"] * 100, config_keys["SL"][3]) / 100
        st.session_state.config["SL"] = new_sl
        
        st.markdown("---")
        # Use the passed poll_interval variable
        st.caption(f"Price Polling Interval: **{poll_interval} seconds**") 

    st.divider()

    # ------------------ Bot Control ------------------
    st.subheader("Bot Engine")
    
    col_status, col_button = st.columns([1, 2])
    
    with col_status:
        status_text = "Running" if state["active"] else "Inactive"
        st.markdown(f"**Status:** <span style='color: {'#10b981' if state['active'] else '#f87171'}'>{status_text}</span>", unsafe_allow_html=True)
        st.caption(f"Current Asset: {current_asset.bybit_symbol}")
    
    with col_button:
        if not state["active"]:
            if st.button("▶️ Start Algo Bot", use_container_width=True, key='start_bot_btn'):
                start_bot()
        else:
            if st.button("⏹ Stop Algo Bot", use_container_width=True, key='stop_bot_btn'):
                stop_bot()

# ===========================================================
# MAIN APP SHELL
# ===========================================================
# CRUCIAL FIX: Ensure this function signature matches the call in main.py
def render_app_shell(page, state, config, ASSETS, current_asset, connection_label, equity, profit, start_bot, stop_bot, reset_state, poll_interval): 
    # Render main content
    render_header(state["symbol_name"], connection_label)

    # PAGES
    # We use a placeholder container to ensure content doesn't jump
    content_container = st.container() 
    with content_container:
        if page == "home":
            render_home_page(state, config, ASSETS, current_asset, equity, profit)
        elif page == "chart":
            render_chart_page(state, config, current_asset)
        elif page == "trades":
            render_trades_page(state)
        elif page == "bot":
            # Pass all necessary arguments to the bot page renderer
            render_bot_page(state, config, ASSETS, current_asset, start_bot, stop_bot, reset_state, poll_interval) 
            
    # Render persistent bottom navigation
    render_bottom_nav(page)
