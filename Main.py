import streamlit as st
import time
from ui import render_app_shell, set_mobile_theme, get_active_page
from bot_core import (
    init_session_state,
    start_bot,
    stop_bot,
    process_price_updates,
    get_equity_and_profit,
    get_connection_status,
    reset_state,
    get_asset_config_and_current_asset,
    POLL_INTERVAL # Correctly imported
)

# ------------------------------------------------------
# APP CONFIG
# ------------------------------------------------------
st.set_page_config(page_title="Algo Trader", layout="wide")

# ------------------------------------------------------
# INITIALIZE SESSION STATE
# ------------------------------------------------------
# Pass session state to the initializer
init_session_state(st.session_state)

state = st.session_state.state
config = st.session_state.config

# ------------------------------------------------------
# APPLY THEME + MOBILE ADAPTATION
# ------------------------------------------------------
set_mobile_theme()

# ------------------------------------------------------
# ROUTER — bottom navigation decides which page is active
# ------------------------------------------------------
page = get_active_page()

# ------------------------------------------------------
# BACKGROUND ENGINE — price queue processor
# ------------------------------------------------------
# Pass session state to the processor
process_price_updates(st.session_state)

# ------------------------------------------------------
# CONNECTION STATUS
# ------------------------------------------------------
# Pass session state to the status checker
connection_label = get_connection_status(st.session_state)

# ------------------------------------------------------
# EQUITY & PROFIT
# ------------------------------------------------------
# Pass session state to the calculator
equity, profit = get_equity_and_profit(st.session_state)

# ------------------------------------------------------
# ASSET CONFIG / CURRENT ASSET
# ------------------------------------------------------
ASSETS, current_asset = get_asset_config_and_current_asset(st.session_state)

# ------------------------------------------------------
# WRAPPED BOT CONTROLS FOR UI
# ------------------------------------------------------
def start_bot_wrapper():
    start_bot(st.session_state)

def stop_bot_wrapper():
    stop_bot(st.session_state)
    
def reset_state_wrapper():
    reset_state(st.session_state)
    
# ------------------------------------------------------
# UI RENDERING
# ------------------------------------------------------
# Ensure all 12 arguments are passed to render_app_shell
render_app_shell(
    page=page,
    state=state,
    config=config,
    ASSETS=ASSETS,
    current_asset=current_asset,
    connection_label=connection_label,
    equity=equity,
    profit=profit,
    start_bot=start_bot_wrapper,
    stop_bot=stop_bot_wrapper,
    reset_state=reset_state_wrapper,
    poll_interval=POLL_INTERVAL 
)

# ------------------------------------------------------
# AUTO-REFRESH
# ------------------------------------------------------
st.rerun()
