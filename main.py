import streamlit as st
import time
from ui import render_app_shell, set_mobile_theme, get_active_page, set_active_page
from bot_core import (
    init_session_state,
    start_bot,
    stop_bot,
    process_price_updates,
    get_equity_and_profit,
    get_connection_status,
    reset_state,
    get_asset_config_and_current_asset,
    POLL_INTERVAL,
)

# APP CONFIG
st.set_page_config(page_title="Algo Trader", layout="wide")

# INIT session state
init_session_state(st.session_state)
state = st.session_state.state
config = st.session_state.config

# THEME
set_mobile_theme()

# ROUTING / NAV
page = get_active_page()

# PROCESS QUEUE (run once per rerun)
process_price_updates(st.session_state)

# CONNECTION + METRICS
connection_label = get_connection_status(st.session_state)
equity, profit = get_equity_and_profit(st.session_state)
ASSETS, current_asset = get_asset_config_and_current_asset(st.session_state)

# WRAPPERS
def start_bot_wrapper():
    start_bot(st.session_state)

def stop_bot_wrapper():
    stop_bot(st.session_state)

def reset_state_wrapper():
    reset_state(st.session_state)

# UI render (ui handles navigation via st.session_state)
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
    poll_interval=POLL_INTERVAL,
)

# SAFE AUTO-REFRESH (native)
# Changing a query param triggers a rerun without external libs.
st.query_params(ts=int(time.time()))
