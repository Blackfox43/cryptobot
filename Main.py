# Main.py
import time
import streamlit as st
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
    POLL_INTERVAL
)

# ------------------------------------------------------
# APP CONFIG
# ------------------------------------------------------
st.set_page_config(page_title="Algo Trader", layout="wide", initial_sidebar_state="collapsed")

# ------------------------------------------------------
# INITIALIZE SESSION STATE (explicit)
# ------------------------------------------------------
# Pass streamlit's session_state to the initializer
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
# We process any queued price updates (safe to call every run)
process_price_updates(st.session_state)

# ------------------------------------------------------
# CONNECTION STATUS & METRICS (read-only helpers)
# ------------------------------------------------------
connection_label = get_connection_status(st.session_state)
equity, profit = get_equity_and_profit(st.session_state)
ASSETS, current_asset = get_asset_config_and_current_asset(st.session_state)

# ------------------------------------------------------
# WRAPPED BOT CONTROLS FOR UI (these call into bot_core with session_state)
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
# This call expects the ui.render_app_shell signature used in your repo.
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
# SAFE AUTO-REFRESH (no query params, compatible across Streamlit versions)
# We store a timestamp in session_state and only call rerun after poll interval
# ------------------------------------------------------
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

# Convert POLL_INTERVAL to a float/sane fallback
try:
    _poll = float(POLL_INTERVAL)
except Exception:
    _poll = 5.0

now = time.time()
if now - st.session_state.last_refresh >= _poll:
    # update timestamp first to avoid immediate rerun loops
    st.session_state.last_refresh = now
    # trigger a safe rerun to pick up new data and UI changes
    st.experimental_rerun()
