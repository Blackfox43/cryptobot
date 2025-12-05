import streamlit as st
from ui import render_app_shell, set_mobile_theme, get_active_page
from bot_core import (
    init_session_state,
    start_bot,
    stop_bot,
    get_state,
)

# ------------------------------------------------------
# APP CONFIG
# ------------------------------------------------------
st.set_page_config(page_title="Algo Trader", layout="wide")

# ------------------------------------------------------
# INITIALIZE SESSION STATE
# ------------------------------------------------------
init_session_state()

state = st.session_state.state

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
process_price_updates()

# ------------------------------------------------------
# CONNECTION STATUS
# ------------------------------------------------------
connection_label = get_connection_status()

# ------------------------------------------------------
# EQUITY & PROFIT
# ------------------------------------------------------
equity, profit = get_equity_and_profit()

# ------------------------------------------------------
# UI RENDERING
# ------------------------------------------------------
render_app_shell(
    page=page,
    state=state,
    connection_label=connection_label,
    equity=equity,
    profit=profit,
    start_bot=start_bot,
    stop_bot=stop_bot
)

# ------------------------------------------------------
# AUTO-REFRESH
# ------------------------------------------------------
st.rerun()
