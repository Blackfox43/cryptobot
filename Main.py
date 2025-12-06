# main.py
import streamlit as st
import time
from bot_core import (
    init_session_state,
    start_multi,
    stop_multi,
    process_price_updates,
    get_multi_state,
    get_asset_config_and_current_asset,
    persist_all,
    POLL_INTERVAL,
)
from ui import set_mobile_theme, get_active_page, render_main_layout

st.set_page_config(page_title="Algo Trader — Pro", layout="wide")

# Initialize
init_session_state(st.session_state)
set_mobile_theme()

# Allow page selection via session state or query param 'asset'
page = get_active_page()
qp = st.experimental_get_query_params()
if page == "dashboard" and qp.get("asset"):
    page = qp.get("asset")[0]

# Process incoming queue
process_price_updates(st.session_state)

# Compute aggregated metrics
multi = get_multi_state(st.session_state)
total_equity = 0.0
for k, s in multi.items():
    total_equity += s.get("balance", 0.0) + s.get("shares", 0.0) * s.get("current_price", 0.0)
total_profit = total_equity - (len(multi) * 10000.0)

ASSETS, current_asset = get_asset_config_and_current_asset(st.session_state)

# Wrappers to pass session_state implicitly
def start_multi_wrapper(st_state_obj, keys):
    # start only valid keys
    valid_keys = [k for k in keys if k in ASSETS]
    start_multi(st.session_state, valid_keys)

def stop_multi_wrapper(st_state_obj=None):
    stop_multi(st.session_state)

def persist_and_reset_wrapper(st_state_obj=None):
    persist_all(st.session_state)

# Render UI
render_main_layout(
    page=page,
    st_state=st.session_state,
    ASSETS=ASSETS,
    current_asset=current_asset,
    connection_label="multi",
    equity=total_equity,
    profit=total_profit,
    start_multi=lambda st_state_obj, keys=st.session_state.strategy.get("multi_assets", []): start_multi_wrapper(st_state_obj, keys),
    stop_multi=lambda st_state_obj=None: stop_multi_wrapper(st_state_obj),
    persist_and_reset=lambda st_state_obj=None: persist_and_reset_wrapper(st_state_obj),
)

# Safe auto-refresh: update a query param to trigger Streamlit rerun without calling query_params()
st.experimental_set_query_params(_ts=int(time.time()))
