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
from ui import set_mobile_theme, get_active_page, set_active_page, render_main_layout

st.set_page_config(page_title="Algo Trader — Pro", layout="wide")

# init
init_session_state(st.session_state)
st_state = st
set_mobile_theme()

# choose page: either 'dashboard' or asset key or 'settings'
page = get_active_page()
# allow hash-based quick navigation for asset detail
if page == "dashboard" and st.experimental_get_query_params().get("asset"):
    page = st.experimental_get_query_params().get("asset")[0]

# process incoming price updates
process_price_updates(st.session_state)

# metrics
multi = get_multi_state(st.session_state)
# compute aggregated equity & profit
total_equity = 0.0
for k, s in multi.items():
    total_equity += s.get("balance",0.0) + s.get("shares",0.0) * s.get("current_price",0.0)
total_profit = total_equity - (len(multi) * 10000.0)  # rough baseline

ASSETS, current_asset = get_asset_config_and_current_asset(st.session_state)

# wrappers
def start_multi_wrapper(st_state_obj, keys):
    start_multi(st_state_obj, keys)

def stop_multi_wrapper(st_state_obj):
    stop_multi(st_state_obj)

def persist_and_reset(st_state_obj):
    # persist balances and reset prices only
    persist_all(st_state_obj)
    # reset price history in session only
    for k in st_state_obj.multi_state:
        st_state_obj.multi_state[k]["prices"] = []
        st_state_obj.multi_state[k]["times"] = []
        st_state_obj.multi_state[k]["sma"] = []

# render UI
render_main_layout(
    page=page,
    st_state=st.session_state,
    ASSETS=ASSETS,
    current_asset=current_asset,
    connection_label="multi",
    equity=total_equity,
    profit=total_profit,
    start_multi=lambda s, keys=st.session_state.strategy.get("multi_assets", []): start_multi(st.session_state, keys),
    stop_multi=lambda s=None: stop_multi(st.session_state),
    persist_and_reset=lambda s=st.session_state: persist_all(st.session_state),
)

# SAFE AUTO-REFRESH (Python 3.11 compatible)
st.query_params["_ts"] = int(time.time())
