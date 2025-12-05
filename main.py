import streamlit as st
import time

# FIX: Import the main_app_loop function from the new bot_core.py file
from bot_core import main_app_loop

# ------------------------------------------------------
# STREAMLIT PAGE CONFIG (Required here for the entry file)
# ------------------------------------------------------
st.set_page_config(page_title="Bybit Algo Trader", layout="wide")

# ------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------
if __name__ == "__main__":
    main_app_loop()
