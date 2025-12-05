# ui.py
"""
UI helpers for CryptoBot. This module depends on Streamlit and imports bot_core for state.
It renders the mobile design system (UI Level 2), navigation, FAB and pages.
"""

import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

# CSS + theme (Level 2)
def render_theme():
    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = "dark"

    dark_bg = "#0d0f12"
    light_bg = "#f2f4f7"
    dark_card = "#15181d"
    light_card = "#ffffff"
    theme_bg = dark_bg if st.session_state.theme_mode == "dark" else light_bg
    theme_card = dark_card if st.session_state.theme_mode == "dark" else light_card
    theme_text = "#e8e8e8" if st.session_state.theme_mode == "dark" else "#1a1c20"

    css = f"""
    <style>
    body {{ background-color: {theme_bg} !important; color: {theme_text} !important; }}
    .block-container {{ padding: 0.6rem 0.6rem 6rem 0.6rem !important; }}
    .mobile-card {{ background: {theme_card}; padding: 1rem; border-radius: 14px; margin-bottom: 0.9rem; box-shadow: 0 4px 14px rgba(0,0,0,0.25); }}
    .navbar {{ position: fixed; bottom: 0; left: 0; width: 100%; height: 62px; background: {theme_card}; display:flex; justify-content: space-around; align-items:center; z-index:9999; }}
    .nav-item {{ text-align:center; width:20%; color:#8a8f99; font-size:0.78rem; }}
    .active-nav {{ color: #4da3ff !important; font-weight:700; }}
    .fab {{ position: fixed; bottom: 80px; right: 18px; width:62px; height:62px; background:#4da3ff; border-radius:50%; display:flex; justify-content:center; align-items:center; color:white; font-size:30px; z-index:9999; cursor:pointer; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Navbar + FAB

def render_navbar(active_page):
    html = f"""
    <div class='navbar'>
        <div class='nav-item {'active-nav' if active_page=='home' else ''}' onclick="window.location.href='/?page=home';">🏠<br>Home</div>
        <div class='nav-item {'active-nav' if active_page=='chart' else ''}' onclick="window.location.href='/?page=chart';">📈<br>Chart</div>
        <div class='nav-item {'active-nav' if active_page=='trades' else ''}' onclick="window.location.href='/?page=trades';">📜<br>Trades</div>
        <div class='nav-item {'active-nav' if active_page=='bot' else ''}' onclick="window.location.href='/?page=bot';">🤖<br>Bot</div>
        <div class='nav-item {'active-nav' if active_page=='settings' else ''}' onclick="window.location.href='/?page=settings';">⚙️<br>Settings</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_fab(is_running):
    icon = "⏹️" if is_running else "▶️"
    html = f"""
    <div class='fab' onclick="window.location.href='/?toggle_bot=true';">{icon}</div>
    """
    st.markdown(html, unsafe_allow_html=True)

# Page renderers (lightweight): they accept the core state dict and some prepared plotly fig

def render_home(core_state, symbol_name, current_price):
    st.markdown("<div class='mobile-card'><h3>Dashboard</h3></div>", unsafe_allow_html=True)
    st.metric(label=symbol_name, value=f"${current_price:.4f}")
    balances = core_state.get('balances', {})
    positions = core_state.get('positions', {})
    st.markdown("<div class='mobile-card'><b>Accounts</b></div>", unsafe_allow_html=True)
    for k, bal in balances.items():
        st.write(f"{k} — ${bal:.2f} — Pos: {positions.get(k,0.0):.6f}")

def render_chart(fig):
    st.markdown("<div class='mobile-card'><h3>Chart</h3></div>", unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)

def render_trades(trade_history):
    st.markdown("<div class='mobile-card'><h3>Trades</h3></div>", unsafe_allow_html=True)
    if not trade_history:
        st.write("No trades yet")
        return
    for rec in trade_history[:200]:
        st.markdown(f"<div class='mobile-card'><b>{rec
