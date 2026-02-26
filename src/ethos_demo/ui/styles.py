"""Global CSS styles for the ETHOS Demo application."""

import streamlit as st

GLOBAL_CSS = (
    "[data-testid='stMainBlockContainer']{max-width:60rem!important;"
    "margin:0 auto!important}"
    "@keyframes _spin{to{transform:rotate(360deg)}}"
    "@keyframes _pulse{0%,100%{opacity:1}50%{opacity:.3}}"
    ".status-row{display:flex;align-items:center;gap:8px;height:28px}"
    ".status-dot{width:10px;height:10px;border-radius:50%;flex-shrink:0}"
    ".status-dot.ok{background:#4caf50}"
    ".status-dot.err{background:#f44336}"
    ".status-text{font-size:0.85em}"
    ".st-key-sidebar_bottom{position:fixed;bottom:1rem;width:inherit}"
    ".st-key-sidebar_bottom button"
    "{padding:0!important;min-height:0!important}"
    ".st-key-ai_working_bar span[data-testid='stIconMaterial']"
    "{animation:_spin .7s linear infinite}"
    ".st-key-sel_ds label p,"
    ".st-key-sel_sc label p,"
    ".st-key-sel_pt label p,"
    ".st-key-sel_outcome label p"
    "{font-size:0.95rem}"
    ".outcome-card{border:1px solid rgba(255,255,255,0.1);"
    "border-radius:12px;padding:1.2em 0.6em;"
    "text-align:center}"
    "div[class*='st-key-card_']{position:relative}"
    ".st-key-refresh_btn{"
    "position:absolute!important;top:10px;right:10px;"
    "z-index:10;width:auto!important}"
    ".st-key-refresh_btn button{"
    "padding:4px 6px!important;min-height:0!important;"
    "background:transparent!important;border:none!important;"
    "box-shadow:none!important;opacity:0.5}"
    ".st-key-refresh_btn button span[data-testid='stIconMaterial']{"
    "font-size:1.4em}"
    ".st-key-refresh_btn button:hover{opacity:1}"
    ".st-key-stop_expl_btn button{"
    "padding:2px 8px!important;min-height:0!important;opacity:0.5}"
    ".st-key-stop_expl_btn button:hover{opacity:1}"
)


def inject_styles() -> None:
    """Inject the global CSS into the Streamlit page."""
    st.markdown(f"<style>{GLOBAL_CSS}</style>", unsafe_allow_html=True)
