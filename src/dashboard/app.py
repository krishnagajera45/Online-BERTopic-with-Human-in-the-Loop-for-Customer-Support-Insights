"""
TwCS Topic Modeling System — Entry Point.

This is the landing page that redirects to Project Overview on first visit.
"""
import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TwCS Topic Modeling",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Redirect to Project Overview ─────────────────────────────────────────────
st.switch_page("pages/0_Project_Overview.py")

