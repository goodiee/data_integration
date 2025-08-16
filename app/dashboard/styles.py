# dashboard/styles.py
import streamlit as st

CSS = """
<style>
.topbar {background:#3f87c0; color:white; padding:12px 16px; border-radius:10px;}
.badge {display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; margin-left:6px;}
.badge-success {background:#198754; color:white;}
.badge-secondary {background:#6c757d; color:white;}
</style>
"""

def inject():
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('<div class="topbar"><span>🧭</span> <strong>COVID-19 Interactive Dashboard</strong></div>',
                unsafe_allow_html=True)
