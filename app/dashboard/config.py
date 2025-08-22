from datetime import datetime
import streamlit as st
import os

def set_page():
    st.set_page_config(page_title="COVID-19 Dashboard", layout="wide")

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

DATA_MIN = "2020-01-01"
DATA_MAX = "2023-12-31"
DATA_MIN_D = datetime.strptime(DATA_MIN, "%Y-%m-%d").date()
DATA_MAX_D = datetime.strptime(DATA_MAX, "%Y-%m-%d").date()
