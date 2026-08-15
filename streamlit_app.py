import streamlit as st
import pandas as pd
import numpy as np


st.set_page_config(
    page_title="Ghanaian Rice Germplasm Intelligence Hub",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>

    
    .main-header {
        font-size: 1.375rem;
        color: #0A2B7A;
        font-weight: 500;
        text-align: center;
        margin-bottom: 0.375rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: var(--text-secondary);
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #2d6a4f;
    }
    </style>
""", unsafe_allow_html=True)

if 'page' not in st.session_state:
    st.session_state['page'] = 'Home'

# Helper function to switch pages
def set_page(page_name):
    st.session_state['page'] = page_name

nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns([4, 1, 1, 1, 1])

with nav_col1:
    st.markdown("<div style= font-size:1.2rem; padding:16px; font-weight:600;'>🌾 GhanaRice ML</div>", unsafe_allow_html=True)

with nav_col2:
    if st.button("Home", key="btn_home", use_container_width=True, type="primary" if st.session_state['page'] == 'Home' else "secondary"):
        set_page("Home")
        st.rerun()

with nav_col3:
    if st.button("Predict", key="btn_predict", use_container_width=True, type="primary" if st.session_state['page'] == 'Predict' else "secondary"):
        set_page("Predict")
        st.rerun()

with nav_col4:
    if st.button("Batch", key="btn_batch", use_container_width=True, type="primary" if st.session_state['page'] == 'Batch' else "secondary"):
        set_page("Batch")
        st.rerun()

with nav_col5:
    if st.button("Dashboard", key="btn_dashboard", use_container_width=True, type="primary" if st.session_state['page'] == 'Dashboard' else "secondary"):
        set_page("Dashboard")
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

if st.session_state['page'] == 'Home':
  st.markdown("<div class='main-header'> Ghanaian Rice Germplasm ML Predictor</div>", unsafe_allow_html=True)
  st.markdown("<div class='sub-header'>Enter early-stage growth measurements — get instant predictions on variety, traits, and stress classification</div>", unsafe_allow_html=True)
