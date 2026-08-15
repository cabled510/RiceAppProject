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
        margin-bottom: 2.813rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #2d6a4f;
    }

    /* Metric Stat Box */

    .stats-wrapper {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 3.75rem;
        margin-bottom: 40px;
    }
    
    .stat-box {
        text-align: center;
        margin-bottom: 50px;
    }
    .stat-number {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0b3c85;
        line-height: 1.1;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #555555;
    }

    /* Feature Grid Section */
    .feature-card {
        padding: 0px 10px;
    }
    .feature-icon-box {
        width: 44px;
        height: 44px;
        background-color: #eef4fc; /* Light blue tint matching the screenshot */
        border-radius: 2px;
        margin-bottom: 18px;
    }
    .feature-title {
        font-size: 1.15rem;
        font-weight: 500;
        color: #111111;
        font-family: serif, 'Times New Roman';
        margin-bottom: 8px;
        line-height: 1.2;
    }
    .feature-desc {
        font-size: 0.95rem;
        color: #111111;
        font-family: serif, 'Times New Roman';
        line-height: 1.45;
        max-width: 280px;
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

st.markdown("""
        <div class="stats-wrapper">
            <div class="stat-box">
                <div class="stat-number">18</div>
                <div class="stat-label">accessions</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">720</div>
                <div class="stat-label">observations</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">6</div>
                <div class="stat-label">algorithms</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">98.1%</div>
                <div class="stat-label">best F1 (task C)</div>
            </div>
        </div>
    """, unsafe_allow_html=True)


with c1:
        st.markdown("""
            <div class="feature-card">
                <div class="feature-icon-box"></div>
                <div class="feature-title">Single prediction</div>
                <div class="feature-desc">Enter one plant's measurements and get variety, trait, and stress predictions instantly</div>
            </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
            <div class="feature-card">
                <div class="feature-icon-box"></div>
                <div class="feature-title">Batch upload</div>
                <div class="feature-desc">Upload a CSV of multiple plants and download an Excel file of all predictions</div>
            </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
            <div class="feature-card">
                <div class="feature-icon-box"></div>
                <div class="feature-title">Model dashboard</div>
                <div class="feature-desc">View performance metrics and feature importance for all six trained algorithms</div>
            </div>
        """, unsafe_allow_html=True)
