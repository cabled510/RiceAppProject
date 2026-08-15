import streamlit as st
import pandas as pd
import numpy as np


st.set_page_config(
    page_title="GhanaRice ML",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize Session State Page Navigation
if 'page' not in st.session_state:
    st.session_state['page'] = 'Home'

def set_page(page_name):
    st.session_state['page'] = page_name

st.markdown("""
    <style>
    .block-container {
        padding-top: 0rem;
        padding-bottom: 2rem;
        padding-left: 0rem;
        padding-right: 0rem;
        max-width: 100%;
    }

    header[data-testid="stHeader"] { display: none !important; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* Target the navigation bar container */
    div[data-testid="stHorizontalBlock"]:has(div.nav-brand-target) {
        background-color: #0b2f6b;
        border-bottom: 2px solid #caa052;
        padding: 15px 4px;
        margin-bottom: 30px;
    }

    /* Brand Name Text Styling */
    .nav-brand-text {
        color: #ffffff;
        font-family: serif, 'Times New Roman', Times;
        font-size: 1.35rem;
        font-weight: 400;
        line-height: 2.2rem;
        display: flex;
        
    }

    /* Style Streamlit Navigation Buttons */
    div[data-testid="stHorizontalBlock"]:has(div.nav-brand-target) button {
        background-color: transparent !important;
        color: #d1d5db !important;
        border: none !important;
        font-family: serif, 'Times New Roman', Times !important;
        font-size: 1.05rem !important;
        box-shadow: none !important;
        height: 2.4rem !important;
        border-radius: 2px !important;
        margin-top: 0px !important;
    }

    div[data-testid="stHorizontalBlock"]:has(div.nav-brand-target) button:hover {
        color: #ffffff !important;
        background-color: rgba(255, 255, 255, 0.08) !important;
    }

    /* Active Tab Style */
    div[data-testid="stHorizontalBlock"]:has(div.nav-brand-target) button[kind="primary"] {
        color: #ffffff !important;
        background-color: #274780 !important;
        font-weight: 500 !important;
    }

    /* Home Page Styling */
    .main-header {
        font-family: serif, 'Times New Roman';
        font-size: 2.2rem;
        color: #0b2f6b;
        text-align: center;
        margin-bottom: 8px;
    }
    .sub-header {
        font-family: serif, 'Times New Roman';
        font-size: 1.1rem;
        color: #444444;
        text-align: center;
        margin-bottom: 40px;
    }
    .stats-wrapper {
        display: flex;
        justify-content: center;
        gap: 62px;
        margin-bottom: 50px;
    }
    .stat-box { text-align: center; }
    .stat-number {
        font-family: serif, 'Times New Roman';
        font-size: 2.2rem;
        font-weight: 600;
        color: #0b2f6b;
    }
    .stat-label {
        font-family: serif, 'Times New Roman';
        font-size: 0.95rem;
        color: #555555;
    }
    .feature-card {
    padding: 0px 10px; 
    margin-left:20px;
    }
    
    .feature-icon-box {
        width: 44px;
        height: 44px;
        background-color: #eef4fc;
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
    }

    /* Predict Page Cards */
    .input-header {
        font-family: serif, 'Times New Roman';
        font-size: 1.15rem;
        font-weight: 500;
        color: #111111;
        margin-bottom: 12px;

    }
    .variety-card {
        border-left: 3px solid #1a4380;
        border-radius: 0 12px 12px 0;
        padding: 8px 0px 8px 18px;
        margin-bottom: 24px;
    }
    .card-label {
        font-family: sans-serif;
        font-size: 0.75rem;
        letter-spacing: 0.8px;
        color: #222222;
        text-transform: uppercase;
        font-weight: 500;
    }
    .confidence-tag {
        float: right;
        background-color: #e8f5e9;
        color: #2e7d32;
        font-size: 0.75rem;
        padding: 2px 8px;
        border-radius: 10px;
        font-family: sans-serif;
    }
    .variety-title {
        font-family: serif, 'Times New Roman';
        font-size: 1.6rem;
        color: #0d2c6c;
        margin: 6px 0 2px 0;
    }
    .variety-subtitle {
        font-family: serif, 'Times New Roman';
        font-size: 0.88rem;
        color: #333333;
    }
    .trait-card {
        border-left: 3px solid #caa052;
        border-radius: 0 12px 12px 0;
        padding: 8px 0px 8px 18px;
        margin-bottom: 24px;
    }
    .metric-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        row-gap: 16px;
        column-gap: 20px;
        margin-top: 14px;
    }
    .metric-val {
        font-family: serif, 'Times New Roman';
        font-size: 1.25rem;
        color: #111111;
    }
    .metric-lbl {
        font-family: serif, 'Times New Roman';
        font-size: 0.82rem;
        color: #444444;
    }
    .treatment-card {
        border-left: 3px solid #2e7d32;
        border-radius: 0 12px 12px 0;
        padding: 8px 0px 8px 18px;
    }
    .treatment-title {
        font-family: serif, 'Times New Roman';
        font-size: 1.25rem;
        color: #2e7d32;
        margin: 8px 0 12px 0;
    }
    .progress-bg {
        background-color: #e0e0e0;
        border-radius: 4px;
        height: 8px;
        width: 100%;
        margin-bottom: 6px;
    }
    .progress-fill {
        background-color: #2e7d32;
        height: 8px;
        border-radius: 4px;
        width: 97%;
    }
    .progress-labels {
        display: flex;
        justify-content: space-between;
        font-family: serif, 'Times New Roman';
        font-size: 0.82rem;
        color: #333333;
    }

    /* Submit Button */
    div.stButton > button:not(div[data-testid="stHorizontalBlock"] button) {
        background-color: #0A2B7A;
        color: white !important;
        border: none !important;
        border-radius: 2px !important;
        font-family: serif, 'Times New Roman' !important;
        font-size: 1.1rem !important;
        height: 2.8rem !important;
        margin-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)

nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns([4, 1, 1, 1, 1])

with nav_col1:
    st.markdown("<div class='nav-brand-target nav-brand-text'>🌾 GhanaRice ML</div>", unsafe_allow_html=True)

with nav_col2:
    st.button("Home", key="btn_home", use_container_width=True, 
              type="primary" if st.session_state['page'] == 'Home' else "secondary", 
              on_click=set_page, args=("Home",))

with nav_col3:
    st.button("Predict", key="btn_predict", use_container_width=True, 
              type="primary" if st.session_state['page'] == 'Predict' else "secondary", 
              on_click=set_page, args=("Predict",))

with nav_col4:
    st.button("Batch", key="btn_batch", use_container_width=True, 
              type="primary" if st.session_state['page'] == 'Batch' else "secondary", 
              on_click=set_page, args=("Batch",))

with nav_col5:
    st.button("Dashboard", key="btn_dashboard", use_container_width=True, 
              type="primary" if st.session_state['page'] == 'Dashboard' else "secondary", 
              on_click=set_page, args=("Dashboard",))


# =============================================================================
# 4. PAGE ROUTING & CONTENT
# =============================================================================
st.markdown("<div style='padding: 0px 60px;'>", unsafe_allow_html=True)

# --- HOME PAGE ---
if st.session_state['page'] == 'Home':
    st.markdown("<div class='main-header'>Ghanaian Rice Germplasm ML Predictor</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Enter early-stage growth measurements — get instant predictions on variety, traits, and stress classification</div>", unsafe_allow_html=True)

    st.markdown("""
        <div class="stats-wrapper">
            <div class="stat-box"><div class="stat-number">18</div><div class="stat-label">accessions</div></div>
            <div class="stat-box"><div class="stat-number">720</div><div class="stat-label">observations</div></div>
            <div class="stat-box"><div class="stat-number">6</div><div class="stat-label">algorithms</div></div>
            <div class="stat-box"><div class="stat-number">98.1%</div><div class="stat-label">best F1 (task C)</div></div>
        </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

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
        

# --- PREDICT PAGE ---
elif st.session_state['page'] == 'Predict':
    col_spacer_left, col_left, col_right, col_spacer_right = st.columns([0.05, 1, 1.1, 0.05], gap="medium")

    with col_left:
        st.markdown("<div class='input-header'>Growth measurements</div>", unsafe_allow_html=True)
        accession = st.selectbox("Accession", ["AGRA", "ADDO1", "GH10887", "Togbei"], index=0)
        treatment = st.selectbox("Treatment", ["Control", "Stress"], index=0)
        base_height = st.text_input("Base height at 3rd leaf (cm)", value="12.4")
        height_feb4 = st.text_input("Height — Feb 4 (cm)", value="18.2")
        height_feb14 = st.text_input("Height — Feb 14 (cm)", value="27.6")

        st.markdown("<br><div class='input-header'>Survival status</div>", unsafe_allow_html=True)
        alive_feb4 = st.toggle("Alive at Feb 4", value=True)
        alive_feb14 = st.toggle("Alive at Feb 14", value=True)
        alive_feb21 = st.toggle("Alive at Feb 21", value=False)

        st.button("Run Prediction", use_container_width=True)

    with col_right:
        st.markdown("""
            <div class="variety-card">
                <span class="confidence-tag">78% confidence</span>
                <div class="card-label">VARIETY PREDICTION</div>
                <div class="variety-title">ADDO1</div>
                <div class="variety-subtitle">Also likely: AGRA (12%) · GH10887 (6%)</div>
            </div>

            <div class="trait-card">
                <span class="confidence-tag" style="background-color: #f5f5f5; color: #555;">R² 0.82</span>
                <div class="card-label">TRAIT PREDICTIONS</div>
                <div class="metric-grid">
                    <div>
                        <div class="metric-val">34.2 cm</div>
                        <div class="metric-lbl">Final height</div>
                    </div>
                    <div>
                        <div class="metric-val">7</div>
                        <div class="metric-lbl">Leaf count</div>
                    </div>
                    <div>
                        <div class="metric-val">14</div>
                        <div class="metric-lbl">Root count</div>
                    </div>
                    <div>
                        <div class="metric-val">22.1 cm</div>
                        <div class="metric-lbl">Root length</div>
                    </div>
                </div>
            </div>

            <div class="treatment-card">
                <span class="confidence-tag">97% confidence</span>
                <div class="card-label">TREATMENT GROUP</div>
                <div class="treatment-title">Control</div>
                <div class="progress-bg"><div class="progress-fill"></div></div>
                <div class="progress-labels">
                    <span>Control 97%</span>
                    <span>Stress 3%</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

# --- BATCH PAGE ---
elif st.session_state['page'] == 'Batch':
    st.title("Batch Upload")
    st.write("Upload a CSV file containing early growth parameters to generate predictions in bulk.")

# --- DASHBOARD PAGE ---
elif st.session_state['page'] == 'Dashboard':
    st.title("Model Dashboard")
    st.write("Explore machine learning model metrics, confusion matrices, and feature importance rankings.")

st.markdown("</div>", unsafe_allow_html=True)
