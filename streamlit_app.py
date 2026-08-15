import streamlit as st
import pandas as pd
import numpy as np

# 1. Page Configuration
st.set_page_config(
    page_title="Ghanaian Rice Germplasm Intelligence Hub",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Custom Styling
st.markdown("""
    <style>
    /* Reset Streamlit outer margins */
    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 2rem !important;
        padding-left: 0rem !important;
        padding-right: 0rem !important;
        max-width: 100% !important;
    }
    header[data-testid="stHeader"] { display: none !important; }

    /* Target the navigation bar container */
    div[data-testid="stHorizontalBlock"]:has(div.nav-brand-target) {
        background-color: #0b2f6b !important;
        border-bottom: 2px solid #caa052 !important;
        padding: 8px 40px !important;
        align-items: center !important;
        margin-bottom: 30px !important;
    }

    /* Brand Name Text Styling */
    .nav-brand-text {
        color: #ffffff;
        font-family: serif, 'Times New Roman', Times;
        font-size: 1.35rem;
        font-weight: 400;
        line-height: 2.2rem;
    }

    /* Remove default Streamlit padding/borders for Nav Buttons */
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

    /* Highlight standard primary button (Active Tab) */
    div[data-testid="stHorizontalBlock"]:has(div.nav-brand-target) button[kind="primary"] {
        color: #ffffff !important;
        background-color: #274780 !important;
        font-weight: 500 !important;
    }
    </style>
""", unsafe_allow_html=True)


# Helper function to switch pages smoothly
def set_page(page_name):
    st.session_state['page'] = page_name

# Column layout for Header + 4 Nav Buttons
nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns([5, 1, 1, 1, 1])

with nav_col1:
    # Anchor class used by CSS selector above
    st.markdown("<div class='nav-brand-target nav-brand-text'>🌾 GhanaRice ML</div>", unsafe_allow_html=True)

with nav_col2:
    st.button(
        "Home", 
        key="btn_home", 
        use_container_width=True, 
        type="primary" if st.session_state['page'] == 'Home' else "secondary",
        on_click=set_page,
        args=("Home",)
    )

with nav_col3:
    st.button(
        "Predict", 
        key="btn_predict", 
        use_container_width=True, 
        type="primary" if st.session_state['page'] == 'Predict' else "secondary",
        on_click=set_page,
        args=("Predict",)
    )

with nav_col4:
    st.button(
        "Batch", 
        key="btn_batch", 
        use_container_width=True, 
        type="primary" if st.session_state['page'] == 'Batch' else "secondary",
        on_click=set_page,
        args=("Batch",)
    )

with nav_col5:
    st.button(
        "Dashboard", 
        key="btn_dashboard", 
        use_container_width=True, 
        type="primary" if st.session_state['page'] == 'Dashboard' else "secondary",
        on_click=set_page,
        args=("Dashboard",)
    )
# 4. Main App Container
st.markdown("<div style='padding: 0px 40px;'>", unsafe_allow_html=True)

# --- PAGE ROUTING ---

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

elif st.session_state['page'] == 'Predict':
    col_left, col_right = st.columns([1, 1], gap="large")

    # Inputs Column
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
        run_pred = st.button("Run prediction", use_container_width=True)

    # Prediction Display Cards Column
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
                <div class="progress-bg">
                    <div class="progress-fill"></div>
                </div>
                <div class="progress-labels">
                    <span>Control 97%</span>
                    <span>Stress 3%</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

elif st.session_state['page'] == 'Batch':
    st.title("Batch Prediction")

elif st.session_state['page'] == 'Dashboard':
    st.title("Model Dashboard")

# Close main app container
st.markdown("</div>", unsafe_allow_html=True)
