import streamlit as st
import pandas as pd
import numpy as np

st.title('🎈 Ghanaian Rice Germplasm ML Predictor')

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
        font-size: 2.2rem;
        color: #1b4332;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #40916c;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #2d6a4f;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'>🌾 Ghanaian Rice Germplasm Intelligence Hub</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Precision Classification, Trait Prediction & Stress Detection System</div>", unsafe_allow_html=True)
