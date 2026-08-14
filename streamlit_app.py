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

    * {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
    }
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

st.markdown("<div class='main-header'>🌾 Ghanaian Rice Germplasm Intelligence Hub</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Enter early-stage growth measurements — get instant predictions on variety, traits, and stress classification</div>", unsafe_allow_html=True)
