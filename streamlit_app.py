import streamlit as st
import pandas as pd
import numpy as np


st.set_page_config(
    page_title="Ghanaian Rice Germplasm Intelligence Hub",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Styling
st.markdown("""
    <style>

    /* Full-width dark blue navigation wrapper */
    .nav-wrapper {
        background: #0A2B7A;
        display: flex;
        justify-content: space-between;
        padding: 0 1.25rem;
        border-bottom: 2px solid #caa052;
        margin-bottom: 30px;
        height: 48px;
    }

    .nav-brand {
        color: #ffffff;
        font-family: serif, 'Times New Roman', Times;
        font-size: 1.35rem;
        font-weight: 400;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .nav-items {
        display: flex;
        align-items: center;
        gap: 12px;
    }


    .nav-link {
        color: #d1d5db;
        font-family: 'Times New Roman';
        font-size: 1.05rem;
        text-decoration: none;
        padding: 6px 18px;
        border-radius: 2px;
        transition: background-color 0.2s ease, color 0.2s ease;
        cursor: pointer;
    }

    .nav-link:hover {
        color: #ffffff;
        background-color: rgba(255, 255, 255, 0.08);
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
        color: var(--text-primary);
        font-family: serif, 'Times New Roman';
        margin-bottom: 4px;
        line-height: 1.2;
    }
    .feature-desc {
        font-size: 0.95rem;
        color: var(--text-secondary);
        font-family: serif, 'Times New Roman';
        line-height: 1.5;
        max-width: 280px;
    }
    
    </style>
""", unsafe_allow_html=True)


st.markdown(f"""
    <div class="nav-wrapper">
        <div class="nav-brand">🌾 GhanaRice ML</div>
        <div class="nav-items">
            <div class="nav-link" href="?page=Home" target="_self">Home</div>
            <div class="nav-link" href="?page=Predict" target="_self">Predict</div>
            <div class="nav-link" href="?page=Batch" target="_self">Batch</div>
            <div class="nav-link" href="?page=Dashboard" target="_self">Dashboard</div>
        </div>
    </div>
""", unsafe_allow_html=True)


st.markdown("<div style='padding: 0px 40px;'>", unsafe_allow_html=True)


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


st.markdown("</div>", unsafe_allow_html=True)
