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
        height: 69px;
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


    /* Base typography for input form */
        .input-header {
            font-family: serif, 'Times New Roman';
            font-size: 1.15rem;
            font-weight: 500;
            color: #111111;
            margin-bottom: 12px;
        }
        
        /* Prediction Card 1: Variety Card (Blue Bracket Accent) */
        .variety-card {
            border-left: 3px solid #1a4380;
            border-radius: 0 12px 12px 0;
            padding: 8px 0px 8px 18px;
            margin-bottom: 24px;
            position: relative;
        }
        .card-label {
            font-family: 'Helvetica Neue', sans-serif;
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

        /* Prediction Card 2: Trait Predictions (Gold Bracket Accent) */
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

        /* Prediction Card 3: Treatment Group (Green Bracket Accent) */
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
        /* Custom Progress Bar */
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

        /* Submit Button Custom Styling */
        div.stButton > button {
            background-color: #0b2f6b !important;
            color: white !important;
            border: none !important;
            border-radius: 2px !important;
            font-family: serif, 'Times New Roman' !important;
            font-size: 1.1rem !important;
            height: 2.8rem !important;
            margin-top: 15px;
        }
        div.stButton > button:hover {
            background-color: #071f48 !important;
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

query_params = st.query_params
if "page" in query_params:
    st.session_state['page'] = query_params["page"]
elif 'page' not in st.session_state:
    st.session_state['page'] = 'Home'

# Helper function to switch pages
def set_page(page_name):
    st.session_state['page'] = page_name

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

elif st.session_state['page'] == 'Predict':
   col_left, col_right = st.columns([1, 1], gap="large")
    
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

    with col_right:
        st.markdown("""
            <div class="variety-card">
                <span class="confidence-tag">78% confidence</span>
                <div class="card-label">VARIETY PREDICTION</div>
                <div class="variety-title">ADDO1</div>
                <div class="variety-subtitle">Also likely: AGRA (12%) · GH10887 (6%)</div>
            </div>
        """, unsafe_allow_html=True)

        # 2. Trait Predictions Card
    st.markdown("""
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
        """, unsafe_allow_html=True)

        # 3. Treatment Group Card
    st.markdown("""
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




