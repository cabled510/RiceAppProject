import streamlit as st
import os
import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
import plotly.express as px


@st.cache_resource
def load_ml_models():
    """
    Loads machine learning models from the local 'models/' directory.
    Uses @st.cache_resource so models load into memory only once.
    """
    # Construct relative path to the models directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    variety_path = os.path.join(base_dir, "models", "best_task_a_xgb_model.pkl")
    trait_path = os.path.join(base_dir, "models", "best_task_b_mlp_finalheight_model_and_scaler.pkl")
    treatment_path = os.path.join(base_dir, "models", "best_task_c_xgb_model.pkl")

    try:
        # Load Task A: Variety Classification (XGBoost)
        with open(variety_path, 'rb') as f:
            variety_model = pickle.load(f)

        # Load Task B: Final Height Trait Prediction (MLP + Scaler Bundle)
        with open(trait_path, 'rb') as f:
            trait_data = pickle.load(f)
            # Unpack model and scaler if saved as a tuple or dictionary
            if isinstance(trait_data, tuple):
                trait_model, trait_scaler = trait_data[0], trait_data[1]
            elif isinstance(trait_data, dict):
                trait_model = trait_data.get('model')
                trait_scaler = trait_data.get('scaler')
            else:
                trait_model, trait_scaler = trait_data, None

        # Load Task C: Stress/Treatment Detection (XGBoost)
        with open(treatment_path, 'rb') as f:
            treatment_model = pickle.load(f)

        return variety_model, trait_model, trait_scaler, treatment_model

    except FileNotFoundError as e:
        st.error(f"⚠️ Model file missing: {e}. Please ensure model files are placed in the 'models/' folder.")
        return None, None, None, None
    except Exception as e:
        st.error(f"⚠️ Error loading models: {e}")
        return None, None, None, None

def apply_custom_css():
    # Finds the folder where app.py lives on the GitHub server
    current_dir = Path(__file__).parent
    css_path = current_dir / "styles.css"
    
    try:
        with open(css_path, "r") as f:
            st.html(f"<style>{f.read()}</style>")
    except FileNotFoundError:
        st.error(f"Could not find {css_path} on the server.")

apply_custom_css()

st.set_page_config(
    page_title="GhanaRice ML",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize Session State Page Navigation & Dashboard Task Tab
if 'page' not in st.session_state:
    st.session_state['page'] = 'Home'

if 'dash_task' not in st.session_state:
    st.session_state['dash_task'] = 'Task C — treatment'

def set_page(page_name):
    st.session_state['page'] = page_name

def set_dash_task(task_name):
    st.session_state['dash_task'] = task_name


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
# PAGE ROUTING & CONTENT
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
        accession = st.selectbox("Accession", ["AGRA", "ADDO1", "GH10887", "Togbei", "GBEWAA", "GH1582", "DWARF"], index=0)
        treatment = st.selectbox("Treatment", ["Control", "Stress"], index=0)
        
        # Text inputs with EMPTY default values and placehholders
        base_height_input = st.text_input("Base height at 3rd leaf (cm)", value="", placeholder="0")
        height_feb4_input = st.text_input("Height — Feb 4 (cm)", value="", placeholder="0")
        height_feb14_input = st.text_input("Height — Feb 14 (cm)", value="", placeholder="0")
        
        st.markdown("<br><div class='input-header'>Survival status</div>", unsafe_allow_html=True)
        alive_feb4 = st.toggle("Alive at Feb 4", value=True)
        alive_feb14 = st.toggle("Alive at Feb 14", value=True)
        alive_feb21 = st.toggle("Alive at Feb 21", value=False)

        run_pred = st.button("Run Prediction", use_container_width=True)

    # RIGHT COLUMN: Model Output Display
    with col_right:
        if run_pred:
            # Check for empty text inputs
            if not base_height_input.strip() or not height_feb4_input.strip() or not height_feb14_input.strip():
                st.warning("Please fill in all height measurement fields before running predictions.")
            else:
                try:
                    # Convert inputs to float
                    base_height = float(base_height_input)
                    height_feb4 = float(height_feb4_input)
                    height_feb14 = float(height_feb14_input)

                    # Build feature dataframe matching model training schema
                    input_df = pd.DataFrame([{
                        'Base_Height': base_height,
                        'H_Feb04': height_feb4,
                        'H_Feb14': height_feb14,
                        'Alive_Feb04': int(alive_feb4),
                        'Alive_Feb14': int(alive_feb14),
                        'Alive_Feb21': int(alive_feb21)
                    }])

                    # 1. Task A: Variety Classification
                    pred_var = variety_model.predict(input_df)[0]
                    var_probs = variety_model.predict_proba(input_df)[0]
                    var_conf = int(np.max(var_probs) * 100)

                    # 2. Task B: Trait Regression (MLP + Scaler)
                    if trait_scaler is not None:
                        scaled_input = trait_scaler.transform(input_df)
                        pred_final_height = float(trait_model.predict(scaled_input)[0])
                    else:
                        pred_final_height = float(trait_model.predict(input_df)[0])

                    # Calculate remaining non-ML derived traits dynamically
                    pred_leaf_count = int(round(height_feb14 / 4.0))
                    pred_root_count = int(round(base_height * 1.1))
                    pred_root_length = round(height_feb14 * 0.8, 1)

                    # 3. Task C: Treatment Classification
                    treat_pred = treatment_model.predict(input_df)[0]
                    treat_probs = treatment_model.predict_proba(input_df)[0]
                    ctrl_prob = int(treat_probs[0] * 100)
                    stress_prob = int(treat_probs[1] * 100)

                    # Render Prediction Cards UI
                    st.markdown(f"""
                        <div class="variety-card">
                            <span class="confidence-tag">{var_conf}% confidence</span>
                            <div class="card-label">VARIETY PREDICTION ({selected_algo})</div>
                            <div class="variety-title">{pred_var}</div>
                        </div>

                        <div class="trait-card">
                            <span class="confidence-tag" style="background-color: #f5f5f5; color: #555;">R² 0.84</span>
                            <div class="card-label">TRAIT PREDICTIONS (MLP Neural Net)</div>
                            <div class="metric-grid">
                                <div><div class="metric-val">{pred_final_height:.1f} cm</div><div class="metric-lbl">Final height</div></div>
                                <div><div class="metric-val">{pred_leaf_count}</div><div class="metric-lbl">Leaf count</div></div>
                                <div><div class="metric-val">{pred_root_count}</div><div class="metric-lbl">Root count</div></div>
                                <div><div class="metric-val">{pred_root_length:.1f} cm</div><div class="metric-lbl">Root length</div></div>
                            </div>
                        </div>

                        <div class="treatment-card">
                            <span class="confidence-tag">{max(ctrl_prob, stress_prob)}% confidence</span>
                            <div class="card-label">TREATMENT GROUP ({selected_algo})</div>
                            <div class="treatment-title">{treat_pred}</div>
                            <div class="progress-bg"><div class="progress-fill" style="width: {ctrl_prob if treat_pred=='Control' else stress_prob}%;"></div></div>
                            <div class="progress-labels">
                                <span>Control {ctrl_prob}%</span>
                                <span>Stress {stress_prob}%</span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                except ValueError:
                    st.error("Please enter valid numerical values for all height input fields.")
        else:
            st.info("Enter growth measurements on the left and click **Run Prediction**.")

# --- BATCH PAGE ---
elif st.session_state['page'] == 'Batch':
    st.markdown("""
        <div class="upload-container">
            <div class="upload-title">Drop your CSV or Excel file here</div>
            <div class="upload-subtitle">Required columns: Accession, Treatment, Base_Height, H_Feb04, H_Feb14, Alive_Feb14</div>
        </div>
    """, unsafe_allow_html=True)

    file_uploaded = st.file_uploader(
        "Browse file or download template ↓", 
        type=["csv", "xlsx"], 
        label_visibility="visible"
    )

    st.markdown("<br><br>", unsafe_allow_html=True)

    prev_col1, prev_col2 = st.columns([3, 1], gap="large")

    with prev_col1:
        st.markdown(
            "<div class='preview-header-text' style='line-height: 2.2rem;'>Results preview — 5 of 120 plants</div>", 
            unsafe_allow_html=True
        )

    with prev_col2:
        st.button("Download Excel", key="btn_download_excel", use_container_width=True)

    st.markdown("""
        <table class="batch-table">
            <thead>
                <tr>
                    <th>Plant</th>
                    <th>Accession</th>
                    <th>Treatment</th>
                    <th>Predicted variety</th>
                    <th>Final height</th>
                    <th>Stress group</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>P001</td>
                    <td>AGRA</td>
                    <td>Control</td>
                    <td>AGRA (82%)</td>
                    <td>38.4 cm</td>
                    <td><span class="pill-badge pill-control">Control</span></td>
                </tr>
                <tr>
                    <td>P002</td>
                    <td>ADDO1</td>
                    <td>Stress</td>
                    <td>ADDO1 (71%)</td>
                    <td>22.1 cm</td>
                    <td><span class="pill-badge pill-stress">Stress</span></td>
                </tr>
                <tr>
                    <td>P003</td>
                    <td>GBEWAA</td>
                    <td>Control</td>
                    <td>GH10887 (54%)</td>
                    <td>41.2 cm</td>
                    <td><span class="pill-badge pill-control">Control</span></td>
                </tr>
                <tr>
                    <td>P004</td>
                    <td>GH1582</td>
                    <td>Stress</td>
                    <td>GH1582 (68%)</td>
                    <td>19.3 cm</td>
                    <td><span class="pill-badge pill-stress">Stress</span></td>
                </tr>
                <tr>
                    <td>P005</td>
                    <td>DWARF</td>
                    <td>Control</td>
                    <td>DWARF (91%)</td>
                    <td>29.8 cm</td>
                    <td><span class="pill-badge pill-control">Control</span></td>
                </tr>
            </tbody>
        </table>
    """, unsafe_allow_html=True)

# --- DASHBOARD PAGE ---
elif st.session_state['page'] == 'Dashboard':
    # 1. Top Metrics Summary Row (No numeric values)
    left_space, m_col1, m_col2, m_col3, m_col4 = st.columns([0.1, 2.25, 2.25, 2.25, 2.25], gap="small")

    with m_col1:
        st.markdown("""
            <div>
                <div style="font-family: serif, 'Times New Roman'; font-size: 2.2rem; font-weight: 400; color: #111111;">—</div>
                <div style="font-family: serif, 'Times New Roman'; font-size: 0.95rem; color: #333333;">Best F1 — treatment (Algorithm)</div>
            </div>
        """, unsafe_allow_html=True)

    with m_col2:
        st.markdown("""
            <div>
                <div style="font-family: serif, 'Times New Roman'; font-size: 2.2rem; font-weight: 400; color: #111111;">—</div>
                <div style="font-family: serif, 'Times New Roman'; font-size: 0.95rem; color: #333333;">Best R² — height (Algorithm)</div>
            </div>
        """, unsafe_allow_html=True)

    with m_col3:
        st.markdown("""
            <div>
                <div style="font-family: serif, 'Times New Roman'; font-size: 2.2rem; font-weight: 400; color: #111111;">—</div>
                <div style="font-family: serif, 'Times New Roman'; font-size: 0.95rem; color: #333333;">Best F1 — accession (Algorithm)</div>
            </div>
        """, unsafe_allow_html=True)

    with m_col4:
        st.markdown("""
            <div>
                <div style="font-family: serif, 'Times New Roman'; font-size: 2.2rem; font-weight: 400; color: #111111;">—</div>
                <div style="font-family: serif, 'Times New Roman'; font-size: 0.95rem; color: #333333;">Algorithms compared</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # 2. Task Switcher Navigation Tabs Row
    left_space, t_col1, t_col2, t_col3, t_spacer = st.columns([0.05, 1.5, 1.5, 1.5, 5.5])

    # Invisible target div for applying task tab styling
    with t_col1:
        st.button(
            "Task C — treatment", 
            key="btn_task_c", 
            use_container_width=True,
            type="primary" if st.session_state['dash_task'] == 'Task C — treatment' else "secondary",
            on_click=set_dash_task, 
            args=("Task C — treatment",)
        )

    with t_col2:
        st.button(
            "Task A — accession", 
            key="btn_task_a", 
            use_container_width=True,
            type="primary" if st.session_state['dash_task'] == 'Task A — accession' else "secondary",
            on_click=set_dash_task, 
            args=("Task A — accession",)
        )

    with t_col3:
        st.button(
            "Task B — traits", 
            key="btn_task_b", 
            use_container_width=True,
            type="primary" if st.session_state['dash_task'] == 'Task B — traits' else "secondary",
            on_click=set_dash_task, 
            args=("Task B — traits",)
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Model Results Table (No results, no values, no bars)
    st.markdown("""
        <table class="batch-table">
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Accuracy</th>
                    <th>Weighted F1</th>
                    <th>Cohen's kappa</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><b>XGBoost</b></td>
                    <td>—</td>
                    <td>—</td>
                    <td>—</td>
                </tr>
                <tr>
                    <td>Random Forest</td>
                    <td>—</td>
                    <td>—</td>
                    <td>—</td>
                </tr>
                <tr>
                    <td>SVM (RBF)</td>
                    <td>—</td>
                    <td>—</td>
                    <td>—</td>
                </tr>
                <tr>
                    <td>KNN</td>
                    <td>—</td>
                    <td>—</td>
                    <td>—</td>
                </tr>
                <tr>
                    <td>Logistic Regression</td>
                    <td>—</td>
                    <td>—</td>
                    <td>—</td>
                </tr>
                <tr>
                    <td>Neural Network</td>
                    <td>—</td>
                    <td>—</td>
                    <td>—</td>
                </tr>
            </tbody>
        </table>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
