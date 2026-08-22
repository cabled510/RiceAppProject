import streamlit as st
import os
import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
import plotly.express as px

# Accession & Treatment Encoding Mappings
ACC_MAP = {
    0: 'ADDO1', 1: 'ADDO2', 2: 'AGRA', 3: 'DWARF', 4: 'GBEWAA', 5: 'GH10887',
    6: 'GH10942', 7: 'GH10950', 8: 'GH11036', 9: 'GH11610', 10: 'GH1528',
    11: 'GH1546', 12: 'GH1570', 13: 'GH1582', 14: 'GH1589', 15: 'GH1827',
    16: 'GH2075', 17: 'GH2123'
}
REV_ACC_MAP = {v: k for k, v in ACC_MAP.items()}


@st.cache_resource
def load_ml_models():
    """
    Loads machine learning models from the local directory or models/ folder.
    Uses @st.cache_resource so models load into memory only once.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    def get_path(filename):
        root_path = os.path.join(base_dir, filename)
        sub_path = os.path.join(base_dir, "models", filename)
        return root_path if os.path.exists(root_path) else sub_path

    variety_path = get_path("best_task_a_xgb_model.pkl")
    trait_path = get_path("best_task_b_mlp_finalheight_model_and_scaler.pkl")
    treatment_path = get_path("best_task_c_xgb_model.pkl")

    try:
        # Load Task A: Variety Classification (XGBoost)
        with open(variety_path, 'rb') as f:
            variety_model = pickle.load(f)

        # Load Task B: Final Height Trait Prediction (MinMaxScaler + MLPRegressor)
        with open(trait_path, 'rb') as f:
            trait_data = pickle.load(f)
            if isinstance(trait_data, tuple):
                trait_scaler, trait_model = trait_data[0], trait_data[1]
            elif isinstance(trait_data, dict):
                trait_scaler = trait_data.get('scaler')
                trait_model = trait_data.get('model')
            else:
                trait_model, trait_scaler = trait_data, None

        # Load Task C: Stress/Treatment Detection (XGBoost)
        with open(treatment_path, 'rb') as f:
            treatment_model = pickle.load(f)

        return variety_model, trait_model, trait_scaler, treatment_model

    except FileNotFoundError as e:
        st.error(f"⚠️ Model file missing: {e}. Please ensure model files are placed in the repository root or 'models/' folder.")
        return None, None, None, None
    except Exception as e:
        st.error(f"⚠️ Error loading models: {e}")
        return None, None, None, None

# Load models once at app startup
variety_model, trait_model, trait_scaler, treatment_model = load_ml_models()


def build_model_inputs(accession_str, treatment_str, base_height, h_feb04, h_feb14, alive_feb04=1, alive_feb14=1, alive_feb21=0):
    acc_label = REV_ACC_MAP.get(accession_str, 0)
    treat_enc = 1 if treatment_str == 'Stress' else 0
    
    growth_feb04 = h_feb04 - base_height
    growth_feb14 = h_feb14 - h_feb04
    rate_base_feb04 = growth_feb04 / 4.0 if 4.0 != 0 else 0.0
    rate_feb04_feb14 = growth_feb14 / 10.0 if 10.0 != 0 else 0.0
    
    growth_feb21 = growth_feb14 * 0.7
    h_feb21 = h_feb14 + growth_feb21
    growth_feb28 = growth_feb14 * 0.5
    h_feb28 = h_feb21 + growth_feb28
    
    rate_feb14_feb21 = growth_feb21 / 7.0
    rate_feb21_feb28 = growth_feb28 / 7.0
    total_growth = h_feb28 - base_height
    growth_accel = rate_feb04_feb14 - rate_base_feb04
    
    alive_feb28 = alive_feb21
    survival_score = (alive_feb14 + alive_feb21 + alive_feb28) / 3.0
    
    leaf_number = max(1, int(round(h_feb14 / 4.0)))
    root_number = max(1, int(round(base_height * 1.1)))
    root_length = round(h_feb14 * 0.8, 1)
    
    # Task A Features
    df_task_a = pd.DataFrame([{
        'Base_Height': base_height,
        'H_Feb04': h_feb04,
        'H_Feb14': h_feb14,
        'H_Feb21': h_feb21,
        'H_Feb28': h_feb28,
        'Growth_Feb04': growth_feb04,
        'Growth_Feb14': growth_feb14,
        'Growth_Feb21': growth_feb21,
        'Growth_Feb28': growth_feb28,
        'Rate_Feb04_Feb14': rate_feb04_feb14,
        'Rate_Feb14_Feb21': rate_feb14_feb21,
        'Rate_Feb21_Feb28': rate_feb21_feb28,
        'Total_Growth': total_growth,
        'Growth_Acceleration': growth_accel,
        'Survival_Score': survival_score,
        'Treatment_Encoded': treat_enc,
        'Alive_Feb14': alive_feb14,
        'Alive_Feb21': alive_feb21,
        'Alive_Feb28': alive_feb28
    }])
    if variety_model and hasattr(variety_model, 'feature_names_in_'):
        df_task_a = df_task_a[variety_model.feature_names_in_]
        
    # Task B Features
    df_task_b = pd.DataFrame([{
        'Base_Height': base_height,
        'H_Feb04': h_feb04,
        'H_Feb14': h_feb14,
        'Growth_Feb04': growth_feb04,
        'Growth_Feb14': growth_feb14,
        'Rate_Base_Feb04': rate_base_feb04,
        'Rate_Feb04_Feb14': rate_feb04_feb14,
        'Accession_Label': acc_label,
        'Treatment_Encoded': treat_enc,
        'Alive_Feb14': alive_feb14
    }])
    if trait_scaler and hasattr(trait_scaler, 'feature_names_in_'):
        df_task_b = df_task_b[trait_scaler.feature_names_in_]
        
    # Task C Features
    df_task_c = pd.DataFrame([{
        'Base_Height': base_height,
        'H_Feb04': h_feb04,
        'H_Feb14': h_feb14,
        'H_Feb21': h_feb21,
        'H_Feb28': h_feb28,
        'Growth_Feb04': growth_feb04,
        'Growth_Feb14': growth_feb14,
        'Growth_Feb21': growth_feb21,
        'Growth_Feb28': growth_feb28,
        'Total_Growth': total_growth,
        'Growth_Acceleration': growth_accel,
        'Survival_Score': survival_score,
        'Alive_Feb14': alive_feb14,
        'Alive_Feb21': alive_feb21,
        'Alive_Feb28': alive_feb28,
        'Leaf_Number': leaf_number,
        'Root_Number': root_number,
        'Root_Length': root_length,
        'Accession_Label': acc_label
    }])
    if treatment_model and hasattr(treatment_model, 'feature_names_in_'):
        df_task_c = df_task_c[treatment_model.feature_names_in_]
        
    return df_task_a, df_task_b, df_task_c, leaf_number, root_number, root_length


def apply_custom_css():
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

# Initialize Session State Page Navigation, Dashboard Task Tab & Prediction History
if 'page' not in st.session_state:
    st.session_state['page'] = 'Home'

if 'dash_task' not in st.session_state:
    st.session_state['dash_task'] = 'Task C — treatment'

if 'last_prediction' not in st.session_state:
    st.session_state['last_prediction'] = None

if 'prediction_history' not in st.session_state:
    st.session_state['prediction_history'] = []

# Persistent Input Session States
if 'pred_accession' not in st.session_state:
    st.session_state['pred_accession'] = 'ADDO1'
if 'pred_treatment' not in st.session_state:
    st.session_state['pred_treatment'] = 'Control'
if 'pred_base_height' not in st.session_state:
    st.session_state['pred_base_height'] = ''
if 'pred_h_feb04' not in st.session_state:
    st.session_state['pred_h_feb04'] = ''
if 'pred_h_feb14' not in st.session_state:
    st.session_state['pred_h_feb14'] = ''
if 'pred_alive_feb4' not in st.session_state:
    st.session_state['pred_alive_feb4'] = True
if 'pred_alive_feb14' not in st.session_state:
    st.session_state['pred_alive_feb14'] = True
if 'pred_alive_feb21' not in st.session_state:
    st.session_state['pred_alive_feb21'] = False


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
                <div class="feature-icon-box">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#0b2f6b" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>
                </div>
                <div class="feature-title">Single prediction</div>
                <div class="feature-desc">Enter one plant's measurements and get variety, trait, and stress predictions instantly</div>
            </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
            <div class="feature-card">
                <div class="feature-icon-box">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#0b2f6b" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
                </div>
                <div class="feature-title">Batch upload</div>
                <div class="feature-desc">Upload a CSV of multiple plants and download an Excel file of all predictions</div>
            </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
            <div class="feature-card">
                <div class="feature-icon-box">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#0b2f6b" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="9" y1="21" x2="9" y2="9"/></svg>
                </div>
                <div class="feature-title">Model dashboard</div>
                <div class="feature-desc">View performance metrics and feature importance for all six trained algorithms</div>
            </div>
        """, unsafe_allow_html=True)

# --- PREDICT PAGE ---
elif st.session_state['page'] == 'Predict':
    col_spacer_left, col_left, col_right, col_spacer_right = st.columns([0.05, 1, 1.1, 0.05], gap="medium")

    with col_left:
        st.markdown("<div class='input-header'>Growth measurements</div>", unsafe_allow_html=True)
        accession_options = list(REV_ACC_MAP.keys())
        accession = st.selectbox("Accession", accession_options, key="pred_accession")
        treatment = st.selectbox("Treatment", ["Control", "Stress"], key="pred_treatment")
        
        base_height_input = st.text_input("Base height at 3rd leaf (cm)", key="pred_base_height", placeholder="e.g. 15.0")
        height_feb4_input = st.text_input("Height — Feb 4 (cm)", key="pred_h_feb04", placeholder="e.g. 22.5")
        height_feb14_input = st.text_input("Height — Feb 14 (cm)", key="pred_h_feb14", placeholder="e.g. 32.0")
        
        st.markdown("<br><div class='input-header'>Survival status</div>", unsafe_allow_html=True)
        alive_feb4 = st.toggle("Alive at Feb 4", key="pred_alive_feb4")
        alive_feb14 = st.toggle("Alive at Feb 14", key="pred_alive_feb14")
        alive_feb21 = st.toggle("Alive at Feb 21", key="pred_alive_feb21")

        run_pred = st.button("Run Prediction", use_container_width=True)

    # RIGHT COLUMN: Model Output Display
    with col_right:
        if run_pred:
            if not base_height_input.strip() or not height_feb4_input.strip() or not height_feb14_input.strip():
                st.warning("Please fill in all height measurement fields before running predictions.")
            elif variety_model is None or trait_model is None or treatment_model is None:
                st.error("Model loading error. Please ensure model files exist.")
            else:
                try:
                    base_height = float(base_height_input)
                    height_feb4 = float(height_feb4_input)
                    height_feb14 = float(height_feb14_input)

                    df_a, df_b, df_c, pred_leaf_count, pred_root_count, pred_root_length = build_model_inputs(
                        accession, treatment, base_height, height_feb4, height_feb14,
                        int(alive_feb4), int(alive_feb14), int(alive_feb21)
                    )

                    # 1. Task A: Variety Classification
                    pred_var_idx = variety_model.predict(df_a)[0]
                    var_probs = variety_model.predict_proba(df_a)[0]
                    pred_var = ACC_MAP.get(pred_var_idx, f"Variety_{pred_var_idx}")
                    var_conf = int(np.max(var_probs) * 100)

                    # 2. Task B: Trait Regression (MLP + Scaler)
                    if trait_scaler is not None:
                        scaled_b = trait_scaler.transform(df_b)
                        pred_final_height = float(trait_model.predict(scaled_b)[0])
                    else:
                        pred_final_height = float(trait_model.predict(df_b)[0])

                    # 3. Task C: Treatment Classification
                    treat_pred_idx = treatment_model.predict(df_c)[0]
                    treat_probs = treatment_model.predict_proba(df_c)[0]
                    treat_pred = "Control" if treat_pred_idx == 0 else "Stress"
                    ctrl_prob = int(treat_probs[0] * 100)
                    stress_prob = int(treat_probs[1] * 100)

                    # Save prediction result to session state
                    pred_result = {
                        'var_conf': var_conf,
                        'pred_var': pred_var,
                        'pred_final_height': pred_final_height,
                        'pred_leaf_count': pred_leaf_count,
                        'pred_root_count': pred_root_count,
                        'pred_root_length': pred_root_length,
                        'treat_pred': treat_pred,
                        'ctrl_prob': ctrl_prob,
                        'stress_prob': stress_prob,
                        'max_treat_conf': max(ctrl_prob, stress_prob)
                    }
                    st.session_state['last_prediction'] = pred_result
                    st.session_state['prediction_history'].append({
                        'Accession': accession,
                        'Treatment': treatment,
                        'Base_Height': base_height,
                        'H_Feb04': height_feb4,
                        'H_Feb14': height_feb14,
                        'Predicted_Variety': pred_var,
                        'Var_Conf': f"{var_conf}%",
                        'Predicted_Final_Height': f"{pred_final_height:.1f} cm",
                        'Predicted_Stress_Group': treat_pred
                    })

                except ValueError:
                    st.error("Please enter valid numerical values for all height input fields.")
                except Exception as ex:
                    st.error(f"Prediction Error: {ex}")

        # Render stored prediction result if available
        if st.session_state['last_prediction'] is not None:
            p = st.session_state['last_prediction']
            st.markdown(f"""
                <div class="variety-card">
                    <span class="confidence-tag">{p['var_conf']}% confidence</span>
                    <div class="card-label">VARIETY PREDICTION (XGBoost)</div>
                    <div class="variety-title">{p['pred_var']}</div>
                </div>

                <div class="trait-card">
                    <span class="confidence-tag" style="background-color: #f5f5f5; color: #555;">R² 0.84</span>
                    <div class="card-label">TRAIT PREDICTIONS (MLP Neural Net)</div>
                    <div class="metric-grid">
                        <div><div class="metric-val">{p['pred_final_height']:.1f} cm</div><div class="metric-lbl">Final height</div></div>
                        <div><div class="metric-val">{p['pred_leaf_count']}</div><div class="metric-lbl">Leaf count</div></div>
                        <div><div class="metric-val">{p['pred_root_count']}</div><div class="metric-lbl">Root count</div></div>
                        <div><div class="metric-val">{p['pred_root_length']:.1f} cm</div><div class="metric-lbl">Root length</div></div>
                    </div>
                </div>

                <div class="treatment-card">
                    <span class="confidence-tag">{p['max_treat_conf']}% confidence</span>
                    <div class="card-label">TREATMENT GROUP (XGBoost)</div>
                    <div class="treatment-title">{p['treat_pred']}</div>
                    <div class="progress-bg"><div class="progress-fill" style="width: {p['ctrl_prob'] if p['treat_pred']=='Control' else p['stress_prob']}%;"></div></div>
                    <div class="progress-labels">
                        <span>Control {p['ctrl_prob']}%</span>
                        <span>Stress {p['stress_prob']}%</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
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
