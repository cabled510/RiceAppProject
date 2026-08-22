# 🌾 Ghanaian Rice Germplasm ML Predictor

A Streamlit machine learning web application for predicting Ghanaian rice accession variety, growth traits, and drought/salinity stress conditions from early-stage growth measurements.

---

## ⚡ Features

- 🎯 **Single Prediction**: Enter early-stage growth measurements (Base height, Feb 4 height, Feb 14 height, survival status) to instantly predict:
  - **Variety Classification** (XGBoost Classifier)
  - **Final Height & Traits** (MLP Neural Network)
  - **Treatment/Stress Group** (XGBoost Classifier)
- 📤 **Batch Upload**: Upload CSV or Excel files with multiple plant records to view and export predictions.
- 📊 **Model Dashboard**: View performance metrics across trained algorithms.

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/cabled510/RiceAppProject.git
cd RiceAppProject
```

### 2. Create and activate virtual environment
```bash
# Windows
python -m venv .venv
source .venv/Scripts/activate

# Mac / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run streamlit_app.py
```

---

## 📂 Project Files

- `streamlit_app.py` — Main Streamlit application
- `styles.css` — Custom CSS styling
- `requirements.txt` — Python package dependencies
- `Preprocessed_Rice_Data.xlsx` — Dataset & accession encoding mappings
- `best_task_a_xgb_model.pkl` — Task A: Variety classification model
- `best_task_b_mlp_finalheight_model_and_scaler.pkl` — Task B: Trait regression model & scaler
- `best_task_c_xgb_model.pkl` — Task C: Stress classification model
