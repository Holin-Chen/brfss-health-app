# BRFSS Population Health Risk App

## Project overview
Population health risk app using CDC BRFSS 2024 data (457K respondents, 301 features).
Two features: (1) population dashboard with US choropleth map, (2) individual
chronic disease risk calculator with SHAP explainability.

## Data
- Raw data: data/brfss_survey_data_2024.csv (BRFSS 2024, public domain)
- Cleaned data: data/brfss_clean.csv (decoded, after Phase 1)
- Features: data/features.pkl (X matrix + y target, after Phase 2)
- Model: models/xgb_diabetes.pkl (trained XGBoost, after Phase 3)

## File structure
src/preprocess.py  — feature selection, decoding, imputation, encoding
src/model.py       — XGBoost training, evaluation, SHAP
app.py             — Streamlit app (2 pages)

## Stack
Python 3.11, pandas, numpy, scikit-learn, xgboost, shap,
streamlit, plotly, folium, imbalanced-learn, joblib

## Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

## Run
streamlit run app.py

## Coding conventions
- Modular: separate files for data, preprocessing, modeling, app
- Always run scripts after writing them and confirm output
- Handle BRFSS coded values (7/9/77/99) as NaN, never model raw codes
- Use joblib for model persistence (not pickle)
- Add type hints and docstrings to all functions
- Health AI disclaimer required on any risk-facing UI

## Key BRFSS target variable
DIABETE4 (2024 coding): 1=Yes, 2=Yes(pregnant only), 3=No, 4=No(pre-diabetes/borderline), 7=Unsure, 9=Refused
Binary encode: 1 = at-risk (1); 2,3,4 = not at-risk (0); 7,9 = NaN
Blood pressure column is BPHIGH4 in 2024 (not BPHIGH6)

## Testing
After each phase, run the relevant script and confirm output shape/metrics.
For the app: run streamlit run app.py and verify both pages load.
