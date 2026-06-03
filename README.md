# Population Health Risk App
### Using CDC BRFSS 2024 Data

🌐 **Live App: [holin-chen-brfss-health-app.streamlit.app](https://holin-chen-brfss-health-app.streamlit.app/)**

A public health web application that visualizes chronic disease prevalence across the United States and predicts individual disease risk using machine learning — built on the CDC Behavioral Risk Factor Surveillance System (BRFSS) 2024 survey of 457,670 respondents.

---

## Features

### 🗺️ Population Dashboard
- **US choropleth map** of disease prevalence by state
- **Demographic breakdowns** across 6 dimensions: Age, Sex, Race/Ethnicity, Income, Education, BMI & Exercise
- Switchable between **6 diseases**: Diabetes, Heart Disease, COPD, Arthritis, Depression, Asthma

### 🧮 Individual Risk Calculator
- Input form for personal health behaviors and conditions
- **XGBoost model** returns a risk score (%) with a gauge showing your score vs. the population average
- **SHAP waterfall chart** explaining the top 10 factors driving your score
- Switchable between **6 diseases**: Diabetes, Heart Disease, COPD, Arthritis, Depression, Asthma

### 📊 Model Comparison
- Side-by-side **ROC-AUC comparison** of XGBoost, Random Forest, and LASSO across all three diseases
- 5-fold stratified cross-validation on a 100K stratified subsample

---

## Dataset

| Item | Detail |
|------|--------|
| Source | [CDC BRFSS 2024](https://www.cdc.gov/brfss/annual_data/annual_2024.html) |
| Respondents | 457,670 across 49 states + DC + territories |
| Variables | 301 behavioral, condition, and demographic features |
| License | Public Domain (US Government Work) |
| Download | [Kaggle](https://www.kaggle.com/datasets/rudritarahman/cdc-brfss-survey-data-2024) |

---

## Model Performance (5-Fold CV ROC-AUC)

| Disease | XGBoost | Random Forest | LASSO |
|---------|---------|---------------|-------|
| Diabetes | 0.801 | 0.798 | 0.794 |
| Heart Disease | 0.816 | 0.813 | 0.820 |
| COPD | 0.828 | 0.825 | 0.830 |
| Arthritis | 0.810 | 0.804 | 0.803 |
| Depression | 0.820 | 0.815 | 0.805 |
| Asthma | 0.689 | 0.691 | 0.694 |

> XGBoost is used for deployment (SHAP TreeExplainer support). All models trained on a 100K stratified subsample; final XGBoost fit on full data.

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data processing | Python, pandas, numpy |
| ML models | XGBoost, scikit-learn (Random Forest, LASSO) |
| Explainability | SHAP |
| App & UI | Streamlit |
| Charts & maps | Plotly |
| Model persistence | joblib |
| Data hosting | Hugging Face Hub |
| Deployment | Streamlit Community Cloud |

---

## Project Structure

```
brfss-health-app/
├── app.py                        # Streamlit app (2 pages)
├── requirements.txt
├── CLAUDE.md                     # AI assistant context file
├── .streamlit/
│   └── config.toml               # Theme settings
├── src/
│   ├── download_data.py          # Fetch files from Hugging Face Hub
│   ├── phase1_eda.py             # EDA & data cleaning
│   ├── preprocess.py             # Feature engineering pipeline
│   ├── model.py                  # Single-target XGBoost training
│   └── train_all_models.py       # Multi-disease + multi-model training
├── data/                         # (not committed — downloaded at runtime)
│   └── dashboard.csv             # Slim 11-column CSV for dashboard
└── models/                       # (not committed — downloaded at runtime)
    ├── xgb_diabetes.pkl
    ├── xgb_heart.pkl
    ├── xgb_copd.pkl
    └── model_comparison.json
```

---

## Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/Holin-Chen/brfss-health-app.git
cd brfss-health-app

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app (data & models auto-download from Hugging Face on first launch)
streamlit run app.py
```

> First launch downloads ~35MB of pre-processed data and models from Hugging Face Hub automatically.

---

## Key BRFSS Variables

| Variable | Description |
|----------|-------------|
| `DIABETE4` | Ever told you have diabetes (target) |
| `CVDCRHD4` | Ever told you have coronary heart disease (target) |
| `CHCCOPD3` | Ever told you have COPD/emphysema (target) |
| `_BMI5` / `_BMI5CAT` | Continuous BMI and BMI category |
| `_AGEG5YR` | Age group (18–24 through 80+) |
| `EXERANY2` | Any physical activity in past 30 days |
| `SMOKDAY2` | Smoking frequency |
| `GENHLTH` | Self-rated general health (1=Excellent … 5=Poor) |
| `_STATE` | State FIPS code (used for choropleth map) |

> Values 7/9/77/99 (Don't know / Refused) are treated as NaN and excluded from modeling.

---

## Disclaimer

> **For informational purposes only. Not medical advice.** This tool uses statistical models trained on population survey data. Predictions reflect population-level patterns, not individual diagnoses. Consult a qualified healthcare provider for personal medical guidance.

---

## Data Sources & Hosting

- **Raw survey data:** [CDC BRFSS 2024](https://www.cdc.gov/brfss/annual_data/annual_2024.html) via Kaggle
- **Pre-processed files:** [Hugging Face Hub — holinchen/brfss-health-app-data](https://huggingface.co/datasets/holinchen/brfss-health-app-data)
- **Source code:** [GitHub — Holin-Chen/brfss-health-app](https://github.com/Holin-Chen/brfss-health-app)
