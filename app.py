"""BRFSS Population Health Risk App — Streamlit (2 pages)."""

import subprocess, sys
from pathlib import Path

# Bootstrap: download data and run preprocessing if first deploy
def _bootstrap():
    from src.download_data import download_if_missing
    download_if_missing()
    if not Path("data/brfss_clean.csv").exists():
        subprocess.run([sys.executable, "src/phase1_eda.py"], check=True)
    if not Path("data/features.pkl").exists():
        subprocess.run([sys.executable, "src/preprocess.py"], check=True)

_bootstrap()

import numpy as np
import pandas as pd
import joblib
import shap
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BRFSS Health Risk App",
    page_icon="🏥",
    layout="wide",
)

# ── Constants ──────────────────────────────────────────────────────────────────
CLEAN_PATH = "data/brfss_clean.csv"
FEATURES_PATH = "data/features.pkl"
MODEL_PATH = "models/xgb_diabetes.pkl"

FIPS_TO_STATE = {
    1:"Alabama",2:"Alaska",4:"Arizona",5:"Arkansas",6:"California",8:"Colorado",
    9:"Connecticut",10:"Delaware",11:"District of Columbia",12:"Florida",
    13:"Georgia",15:"Hawaii",16:"Idaho",17:"Illinois",18:"Indiana",19:"Iowa",
    20:"Kansas",21:"Kentucky",22:"Louisiana",23:"Maine",24:"Maryland",
    25:"Massachusetts",26:"Michigan",27:"Minnesota",28:"Mississippi",
    29:"Missouri",30:"Montana",31:"Nebraska",32:"Nevada",33:"New Hampshire",
    34:"New Jersey",35:"New Mexico",36:"New York",37:"North Carolina",
    38:"North Dakota",39:"Ohio",40:"Oklahoma",41:"Oregon",42:"Pennsylvania",
    44:"Rhode Island",45:"South Carolina",46:"South Dakota",47:"Tennessee",
    48:"Texas",49:"Utah",50:"Vermont",51:"Virginia",53:"Washington",
    54:"West Virginia",55:"Wisconsin",56:"Wyoming",66:"Guam",72:"Puerto Rico",
    78:"Virgin Islands",
}

AGE_LABELS = {
    1:"18-24",2:"25-29",3:"30-34",4:"35-39",5:"40-44",
    6:"45-49",7:"50-54",8:"55-59",9:"60-64",10:"65-69",
    11:"70-74",12:"75-79",13:"80+",
}

# ── Loaders (cached) ───────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading survey data…")
def load_clean() -> pd.DataFrame:
    df = pd.read_csv(CLEAN_PATH, low_memory=False)
    df["_STATE"] = pd.to_numeric(df["_STATE"], errors="coerce")
    df["DIABETE4_bin"] = df["DIABETE4"].map(
        {"Yes": 1, "Yes_pregnant_only": 0, "No": 0, "No_prediabetes": 0}
    )
    df["state_name"] = df["_STATE"].map(FIPS_TO_STATE)
    df["age_label"] = df["_AGEG5YR"]
    return df

@st.cache_resource(show_spinner="Loading model…")
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource(show_spinner="Loading feature pipeline…")
def load_pipeline():
    return joblib.load(FEATURES_PATH)

# ── Sidebar navigation ─────────────────────────────────────────────────────────
page = st.sidebar.radio(
    "Navigate",
    ["🗺️ Population Dashboard", "🧮 Individual Risk Calculator"],
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Population Dashboard
# ══════════════════════════════════════════════════════════════════════════════
if page == "🗺️ Population Dashboard":
    st.title("🗺️ Population Health Dashboard")
    st.caption("CDC BRFSS 2024 — 457,670 respondents across US states and territories")

    df = load_clean()

    # ── Choropleth ─────────────────────────────────────────────────────────────
    st.subheader("Diabetes Prevalence by State")
    state_prev = (
        df.dropna(subset=["DIABETE4_bin", "state_name"])
        .groupby("state_name")
        .agg(prevalence=("DIABETE4_bin", "mean"), respondents=("DIABETE4_bin", "count"))
        .reset_index()
    )
    state_prev["prevalence_pct"] = (state_prev["prevalence"] * 100).round(1)

    fig_map = px.choropleth(
        state_prev,
        locations="state_name",
        locationmode="USA-states",
        color="prevalence_pct",
        hover_name="state_name",
        hover_data={"respondents": True, "prevalence_pct": True},
        color_continuous_scale="Reds",
        scope="usa",
        labels={"prevalence_pct": "Diabetes %"},
        title="Self-Reported Diabetes Prevalence (%) — BRFSS 2024",
    )
    fig_map.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0}, height=480)
    st.plotly_chart(fig_map, width='stretch')

    # ── Top 10 states ──────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top 10 Highest Prevalence States**")
        top10 = state_prev.nlargest(10, "prevalence_pct")[["state_name", "prevalence_pct", "respondents"]]
        top10.columns = ["State", "Diabetes %", "Respondents"]
        st.dataframe(top10.reset_index(drop=True), hide_index=True, width='stretch')
    with col2:
        st.markdown("**Top 10 Lowest Prevalence States**")
        bot10 = state_prev.nsmallest(10, "prevalence_pct")[["state_name", "prevalence_pct", "respondents"]]
        bot10.columns = ["State", "Diabetes %", "Respondents"]
        st.dataframe(bot10.reset_index(drop=True), hide_index=True, width='stretch')

    st.divider()

    # ── Risk factors by age ────────────────────────────────────────────────────
    st.subheader("Diabetes Prevalence by Age Group")
    age_prev = (
        df.dropna(subset=["DIABETE4_bin", "age_label"])
        .groupby("age_label")["DIABETE4_bin"]
        .mean()
        .mul(100)
        .reset_index()
        .rename(columns={"age_label": "Age Group", "DIABETE4_bin": "Diabetes %"})
    )
    age_order = ["18-24","25-29","30-34","35-39","40-44","45-49","50-54","55-59","60-64","65-69","70-74","75-79","80+"]
    age_prev["Age Group"] = pd.Categorical(age_prev["Age Group"], categories=age_order, ordered=True)
    age_prev = age_prev.sort_values("Age Group").dropna(subset=["Age Group"])

    fig_age = px.bar(
        age_prev, x="Age Group", y="Diabetes %",
        color="Diabetes %", color_continuous_scale="Reds",
        title="Diabetes Prevalence by Age Group",
        text_auto=".1f",
    )
    fig_age.update_layout(coloraxis_showscale=False, height=380)
    st.plotly_chart(fig_age, width='stretch')

    st.divider()

    # ── BMI & Exercise breakdown ───────────────────────────────────────────────
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Prevalence by BMI Category")
        bmi_prev = (
            df.dropna(subset=["DIABETE4_bin", "_BMI5CAT"])
            .groupby("_BMI5CAT")["DIABETE4_bin"]
            .mean().mul(100).reset_index()
            .rename(columns={"_BMI5CAT": "BMI Category", "DIABETE4_bin": "Diabetes %"})
        )
        bmi_order = ["Underweight", "Normal", "Overweight", "Obese"]
        bmi_prev["BMI Category"] = pd.Categorical(bmi_prev["BMI Category"], categories=bmi_order, ordered=True)
        bmi_prev = bmi_prev.sort_values("BMI Category")
        fig_bmi = px.bar(bmi_prev, x="BMI Category", y="Diabetes %", text_auto=".1f",
                         color="Diabetes %", color_continuous_scale="Blues")
        fig_bmi.update_layout(coloraxis_showscale=False, height=320)
        st.plotly_chart(fig_bmi, width='stretch')

    with col4:
        st.subheader("Prevalence by Exercise Habit")
        ex_prev = (
            df.dropna(subset=["DIABETE4_bin", "EXERANY2"])
            .groupby("EXERANY2")["DIABETE4_bin"]
            .mean().mul(100).reset_index()
            .rename(columns={"EXERANY2": "Exercises", "DIABETE4_bin": "Diabetes %"})
        )
        fig_ex = px.bar(ex_prev, x="Exercises", y="Diabetes %", text_auto=".1f",
                        color="Diabetes %", color_continuous_scale="Greens")
        fig_ex.update_layout(coloraxis_showscale=False, height=320)
        st.plotly_chart(fig_ex, width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Individual Risk Calculator
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.title("🧮 Individual Diabetes Risk Calculator")
    st.info(
        "⚠️ **Disclaimer: For informational purposes only. Not medical advice.** "
        "This tool uses a statistical model trained on population survey data. "
        "Consult a qualified healthcare provider for personal medical guidance.",
        icon="⚠️",
    )

    pipeline_data = load_pipeline()
    model = load_model()
    transformer = pipeline_data["transformer"]
    numeric_cols = pipeline_data["numeric_cols"]
    categorical_cols = pipeline_data["categorical_cols"]
    feature_names = pipeline_data["feature_names"]

    st.subheader("Enter Your Health Information")

    col_a, col_b = st.columns(2)

    with col_a:
        age_group = st.selectbox("Age Group", options=list(AGE_LABELS.keys()),
                                 format_func=lambda x: AGE_LABELS[x], index=4)
        bmi_cat = st.selectbox("BMI Category",
                               options=[1.0, 2.0, 3.0, 4.0],
                               format_func=lambda x: {1.0:"Underweight",2.0:"Normal",3.0:"Overweight",4.0:"Obese"}[x],
                               index=1)
        bmi_val = st.number_input("BMI (numeric)", min_value=10.0, max_value=99.0, value=25.0, step=0.5)
        exercise = st.selectbox("Any physical activity in past 30 days?",
                                options=[1.0, 2.0], format_func=lambda x: "Yes" if x == 1.0 else "No")
        smoking = st.selectbox("Smoking frequency",
                               options=[1.0, 2.0, 3.0],
                               format_func=lambda x: {1.0:"Every day",2.0:"Some days",3.0:"Not at all"}[x],
                               index=2)
        gen_health = st.selectbox("General health",
                                  options=[1.0, 2.0, 3.0, 4.0, 5.0],
                                  format_func=lambda x: {1.0:"Excellent",2.0:"Very good",3.0:"Good",4.0:"Fair",5.0:"Poor"}[x],
                                  index=2)

    with col_b:
        kidney = st.selectbox("Ever told you have kidney disease?",
                              options=[1.0, 2.0], format_func=lambda x: "Yes" if x == 1.0 else "No", index=1)
        heart = st.selectbox("Ever told you have coronary heart disease?",
                             options=[1.0, 2.0], format_func=lambda x: "Yes" if x == 1.0 else "No", index=1)
        copd = st.selectbox("Ever told you have COPD/emphysema?",
                            options=[1.0, 2.0], format_func=lambda x: "Yes" if x == 1.0 else "No", index=1)
        depression = st.selectbox("Ever told you have a depressive disorder?",
                                  options=[1.0, 2.0], format_func=lambda x: "Yes" if x == 1.0 else "No", index=1)
        diff_walk = st.selectbox("Difficulty walking or climbing stairs?",
                                 options=[1.0, 2.0], format_func=lambda x: "Yes" if x == 1.0 else "No", index=1)
        alc_days = st.slider("Days had alcohol in past 30 days", 0, 30, 0)
        phys_days = st.slider("Days physical health was not good (past 30)", 0, 30, 0)
        ment_days = st.slider("Days mental health was not good (past 30)", 0, 30, 0)

    if st.button("Calculate My Risk", type="primary", width='stretch'):

        # Build input row matching training columns
        num_values = {
            "_BMI5": bmi_val * 100,   # stored as BMI * 100 in BRFSS
            "POORHLTH": float(phys_days),
            "MENTHLTH": float(ment_days),
            "PHYSHLTH": float(phys_days),
            "ALCDAY4": float(alc_days),
            "_AGEG5YR": float(age_group),
            "CHILDREN": 0.0,
            "INCOME3": 5.0,
            "EDUCA": 4.0,
        }
        cat_values = {
            "_BMI5CAT": bmi_cat,
            "EXERANY2": exercise,
            "SMOKDAY2": smoking,
            "GENHLTH": gen_health,
            "CVDCRHD4": heart,
            "CHCOCNC1": 2.0,
            "CHCCOPD3": copd,
            "CHCKDNY2": kidney,
            "ADDEPEV3": depression,
            "HAVARTH4": 2.0,
            "BLIND": 2.0,
            "DECIDE": 2.0,
            "DIFFWALK": diff_walk,
            "_RFHLTH": 1.0 if gen_health <= 2.0 else 2.0,
            "_TOTINDA": 2.0 if exercise == 1.0 else 1.0,
            "_RFSMOK3": 1.0 if smoking in [1.0, 2.0] else 2.0,
            "_RFBING6": 1.0 if alc_days >= 5 else 2.0,
        }

        # Build input DataFrame
        row_num = pd.DataFrame([[num_values.get(c, 0.0) for c in numeric_cols]], columns=numeric_cols)
        row_cat = pd.DataFrame([[cat_values.get(c, 2.0) for c in categorical_cols]], columns=categorical_cols)
        row_cat = row_cat.astype("object")
        row_all = pd.concat([row_num, row_cat], axis=1)

        X_input = transformer.transform(row_all)
        prob = model.predict_proba(X_input)[0][1]
        risk_pct = prob * 100

        # ── Result display ─────────────────────────────────────────────────────
        st.divider()
        color = "#d62728" if risk_pct >= 30 else "#ff7f0e" if risk_pct >= 15 else "#2ca02c"
        st.markdown(
            f"<h2 style='text-align:center; color:{color};'>Estimated Diabetes Risk: {risk_pct:.1f}%</h2>",
            unsafe_allow_html=True,
        )

        level = "High" if risk_pct >= 30 else "Moderate" if risk_pct >= 15 else "Low"
        st.markdown(f"<p style='text-align:center; font-size:1.1em;'>Risk level: <strong>{level}</strong></p>",
                    unsafe_allow_html=True)

        # Population context gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_pct,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 15], "color": "#d4edda"},
                    {"range": [15, 30], "color": "#fff3cd"},
                    {"range": [30, 100], "color": "#f8d7da"},
                ],
                "threshold": {"line": {"color": "black", "width": 3}, "value": 14.4},
            },
            title={"text": "Your Risk vs Population Average (14.4% black line)"},
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, width='stretch')

        # ── SHAP waterfall ─────────────────────────────────────────────────────
        st.subheader("What's driving your score? (Top 5 factors)")
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_input)[0]
        top5_idx = np.argsort(np.abs(shap_vals))[::-1][:5]

        shap_df = pd.DataFrame({
            "Feature": [feature_names[i] for i in top5_idx],
            "SHAP Value": [shap_vals[i] for i in top5_idx],
        })
        shap_df["Direction"] = shap_df["SHAP Value"].apply(
            lambda v: "Increases risk ▲" if v > 0 else "Decreases risk ▼"
        )
        shap_df["Color"] = shap_df["SHAP Value"].apply(lambda v: "#d62728" if v > 0 else "#2ca02c")

        fig_shap = go.Figure(go.Bar(
            x=shap_df["SHAP Value"],
            y=shap_df["Feature"],
            orientation="h",
            marker_color=shap_df["Color"],
            text=shap_df["Direction"],
            textposition="outside",
        ))
        fig_shap.update_layout(
            title="SHAP Contribution of Top 5 Factors",
            xaxis_title="SHAP Value (impact on risk score)",
            height=320,
            margin={"l": 160},
        )
        st.plotly_chart(fig_shap, width='stretch')

        st.caption(
            "SHAP values show how each factor pushed your risk score up (red) or down (green) "
            "relative to the average. Longer bar = stronger influence."
        )

        st.divider()
        st.warning(
            "⚠️ **For informational purposes only. Not medical advice.** "
            "This prediction is based on population-level statistics. "
            "Please consult your doctor for a proper medical evaluation."
        )
