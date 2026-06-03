"""BRFSS Population Health Risk App — Streamlit (2 pages)."""

from pathlib import Path

# Bootstrap: download pre-processed files from Hugging Face Hub
def _bootstrap():
    from src.download_data import download_if_missing
    download_if_missing()

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
DASHBOARD_PATH = "data/dashboard.csv"
MODEL_PATH = "models/xgb_diabetes.pkl"

DISEASE_CONFIG = {
    "Diabetes":      {"col": "DIABETE4",  "model": "models/xgb_diabetes.pkl", "transformer": "models/xgb_diabetes_transformer.pkl", "at_risk_label": "Yes", "pop_avg": 14.4},
    "Heart Disease": {"col": "CVDCRHD4",  "model": "models/xgb_heart.pkl",    "transformer": "models/xgb_heart_transformer.pkl",    "at_risk_label": "Yes", "pop_avg": 6.3},
    "COPD":          {"col": "CHCCOPD3",  "model": "models/xgb_copd.pkl",      "transformer": "models/xgb_copd_transformer.pkl",     "at_risk_label": "Yes", "pop_avg": 8.1},
}

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

INCOME_ORDER = ["<$15k","$15k-$20k","$20k-$25k","$25k-$35k","$35k-$50k",
                "$50k-$75k","$75k-$100k","$100k-$150k","$150k-$200k","$200k+"]

EDUCA_ORDER = ["No school","Elementary","Some high school",
               "High school graduate","Some college","College graduate"]

FEATURE_LABELS = {
    "_BMI5":"BMI (numeric)","_AGEG5YR":"Age Group","POORHLTH":"Days Poor Health Limited Activity",
    "PHYSHLTH":"Days Physical Health Not Good","MENTHLTH":"Days Mental Health Not Good",
    "ALCDAY4":"Days Had Alcohol (past 30 days)","CHILDREN":"Number of Children in Household",
    "INCOME3":"Household Income Level","EDUCA":"Education Level",
    "_BMI5CAT_1.0":"BMI: Underweight","_BMI5CAT_2.0":"BMI: Normal Weight",
    "_BMI5CAT_3.0":"BMI: Overweight","_BMI5CAT_4.0":"BMI: Obese",
    "EXERANY2_1.0":"Exercises Regularly: Yes","EXERANY2_2.0":"Exercises Regularly: No",
    "SMOKDAY2_1.0":"Smokes Every Day","SMOKDAY2_2.0":"Smokes Some Days","SMOKDAY2_3.0":"Does Not Smoke",
    "GENHLTH_1.0":"General Health: Excellent","GENHLTH_2.0":"General Health: Very Good",
    "GENHLTH_3.0":"General Health: Good","GENHLTH_4.0":"General Health: Fair","GENHLTH_5.0":"General Health: Poor",
    "CVDCRHD4_1.0":"Coronary Heart Disease: Yes","CVDCRHD4_2.0":"Coronary Heart Disease: No",
    "CHCOCNC1_1.0":"Any Cancer: Yes","CHCOCNC1_2.0":"Any Cancer: No",
    "CHCCOPD3_1.0":"COPD / Emphysema: Yes","CHCCOPD3_2.0":"COPD / Emphysema: No",
    "CHCKDNY2_1.0":"Kidney Disease: Yes","CHCKDNY2_2.0":"Kidney Disease: No",
    "ADDEPEV3_1.0":"Depressive Disorder: Yes","ADDEPEV3_2.0":"Depressive Disorder: No",
    "HAVARTH4_1.0":"Arthritis: Yes","HAVARTH4_2.0":"Arthritis: No",
    "BLIND_1.0":"Blind / Serious Vision Difficulty: Yes","BLIND_2.0":"Blind / Serious Vision Difficulty: No",
    "DECIDE_1.0":"Difficulty Concentrating / Deciding: Yes","DECIDE_2.0":"Difficulty Concentrating / Deciding: No",
    "DIFFWALK_1.0":"Difficulty Walking / Climbing Stairs: Yes","DIFFWALK_2.0":"Difficulty Walking / Climbing Stairs: No",
    "_RFHLTH_1.0":"Good or Better Health: Yes","_RFHLTH_2.0":"Good or Better Health: No",
    "_TOTINDA_1.0":"No Leisure-Time Physical Activity: Yes","_TOTINDA_2.0":"No Leisure-Time Physical Activity: No",
    "_RFSMOK3_1.0":"Current Smoker: Yes","_RFSMOK3_2.0":"Current Smoker: No",
    "_RFBING6_1.0":"Binge Drinker: Yes","_RFBING6_2.0":"Binge Drinker: No",
}

def readable_feature(name: str) -> str:
    return FEATURE_LABELS.get(name, name)

# ── Loaders (cached) ───────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading survey data…")
def load_dashboard() -> pd.DataFrame:
    df = pd.read_csv(DASHBOARD_PATH, low_memory=False)
    df["_STATE"] = pd.to_numeric(df["_STATE"], errors="coerce")
    df["state_name"] = df["_STATE"].map(FIPS_TO_STATE)
    df["age_label"] = df["_AGEG5YR"]
    for disease, cfg in DISEASE_CONFIG.items():
        col = cfg["col"]
        df[f"{disease}_bin"] = (df[col] == cfg["at_risk_label"]).astype(float)
        df.loc[df[col].isna(), f"{disease}_bin"] = np.nan
    return df

@st.cache_resource(show_spinner="Loading model…")
def load_model_and_transformer(disease: str):
    cfg = DISEASE_CONFIG[disease]
    model = joblib.load(cfg["model"])
    pipeline = joblib.load(cfg["transformer"])
    return model, pipeline

# ── Sidebar navigation ─────────────────────────────────────────────────────────
page = st.sidebar.radio("Navigate", ["🗺️ Population Dashboard", "🧮 Individual Risk Calculator"])

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Population Dashboard
# ══════════════════════════════════════════════════════════════════════════════
if page == "🗺️ Population Dashboard":
    st.title("🗺️ Population Health Dashboard")
    st.caption("CDC BRFSS 2024 — 457,670 respondents across US states and territories")

    df = load_dashboard()

    disease = st.selectbox("Select disease", list(DISEASE_CONFIG.keys()), key="dash_disease")
    bin_col = f"{disease}_bin"
    cfg = DISEASE_CONFIG[disease]

    # ── Choropleth ─────────────────────────────────────────────────────────────
    st.subheader(f"{disease} Prevalence by State")
    state_prev = (
        df.dropna(subset=[bin_col, "state_name"])
        .groupby("state_name")
        .agg(prevalence=(bin_col, "mean"), respondents=(bin_col, "count"))
        .reset_index()
    )
    state_prev["prevalence_pct"] = (state_prev["prevalence"] * 100).round(1)

    fig_map = px.choropleth(
        state_prev, locations="state_name", locationmode="USA-states",
        color="prevalence_pct", hover_name="state_name",
        hover_data={"respondents": True, "prevalence_pct": True},
        color_continuous_scale="Reds", scope="usa",
        labels={"prevalence_pct": f"{disease} %"},
        title=f"Self-Reported {disease} Prevalence (%) — BRFSS 2024",
    )
    fig_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0}, height=460)
    st.plotly_chart(fig_map, width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top 10 Highest Prevalence States**")
        top10 = state_prev.nlargest(10, "prevalence_pct")[["state_name","prevalence_pct","respondents"]]
        top10.columns = ["State", f"{disease} %", "Respondents"]
        st.dataframe(top10.reset_index(drop=True), hide_index=True, width="stretch")
    with col2:
        st.markdown("**Top 10 Lowest Prevalence States**")
        bot10 = state_prev.nsmallest(10, "prevalence_pct")[["state_name","prevalence_pct","respondents"]]
        bot10.columns = ["State", f"{disease} %", "Respondents"]
        st.dataframe(bot10.reset_index(drop=True), hide_index=True, width="stretch")

    st.divider()

    # ── Demographic breakdowns ─────────────────────────────────────────────────
    st.subheader(f"{disease} Prevalence by Demographic Group")

    def prevalence_bar(df, group_col, title, order=None, color_scale="Reds", height=350):
        grp = (
            df.dropna(subset=[bin_col, group_col])
            .groupby(group_col)[bin_col]
            .mean().mul(100).reset_index()
            .rename(columns={group_col: group_col, bin_col: "Prevalence %"})
        )
        if order:
            grp[group_col] = pd.Categorical(grp[group_col], categories=order, ordered=True)
            grp = grp.sort_values(group_col).dropna(subset=[group_col])
        else:
            grp = grp.sort_values("Prevalence %", ascending=False)
        fig = px.bar(grp, x=group_col, y="Prevalence %", text_auto=".1f",
                     color="Prevalence %", color_continuous_scale=color_scale, title=title)
        fig.update_layout(coloraxis_showscale=False, height=height)
        return fig

    age_order = ["18-24","25-29","30-34","35-39","40-44","45-49","50-54","55-59","60-64","65-69","70-74","75-79","80+"]

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Age","Sex","Race/Ethnicity","Income","Education","BMI & Exercise"])

    with tab1:
        st.plotly_chart(prevalence_bar(df, "age_label", f"{disease} by Age Group", order=age_order), width="stretch")

    with tab2:
        st.plotly_chart(prevalence_bar(df, "SEXVAR", f"{disease} by Sex", color_scale="Blues"), width="stretch")

    with tab3:
        st.plotly_chart(prevalence_bar(df, "_RACE", f"{disease} by Race/Ethnicity", color_scale="Purples", height=380), width="stretch")

    with tab4:
        st.plotly_chart(prevalence_bar(df, "INCOME3", f"{disease} by Household Income",
                                       order=INCOME_ORDER, color_scale="Greens", height=380), width="stretch")

    with tab5:
        st.plotly_chart(prevalence_bar(df, "EDUCA", f"{disease} by Education Level",
                                       order=EDUCA_ORDER, color_scale="Oranges"), width="stretch")

    with tab6:
        c1, c2 = st.columns(2)
        with c1:
            bmi_order = ["Underweight","Normal","Overweight","Obese"]
            st.plotly_chart(prevalence_bar(df, "_BMI5CAT", f"{disease} by BMI Category",
                                           order=bmi_order, color_scale="Blues"), width="stretch")
        with c2:
            st.plotly_chart(prevalence_bar(df, "EXERANY2", f"{disease} by Exercise Habit",
                                           color_scale="Greens"), width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Individual Risk Calculator
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.title("🧮 Individual Disease Risk Calculator")
    st.info(
        "⚠️ **Disclaimer: For informational purposes only. Not medical advice.** "
        "This tool uses a statistical model trained on population survey data. "
        "Consult a qualified healthcare provider for personal medical guidance.",
        icon="⚠️",
    )

    disease = st.selectbox("Select disease to predict", list(DISEASE_CONFIG.keys()), key="calc_disease")
    cfg = DISEASE_CONFIG[disease]
    model, pipeline_data = load_model_and_transformer(disease)
    transformer  = pipeline_data["transformer"]
    feature_names = pipeline_data["feature_names"]
    numeric_cols  = pipeline_data["numeric_cols"]
    categorical_cols = pipeline_data["categorical_cols"]

    st.subheader("Enter Your Health Information")
    col_a, col_b = st.columns(2)

    with col_a:
        age_group  = st.selectbox("Age Group", options=list(AGE_LABELS.keys()),
                                  format_func=lambda x: AGE_LABELS[x], index=4)
        bmi_cat    = st.selectbox("BMI Category", options=[1.0,2.0,3.0,4.0],
                                  format_func=lambda x: {1.0:"Underweight",2.0:"Normal",3.0:"Overweight",4.0:"Obese"}[x], index=1)
        bmi_val    = st.number_input("BMI (numeric)", min_value=10.0, max_value=99.0, value=25.0, step=0.5)
        exercise   = st.selectbox("Any physical activity in past 30 days?",
                                  options=[1.0,2.0], format_func=lambda x: "Yes" if x==1.0 else "No")
        smoking    = st.selectbox("Smoking frequency", options=[1.0,2.0,3.0],
                                  format_func=lambda x: {1.0:"Every day",2.0:"Some days",3.0:"Not at all"}[x], index=2)
        gen_health = st.selectbox("General health", options=[1.0,2.0,3.0,4.0,5.0],
                                  format_func=lambda x: {1.0:"Excellent",2.0:"Very good",3.0:"Good",4.0:"Fair",5.0:"Poor"}[x], index=2)

    with col_b:
        kidney     = st.selectbox("Kidney disease?",      options=[1.0,2.0], format_func=lambda x: "Yes" if x==1.0 else "No", index=1)
        heart      = st.selectbox("Coronary heart disease?", options=[1.0,2.0], format_func=lambda x: "Yes" if x==1.0 else "No", index=1)
        copd_val   = st.selectbox("COPD / Emphysema?",    options=[1.0,2.0], format_func=lambda x: "Yes" if x==1.0 else "No", index=1)
        depression = st.selectbox("Depressive disorder?", options=[1.0,2.0], format_func=lambda x: "Yes" if x==1.0 else "No", index=1)
        diff_walk  = st.selectbox("Difficulty walking?",  options=[1.0,2.0], format_func=lambda x: "Yes" if x==1.0 else "No", index=1)
        alc_days   = st.slider("Days had alcohol (past 30 days)", 0, 30, 0)
        phys_days  = st.slider("Days physical health not good (past 30)", 0, 30, 0)
        ment_days  = st.slider("Days mental health not good (past 30)", 0, 30, 0)

    if st.button("Calculate My Risk", type="primary", use_container_width=True):

        num_values = {
            "_BMI5": bmi_val * 100,
            "POORHLTH": float(phys_days),
            "MENTHLTH": float(ment_days),
            "PHYSHLTH": float(phys_days),
            "ALCDAY4": float(alc_days),
            "_AGEG5YR": float(age_group),
            "CHILDREN": 0.0, "INCOME3": 5.0, "EDUCA": 4.0,
        }
        cat_values = {
            "_BMI5CAT": bmi_cat, "EXERANY2": exercise, "SMOKDAY2": smoking,
            "GENHLTH": gen_health, "CVDCRHD4": heart, "CHCOCNC1": 2.0,
            "CHCCOPD3": copd_val, "CHCKDNY2": kidney, "ADDEPEV3": depression,
            "HAVARTH4": 2.0, "BLIND": 2.0, "DECIDE": 2.0, "DIFFWALK": diff_walk,
            "_RFHLTH": 1.0 if gen_health <= 2.0 else 2.0,
            "_TOTINDA": 2.0 if exercise == 1.0 else 1.0,
            "_RFSMOK3": 1.0 if smoking in [1.0,2.0] else 2.0,
            "_RFBING6": 1.0 if alc_days >= 5 else 2.0,
        }

        row_num = pd.DataFrame([[num_values.get(c, 0.0) for c in numeric_cols]], columns=numeric_cols)
        row_cat = pd.DataFrame([[cat_values.get(c, 2.0) for c in categorical_cols]], columns=categorical_cols).astype("object")
        X_input = transformer.transform(pd.concat([row_num, row_cat], axis=1))

        prob = model.predict_proba(X_input)[0][1]
        risk_pct = prob * 100
        pop_avg = cfg["pop_avg"]

        # ── Result display ─────────────────────────────────────────────────────
        st.divider()
        color = "#d62728" if risk_pct >= pop_avg * 2 else "#ff7f0e" if risk_pct >= pop_avg else "#2ca02c"
        st.markdown(
            f"<h2 style='text-align:center;color:{color};'>Estimated {disease} Risk: {risk_pct:.1f}%</h2>",
            unsafe_allow_html=True,
        )
        level = "High" if risk_pct >= pop_avg * 2 else "Moderate" if risk_pct >= pop_avg else "Low"
        st.markdown(f"<p style='text-align:center;font-size:1.1em;'>Risk level: <strong>{level}</strong></p>",
                    unsafe_allow_html=True)

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=risk_pct, number={"suffix":"%"},
            gauge={
                "axis": {"range":[0,100]},
                "bar": {"color": color},
                "steps": [
                    {"range":[0, pop_avg],        "color":"#d4edda"},
                    {"range":[pop_avg, pop_avg*2], "color":"#fff3cd"},
                    {"range":[pop_avg*2, 100],     "color":"#f8d7da"},
                ],
                "threshold": {"line":{"color":"black","width":3}, "value": pop_avg},
            },
            title={"text": f"Your Risk vs Population Average ({pop_avg}% — black line)"},
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, width="stretch")

        # ── SHAP waterfall ─────────────────────────────────────────────────────
        st.subheader("What's driving your score? (Top 10 factors)")
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_input)[0]
        top10_idx = np.argsort(np.abs(shap_vals))[::-1][:10]

        shap_df = pd.DataFrame({
            "Feature": [readable_feature(feature_names[i]) for i in top10_idx],
            "SHAP Value": [shap_vals[i] for i in top10_idx],
        })
        shap_df["Direction"] = shap_df["SHAP Value"].apply(lambda v: "Increases risk ▲" if v > 0 else "Decreases risk ▼")
        shap_df["Color"] = shap_df["SHAP Value"].apply(lambda v: "#d62728" if v > 0 else "#2ca02c")

        fig_shap = go.Figure(go.Bar(
            x=shap_df["SHAP Value"], y=shap_df["Feature"],
            orientation="h", marker_color=shap_df["Color"],
            text=shap_df["Direction"], textposition="outside",
        ))
        fig_shap.update_layout(
            title=f"SHAP Contribution of Top 10 Factors — {disease}",
            xaxis_title="SHAP Value (impact on risk score)",
            height=520, margin={"l":220},
        )
        st.plotly_chart(fig_shap, width="stretch")

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
