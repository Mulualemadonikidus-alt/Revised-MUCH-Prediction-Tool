import math
import streamlit as st
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIGURATION & PAGE TITLE
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Revised TASH MUCH Prediction Tool (Model A)",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# MODEL COEFFICIENTS & CONSTANTS (Model A Source of Truth)
# ---------------------------------------------------------------------------
INTERCEPT = -8.297
B_SBP = 0.063
B_DIABETES = 0.782
B_MED_COUNT = 0.000
B_AGE = 0.012

SBP_MIN, SBP_MAX = 90, 139
AGE_MIN, AGE_MAX = 18, 100

# Simplified integer score reference baselines
REF_SBP = 100
REF_AGE = 30
BASE_UNIT = B_AGE * 10  # 0.12

POINTS_PER_10_SBP = round((B_SBP * 10) / BASE_UNIT)  # 5
POINTS_DIABETES = round(B_DIABETES / BASE_UNIT)     # 7
POINTS_PER_10_AGE = round((B_AGE * 10) / BASE_UNIT) # 1
POINTS_PER_MED = round(B_MED_COUNT / BASE_UNIT)     # 0

# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------
def calculate_much_probability(sbp: float, diabetes: int, med_count: int, age: float) -> dict:
    """Computes continuous logistic regression probability for Model A."""
    contrib_sbp = B_SBP * sbp
    contrib_diabetes = B_DIABETES * diabetes
    contrib_med_count = B_MED_COUNT * med_count
    contrib_age = B_AGE * age
    
    log_odds = INTERCEPT + contrib_sbp + contrib_diabetes + contrib_med_count + contrib_age
    probability = 1.0 / (1.0 + math.exp(-log_odds))
    
    return {
        "contrib_sbp": contrib_sbp,
        "contrib_diabetes": contrib_diabetes,
        "contrib_med_count": contrib_med_count,
        "contrib_age": contrib_age,
        "constant": INTERCEPT,
        "log_odds": log_odds,
        "probability": probability
    }

def calculate_simplified_score(sbp: float, diabetes: int, med_count: int, age: float) -> dict:
    """Computes bedside integer point score approximation for Model A."""
    pts_sbp = round(POINTS_PER_10_SBP * (sbp - REF_SBP) / 10)
    pts_diabetes = POINTS_DIABETES if diabetes == 1 else 0
    pts_age = round(POINTS_PER_10_AGE * (age - REF_AGE) / 10)
    pts_med = POINTS_PER_MED * med_count
    
    total_pts = pts_sbp + pts_diabetes + pts_age + pts_med
    
    if total_pts < 10:
        tier = "Low Risk (<10 pts)"
        color = "green"
    elif total_pts < 20:
        tier = "Moderate Risk (10–19 pts)"
        color = "orange"
    else:
        tier = "High Risk (≥20 pts) → ABPM Referral Indicated"
        color = "red"
        
    return {
        "pts_sbp": pts_sbp,
        "pts_diabetes": pts_diabetes,
        "pts_age": pts_age,
        "pts_med": pts_med,
        "total": total_pts,
        "tier": tier,
        "color": color
    }

# ---------------------------------------------------------------------------
# UI LAYOUT & HEADER
# ---------------------------------------------------------------------------
st.title("Revised TASH MUCH Prediction Tool")
st.subheader("Model A — Core Parsimonious Clinical Model")

st.markdown(
    """
    This tool estimates the probability of **Masked Uncontrolled Hypertension (MUCH)** versus 
    **Sustained Controlled Hypertension (SCH)** in treated hypertensive patients with controlled office 
    blood pressure (<140/90 mmHg, ISH 2020 criteria).
    
    *Note: This tool supersedes the earlier prototype (office SBP + diabetes + ACEI + CCB) hosted on GitHub Pages. 
    It preserves the original UI workflow while updating all coefficients and validation parameters to match Model A.*
    """
)
st.divider()

# ---------------------------------------------------------------------------
# PATIENT INPUT FORM
# ---------------------------------------------------------------------------
col_in1, col_in2 = st.columns(2)

with col_in1:
    sbp = st.number_input(
        "Office Systolic BP (mmHg)",
        min_value=float(SBP_MIN),
        max_value=float(SBP_MAX),
        value=130.0,
        step=1.0,
        help="Must be within the controlled office BP range (90–139 mmHg per ISH 2020)."
    )
    age = st.number_input(
        "Age (years)",
        min_value=float(AGE_MIN),
        max_value=float(AGE_MAX),
        value=55.0,
        step=1.0
    )

with col_in2:
    diabetes_str = st.radio("Diabetes Status", ["Absent", "Present"], index=1)
    diabetes = 1 if diabetes_str == "Present" else 0
    
    med_count = st.number_input(
        "Number of Antihypertensive Medications",
        min_value=0,
        max_value=10,
        value=2,
        step=1,
        help="Note: Fitted coefficient B=0.000 (p=1.000); retained for clinical parsimony."
    )

# ---------------------------------------------------------------------------
# COMPUTATION & OUTPUTS
# ---------------------------------------------------------------------------
res = calculate_much_probability(sbp, diabetes, med_count, age)
score = calculate_simplified_score(sbp, diabetes, med_count, age)

st.divider()
st.header("1. Continuous Logistic Regression Model (Authoritative)")

m_col1, m_col2, m_col3 = st.columns([1.5, 1, 1])

with m_col1:
    st.metric("Predicted P(MUCH)", f"{res['probability'] * 100:.1f}%", delta=None)
    st.caption(f"Log-odds (Linear Predictor): **{res['log_odds']:+.4f}**")

with m_col2:
    r50 = "MUCH" if res['probability'] >= 0.50 else "SCH"
    st.markdown(f"**Classification (pt = 0.50)**\n*(Study Anchor)*")
    if r50 == "MUCH":
        st.error(f"**{r50}**")
    else:
        st.success(f"**{r50}**")

with m_col3:
    r30 = "MUCH" if res['probability'] >= 0.30 else "SCH"
    st.markdown(f"**Classification (pt = 0.30)**\n*(Primary-Care Screen)*")
    if r30 == "MUCH":
        st.warning(f"**{r30}**")
    else:
        st.success(f"**{r30}**")

with st.expander("View Logistic Regression Coefficient Breakdown", expanded=True):
    breakdown_df = pd.DataFrame({
        "Predictor Variable": ["Office SBP", "Diabetes", "Medication Count", "Age", "Constant"],
        "Input Value": [f"{sbp} mmHg", f"{diabetes}", f"{med_count}", f"{age} yrs", "1.0"],
        "Model Coefficient (B)": [f"{B_SBP:+.3f}", f"{B_DIABETES:+.3f}", f"{B_MED_COUNT:+.3f}", f"{B_AGE:+.3f}", f"{INTERCEPT:+.3f}"],
        "Log-Odds Contribution": [f"{res['contrib_sbp']:+.4f}", f"{res['contrib_diabetes']:+.4f}", f"{res['contrib_med_count']:+.4f}", f"{res['contrib_age']:+.4f}", f"{res['constant']:+.4f}"]
    })
    st.table(breakdown_df)

# ---------------------------------------------------------------------------
# SIMPLIFIED BEDSIDE SCORE
# ---------------------------------------------------------------------------
st.divider()
st.header("2. Simplified Bedside Integer-Point Score")

sc_col1, sc_col2 = st.columns([1, 2])

with sc_col1:
    st.metric("Total Bedside Score", f"{score['total']} pts")
    if score['color'] == "red":
        st.error(f"**Tier:** {score['tier']}")
    elif score['color'] == "orange":
        st.warning(f"**Tier:** {score['tier']}")
    else:
        st.success(f"**Tier:** {score['tier']}")

with sc_col2:
    pts_df = pd.DataFrame({
        "Component": ["Office SBP (+5 pts / 10 mmHg > 100)", "Diabetes (+7 pts)", "Age (+1 pt / 10 yrs > 30)", "Medication Count (+0 pts)"],
        "Points Allocated": [f"{score['pts_sbp']:+d}", f"{score['pts_diabetes']:+d}", f"{score['pts_age']:+d}", f"{score['pts_med']:+d}"]
    })
    st.table(pts_df)

# ---------------------------------------------------------------------------
# MODEL PERFORMANCE & VALIDATION METRICS
# ---------------------------------------------------------------------------
st.divider()
st.header("3. Model Performance & Validation Dossier")

p_col1, p_col2, p_col3, p_col4 = st.columns(4)
p_col1.metric("Optimism-Corrected AUC", "0.6146", help="Harrell Bootstrap B=1000 (Primary Standard)")
p_col2.metric("Apparent AUC", "0.6604", help="DeLong 95% CI: 0.5632–0.7575")
p_col3.metric("Calibration Slope", "0.6893", help="Optimism-corrected calibration slope")
p_col4.metric("Brier Score", "0.2405", help="Optimism-corrected Brier score")

st.info(
    """
    **Validation Status & Statistical Summary:**
    - **Development Cohort:** Derived from N=129 complete cases at Tikur Anbessa Specialized Hospital (TASH).
    - **Outcome Prevalence:** Study empirical MUCH prevalence = 58.1% (tertiary care setting).
    - **Predictor Significance:** Office SBP is the sole individually significant predictor (p=0.006). Diabetes is borderline (p=0.079). Medication count (p=1.000) and Age (p=0.491) are retained on clinical grounds.
    - **AUC Discrepancy Note:** An earlier draft reported an AUC of 0.6940; this was confirmed to be an unlabeled N=65 echo-subgroup scoring artifact. The true full-cohort optimism-corrected AUC is **0.6146**.
    """
)

# ---------------------------------------------------------------------------
# CLINICAL DISCLAIMER
# ---------------------------------------------------------------------------
st.warning(
    """
    **CLINICAL DISCLAIMER:** This tool implements Model A, a binary logistic regression model developed on a single 
    tertiary-referral cohort (TASH, N=129; MUCH prevalence 58.1%) using ISH 2020 criteria. It has undergone internal 
    (bootstrap) validation only and has **NOT** been externally or temporally validated. Results should be interpreted 
    by a qualified clinician alongside the full clinical picture and are not a substitute for 24-hour Ambulatory Blood Pressure 
    Monitoring (ABPM).
    """
)
