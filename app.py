import math
import streamlit as st
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIGURATION & PAGE TITLE
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TASH MUCH Prediction Tool (Model A & Sullivan Score)",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# MODEL COEFFICIENTS & CONSTANTS (Refit Raw & Shrunk Model A)
# ---------------------------------------------------------------------------
# Raw Model A Maximum-Likelihood Coefficients (Table 1.1)
RAW_INTERCEPT = -8.29658
RAW_B_SBP = 0.06273
RAW_B_DIABETES = 0.78240
RAW_B_MED_COUNT = 0.00010
RAW_B_AGE = 0.01155

# Bootstrap Shrinkage & Offset-Recalibrated Model A Coefficients (Table 3.1 & 3.2)
SHRINKAGE_FACTOR_GAMMA = 0.7709
SHRUNK_INTERCEPT = -6.3221
SHRUNK_B_SBP = 0.04836
SHRUNK_B_DIABETES = 0.60316
SHRUNK_B_MED_COUNT = 0.00008
SHRUNK_B_AGE = 0.00891

# Authoritative Score-to-Probability Recalibration Model (Section 5.2)
SCORE_RECAL_INTERCEPT = -1.8494
SCORE_RECAL_SLOPE = 0.1089

# Predictor Range Boundaries
SBP_MIN, SBP_MAX = 90.0, 139.0
AGE_MIN, AGE_MAX = 18.0, 100.0

# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------
def calculate_much_probability(sbp: float, diabetes: int, med_count: int, age: float) -> dict:
    """Computes continuous logistic regression probability for Raw and Shrunk Model A."""
    # Raw Model Calculations
    raw_contrib_sbp = RAW_B_SBP * sbp
    raw_contrib_diabetes = RAW_B_DIABETES * diabetes
    raw_contrib_med_count = RAW_B_MED_COUNT * med_count
    raw_contrib_age = RAW_B_AGE * age
    raw_log_odds = RAW_INTERCEPT + raw_contrib_sbp + raw_contrib_diabetes + raw_contrib_med_count + raw_contrib_age
    raw_prob = 1.0 / (1.0 + math.exp(-raw_log_odds))

    # Shrunk & Recalibrated Model Calculations
    shrunk_contrib_sbp = SHRUNK_B_SBP * sbp
    shrunk_contrib_diabetes = SHRUNK_B_DIABETES * diabetes
    shrunk_contrib_med_count = SHRUNK_B_MED_COUNT * med_count
    shrunk_contrib_age = SHRUNK_B_AGE * age
    shrunk_log_odds = SHRUNK_INTERCEPT + shrunk_contrib_sbp + shrunk_contrib_diabetes + shrunk_contrib_med_count + shrunk_contrib_age
    shrunk_prob = 1.0 / (1.0 + math.exp(-shrunk_log_odds))

    return {
        "raw_log_odds": raw_log_odds,
        "raw_prob": raw_prob,
        "shrunk_log_odds": shrunk_log_odds,
        "shrunk_prob": shrunk_prob,
        "raw_contrib_sbp": raw_contrib_sbp,
        "raw_contrib_diabetes": raw_contrib_diabetes,
        "raw_contrib_med_count": raw_contrib_med_count,
        "raw_contrib_age": raw_contrib_age,
        "shrunk_contrib_sbp": shrunk_contrib_sbp,
        "shrunk_contrib_diabetes": shrunk_contrib_diabetes,
        "shrunk_contrib_med_count": shrunk_contrib_med_count,
        "shrunk_contrib_age": shrunk_contrib_age,
    }

def calculate_sullivan_score(sbp: float, diabetes: int, med_count: int, age: float) -> dict:
    """Computes Sullivan-method integer point score and recalibrated probability (Section 4 & 5.2)."""
    # 1. SBP: 90–99 mmHg = 0 pts; +5 pts for every 10 mmHg above 99 mmHg
    pts_sbp = int(max(0.0, (sbp - 90.0) // 10.0)) * 5

    # 2. Diabetes: Absent = 0 pts, Present = 7 pts
    pts_diabetes = 7 if diabetes == 1 else 0

    # 3. Age: 18–44 yrs = 0 pts; +1 pt for every decade after 44 yrs
    # Categorization mapping: <45y = 0; 45-54y = 1; 55-64y = 2; 65-74y = 3; >=75y = 4
    if age < 45.0:
        pts_age = 0
    else:
        pts_age = int((age - 45.0) // 10.0) + 1
       
    # 4. Antihypertensive Medications: Retained at 0 pts (§4.4)
    pts_med = 0

    total_pts = pts_sbp + pts_diabetes + pts_age + pts_med

    # Authoritative Score-to-Probability Recalibration Equation (§5.2)
    score_log_odds = SCORE_RECAL_INTERCEPT + (SCORE_RECAL_SLOPE * total_pts)
    score_prob = 1.0 / (1.0 + math.exp(-score_log_odds))

    # Risk Tiers aligned with TRIPOD derivation thresholds (§7)
    if total_pts < 10:
        tier = "Low Risk (<10 pts; P < 0.30)"
        color = "green"
    elif total_pts < 20:
        tier = "Intermediate Risk (10–19 pts; P: 0.30–0.58)"
        color = "orange"
    else:
        tier = "High Risk (≥20 pts; P ≥ 0.58) → ABPM Referral Indicated"
        color = "red"

    return {
        "pts_sbp": pts_sbp,
        "pts_diabetes": pts_diabetes,
        "pts_age": pts_age,
        "pts_med": pts_med,
        "total": total_pts,
        "score_prob": score_prob,
        "score_log_odds": score_log_odds,
        "tier": tier,
        "color": color
    }

# ---------------------------------------------------------------------------
# UI LAYOUT & HEADER
# ---------------------------------------------------------------------------
st.title("Redeveloped TASH MUCH Prediction & Integer Score Tool")
st.subheader("Model A — TRIPOD-Aligned Sullivan Derivation")

st.markdown(
    """
    This application predicts the probability of **Masked Uncontrolled Hypertension (MUCH)** versus 
    **Sustained Controlled Hypertension (SCH)** in treated hypertensive patients with controlled office 
    blood pressure (<140/90 mmHg, ISH 2020 criteria).
    
    *Note: All model parameters reflect direct maximum-likelihood refitting and Harrell bootstrap shrinkage (γ = 0.7709) 
    on the raw analytic dataset (N=129). Integer score predictions use the authoritative single-predictor score recalibration model.*
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
        help="Controlled office SBP range (90–139 mmHg per ISH 2020)."
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
        help="Retained at 0 integer points per §4.4 (shrunk B=0.00008, p=1.000)."
    )

# ---------------------------------------------------------------------------
# COMPUTATION & OUTPUTS
# ---------------------------------------------------------------------------
res = calculate_much_probability(sbp, diabetes, med_count, age)
score = calculate_sullivan_score(sbp, diabetes, med_count, age)

st.divider()
st.header("1. Sullivan Bedside Integer-Point Score (Primary Direct Clinical Instrument)")

sc_col1, sc_col2 = st.columns([1, 2])

with sc_col1:
    st.metric("Total Bedside Score", f"{score['total']} pts")
    st.metric("Recalibrated P(MUCH | Score)", f"{score['score_prob'] * 100:.1f}%")
    
    if score['color'] == "red":
        st.error(f"**Tier:** {score['tier']}")
    elif score['color'] == "orange":
        st.warning(f"**Tier:** {score['tier']}")
    else:
        st.success(f"**Tier:** {score['tier']}")

with sc_col2:
    pts_df = pd.DataFrame({
        "Predictor Component": [
            "Office SBP (90–99 mmHg = 0; +5 pts / 10 mmHg > 99)",
            "Diabetes Status (Present = +7 pts)",
            "Age (18–44 yrs = 0; +1 pt / 10 yrs > 44)",
            "Medication Count (0 pts per agent)"
        ],
        "Input Value": [f"{sbp} mmHg", f"{diabetes_str}", f"{age} yrs", f"{med_count}"],
        "Points Allocated": [f"+{score['pts_sbp']}", f"+{score['pts_diabetes']}", f"+{score['pts_age']}", f"+{score['pts_med']}"]
    })
    st.table(pts_df)
    st.caption("Score Recalibration Formula: logit(P) = -1.8494 + (0.1089 × Score)")

# ---------------------------------------------------------------------------
# CONTINUOUS MODEL COMPARISON
# ---------------------------------------------------------------------------
st.divider()
st.header("2. Continuous Logistic Regression Model (Raw vs. Shrunk & Recalibrated)")

m_col1, m_col2, m_col3 = st.columns([1.5, 1, 1])

with m_col1:
    st.metric("Shrunk & Recalibrated P(MUCH)", f"{res['shrunk_prob'] * 100:.1f}%")
    st.caption(f"Raw Model P(MUCH): **{res['raw_prob'] * 100:.1f}%**")

with m_col2:
    r50 = "MUCH" if res['shrunk_prob'] >= 0.58 else "SCH"
    st.markdown("**Classification (pt = 0.58)**\n*(Youden-Optimal Threshold)*")
    if r50 == "MUCH":
        st.error(f"**{r50}**")
    else:
        st.success(f"**{r50}**")

with m_col3:
    r30 = "MUCH" if res['shrunk_prob'] >= 0.30 else "SCH"
    st.markdown("**Classification (pt = 0.30)**\n*(Rule-Out Screen Threshold)*")
    if r30 == "MUCH":
        st.warning(f"**{r30}**")
    else:
        st.success(f"**{r30}**")

with st.expander("View Full Coefficient & Contribution Comparison Table", expanded=False):
    breakdown_df = pd.DataFrame({
        "Predictor Variable": ["Office SBP", "Diabetes", "Medication Count", "Age", "Intercept / Constant"],
        "Raw Coefficient (B)": [f"{RAW_B_SBP:+.5f}", f"{RAW_B_DIABETES:+.5f}", f"{RAW_B_MED_COUNT:+.5f}", f"{RAW_B_AGE:+.5f}", f"{RAW_INTERCEPT:+.5f}"],
        "Shrunk Coefficient (γ = 0.7709)": [f"{SHRUNK_B_SBP:+.5f}", f"{SHRUNK_B_DIABETES:+.5f}", f"{SHRUNK_B_MED_COUNT:+.5f}", f"{SHRUNK_B_AGE:+.5f}", f"{SHRUNK_INTERCEPT:+.5f}"],
        "Shrunk Log-Odds Contribution": [f"{res['shrunk_contrib_sbp']:+.4f}", f"{res['shrunk_contrib_diabetes']:+.4f}", f"{res['shrunk_contrib_med_count']:+.4f}", f"{res['shrunk_contrib_age']:+.4f}", f"{SHRUNK_INTERCEPT:+.4f}"]
    })
    st.table(breakdown_df)

# ---------------------------------------------------------------------------
# MODEL PERFORMANCE & VALIDATION DOSSIER
# ---------------------------------------------------------------------------
st.divider()
st.header("3. Model Performance & TRIPOD Validation Dossier")

p_col1, p_col2, p_col3, p_col4 = st.columns(4)
p_col1.metric("Optimism-Corrected Score AUC", "0.648", help="95% CI: 0.550–0.745 (Harrell Bootstrap B=1000)")
p_col2.metric("Continuous Model Optimism AUC", "0.616", help="95% CI: 0.519–0.713; Apparent AUC = 0.660")
p_col3.metric("Shrinkage Factor (γ)", "0.7709", help="Calibration slope shrinkage factor applied to raw coefficients")
p_col4.metric("Score Calibration Slope", "1.1446", help="Optimism-corrected score calibration slope")

st.info(
    """
    **Derivation Summary & Statistical Notes:**
    - **Development Cohort:** N=129 complete cases from Tikur Anbessa Specialized Hospital (TASH).
    - **Outcome Prevalence:** Observed MUCH prevalence = 58.14% (75/129 cases).
    - **Hosmer–Lemeshow Test:** χ² = 9.24 (df=8, p=0.323), indicating no evidence of gross miscalibration. Apparent Brier Score = 0.223.
    - **Risk Stratum Boundaries:** Low Risk (<10 pts; P < 0.30), Intermediate Risk (10–19 pts; P: 0.30–0.58), High Risk (≥20 pts; P ≥ 0.58).
    - **Low-Risk Stratum Limitation:** Only 3.1% (4/129) of the derivation sample score <10 due to high baseline prevalence in tertiary care.
    """
)

# ---------------------------------------------------------------------------
# CLINICAL DISCLAIMER
# ---------------------------------------------------------------------------
st.warning(
    """
    **CLINICAL DISCLAIMER:** This tool implements Model A and its derived Sullivan integer score developed on a single 
    tertiary-referral cohort (TASH, N=129; MUCH prevalence 58.1%) using ISH 2020 criteria. It has undergone internal 
    (bootstrap) validation only and has **NOT** been externally validated. Results should be interpreted 
    by a qualified clinician alongside the full clinical context and are not a substitute for 24-hour Ambulatory Blood Pressure Monitoring (ABPM).
    """
)
