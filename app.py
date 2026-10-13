import math
import streamlit as st
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIGURATION & PAGE TITLE
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TASH MUCH Prediction Tool (Model 2 & Integer Score)",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# MODEL COEFFICIENTS & CONSTANTS
# Source: prediction_model_report (2026-10-11), Appendix A "Model 2
# (shrinkage-corrected)" and Section "Simplified Integer Score".
# Model 2 = 1 + officesy + diabetes (unpenalized ML, then uniform
# bootstrap-slope shrinkage with intercept re-estimation).
# ---------------------------------------------------------------------------
# Shrinkage-corrected Model 2 (Appendix A, Model 2)
SHRUNK_INTERCEPT = -7.6417
SHRUNK_B_SBP = 0.0646
SHRUNK_B_DIABETES = 0.4538
SHRINKAGE_FACTOR = 0.964          # Table 1: Model 2 shrinkage [95% CI 0.494-1.944]

# Sullivan integer score sheet (Score Sheet table)
#   Office SBP: <120 = 0 ; 120-<130 = 2 ; >=130 = 3
#   Diabetes  : absent = 0 ; present = 1
SBP_CUT_LOW = 120.0
SBP_CUT_HIGH = 130.0
PTS_SBP_LOW, PTS_SBP_MID, PTS_SBP_HIGH = 0, 2, 3
PTS_DIABETES = 1
MAX_SCORE = PTS_SBP_HIGH + PTS_DIABETES  # 4

# Score-to-risk lookup (Risk Table; "recalibrate" mapping)
SCORE_RISK = {0: 0.296, 1: 0.417, 2: 0.549, 3: 0.674, 4: 0.779}

# Analyst-specified operating points (Recommended Operating Point section)
REFERRAL_THRESHOLD = 0.25   # referral threshold; limited ABPM capacity
SHARED_DECISION_UPPER = 0.50  # shared-decision zone = 25-50%

# Predictor input ranges (controlled office SBP, carried over from the
# previous tool; the report does not state a separate range)
SBP_MIN, SBP_MAX = 90.0, 139.0

# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------
def classify_risk(prob: float) -> tuple:
    """Maps a probability onto the report's operating-point zones."""
    if prob < REFERRAL_THRESHOLD:
        return ("Below referral threshold (P < 25%) → ABPM not routinely indicated", "green")
    elif prob < SHARED_DECISION_UPPER:
        return ("Shared-decision zone (P 25–<50%) → discuss ABPM with patient", "orange")
    else:
        return ("Above shared-decision zone (P ≥ 50%) → ABPM referral recommended", "red")


def calculate_much_probability(sbp: float, diabetes: int) -> dict:
    """Shrinkage-corrected Model 2 logistic regression probability."""
    contrib_sbp = SHRUNK_B_SBP * sbp
    contrib_diabetes = SHRUNK_B_DIABETES * diabetes
    log_odds = SHRUNK_INTERCEPT + contrib_sbp + contrib_diabetes
    prob = 1.0 / (1.0 + math.exp(-log_odds))
    return {
        "log_odds": log_odds,
        "prob": prob,
        "contrib_sbp": contrib_sbp,
        "contrib_diabetes": contrib_diabetes,
    }


def calculate_integer_score(sbp: float, diabetes: int) -> dict:
    """Computes the Model 2 Sullivan integer score and its tabulated risk."""
    if sbp < SBP_CUT_LOW:
        pts_sbp, sbp_cat = PTS_SBP_LOW, "<120"
    elif sbp < SBP_CUT_HIGH:
        pts_sbp, sbp_cat = PTS_SBP_MID, "120–<130"
    else:
        pts_sbp, sbp_cat = PTS_SBP_HIGH, "≥130"

    pts_diabetes = PTS_DIABETES if diabetes == 1 else 0
    total = pts_sbp + pts_diabetes
    score_prob = SCORE_RISK[total]
    tier, color = classify_risk(score_prob)

    return {
        "pts_sbp": pts_sbp,
        "sbp_cat": sbp_cat,
        "pts_diabetes": pts_diabetes,
        "total": total,
        "score_prob": score_prob,
        "tier": tier,
        "color": color,
    }


def show_tier(color: str, text: str):
    if color == "red":
        st.error(f"**Zone:** {text}")
    elif color == "orange":
        st.warning(f"**Zone:** {text}")
    else:
        st.success(f"**Zone:** {text}")

# ---------------------------------------------------------------------------
# UI LAYOUT & HEADER
# ---------------------------------------------------------------------------
st.title("Revised TASH MUCH Prediction & Integer Score Tool")
st.subheader("Model 2 (Office SBP + Diabetes) — Sullivan Integer Score")

st.markdown(
    """
    This application estimates the probability of **Masked Uncontrolled Hypertension (MUCH)** in treated
    hypertensive patients with controlled office blood pressure (<140/90 mmHg), to help prioritise referral for
    24-hour ambulatory blood pressure monitoring (ABPM) when ABPM capacity is limited.

    *Note: Parameters come from the shrinkage-corrected Model 2 (office SBP + diabetes; N=113, 67 events) and the
    linear integer score derived from it using the Sullivan, Massaro & D'Agostino method. Probabilities for the
    integer score are taken from the score-to-risk recalibration table.*
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
        help="Controlled office SBP range (90–139 mmHg)."
    )

with col_in2:
    diabetes_str = st.radio("Diabetes Status", ["Absent", "Present"], index=1, horizontal=True)
    diabetes = 1 if diabetes_str == "Present" else 0

# ---------------------------------------------------------------------------
# COMPUTATION
# ---------------------------------------------------------------------------
res = calculate_much_probability(sbp, diabetes)
score = calculate_integer_score(sbp, diabetes)
reg_tier, reg_color = classify_risk(res["prob"])

# ---------------------------------------------------------------------------
# SECTION 1: INTEGER SCORE
# ---------------------------------------------------------------------------
st.divider()
st.header("1. Bedside Integer-Point Score (Primary Clinical Instrument)")

sc_col1, sc_col2 = st.columns([1, 2])

with sc_col1:
    st.metric("Total Bedside Score", f"{score['total']} / {MAX_SCORE} pts")
    st.metric("Estimated P(MUCH | Score)", f"{score['score_prob'] * 100:.1f}%")
    show_tier(score["color"], score["tier"])

with sc_col2:
    pts_df = pd.DataFrame({
        "Predictor Component": [
            "Office SBP (<120 = 0; 120–<130 = 2; ≥130 = 3)",
            "Diabetes Status (Present = +1 pt)",
        ],
        "Input Value": [f"{sbp:g} mmHg  (category {score['sbp_cat']})", diabetes_str],
        "Points Allocated": [f"+{score['pts_sbp']}", f"+{score['pts_diabetes']}"],
    })
    st.table(pts_df)

    risk_df = pd.DataFrame({
        "Total Points": list(SCORE_RISK.keys()),
        "Estimated Risk of MUCH": [f"{p * 100:.1f}%" for p in SCORE_RISK.values()],
        "Zone": [classify_risk(p)[0].split(" →")[0] for p in SCORE_RISK.values()],
        "Your Score": ["◀" if k == score["total"] else "" for k in SCORE_RISK],
    })
    st.table(risk_df)
    st.caption("Score-to-risk lookup from the recalibrated risk table (report, Simplified Integer Score section).")

# ---------------------------------------------------------------------------
# SECTION 2: CONTINUOUS MODEL
# ---------------------------------------------------------------------------
st.divider()
st.header("2. Continuous Logistic Regression Model (Shrinkage-Corrected Model 2)")

m_col1, m_col2, m_col3 = st.columns([1.5, 1, 1])

with m_col1:
    st.metric("Shrunken Model 2 P(MUCH)", f"{res['prob'] * 100:.1f}%")
    st.caption(f"Log-odds = {res['log_odds']:+.4f}  |  logit(P) = "
               f"{SHRUNK_INTERCEPT:+.4f} + {SHRUNK_B_SBP:.4f}×SBP + {SHRUNK_B_DIABETES:.4f}×Diabetes")

with m_col2:
    st.markdown("**Referral threshold (P ≥ 25%)**\n*(ABPM-capacity-driven)*")
    if res["prob"] >= REFERRAL_THRESHOLD:
        st.warning("**Refer / consider ABPM**")
    else:
        st.success("**Below threshold**")

with m_col3:
    st.markdown("**Upper shared-decision bound (P ≥ 50%)**\n*(ABPM referral recommended)*")
    if res["prob"] >= SHARED_DECISION_UPPER:
        st.error("**ABPM recommended**")
    else:
        st.success("**Not above upper bound**")

show_tier(reg_color, reg_tier)

with st.expander("View Coefficient & Contribution Table", expanded=False):
    breakdown_df = pd.DataFrame({
        "Predictor Variable": ["Office SBP", "Diabetes", "Intercept / Constant"],
        "Shrunken Coefficient (B)": [f"{SHRUNK_B_SBP:+.4f}", f"{SHRUNK_B_DIABETES:+.4f}", f"{SHRUNK_INTERCEPT:+.4f}"],
        "Log-Odds Contribution": [f"{res['contrib_sbp']:+.4f}", f"{res['contrib_diabetes']:+.4f}", f"{SHRUNK_INTERCEPT:+.4f}"],
    })
    st.table(breakdown_df)
    st.caption(f"Uniform bootstrap shrinkage factor = {SHRINKAGE_FACTOR} (95% CI 0.494–1.944); intercept re-estimated after shrinkage.")

# ---------------------------------------------------------------------------
# SECTION 3: PERFORMANCE & VALIDATION DOSSIER
# ---------------------------------------------------------------------------
st.divider()
st.header("3. Model Performance & Validation Dossier")

p_col1, p_col2, p_col3, p_col4 = st.columns(4)
p_col1.metric("Optimism-Corrected AUC (Integer Score)", "0.622",
              help="Apparent 0.638; DeLong 95% CI 0.537–0.740 (apparent). Full-process bootstrap, B=1000.")
p_col2.metric("Optimism-Corrected AUC (Shrunken Model 2)", "0.625",
              help="Apparent 0.644; DeLong 95% CI 0.539–0.750 (apparent).")
p_col3.metric("Calibration Slope (Score)", "1.014",
              help="Optimism-corrected; resample range 0.469–2.314. Model 2 regression slope = 0.910.")
p_col4.metric("Brier Score (Score)", "0.2343",
              help="Optimism-corrected; shrunken regression = 0.2326. Scaled Brier (score) = 0.038.")

tab_diag, tab_dca = st.tabs(["Integer-Score Threshold Diagnostics", "Decision-Curve Net Benefit"])

with tab_diag:
    diag_df = pd.DataFrame({
        "Probability Cut-off": ["0.25", "0.33", "0.50"],
        "Sensitivity [95% CI]": ["1.000 [0.946, 1.000]", "0.955 [0.876, 0.985]", "0.925 [0.837, 0.968]"],
        "Specificity [95% CI]": ["0.000 [0.000, 0.077]", "0.196 [0.107, 0.332]", "0.261 [0.156, 0.403]"],
        "PPV (prev. 59.3%)": ["0.593", "0.634", "0.646"],
        "NPV (prev. 59.3%)": ["N/A", "0.750", "0.706"],
        "LR+": ["1.00", "1.19", "1.25"],
        "LR−": ["0.69", "0.23", "0.29"],
        "PPV (prev. 30%)": ["0.300", "0.337", "0.349"],
        "NPV (prev. 30%)": ["N/A", "0.911", "0.891"],
    })
    st.dataframe(diag_df, hide_index=True)
    st.caption("Sensitivity, specificity and LRs are prevalence-invariant; PPV/NPV do not transport to settings with a different MUCH prevalence.")

with tab_dca:
    dca_df = pd.DataFrame({
        "Threshold": ["0.25", "0.33", "0.50"],
        "NB (Integer Score)": ["0.4572", "0.4051", "0.2478"],
        "NB (Shrunken Model 2)": ["0.4631", "0.4099", "0.2478"],
        "NB (Refer All)": ["0.4572", "0.3924", "0.1858"],
        "NB (Refer None)": ["0.0000", "0.0000", "0.0000"],
        "Tests avoided /100 (Score)": ["-0.0", "2.6", "6.2"],
        "Tests avoided /100 (Model 2)": ["1.8", "3.5", "6.2"],
    })
    st.dataframe(dca_df, hide_index=True)
    st.caption("Net benefit of the integer score equals refer-all at the 25% threshold (no tests avoided).")

st.info(
    """
    **Derivation Summary & Statistical Notes:**
    - **Development Cohort:** N=113 complete cases; 67 events (observed MUCH prevalence 59.3%). EPV = 15.33 (3 candidate predictor df).
    - **Model Selection:** Model 2 (office SBP + diabetes) was compared with Model 1 (SBP only) and Model 3 (+ number of medications); differences in AUC were not statistically significant (DeLong p = 0.389 for Model 1 vs 2).
    - **Functional Form:** Restricted cubic spline test for office SBP showed no evidence of non-linearity (LR χ² = 0.34, p = 0.560); this test has low power in small samples.
    - **Integer Score Construction:** Point constant B = β(SBP) × 10; shrinkage method = bootstrap slope; full-process bootstrap (1000 resamples, 0 non-convergent) with mean resample shrinkage 0.913.
    - **Operating Points:** Referral threshold 25%; shared-decision zone 25–50% (rationale: balance of missed MUCH diagnoses vs. negative ABPM tests; limited test capacity).
    - **Discrimination Is Modest:** Corrected AUC ≈ 0.62; the score is a triage aid, not a diagnostic test.
    - **Low-Risk Limitation:** Every attainable integer score maps to a risk ≥ 25% (score 0 = 29.6%), so the score alone cannot place a patient below the 25% referral threshold; it works as a rank-ordering/prioritisation tool across the 25–50% and ≥ 50% zones.
    """
)

# ---------------------------------------------------------------------------
# CLINICAL DISCLAIMER
# ---------------------------------------------------------------------------
st.warning(
    """
    **CLINICAL DISCLAIMER:** This tool implements shrinkage-corrected Model 2 and its derived Sullivan integer score,
    developed on a single tertiary-referral cohort (N=113; MUCH prevalence 59.3%). It has undergone internal
    (bootstrap) validation only and has **NOT** been externally validated. Results should be interpreted
    by a qualified clinician alongside the full clinical context and are not a substitute for 24-hour Ambulatory
    Blood Pressure Monitoring (ABPM).
    """
)

