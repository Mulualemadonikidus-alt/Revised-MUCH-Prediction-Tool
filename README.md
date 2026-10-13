# TASH MUCH Prediction Tool — Model 2 & Integer Score

A Streamlit web app that estimates the probability of **Masked Uncontrolled Hypertension (MUCH)** in treated hypertensive patients whose office blood pressure is controlled (<140/90 mmHg). It is intended to help prioritise referral for 24-hour ambulatory blood pressure monitoring (ABPM) where ABPM capacity is limited.

The app implements:

1. **A Sullivan-method integer bedside score (0–4)** — the primary clinical instrument.
2. **The shrinkage-corrected logistic regression (Model 2)** that the score was derived from.

> **Status: research prototype.** Internally validated (bootstrap) only. **Not externally validated. Not a medical device.** See [Limitations](#limitations) and [Disclaimer](#disclaimer).

---

## Model summary

**Model 2** uses two predictors: office systolic BP (SBP) and diabetes status.

```
logit(P) = −7.6417 + 0.0646 × SBP(mmHg) + 0.4538 × Diabetes(0/1)
```

Coefficients are from unpenalised maximum likelihood followed by uniform bootstrap-slope shrinkage (factor 0.964) with intercept re-estimation.

### Integer score sheet

| Predictor | Category | Points |
|---|---|---|
| Office SBP | < 120 mmHg | 0 |
| Office SBP | 120 to < 130 mmHg | 2 |
| Office SBP | ≥ 130 mmHg | 3 |
| Diabetes | Absent | 0 |
| Diabetes | Present | 1 |

### Score-to-risk table (recalibrated)

| Total points | Estimated risk of MUCH |
|---|---|
| 0 | 29.6% |
| 1 | 41.7% |
| 2 | 54.9% |
| 3 | 67.4% |
| 4 | 77.9% |

### Operating points

| Predicted risk | Zone |
|---|---|
| < 25% | Below referral threshold — ABPM not routinely indicated |
| 25% to < 50% | Shared-decision zone — discuss ABPM with the patient |
| ≥ 50% | ABPM referral recommended |

The 25% referral threshold and 25–50% shared-decision zone were specified by the analyst, reflecting the balance between missed MUCH diagnoses and negative ABPM tests, and limited test capacity.

## Derivation and validation (internal)

| Item | Value |
|---|---|
| Cohort | N = 113 complete cases, 67 events (MUCH prevalence 59.3%) |
| Events per variable | 15.33 |
| Model comparison | Model 1 (SBP) vs Model 2 (SBP + diabetes) vs Model 3 (+ number of medications); AUC differences not significant (DeLong) |
| Linearity check (SBP, RCS) | LR χ² = 0.34, p = 0.560 |
| Validation | Full-process Harrell bootstrap, 1000 resamples |
| Corrected AUC — integer score | 0.622 (apparent 0.638) |
| Corrected AUC — shrunken Model 2 | 0.625 (apparent 0.644) |
| Calibration slope — score / Model 2 | 1.014 / 0.910 |
| Corrected Brier — score / Model 2 | 0.2343 / 0.2326 |

Threshold diagnostics and decision-curve net benefit tables are shown inside the app (section 3).

## Limitations

- **Modest discrimination** (corrected AUC ≈ 0.62). Treat the output as a triage aid, not a diagnostic test.
- **The integer score cannot place a patient below the 25% referral threshold** — every attainable score maps to a risk ≥ 25% (score 0 = 29.6%). It ranks patients across the shared-decision and referral zones but cannot rule MUCH out. Only the continuous model can fall below 25%.
- **Small, single-centre, tertiary-referral cohort** with high baseline prevalence; PPV/NPV do not transport to other prevalences.
- **No external validation.**
- Inputs are restricted to controlled office SBP of 90–139 mmHg.
- The SBP score categories are coded as <120, 120 to <130, and ≥130 (the source report labels them "<120", "120-130", ">=130").

## Installation

Requires Python 3.9+.

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Usage

```bash
streamlit run TASH_MUCH_prediction_tool_Model2.py
```

The app opens at `http://localhost:8501`. Enter office SBP and diabetes status; the app returns:

1. The integer score, its estimated risk, the risk zone, and the full score-to-risk table.
2. The continuous Model 2 probability with a coefficient/contribution breakdown.
3. A validation dossier (performance metrics, threshold diagnostics, decision-curve net benefit).

### Deploying on Streamlit Community Cloud

1. Push this repository to GitHub.
2. In [share.streamlit.io](https://share.streamlit.io), choose **New app**, select the repository and branch, and set the main file to `TASH_MUCH_prediction_tool_Model2.py`.
3. Deploy. Dependencies are installed from `requirements.txt`.

## Repository structure

```
.
├── TASH_MUCH_prediction_tool_Model2.py   # Streamlit application
├── requirements.txt                      # Python dependencies
├── .gitignore
└── README.md
```

## Updating the model

All coefficients, score points, risk-table values and thresholds are defined as constants at the top of the script (`SHRUNK_*`, `PTS_*`, `SCORE_RISK`, `REFERRAL_THRESHOLD`, `SHARED_DECISION_UPPER`). If the model is refitted or externally recalibrated, update those constants and the validation text in section 3.

## Disclaimer

This tool is provided for research and educational purposes. It implements a model developed on a single tertiary-referral cohort and validated internally only. Results must be interpreted by a qualified clinician in the context of the full clinical picture and are **not** a substitute for 24-hour ambulatory blood pressure monitoring or clinical judgement. The authors accept no liability for clinical decisions made using this tool.

## Citation

If you use this tool, please cite the associated manuscript (details to be added on publication):

> Gebreselassie MA, et al. *[Manuscript title]*. [Journal, year].

## Author

Dr. Mulualem Alemayehu Gebreselassie, MD — Interventional Cardiologist, International Cardiovascular & Medical Centre (ICMC), Addis Ababa, Ethiopia.

## License

*Add a license before making the repository public* (e.g. MIT for open use, or a more restrictive licence if you plan to commercialise). Without a licence file, the code is all-rights-reserved by default.

