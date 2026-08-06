# Revised TASH Masked Uncontrolled Hypertension (MUCH) Prediction Tool (Model A)

This repository contains the official Streamlit implementation of **Model A ("Core Parsimonious Clinical Model")** for predicting Masked Uncontrolled Hypertension (MUCH) versus Sustained Controlled Hypertension (SCH) in treated hypertensive patients presenting with controlled office blood pressure (office BP < 140/90 mmHg, ISH 2020 criteria)[cite: 13, 14].

## 1. Model Equation & Coefficients
The authoritative multivariable logistic regression formula transcribed directly from SPSS LRA output[cite: 13, 14, 16]:

$$\text{logit}[P(\text{MUCH})] = -8.297 + 0.063 \times \text{Office SBP} + 0.782 \times \text{Diabetes} + 0.000 \times \text{MedCount} + 0.012 \times \text{Age}$$[cite: 13, 14, 16]

| Predictor Variable | Coefficient (B) | Std. Error | Wald | p-value | Odds Ratio (95% CI) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Office SBP (mmHg)** | +0.063 | 0.023 | 7.641 | 0.006 | 1.065 (1.018–1.113) |[cite: 13, 14, 16]
| **Diabetes (1=Yes, 0=No)** | +0.782 | 0.446 | 3.077 | 0.079 | 2.187 (0.912–5.242) |[cite: 13, 14, 16]
| **Medication Count** | +0.000 | 0.215 | 0.000 | 1.000 | 1.000 (0.656–1.524) |[cite: 13, 14, 16]
| **Age (years)** | +0.012 | 0.017 | 0.475 | 0.491 | 1.012 (0.979–1.045) |[cite: 13, 14, 16]
| **Constant** | -8.297 | 3.110 | 7.117 | 0.008 | — |[cite: 13, 14, 16]

## 2. Validation & Discrimination Summary
- **Analytic Cohort**: N = 129 complete cases (Tikur Anbessa Specialized Hospital, Addis Ababa, Ethiopia)[cite: 13, 14, 15].
- **Prevalence of MUCH**: 58.1%[cite: 13, 14, 15].
- **Apparent AUC**: 0.6604 (DeLong 95% CI: 0.5632–0.7575)[cite: 13, 14, 15].
- **Optimism-Corrected AUC (Headline Metric)**: 0.6146 (Harrell Bootstrap, B=1000)[cite: 13, 14, 15].
- **Calibration Slope**: 0.6893[cite: 13, 14, 15].
- **Brier Score**: 0.2405[cite: 13, 14, 15].

## 3. Local Setup & Execution
1. Clone repository:
   ```bash
   git clone [https://github.com/](https://github.com/)<YOUR_GITHUB_USERNAME>/Revised-TASH-MUCH-Prediction-Tool.git
   cd Revised-TASH-MUCH-Prediction-Tool