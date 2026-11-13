# Predicting-30-Day-Hospital-Readmission-in-Diabetic-Patients
An end-to-end machine learning project that predicts which diabetic inpatients are likely to be readmitted within 30 days of discharge, and translates the predictions into hospital-ready risk tiers and a cost/ROI estimate.

**Author:** Shubhi | BDS, MBA (Healthcare Administration), IIHMR University
**Tech:** Python, pandas, scikit-learn, imbalanced-learn (SMOTE), matplotlib, Jupyter

---

## Overview

Thirty-day readmission is a core hospital quality metric and a driver of reimbursement penalties (e.g., CMS Hospital Readmissions Reduction Program). This project builds a risk-stratification model so a case-management team can focus limited follow-up capacity (calls, home-health referrals, medication reconciliation) on the patients most likely to bounce back.

It operationalizes the research question from:

> Strack, B., DeShazo, J.P., Gennings, C., Olmo Ortiz, J.L., Ventura, S., Cios, K.J., & Clore, J. (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records.* BioMed Research International, 2014:781670.

## Dataset

- **Source:** [Diabetes 130-US Hospitals for Years 1999-2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) (UCI Machine Learning Repository)
- **Size:** 101,766 encounters, 50 columns, 130 US hospitals
- **Target:** `readmitted_30d` = 1 if `readmitted == "<30"`, else 0
- **Base rate:** 11.2% of encounters were readmitted within 30 days

The raw CSV is not included in this repository. Download it from the link above and update the path in the notebook's data-loading cell.

## Approach

1. **Cleaning:** replaced `?` with missing values; built the binary 30-day target.
2. **Features:** 8 numeric utilization/clinical variables (e.g., `time_in_hospital`, `number_inpatient`, `number_emergency`, `num_medications`) and 9 categorical variables (age band, admission type/source, discharge disposition, glucose serum, HbA1c result, insulin, diabetes medication, medication change). 82 features after encoding.
3. **Fairness by design:** `race` and `gender` were deliberately excluded from the model inputs. The EDA audit showed small differences in readmission rates across groups (about 9.6% to 11.3% by race), and using demographics as direct inputs to a clinical risk score raises disparate-impact concerns.
4. **Preprocessing:** median/most-frequent imputation, standardization of numeric features, one-hot encoding of categoricals, stratified 75/25 train-test split (76,324 train / 25,442 test).
5. **Class imbalance:** SMOTE on the training set (76,324 to 135,612 rows, 50% positive).
6. **Models:** Logistic Regression and Random Forest on SMOTE-balanced data, plus a **class-weighted logistic regression with Platt-scaling calibration** for decision-facing probabilities. SMOTE improves ranking but distorts predicted probabilities, so the calibrated model powers the risk tiers.
7. **Evaluation:** ROC-AUC, precision/recall, confusion matrix at a 0.20 threshold (near the base rate rather than the default 0.5).

## Results

All results are on the held-out test set of 25,442 encounters.

### Model comparison (ROC-AUC)

| Model | ROC-AUC |
|---|---|
| Logistic Regression (SMOTE) | 0.665 |
| Random Forest (SMOTE) | 0.653 |
| **Calibrated Logistic Regression (selected)** | **0.672** |

Published studies on this dataset typically report AUCs of roughly 0.65 to 0.70 for 30-day readmission, so this is in the expected range for a problem that depends heavily on post-discharge factors not captured in the inpatient chart.

### Classification performance at threshold 0.20

| Metric | Not readmitted | Readmitted <30d |
|---|---|---|
| Precision | 0.901 | 0.295 |
| Recall | 0.949 | 0.169 |
| F1-score | 0.924 | 0.215 |

Overall accuracy is 86.2%, but accuracy is misleading with an 11% base rate. The model is best read as a **prioritization tool**, not a classifier that catches every readmission.

### Risk tiers (calibrated probabilities)

| Risk tier | Patients | Actual 30-day readmission rate |
|---|---|---|
| Low (<10%) | 14,044 | 6.7% |
| Medium (10-20%) | 9,766 | 14.6% |
| High (>20%) | 1,632 | 29.5% |

The High tier is readmitted at roughly **4.4x the rate of the Low tier** and about 2.6x the cohort average, which is what makes targeted follow-up worthwhile.

### Illustrative ROI (test cohort)

Assumptions (illustrative, not measured): $15,000 average cost per readmission, $300 per high-risk patient for a follow-up program, 20% relative reduction in readmissions among those targeted.

| Item | Value |
|---|---|
| High-risk patients flagged | 1,632 |
| Observed readmissions among them | 481 |
| Readmissions avoided (20% reduction) | ~96 |
| Gross savings | $1,443,000 |
| Program cost | $489,600 |
| **Net savings** | **$953,400** |

These figures show the *method* of building a business case. A real engagement would replace the assumptions with the client's own cost data and a piloted effect size.

## Repository structure

```
.
├── Diabetes_Readmission_Risk_Model.ipynb   # Main notebook (EDA, modeling, evaluation, ROI)
├── outputs/                                # Saved charts and model spec (ROC curve, calculator_spec.json)
└── README.md
```

## How to run

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install pandas numpy scikit-learn matplotlib imbalanced-learn jupyter
# Download diabetic_data.csv from UCI and update the path in the notebook's load cell
jupyter notebook Diabetes_Readmission_Risk_Model.ipynb
```

Create an `outputs/` folder before running if it does not exist, since the notebook saves figures and the model spec there.

## Limitations

- **Modest discriminative power:** AUC of 0.672 and 16.9% recall at the 0.20 threshold mean many readmissions are missed. A lower threshold would raise recall at the cost of more false alarms; the right trade-off depends on follow-up capacity.
- **Possible patient-level leakage:** the dataset contains multiple encounters per patient, and the train/test split is at the encounter level. A patient-grouped split would give a more conservative estimate.
- **Historical US data (1999-2008):** coding practices, care patterns, and populations differ from current Indian or modern US settings. Local recalibration and revalidation would be required.
- **Illustrative ROI:** cost, program cost, and effect-size figures are assumptions, not observed values.
- **Fairness:** excluding race and gender does not eliminate proxy effects (e.g., payer or admission source). A deployed model should be audited for subgroup error rates.
- **Decision support only:** outputs are a prioritization signal for human case managers, not an automated clinical or discharge decision. Any real deployment needs institutional review and data governance (HIPAA in the US; DPDP Act 2023 / ABDM in India).

## Skills demonstrated

Healthcare data cleaning and EDA, handling class imbalance (SMOTE), probability calibration (Platt scaling), model evaluation beyond accuracy, interpretability (odds ratios, feature importance), fairness-aware feature selection, and translating model output into a cost/ROI business case.

## Reference

Strack et al. (2014). *BioMed Research International*, 2014:781670. Dataset: Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). Diabetes 130-US Hospitals for Years 1999-2008. UCI Machine Learning Repository.
