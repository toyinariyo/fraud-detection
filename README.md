# Fraud Detection — End-to-End Classification Pipeline

A production-oriented fraud detection system built on the [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). Designed to reflect real-world financial crime prevention workflows: SQL-backed data access, rigorous imbalance handling, threshold-tuned evaluation, and SHAP-based explainability for regulatory compliance.

---

## Project Structure

```
fraud-detection/
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory analysis & class imbalance
│   ├── 02_modeling.ipynb         # Model training, SMOTE, PR-curve evaluation
│   └── 03_explainability.ipynb   # SHAP values, waterfall plots, reason codes
├── data/                         # SQLite database (generated, not versioned)
├── models/                       # Saved model artifacts (generated, not versioned)
├── setup_db.py                   # One-time script: CSV → SQLite
└── requirements.txt
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place creditcard.csv in the project root, then build the database
python setup_db.py

# 3. Run notebooks in order
jupyter notebook notebooks/01_eda.ipynb
```

> The dataset is available from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). It is not included in this repository.

---

## Dataset

284,807 credit card transactions (September 2013, European cardholders).

| Feature | Description |
|---|---|
| `V1`–`V28` | PCA-transformed transaction attributes (privacy-preserving) |
| `Amount` | Transaction value |
| `Time` | Seconds elapsed since first transaction in dataset |
| `is_fraud` | Target label — `1` = fraud, `0` = legitimate |

**Class imbalance:** 492 fraud cases (0.17%) out of 284,807 transactions — 1 in every 579 transactions.

---

## Approach

### Why not accuracy?

With a 0.17% fraud rate, a classifier that labels everything as legitimate achieves **99.83% accuracy** and catches **zero fraud**. All models are evaluated on **Precision-Recall AUC** — the appropriate metric when the positive class is rare and costly to miss.

| Metric | Meaning in business terms |
|---|---|
| **Precision** | Of all transactions we block, what fraction are genuinely fraudulent? (false alarm rate) |
| **Recall** | Of all actual fraud cases, what fraction do we catch? (miss rate) |
| **F1** | Harmonic mean — balances precision and recall |
| **PR-AUC** | Area under the full precision-recall curve — summarises performance at all operating thresholds |

### Class Imbalance Handling

- **SMOTE** (Synthetic Minority Over-sampling) applied to training data only — never to the test set, which remains at the true 0.17% prevalence to simulate production conditions
- **`scale_pos_weight`** in XGBoost set to the negative/positive ratio (~577) to penalise missed fraud more heavily
- **Threshold tuning**: the decision threshold is treated as a business decision, not a statistical default. A cost table (£200 per missed fraud, £5 per blocked legitimate customer) quantifies the trade-off across all operating points

### Models

| Model | PR-AUC | Notes |
|---|---|---|
| Logistic Regression | 0.74 | Interpretable baseline; suitable for regulatory review |
| Random Forest | 0.83 | Captures non-linear patterns; robust to outliers |
| **XGBoost** | **0.88** | Best performance; SHAP-compatible |

**XGBoost at optimal threshold (0.74):** Precision 89% · Recall 84% · F1 0.863

---

## Explainability

Explainability is a first-class requirement in regulated financial environments. The third notebook provides:

- **Global feature importance** (mean |SHAP|) — which transaction characteristics drive fraud predictions overall
- **Beeswarm plot** — direction and magnitude of each feature's impact across the dataset
- **Individual waterfall plots** — why a specific transaction was flagged or approved (critical for fraud analyst triage and customer dispute handling)
- **Dependence plots** — feature interaction analysis, informing rule-based fallback logic
- **Reason code generator** — prototype converting SHAP values into human-readable decline codes, as required for Suspicious Activity Reports (SARs) and GDPR Article 22 compliance

---

## Key Results

```
=== XGBoost (threshold=0.74) ===
              precision    recall  f1-score
  Legitimate       1.00      1.00      1.00
       Fraud       0.89      0.84      0.86

PR-AUC : 0.8751
ROC-AUC: 0.9814
```

---

## Requirements

```
pandas · numpy · scikit-learn · imbalanced-learn · xgboost · shap · matplotlib · seaborn · joblib · jupyter
```

Python 3.9+
