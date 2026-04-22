# Fraud Detection — Validation-Safe Classification Pipeline

A portfolio fraud detection prototype built on the [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). The project focuses on realistic model evaluation for rare-event classification: SQL-backed data access, chronological validation, imbalance handling, threshold tuning on validation data, and SHAP-based model interpretation.

---

## Project Structure

```
fraud-detection/
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory analysis & class imbalance
│   ├── 02_modeling.ipynb         # Time-aware split, model training, validation/test evaluation
│   └── 03_explainability.ipynb   # SHAP values and honest explanation summaries
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

- **Chronological split**: first 60% of transactions for training, next 20% for validation/model selection, final 20% held out for one final test evaluation
- **SMOTE** applied to training data only — never to validation or test data
- **`scale_pos_weight`** in XGBoost set from the training split's negative/positive ratio
- **Threshold tuning**: the decision threshold is selected on validation data, then frozen before final test evaluation. A cost table (£200 per missed fraud, £5 per blocked legitimate customer) quantifies the trade-off on validation data

### Models

| Model | Validation PR-AUC | Notes |
|---|---|---|
| Logistic Regression | 0.708 | Fast baseline |
| Random Forest | 0.744 | Non-linear benchmark |
| **XGBoost** | **0.788** | Selected model; SHAP-compatible |

**Final untouched test result, XGBoost at frozen validation threshold (0.954):** Precision 92% · Recall 73% · F1 0.81 · PR-AUC 0.797

---

## Explainability

The third notebook uses SHAP to inspect how the selected model behaves. Because `V1`-`V28` are anonymized PCA components, the project does **not** invent business meanings for them or claim customer-facing decline explanations.

- **Global feature importance** (mean |SHAP|) — which anonymized features drive predictions overall
- **Beeswarm plot** — direction and magnitude of each feature's impact across the dataset
- **Individual waterfall plots** — why a specific transaction was flagged or approved
- **Dependence plots** — feature interaction analysis
- **Explanation summaries** — compact summaries using actual dataset feature names such as `V14`, `V10`, and `Amount_log`

---

## Key Results

```
=== Final Test: XGBoost (frozen threshold=0.954) ===
              precision    recall  f1-score
  Legitimate       1.00      1.00      1.00
       Fraud       0.92      0.73      0.81

Test PR-AUC : 0.7967
Test ROC-AUC: 0.9743

Confusion matrix:
  Legit approved : 56,882
  Legit blocked  : 5
  Fraud missed   : 20
  Fraud caught   : 55
```

---

## Requirements

```
pandas · numpy · scikit-learn · imbalanced-learn · xgboost · shap · matplotlib · seaborn · joblib · jupyter
```

Python 3.9+
