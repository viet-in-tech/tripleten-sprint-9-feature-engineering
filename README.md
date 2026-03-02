# Sprint 9 — Feature Engineering
### Beta Bank Customer Churn Prediction

**TripleTen Data Science Program**

**Completed:** January 24, 2026

---

## Project Overview

Beta Bank is losing customers. Research shows retaining existing customers is significantly cheaper than acquiring new ones. The goal of this project is to build a machine learning model that predicts whether a bank customer will churn, using historical behavioral data.

**Project requirement:** Achieve a minimum **F1 score ≥ 0.59** on the held-out test set.

---

## Dataset

- **Source:** `/datasets/Churn.csv`
- **Size:** 10,000 customers × 14 features
- **Target column:** `Exited` (0 = Stayed, 1 = Churned)
- **Class distribution:** 79.6% Stayed / 20.4% Churned (significant imbalance)
- **Notable issue:** 909 missing values in the `Tenure` column (9.1%)

---

## Workflow

### Phase 1 — Import Libraries & Load Data
- pandas, numpy, matplotlib, seaborn
- scikit-learn: model selection, preprocessing, classifiers, metrics

### Phase 2 — Exploratory Data Analysis
- Dataset shape, data types, statistical summary
- Missing value analysis
- Class distribution visualization

### Phase 3 — Data Preparation
- **Missing values:** `Tenure` filled with median (robust to outliers; missing values confirmed random)
- **Feature removal:** Dropped `RowNumber`, `CustomerId`, `Surname` (identifiers with no predictive value)
- **Encoding:** Binary encoding for `Gender`; one-hot encoding for `Geography`

### Phase 4 — Class Balance & Baseline Models
- Examined 4:1 class imbalance in the target variable
- Stratified 60/20/20 train/validation/test split
- Scaled features with `StandardScaler` (fit on train only)
- Trained baseline models with no imbalance correction:

| Model               | F1 Score | ROC-AUC |
|---------------------|----------|---------|
| Logistic Regression | 0.3190   | —       |
| Decision Tree       | 0.5573   | —       |
| Random Forest       | 0.5573   | —       |

### Phase 5 — Imbalance Handling & Model Improvement
Tested three techniques on Random Forest:

| Approach        | F1 Score | ROC-AUC | Passes Target |
|----------------|----------|---------|---------------|
| Class Weights   | 0.6381   | 0.8595  | ✅ Yes        |
| Upsampling      | 0.6173   | —       | ✅ Yes        |
| Downsampling    | 0.5986   | —       | ✅ Yes        |

**Selected model:** Random Forest with `class_weight='balanced'` (highest F1, no data loss)

**Hyperparameter tuning:** GridSearchCV (3-fold CV) over `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`

Best params: `max_depth=10`, `min_samples_leaf=2`, `min_samples_split=5`, `n_estimators=100`

---

## Final Test Results

| Metric    | Score  |
|-----------|--------|
| F1 Score  | 0.6197 |
| ROC-AUC   | 0.8618 |

**Target F1 ≥ 0.59 → PASSED** (margin: +0.0297)

---

## Technologies

- Python 3.8
- pandas, numpy
- matplotlib, seaborn
- scikit-learn (LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GridSearchCV, StandardScaler)

---

## Project Status

✅ Approved — TripleTen Data Science Program, Sprint 9
