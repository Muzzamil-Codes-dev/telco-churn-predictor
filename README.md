# 📉 Telco Customer Churn — End-to-End ML Pipeline

A production-style machine learning project demonstrating the full data science lifecycle: from raw data ingestion and cleaning, through feature engineering, to a two-stage predictive model (regression + classification) with comprehensive evaluation.

---

## 🎯 Problem Statement

Customer churn — the loss of subscribers to competitors — is a critical commercial problem for telecoms businesses. Acquiring a new customer costs **5–7× more** than retaining an existing one. This project builds a system that:

1. **Scores** each customer with a continuous churn risk value (0–1)
2. **Classifies** customers as likely churners or not
3. **Explains** which factors drive churn risk, enabling targeted retention campaigns

**Dataset:** IBM Telco Customer Churn — 7,043 customers, 21 features

---

## 📊 Results

| Model | Metric | Score |
|-------|--------|-------|
| GBM Risk Regressor | R² | 0.308 |
| GBM Risk Regressor | RMSE | 0.367 |
| XGBoost Classifier | **ROC-AUC** | **0.823** |
| XGBoost Classifier | F1 | 0.597 |
| XGBoost Classifier | Recall | 0.655 |

> A ROC-AUC of 0.823 means the model correctly ranks a random churner above a random non-churner 82% of the time — substantially better than the 0.5 random baseline.

---

## 🗂️ Project Structure

```
telco-churn-ml/
├── data/
│   ├── telco_raw.csv              # Raw IBM dataset
│   ├── telco_processed.csv        # After cleaning & encoding
│   └── telco_features.csv         # After feature engineering
│
├── notebook/
│   └── telco_churn_analysis.ipynb # Full narrative walkthrough
│
├── src/
│   ├── ingest.py                  # Data loading & quality audit
│   ├── preprocess.py              # Cleaning, encoding, scaling
│   ├── features.py                # Feature engineering (10 features)
│   ├── train.py                   # Model training & tuning
│   ├── evaluate.py                # Visualisations & metrics
│   └── main.py                    # End-to-end pipeline runner
│
├── models/
│   ├── churn_risk_regressor.pkl   # Fitted GBM regressor
│   ├── churn_classifier.pkl       # Fitted XGBoost classifier
│   └── scaler.pkl                 # Fitted RobustScaler
│
├── reports/                       # Generated visualisation plots
├── requirements.txt
└── README.md
```

---

## 🔬 Methodology

### Stage 1 — Preprocessing

| Step | Method | Rationale |
|------|--------|-----------|
| Missing values | Median imputation | Right-skewed TotalCharges; median is outlier-robust |
| Binary encoding | 0/1 integer mapping | Yes/No and service columns |
| Nominal encoding | One-hot (drop_first) | InternetService, Contract, PaymentMethod |
| Feature scaling | **RobustScaler** | Preferred over StandardScaler for skewed billing distributions |

### Stage 2 — Feature Engineering (10 features)

| Feature | Business Logic |
|---------|---------------|
| `avg_monthly_spend` | TotalCharges ÷ (tenure + ε) — removes new-customer distortion |
| `spend_vs_expected` | MonthlyCharges − avg_monthly_spend — recent upgrade/downgrade signal |
| `n_services` | Count of active add-ons — switching cost proxy |
| `tenure_x_contract_monthly` | Interaction term: new customer × month-to-month — highest risk combination |
| `charges_per_service` | Cost-per-service — overpaying signal |
| `has_security_bundle` | Bundle ownership — engagement proxy |
| `payment_auto` | Auto-pay flag — cancellation friction |
| + 3 more | See `src/features.py` for full documentation |

### Stage 3 — Modelling

**Stage 1 (Regression):** `GradientBoostingRegressor` with 5-fold CV GridSearch on R²
- Binary target treated as continuous to produce calibrated risk scores
- Useful for *ranking* customers by risk — not just binary prediction

**Stage 2 (Classification):** `XGBClassifier` with SMOTE oversampling
- SMOTE applied to training set only (never test set) to correct 73/26 class imbalance
- 5-fold stratified CV GridSearch on ROC-AUC
- XGBoost: industry-standard gradient boosting, robust on tabular data

---

## 🔑 Key Findings

1. **Contract type** is the single most predictive feature — two-year contract customers churn at a fraction of the rate of month-to-month customers
2. **Fibre optic internet** customers churn more, likely due to higher monthly costs and competitive alternatives
3. **Electronic check payment** (non-automatic) is a strong churn signal — low-friction customers are easier to lose
4. **Engineered interaction feature** `tenure_x_contract_monthly` was the top feature in the regression stage, validating the feature engineering investment

---

## ⚙️ Running the Project

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the full pipeline
```bash
python src/main.py
```

### Or run the notebook
```bash
jupyter notebook notebooks/telco_churn_analysis.ipynb
```

The pipeline will:
1. Load and audit the raw data
2. Preprocess and encode features
3. Engineer 10 domain features
4. Train and tune both models
5. Generate 7 evaluation plots in `/reports/`

---

## 📦 Dependencies

```
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
matplotlib
seaborn
jupyter
```

---

## 📈 Sample Visualisations

All plots are saved to `/reports/` after running the pipeline:

- `1_class_balance.png` — Target distribution
- `2_correlation_heatmap.png` — Feature correlations
- `3_feature_distributions.png` — Churners vs non-churners
- `4_risk_score_distribution.png` — Stage 1 output distribution
- `5_roc_curve.png` — Stage 2 ROC-AUC curve
- `6_confusion_matrix.png` — Stage 2 confusion matrix
- `7_feature_importance.png` — XGBoost feature importances

---

## 🧠 Skills Demonstrated

- **Data wrangling**: dtype fixing, disguised null detection, pandas pipeline design
- **Preprocessing**: RobustScaler, one-hot encoding, binary encoding, median imputation
- **Feature engineering**: domain-motivated features, interaction terms, ratio features
- **Class imbalance**: SMOTE oversampling with correct train/test isolation
- **Hyperparameter tuning**: GridSearchCV with stratified k-fold cross-validation
- **Model evaluation**: ROC-AUC, F1, precision/recall, confusion matrix
- **Software engineering**: modular src/ layout, reproducible pipeline, persisted artefacts

---

*Built with Python 3.12 · scikit-learn · XGBoost · pandas · matplotlib*
