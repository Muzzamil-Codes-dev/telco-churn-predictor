"""
features.py
-----------
Domain-driven feature engineering on top of the preprocessed dataframe.
Each new feature is motivated by a business hypothesis about churn behaviour.

New features created
--------------------
1.  avg_monthly_spend          — TotalCharges / (tenure + 1)
                                  Captures actual average spend; avoids div-by-zero
                                  for brand-new customers.

2.  spend_vs_expected          — MonthlyCharges − avg_monthly_spend
                                  Positive → customer recently upgraded (upsell risk).
                                  Negative → customer has been downgrading (churn risk).

3.  n_services                 — Sum of all active service add-ons.
                                  More services = higher switching cost = lower churn.

4.  service_density            — n_services / MonthlyCharges (pre-scale)
                                  Value-for-money signal: high services at low cost
                                  → satisfied customer.

5.  is_long_tenure             — 1 if tenure > 24 months (2 years).
                                  Loyalty indicator; tenure is non-linear w.r.t. churn.

6.  is_new_customer            — 1 if tenure ≤ 3 months.
                                  New customers have the highest churn risk.

7.  tenure_x_contract_monthly  — Interaction: short-tenure × month-to-month contract.
                                  High-risk combination.

8.  charges_per_service        — MonthlyCharges / (n_services + 1)
                                  Cost-per-service; high value may indicate overpaying.

9.  has_security_bundle        — 1 if both OnlineSecurity and DeviceProtection are active.
                                  Bundle ownership correlates with engagement.

10. payment_auto               — 1 if payment method is automatic (bank transfer / credit card).
                                  Auto-pay customers churn less (friction to cancel).

Note: all features are computed on the *already-processed* dataframe (post-OHE, post-scale
for continuous cols). The continuous features used here (tenure, MonthlyCharges,
TotalCharges) are in their *scaled* form, which is fine for all ratio/interaction features
because we are comparing within the same scaled space.
"""

import pandas as pd
import numpy as np
import os

FEATURES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "telco_features.csv")

# Service add-on column names (as they appear after preprocessing)
SERVICE_COLS = [
    "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]


def build_features(X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
    """
    Engineer domain features on top of the preprocessed feature matrix.

    Parameters
    ----------
    X : pd.DataFrame  — preprocessed feature matrix (no target column)
    y : pd.Series     — optional, appended to output if provided

    Returns
    -------
    X_eng : pd.DataFrame with all original + new engineered features
    """
    X = X.copy()

    # ── 1. Average monthly spend ───────────────────────────────────────────
    X["avg_monthly_spend"] = X["TotalCharges"] / (X["tenure"] + 1e-6)

    # ── 2. Spend vs expected ───────────────────────────────────────────────
    X["spend_vs_expected"] = X["MonthlyCharges"] - X["avg_monthly_spend"]

    # ── 3. Number of active services ──────────────────────────────────────
    X["n_services"] = X[SERVICE_COLS].sum(axis=1)

    # ── 4. Service density ────────────────────────────────────────────────
    X["service_density"] = X["n_services"] / (X["MonthlyCharges"].abs() + 1e-6)

    # ── 5 & 6. Tenure bands ───────────────────────────────────────────────
    # tenure is scaled; use raw rank-based threshold via quantile
    tenure_q33 = X["tenure"].quantile(0.33)
    tenure_q67 = X["tenure"].quantile(0.67)
    X["is_long_tenure"]   = (X["tenure"] >= tenure_q67).astype(int)
    X["is_new_customer"]  = (X["tenure"] <= tenure_q33).astype(int)

    # ── 7. Interaction: new customer on month-to-month contract ───────────
    # OHE creates 'Contract_One year' and 'Contract_Two year'; absence = month-to-month
    if "Contract_One year" in X.columns and "Contract_Two year" in X.columns:
        is_monthly = ((X["Contract_One year"] == 0) & (X["Contract_Two year"] == 0)).astype(int)
    else:
        is_monthly = pd.Series(0, index=X.index)
    X["tenure_x_contract_monthly"] = X["is_new_customer"] * is_monthly

    # ── 8. Charges per service ────────────────────────────────────────────
    X["charges_per_service"] = X["MonthlyCharges"] / (X["n_services"] + 1)

    # ── 9. Security bundle ────────────────────────────────────────────────
    X["has_security_bundle"] = (
        (X["OnlineSecurity"] == 1) & (X["DeviceProtection"] == 1)
    ).astype(int)

    # ── 10. Auto payment ──────────────────────────────────────────────────
    auto_cols = [c for c in X.columns if "PaymentMethod" in c and
                 ("Bank transfer" in c or "Credit card" in c)]
    if auto_cols:
        X["payment_auto"] = X[auto_cols].max(axis=1)
    else:
        X["payment_auto"] = 0

    n_new = 10
    print(f"[features] Added {n_new} engineered features → {X.shape[1]} total features")

    return X


def save_features(X: pd.DataFrame, y: pd.Series) -> None:
    out = X.copy()
    out["Churn"] = y.values
    os.makedirs(os.path.dirname(FEATURES_PATH), exist_ok=True)
    out.to_csv(FEATURES_PATH, index=False)
    print(f"[features] Saved → {FEATURES_PATH}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from ingest import load_raw
    from preprocess import run_pipeline

    df_raw = load_raw()
    X, y, _, _ = run_pipeline(df_raw, save=False)
    X_eng = build_features(X, y)
    print(X_eng.describe().T[["mean", "std", "min", "max"]].to_string())
    save_features(X_eng, y)
