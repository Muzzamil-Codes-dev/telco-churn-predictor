"""
preprocess.py
-------------
Clean the raw Telco dataframe:
  1. Fix TotalCharges dtype (contains " " for new customers with 0 tenure)
  2. Impute the resulting NaNs with the median
  3. Drop the non-informative customerID column
  4. Encode the binary target: Churn → 1, No Churn → 0
  5. Encode binary yes/no features as 0/1
  6. One-hot encode remaining categorical features
  7. Scale continuous features with RobustScaler (handles outliers better
     than StandardScaler for skewed billing distributions)

Returns X (features) and y (target) as numpy arrays alongside
a processed DataFrame and the fitted scaler for later inference.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import os
import pickle

PROCESSED_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "telco_processed.csv")
SCALER_PATH    = os.path.join(os.path.dirname(__file__), "..", "models", "scaler.pkl")


# ── Binary columns that are Yes/No strings ────────────────────────────────────
BINARY_COLS = [
    "Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn",
]

# Service columns that have 'No internet service' / 'No phone service' in addition to Yes/No
SERVICE_COLS = [
    "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]

# Nominal categoricals that need one-hot encoding
OHE_COLS = ["InternetService", "Contract", "PaymentMethod"]

# Continuous numeric features to scale
CONTINUOUS_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix dtypes and drop uninformative columns."""
    df = df.copy()

    # Drop ID column — no predictive value
    df.drop(columns=["customerID"], inplace=True)

    # Encode gender as binary (Male=1, Female=0)
    df["gender"] = (df["gender"] == "Male").astype(int)

    # TotalCharges is object because new customers have ' ' instead of 0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Impute with median (only ~11 rows affected)
    median_tc = df["TotalCharges"].median()
    n_imputed = df["TotalCharges"].isna().sum()
    df["TotalCharges"] = df["TotalCharges"].fillna(median_tc)
    print(f"[preprocess] TotalCharges: imputed {n_imputed} NaN(s) with median={median_tc:.2f}")

    return df


def encode_binaries(df: pd.DataFrame) -> pd.DataFrame:
    """Map Yes → 1, No → 0 for binary string columns."""
    df = df.copy()
    for col in BINARY_COLS:
        df[col] = (df[col] == "Yes").astype(int)
    return df


def encode_services(df: pd.DataFrame) -> pd.DataFrame:
    """
    Service features have three values: 'Yes', 'No', and
    'No internet/phone service'.  Map Yes → 1, everything else → 0.
    This is semantically correct: the service is either active or it isn't.
    """
    df = df.copy()
    for col in SERVICE_COLS:
        df[col] = (df[col] == "Yes").astype(int)
    return df


def one_hot(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode nominal categoricals, drop first to avoid multicollinearity."""
    df = pd.get_dummies(df, columns=OHE_COLS, drop_first=True, dtype=int)
    print(f"[preprocess] After OHE: {df.shape[1]} columns")
    return df


def scale_continuous(df: pd.DataFrame, fit: bool = True, scaler: RobustScaler = None):
    """
    Apply RobustScaler to continuous features.
    If fit=True, fits a new scaler and saves it to disk.
    If fit=False, expects a pre-fitted scaler to be passed in.
    """
    df = df.copy()
    if fit:
        scaler = RobustScaler()
        df[CONTINUOUS_COLS] = scaler.fit_transform(df[CONTINUOUS_COLS])
        os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)
        print(f"[preprocess] Scaler fitted and saved → {SCALER_PATH}")
    else:
        df[CONTINUOUS_COLS] = scaler.transform(df[CONTINUOUS_COLS])
    return df, scaler


def run_pipeline(df: pd.DataFrame, save: bool = True):
    """
    Execute the full preprocessing pipeline.

    Returns
    -------
    X : pd.DataFrame   feature matrix
    y : pd.Series      binary target
    df_processed : pd.DataFrame  full processed frame
    scaler : fitted RobustScaler
    """
    df = clean(df)
    df = encode_binaries(df)
    df = encode_services(df)
    df = one_hot(df)
    df, scaler = scale_continuous(df, fit=True)

    y = df.pop("Churn")
    X = df

    print(f"[preprocess] Final feature matrix: {X.shape}")
    print(f"[preprocess] Target balance: {y.value_counts().to_dict()}")

    if save:
        os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
        full = X.copy()
        full["Churn"] = y
        full.to_csv(PROCESSED_PATH, index=False)
        print(f"[preprocess] Saved → {PROCESSED_PATH}")

    return X, y, df, scaler


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from ingest import load_raw
    df_raw = load_raw()
    X, y, df_proc, scaler = run_pipeline(df_raw)
    print("\nSample processed features:")
    print(X.head(3).to_string())
