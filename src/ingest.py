"""
ingest.py
---------
Load the raw Telco Customer Churn CSV and perform an initial
data-quality audit: shape, dtypes, missing values, class balance.
"""

import pandas as pd
import numpy as np
import os

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "telco_raw.csv")


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    """Read the raw CSV and return a DataFrame."""
    df = pd.read_csv(path)
    print(f"[ingest] Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def audit(df: pd.DataFrame) -> None:
    """Print a structured data-quality report."""
    print("\n── SHAPE ─────────────────────────────────────")
    print(f"  {df.shape[0]:,} rows  ×  {df.shape[1]} columns")

    print("\n── DTYPES ────────────────────────────────────")
    print(df.dtypes.to_string())

    print("\n── MISSING VALUES ────────────────────────────")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("  No null values found (check for disguised nulls below)")
    else:
        print(missing.to_string())

    # Check for space-only strings that mask missing values
    print("\n── DISGUISED NULLS (whitespace strings) ──────")
    for col in df.select_dtypes(include="object").columns:
        n = (df[col].str.strip() == "").sum()
        if n:
            print(f"  {col}: {n} blank strings")

    print("\n── TARGET DISTRIBUTION ───────────────────────")
    vc = df["Churn"].value_counts()
    pct = df["Churn"].value_counts(normalize=True) * 100
    print(pd.DataFrame({"count": vc, "pct": pct.round(1)}).to_string())

    print("\n── NUMERIC SUMMARY ───────────────────────────")
    print(df.describe().T.to_string())


if __name__ == "__main__":
    df = load_raw()
    audit(df)
