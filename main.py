"""
main.py
-------
End-to-end pipeline runner.
Executes all stages in sequence:
  1. Ingest raw data
  2. Preprocess
  3. Feature engineering
  4. Train (regression + classification)
  5. Evaluate and generate report plots

Usage:
  python src/main.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from ingest    import load_raw, audit
from preprocess import run_pipeline
from features  import build_features, save_features
from train     import run_training
from evaluate  import run_all_plots


def main():
    print("=" * 60)
    print("  TELCO CUSTOMER CHURN — END-TO-END ML PIPELINE")
    print("=" * 60)

    # 1. Ingest
    print("\n[1/5] INGESTING DATA")
    df_raw = load_raw()
    audit(df_raw)

    # 2. Preprocess
    print("\n[2/5] PREPROCESSING")
    X, y, _, scaler = run_pipeline(df_raw, save=True)

    # 3. Feature engineering
    print("\n[3/5] FEATURE ENGINEERING")
    X_eng = build_features(X)
    save_features(X_eng, y)

    # 4. Train
    print("\n[4/5] TRAINING MODELS")
    reg, clf, reg_metrics, clf_metrics, X_train, X_test, y_train, y_test = \
        run_training(X_eng, y)

    # 5. Evaluate
    print("\n[5/5] GENERATING EVALUATION PLOTS")
    run_all_plots(X_eng, y, reg, clf, X_test, y_test)

    # Summary
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE — SUMMARY")
    print("=" * 60)
    print(f"\n  Stage 1 — Regression (Churn Risk Score)")
    print(f"    MAE  : {reg_metrics['mae']:.4f}")
    print(f"    RMSE : {reg_metrics['rmse']:.4f}")
    print(f"    R²   : {reg_metrics['r2']:.4f}")
    print(f"\n  Stage 2 — Classification (Binary Churn)")
    print(f"    ROC-AUC   : {clf_metrics['roc_auc']:.4f}")
    print(f"    F1        : {clf_metrics['f1']:.4f}")
    print(f"    Precision : {clf_metrics['precision']:.4f}")
    print(f"    Recall    : {clf_metrics['recall']:.4f}")
    print(f"\n  Outputs:")
    print(f"    data/telco_processed.csv")
    print(f"    data/telco_features.csv")
    print(f"    models/churn_risk_regressor.pkl")
    print(f"    models/churn_classifier.pkl")
    print(f"    models/scaler.pkl")
    print(f"    reports/  (7 visualisation plots)")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
