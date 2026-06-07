"""
train.py
--------
Two-stage ML pipeline:

  Stage 1 — Regression
    Predict a continuous churn risk score using a GradientBoostingRegressor.
    Target: Churn (0/1) treated as a probability to regress.
    Evaluation: MAE, RMSE, R².
    This score represents how 'churn-like' a customer is.

  Stage 2 — Classification
    Predict binary churn using an XGBoost classifier with SMOTE oversampling
    to handle class imbalance (~26% churn rate).
    Evaluation: ROC-AUC, F1, Precision, Recall, full classification report.

Both models are:
  - Tuned with GridSearchCV (5-fold stratified CV)
  - Persisted to /models/ as .pkl files
  - Evaluated on a held-out 20% test set
"""

import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
REG_PATH   = os.path.join(MODELS_DIR, "churn_risk_regressor.pkl")
CLF_PATH   = os.path.join(MODELS_DIR, "churn_classifier.pkl")

RANDOM_STATE = 42


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    return train_test_split(X, y, test_size=test_size,
                            random_state=RANDOM_STATE, stratify=y)


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 1 — REGRESSION (Churn Risk Score)
# ─────────────────────────────────────────────────────────────────────────────

def train_regressor(X_train, y_train):
    """
    Gradient Boosting Regressor on binary target treated as continuous.
    Hyperparameters tuned via GridSearchCV.
    """
    print("\n══ STAGE 1: CHURN RISK REGRESSOR ══════════════════════════════")

    param_grid = {
        "n_estimators":    [100, 200],
        "max_depth":       [3, 4],
        "learning_rate":   [0.05, 0.1],
        "subsample":       [0.8, 1.0],
    }

    base = GradientBoostingRegressor(random_state=RANDOM_STATE)
    cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Note: StratifiedKFold requires integer y; we pass y as int
    gs = GridSearchCV(base, param_grid, cv=cv, scoring="r2",
                      n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train.astype(int))

    print(f"  Best params : {gs.best_params_}")
    print(f"  CV R²       : {gs.best_score_:.4f}")

    return gs.best_estimator_


def evaluate_regressor(model, X_test, y_test):
    preds = model.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    r2    = r2_score(y_test, preds)

    print(f"\n  ── Test-set metrics ──────────────────────")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2:.4f}")

    return {"mae": mae, "rmse": rmse, "r2": r2}


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 2 — CLASSIFICATION (Binary Churn Prediction)
# ─────────────────────────────────────────────────────────────────────────────

def train_classifier(X_train, y_train):
    """
    XGBoost classifier with SMOTE oversampling.
    Optimised for ROC-AUC via GridSearchCV.
    """
    print("\n══ STAGE 2: CHURN CLASSIFIER ═══════════════════════════════════")

    # Oversample minority class (Churn=1) to balance training set
    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"  After SMOTE: {pd.Series(y_res).value_counts().to_dict()}")

    param_grid = {
        "n_estimators":    [100, 200],
        "max_depth":       [3, 5],
        "learning_rate":   [0.05, 0.1],
        "subsample":       [0.8, 1.0],
        "colsample_bytree":[0.8, 1.0],
    }

    base = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    gs = GridSearchCV(base, param_grid, cv=cv, scoring="roc_auc",
                      n_jobs=-1, verbose=0)
    gs.fit(X_res, y_res)

    print(f"  Best params : {gs.best_params_}")
    print(f"  CV ROC-AUC  : {gs.best_score_:.4f}")

    return gs.best_estimator_


def evaluate_classifier(model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    auc  = roc_auc_score(y_test, proba)
    f1   = f1_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec  = recall_score(y_test, preds)

    print(f"\n  ── Test-set metrics ──────────────────────")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"\n  Classification Report:\n")
    print(classification_report(y_test, preds, target_names=["No Churn", "Churn"]))

    print(f"  Confusion Matrix:\n  {confusion_matrix(y_test, preds)}")

    return {"roc_auc": auc, "f1": f1, "precision": prec, "recall": rec}


# ─────────────────────────────────────────────────────────────────────────────
#  FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

def feature_importance(model, feature_names: list, top_n: int = 15) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        fi = pd.Series(model.feature_importances_, index=feature_names)
        fi = fi.sort_values(ascending=False).head(top_n)
        print(f"\n  ── Top {top_n} Features ───────────────────────")
        print(fi.to_string())
        return fi.reset_index().rename(columns={"index": "feature", 0: "importance"})
    return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_training(X: pd.DataFrame, y: pd.Series):
    os.makedirs(MODELS_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test = split(X, y)
    print(f"[train] Train: {X_train.shape} | Test: {X_test.shape}")

    # ── Regressor ─────────────────────────────────────────────────────────
    reg = train_regressor(X_train, y_train)
    reg_metrics = evaluate_regressor(reg, X_test, y_test)
    feature_importance(reg, X.columns.tolist())
    with open(REG_PATH, "wb") as f:
        pickle.dump(reg, f)
    print(f"\n  Regressor saved → {REG_PATH}")

    # ── Classifier ────────────────────────────────────────────────────────
    clf = train_classifier(X_train, y_train)
    clf_metrics = evaluate_classifier(clf, X_test, y_test)
    feature_importance(clf, X.columns.tolist())
    with open(CLF_PATH, "wb") as f:
        pickle.dump(clf, f)
    print(f"\n  Classifier saved → {CLF_PATH}")

    return reg, clf, reg_metrics, clf_metrics, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from ingest import load_raw
    from preprocess import run_pipeline
    from features import build_features

    df_raw = load_raw()
    X, y, _, _ = run_pipeline(df_raw, save=False)
    X_eng = build_features(X)
    run_training(X_eng, y)
