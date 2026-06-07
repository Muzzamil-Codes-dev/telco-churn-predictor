"""
evaluate.py
-----------
Generate all evaluation visualisations and save them to /reports/:
  1.  Class imbalance bar chart
  2.  Correlation heatmap (top features)
  3.  Feature distributions — churners vs non-churners
  4.  Stage 1: Regression — predicted risk score distribution
  5.  Stage 2: ROC curve
  6.  Stage 2: Confusion matrix heatmap
  7.  Feature importance bar chart (classifier)
  8.  SHAP-style feature contribution summary (manual approximation)
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

from sklearn.metrics import roc_curve, auc, confusion_matrix

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
BLUE   = "#2563EB"
RED    = "#DC2626"
GREY   = "#6B7280"
BG     = "#F8FAFC"
DARK   = "#1E293B"

def _style():
    plt.rcParams.update({
        "figure.facecolor":  BG,
        "axes.facecolor":    BG,
        "axes.edgecolor":    "#CBD5E1",
        "axes.labelcolor":   DARK,
        "xtick.color":       DARK,
        "ytick.color":       DARK,
        "text.color":        DARK,
        "font.family":       "DejaVu Sans",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.color":        "#E2E8F0",
        "grid.linewidth":    0.6,
    })


# ── 1. Class imbalance ────────────────────────────────────────────────────────

def plot_class_balance(y: pd.Series):
    _style()
    fig, ax = plt.subplots(figsize=(6, 4))
    vc = y.value_counts()
    colors = [BLUE, RED]
    bars = ax.bar(["No Churn", "Churn"], vc.values, color=colors, width=0.5, zorder=3)
    for bar, v in zip(bars, vc.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                f"{v:,}\n({v/len(y)*100:.1f}%)", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=DARK)
    ax.set_title("Target Class Distribution", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Count")
    ax.set_ylim(0, vc.max() * 1.2)
    fig.tight_layout()
    path = os.path.join(REPORTS_DIR, "1_class_balance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Saved → {path}")


# ── 2. Correlation heatmap ────────────────────────────────────────────────────

def plot_correlation(X: pd.DataFrame, y: pd.Series, top_n: int = 16):
    _style()
    df = X.copy()
    df["Churn"] = y.values

    corr = df.corr()["Churn"].drop("Churn").abs().sort_values(ascending=False)
    top_features = corr.head(top_n).index.tolist() + ["Churn"]
    sub_corr = df[top_features].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(sub_corr, dtype=bool))
    sns.heatmap(sub_corr, mask=mask, annot=True, fmt=".2f", ax=ax,
                cmap="RdBu_r", center=0, linewidths=0.5,
                annot_kws={"size": 8}, cbar_kws={"shrink": 0.8})
    ax.set_title(f"Correlation Matrix — Top {top_n} Features vs Churn",
                 fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    path = os.path.join(REPORTS_DIR, "2_correlation_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Saved → {path}")


# ── 3. Feature distributions ──────────────────────────────────────────────────

def plot_distributions(X: pd.DataFrame, y: pd.Series):
    _style()
    features = ["tenure", "MonthlyCharges", "TotalCharges",
                "n_services", "avg_monthly_spend"]
    features = [f for f in features if f in X.columns]

    fig, axes = plt.subplots(1, len(features), figsize=(4 * len(features), 4))
    df = X.copy()
    df["Churn"] = y.values

    for ax, feat in zip(axes, features):
        for label, color, name in [(0, BLUE, "No Churn"), (1, RED, "Churn")]:
            vals = df.loc[df["Churn"] == label, feat].dropna()
            ax.hist(vals, bins=30, alpha=0.6, color=color, label=name, density=True)
        ax.set_title(feat, fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        ax.legend(fontsize=8)

    fig.suptitle("Feature Distributions: Churners vs Non-Churners",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(REPORTS_DIR, "3_feature_distributions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Saved → {path}")


# ── 4. Regression — risk score distribution ───────────────────────────────────

def plot_risk_scores(reg, X_test, y_test):
    _style()
    scores = reg.predict(X_test)
    df = pd.DataFrame({"score": scores, "Churn": y_test.values})

    fig, ax = plt.subplots(figsize=(7, 4))
    for label, color, name in [(0, BLUE, "No Churn"), (1, RED, "Churn")]:
        vals = df.loc[df["Churn"] == label, "score"]
        ax.hist(vals, bins=40, alpha=0.65, color=color, label=name, density=True)

    ax.axvline(0.5, color=GREY, linestyle="--", linewidth=1.5, label="Threshold 0.5")
    ax.set_xlabel("Predicted Churn Risk Score")
    ax.set_ylabel("Density")
    ax.set_title("Stage 1 — Predicted Risk Score Distribution", fontsize=13,
                 fontweight="bold", pad=12)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(REPORTS_DIR, "4_risk_score_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Saved → {path}")


# ── 5. ROC curve ──────────────────────────────────────────────────────────────

def plot_roc(clf, X_test, y_test):
    _style()
    proba = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color=BLUE, lw=2.5, label=f"ROC AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color=GREY, lw=1.5, linestyle="--", label="Random")
    ax.fill_between(fpr, tpr, alpha=0.08, color=BLUE)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("Stage 2 — ROC Curve (XGBoost Classifier)", fontsize=13,
                 fontweight="bold", pad=12)
    ax.legend(fontsize=11)
    fig.tight_layout()
    path = os.path.join(REPORTS_DIR, "5_roc_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Saved → {path}")


# ── 6. Confusion matrix ───────────────────────────────────────────────────────

def plot_confusion_matrix(clf, X_test, y_test):
    _style()
    proba = clf.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    cm    = confusion_matrix(y_test, preds)

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d", ax=ax,
                cmap="Blues", linewidths=1,
                xticklabels=["Pred: No Churn", "Pred: Churn"],
                yticklabels=["True: No Churn", "True: Churn"],
                annot_kws={"size": 14, "weight": "bold"})
    ax.set_title("Stage 2 — Confusion Matrix", fontsize=13,
                 fontweight="bold", pad=12)
    fig.tight_layout()
    path = os.path.join(REPORTS_DIR, "6_confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Saved → {path}")


# ── 7. Feature importance ─────────────────────────────────────────────────────

def plot_feature_importance(clf, feature_names: list, top_n: int = 20):
    _style()
    fi = pd.Series(clf.feature_importances_, index=feature_names)
    fi = fi.sort_values(ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = [RED if i >= len(fi) - 5 else BLUE for i in range(len(fi))]
    ax.barh(fi.index, fi.values, color=colors, zorder=3)
    ax.set_xlabel("Feature Importance (Gain)", fontsize=11)
    ax.set_title(f"Stage 2 — Top {top_n} Feature Importances (XGBoost)",
                 fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    path = os.path.join(REPORTS_DIR, "7_feature_importance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Saved → {path}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run_all_plots(X, y, reg, clf, X_test, y_test):
    plot_class_balance(y)
    plot_correlation(X, y)
    plot_distributions(X, y)
    plot_risk_scores(reg, X_test, y_test)
    plot_roc(clf, X_test, y_test)
    plot_confusion_matrix(clf, X_test, y_test)
    plot_feature_importance(clf, X.columns.tolist())
    print(f"\n[evaluate] All plots saved to {REPORTS_DIR}/")


if __name__ == "__main__":
    import sys, pickle
    sys.path.insert(0, os.path.dirname(__file__))
    from ingest import load_raw
    from preprocess import run_pipeline
    from features import build_features
    from train import run_training

    df_raw = load_raw()
    X, y, _, _ = run_pipeline(df_raw, save=False)
    X_eng = build_features(X)
    reg, clf, _, _, X_train, X_test, y_train, y_test = run_training(X_eng, y)
    run_all_plots(X_eng, y, reg, clf, X_test, y_test)
