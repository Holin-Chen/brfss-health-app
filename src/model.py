"""Phase 3 — XGBoost training, evaluation, and SHAP for BRFSS diabetes risk."""

import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)
from xgboost import XGBClassifier

FEATURES_PATH = "data/features.pkl"
MODEL_PATH = "models/xgb_diabetes.pkl"
SHAP_PLOT_PATH = "models/shap_summary.png"


def load_features() -> tuple[np.ndarray, np.ndarray, list[str]]:
    data = joblib.load(FEATURES_PATH)
    return data["X"], data["y"], data["feature_names"]


def train_model(X: np.ndarray, y: np.ndarray) -> XGBClassifier:
    """Train XGBoost with scale_pos_weight to handle class imbalance."""
    neg, pos = (y == 0).sum(), (y == 1).sum()
    scale_pos_weight = neg / pos
    print(f"scale_pos_weight: {scale_pos_weight:.2f}  (neg={neg:,}, pos={pos:,})")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    return model


def evaluate(model: XGBClassifier, X: np.ndarray, y: np.ndarray) -> None:
    """5-fold stratified cross-validation then full-fit evaluation."""
    print("\n--- 5-Fold Stratified Cross-Validation ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        model, X, y, cv=cv,
        scoring=["roc_auc", "precision", "recall"],
        n_jobs=-1,
    )
    print(f"ROC-AUC : {cv_results['test_roc_auc'].mean():.4f} ± {cv_results['test_roc_auc'].std():.4f}")
    print(f"Precision: {cv_results['test_precision'].mean():.4f} ± {cv_results['test_precision'].std():.4f}")
    print(f"Recall  : {cv_results['test_recall'].mean():.4f} ± {cv_results['test_recall'].std():.4f}")

    print("\n--- Full-Dataset Fit (for SHAP + saved model) ---")
    model.fit(X, y)

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    print(f"Train ROC-AUC: {roc_auc_score(y, y_prob):.4f}")
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=["Not at risk", "At risk"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))


def generate_shap(model: XGBClassifier, X: np.ndarray, feature_names: list[str]) -> None:
    """Generate global SHAP summary plot using a 5k sample."""
    print("\n--- SHAP Global Feature Importance ---")
    sample_idx = np.random.RandomState(42).choice(len(X), size=min(5000, len(X)), replace=False)
    X_sample = X[sample_idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(SHAP_PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP summary plot saved to {SHAP_PLOT_PATH}")

    # Top 10 features by mean |SHAP|
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:10]
    print("\nTop 10 features by mean |SHAP value|:")
    for rank, i in enumerate(top_idx, 1):
        print(f"  {rank:2d}. {feature_names[i]:<35} {mean_abs[i]:.4f}")


def shap_for_single(model: XGBClassifier, x_row: np.ndarray, feature_names: list[str]) -> dict:
    """Return SHAP values for a single prediction row (used by the app)."""
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(x_row.reshape(1, -1))[0]
    return dict(zip(feature_names, shap_vals))


def main() -> None:
    X, y, feature_names = load_features()
    print(f"Loaded features: X={X.shape}, y={y.shape}, features={len(feature_names)}")

    model = train_model(X, y)
    evaluate(model, X, y)
    generate_shap(model, X, feature_names)

    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
