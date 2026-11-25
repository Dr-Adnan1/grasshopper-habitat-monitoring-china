"""
Evaluation: AUC on test, train metrics, ROC plot.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score,
    precision_score, recall_score, f1_score,
)

from src.config import FIGURES_DIR


SCALED_MODELS = {"MLP", "Logistic Regression"}


def _inputs_for(name, X_test, X_test_scaled):
    return X_test_scaled if name in SCALED_MODELS else X_test


def evaluate_all(search_results, X_train, X_test,
                 y_train, y_test, X_train_scaled, X_test_scaled):
    """
    Returns:
        auc_scores: {model: auc}
        train_metrics: {model: {metric: value}}
        probas: {model: y_test_proba}
    """
    auc_scores, train_metrics, probas = {}, {}, {}

    for name, search in search_results.items():
        model = search.best_estimator_

        X_tr = X_train_scaled if name in SCALED_MODELS else X_train
        X_te = _inputs_for(name, X_test, X_test_scaled)

        proba = model.predict_proba(X_te)[:, 1]
        y_tr_pred = model.predict(X_tr)

        auc_scores[name] = roc_auc_score(y_test, proba)
        train_metrics[name] = {
            "Accuracy":  accuracy_score(y_train, y_tr_pred),
            "Precision": precision_score(y_train, y_tr_pred, zero_division=0),
            "Recall":    recall_score(y_train, y_tr_pred, zero_division=0),
            "F1-Score":  f1_score(y_train, y_tr_pred, zero_division=0),
        }
        probas[name] = proba

    _plot_roc(probas, y_test, auc_scores)
    return auc_scores, train_metrics, probas


def _plot_roc(probas, y_test, auc_scores):
    fig, ax = plt.subplots(figsize=(10, 8))
    for name, proba in probas.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(fpr, tpr, label=f"{name}: (AUC={auc_scores[name]:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=2)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (All Models)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.4)
    out = FIGURES_DIR / "roc_curve_all_models.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ROC → {out}")