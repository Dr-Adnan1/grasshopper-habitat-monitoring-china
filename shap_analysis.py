"""
Multi-model SHAP analysis (tree, linear, kernel explainers)
+ comparative feature-importance plots.
"""
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from src.config import SHAP_DIR, SCALED_MODELS


def _feature_names(X):
    return list(X.columns) if hasattr(X, "columns") else None


def _summary_plots(model_name, shap_vals, X_data, feature_names):
    slug = model_name.replace(" ", "_").lower()

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals, X_data,
                      feature_names=feature_names, show=False)
    plt.title(f"SHAP Summary — {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(SHAP_DIR / f"shap_summary_{slug}.png",
                dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals, X_data,
                      feature_names=feature_names, show=False,
                      plot_type="bar")
    plt.title(f"SHAP Feature Importance — {model_name}",
              fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(SHAP_DIR / f"shap_bar_{slug}.png",
                dpi=300, bbox_inches="tight")
    plt.close()


def run_shap_for_all(search_results, X_test, X_test_scaled, feature_names):
    """
    Returns dict: {model_name: importance_df}
    """
    results = {}
    for name, search in search_results.items():
        model = search.best_estimator_
        X_use = X_test_scaled if name in SCALED_MODELS else X_test

        print(f"\nSHAP → {name}")
        try:
            if name in {"Random Forest", "XGBoost"}:
                explainer = shap.TreeExplainer(model)
                sv = explainer.shap_values(X_use)
            elif name == "Logistic Regression":
                explainer = shap.LinearExplainer(model, X_use)
                sv = explainer.shap_values(X_use)
            else:  # MLP
                explainer = shap.KernelExplainer(
                    model.predict_proba, X_use[:100]
                )
                sv = explainer.shap_values(X_use[:200])

            sv_pos = sv[1] if isinstance(sv, list) else sv

            _summary_plots(name, sv_pos, X_use, feature_names)

            imp = (
                pd.DataFrame({
                    "feature": feature_names,
                    "shap_importance": np.abs(sv_pos).mean(axis=0),
                })
                .sort_values("shap_importance", ascending=False)
                .reset_index(drop=True)
            )
            slug = name.replace(" ", "_").lower()
            imp.to_csv(SHAP_DIR / f"shap_importance_{slug}.csv", index=False)

            print(f"  top-5: {imp.head(5).to_dict('records')}")
            results[name] = imp
        except Exception as e:
            print(f"  ⚠️  SHAP failed for {name}: {e}")
    return results