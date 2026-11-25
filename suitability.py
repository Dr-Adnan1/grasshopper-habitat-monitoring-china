"""
Low / Moderate / High suitability classification + CSV export.
"""
import numpy as np
import pandas as pd

from src.config import (
    SUITABILITY_LOW_MAX, SUITABILITY_MODERATE_MAX,
    TABLES_DIR, SCALED_MODELS,
)


def classify_suitability(probabilities):
    classes = np.empty(len(probabilities), dtype=object)
    classes[probabilities <= SUITABILITY_LOW_MAX] = "Low"
    classes[(probabilities > SUITABILITY_LOW_MAX) &
            (probabilities <= SUITABILITY_MODERATE_MAX)] = "Moderate"
    classes[probabilities > SUITABILITY_MODERATE_MAX] = "High"
    return classes


def export_suitability(probas, X_test, y_test, X_test_scaled,
                       prefix, name):
    """CSV with predicted probability + class per test point."""
    proba = probas[name]
    classes = classify_suitability(proba)

    base = X_test_scaled if name in SCALED_MODELS else X_test
    out = pd.DataFrame(base, columns=X_test.columns).reset_index(drop=True)
    out["true"]              = y_test.reset_index(drop=True)
    out["probability"]       = proba
    out["suitability_class"] = classes

    slug = name.replace(" ", "_").lower()
    out_path = TABLES_DIR / f"{prefix}{slug}_suitability.csv"
    out.to_csv(out_path, index=False)

    counts = pd.Series(classes).value_counts().to_dict()
    print(f"  {name}: {counts}  →  {out_path}")
    return counts