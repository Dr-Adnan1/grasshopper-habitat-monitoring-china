"""
End-to-end pipeline for habitat suitability modeling.

Usage
-----
python -m scripts.run_pipeline \
    --raster-folder data/raw \
    --occurrence data/raw/raza_LL.xls \
    --prefix raza_
"""
import argparse
import joblib
import pandas as pd

from src.config import (
    DEFAULT_RASTER_FOLDER, DEFAULT_OCCURRENCE, DEFAULT_PREFIX,
    MODELS_DIR, TABLES_DIR,
)
from src.raster_io import load_multiple_rasters_aligned
from src.data_loader import build_training_table, prepare_xy, split_and_scale
from src.train import train_all
from src.evaluate import evaluate_all
from src.suitability import export_suitability
from src.shap_analysis import run_shap_for_all

try:
    import shap  # noqa: F401
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raster-folder", default=str(DEFAULT_RASTER_FOLDER))
    p.add_argument("--occurrence",   default=str(DEFAULT_OCCURRENCE))
    p.add_argument("--prefix",       default=DEFAULT_PREFIX)
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Rasters   : {args.raster_folder}")
    print(f"Occurrence: {args.occurrence}")

    # 1. Rasters
    raster_data, transform, crs, bounds, band_names = \
        load_multiple_rasters_aligned(args.raster_folder)
    print(f"Stacked shape: {raster_data.shape}  ({len(band_names)} bands)")

    # 2. Points + X / y
    env_vals, occ_df = build_training_table(
        raster_data, transform, crs, args.occurrence
    )
    X, y, feature_names = prepare_xy(env_vals, occ_df, band_names)
    print(f"X: {X.shape}   y-balance: {y.value_counts().to_dict()}")

    # 3. Split + scale
    (X_train, X_test, y_train, y_test,
     X_train_scaled, X_test_scaled, scaler) = split_and_scale(X, y)

    # 4. Train
    search_results = train_all(X_train, y_train, X_train_scaled)

    # 5. Evaluate
    auc_scores, train_metrics, probas = evaluate_all(
        search_results, X_train, X_test,
        y_train, y_test, X_train_scaled, X_test_scaled,
    )

    # 6. Suitability CSVs
    print("\nSuitability class distribution (test set):")
    for name in search_results:
        export_suitability(
            probas, X_test, y_test, X_test_scaled, args.prefix, name,
        )

    # 7. SHAP
    if HAS_SHAP:
        run_shap_for_all(search_results, X_test, X_test_scaled, feature_names)
    else:
        print("SHAP not available — skipping.")

    # 8. Persist artifacts
    for name, search in search_results.items():
        slug = name.replace(" ", "_").lower()
        joblib.dump(search.best_estimator_,
                    MODELS_DIR / f"{args.prefix}{slug}_model.pkl")
        joblib.dump(search.best_params_,
                    MODELS_DIR / f"{args.prefix}{slug}_best_params.pkl")
    joblib.dump(scaler, MODELS_DIR / f"{args.prefix}scaler.pkl")

    # 9. Summary tables
    pd.DataFrame(auc_scores, index=["test_auc"]).T.to_csv(
        TABLES_DIR / f"{args.prefix}auc_scores.csv"
    )
    pd.DataFrame(train_metrics).T.to_csv(
        TABLES_DIR / f"{args.prefix}train_metrics.csv"
    )
    print(f"\n✅ Done. Artifacts in: {MODELS_DIR.parent}")


if __name__ == "__main__":
    main()