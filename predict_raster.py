"""
Apply a saved model to a full raster stack → suitability GeoTIFF.

Usage:
python -m scripts.predict_raster \
    --model outputs/models/raza_rf_model.pkl \
    --raster-folder data/raw \
    --output outputs/maps/raza_suitability_rf.tif
"""
import argparse
import joblib
import numpy as np
import rasterio

from src.raster_io import load_multiple_rasters_aligned
from src.config import MODELS_DIR, MAPS_DIR


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--raster-folder", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--scaler", default=None,
                   help="Path to scaler.pkl (required for MLP / LR)")
    return p.parse_args()


def main():
    args = parse_args()
    model = joblib.load(args.model)
    scaler = joblib.load(args.scaler) if args.scaler else None

    stack, transform, crs, bounds, _ = \
        load_multiple_rasters_aligned(args.raster_folder)
    H, W, F = stack.shape[1], stack.shape[2], stack.shape[0]
    flat = stack.reshape(F, -1).T.astype(np.float32)

    if scaler is not None:
        flat = scaler.transform(flat)

    proba = model.predict_proba(flat)[:, 1].reshape(H, W).astype(np.float32)

    with rasterio.open(
        args.output, "w", driver="GTiff",
        height=H, width=W, count=1, dtype="float32",
        crs=crs, transform=transform,
    ) as dst:
        dst.write(proba, 1)
    print(f"Saved suitability raster → {args.output}")


if __name__ == "__main__":
    main()