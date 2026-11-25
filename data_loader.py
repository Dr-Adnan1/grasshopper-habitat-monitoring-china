"""
Occurrence loading + train/test preparation.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.raster_io import extract_values_at_points
from src.config import RANDOM_STATE, TEST_SIZE


def load_occurrence_data(path):
    """Load .xls/.xlsx/.csv occurrence file; require lon/lat + species."""
    if str(path).lower().endswith((".xls", ".xlsx")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    df.columns = df.columns.str.lower().str.strip()

    if "lon" not in df.columns or "lat" not in df.columns:
        if "longitude" in df.columns and "latitude" in df.columns:
            df = df.rename(columns={"longitude": "lon", "latitude": "lat"})
        else:
            raise ValueError("Need lon/lat or longitude/latitude columns")
    return df


def build_training_table(raster_data, transform, raster_crs,
                         occurrence_path):
    """
    Extract predictor values at occurrence points and return
    (X, y, feature_names) with rows where extraction failed removed.
    """
    occ_df = load_occurrence_data(occurrence_path)
    if "species" not in occ_df.columns:
        raise ValueError("Missing 'species' column (0/1 presence-absence)")

    print(f"Extracting predictor values at {len(occ_df)} points ...")
    env_vals = extract_values_at_points(
        raster_data, transform, raster_crs, occ_df
    )

    # feature names = band names without .tif extension (set later)
    return env_vals, occ_df


def prepare_xy(env_vals, occ_df, band_names):
    """
    Build X / y, drop failed rows, ensure binary labels.
    """
    feature_names = [b.rsplit(".", 1)[0] for b in band_names]
    env_df = pd.DataFrame(env_vals, columns=feature_names)
    env_df["Species"] = occ_df["species"].values

    valid = ~np.isnan(env_df.iloc[:, :-1]).all(axis=1)
    env_df = env_df.loc[valid].reset_index(drop=True)

    if env_df["Species"].dtype == object:
        uniq = env_df["Species"].unique()
        if len(uniq) != 2:
            raise ValueError("'species' column must be binary")
        env_df["Species"] = env_df["Species"].map({uniq[0]: 0, uniq[1]: 1})

    X = env_df.drop(columns=["Species"]).astype(np.float32)
    y = env_df["Species"].astype(int)
    return X, y, X.columns.tolist()


def split_and_scale(X, y):
    """70/30 stratified split + StandardScaler for MLP / LR."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    return (
        X_train, X_test, y_train, y_test,
        scaler.transform(X_train), scaler.transform(X_test), scaler,
    )