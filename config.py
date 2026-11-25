"""
Central configuration for the multi-model habitat suitability pipeline.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW     = PROJECT_ROOT / "data" / "raw"
DATA_PROC    = PROJECT_ROOT / "data" / "processed"

OUTPUTS      = PROJECT_ROOT / "outputs"
MODELS_DIR   = OUTPUTS / "models"
FIGURES_DIR  = OUTPUTS / "figures"
MAPS_DIR     = OUTPUTS / "maps"
SHAP_DIR     = OUTPUTS / "shap"
TABLES_DIR   = OUTPUTS / "tables"

for _d in (MODELS_DIR, FIGURES_DIR, MAPS_DIR, SHAP_DIR, TABLES_DIR, DATA_PROC):
    _d.mkdir(parents=True, exist_ok=True)

# ---------- Reproducibility ----------
RANDOM_STATE        = 42
XGB_RANDOM_STATE    = 123
TEST_SIZE           = 0.30
CV_FOLDS            = 5
SCORING             = "neg_log_loss"

# ---------- Default data paths ----------
DEFAULT_RASTER_FOLDER = DATA_RAW
DEFAULT_OCCURRENCE    = DATA_RAW / "raza_LL.xls"
DEFAULT_PREFIX        = "raza_"

# ---------- Grid-search spaces ----------
RF_GRID = {
    "n_estimators":      [100, 300, 500],
    "max_depth":         [10, 20, None],
    "min_samples_split": [2, 5],
}

MLP_GRID = {
    "hidden_layer_sizes": [(64,), (128,), (256,), (256, 128, 64)],
    "learning_rate_init": [0.001, 0.01],
    "alpha":              [0.0001, 0.001],
}

XGB_GRID = {
    "n_estimators":     [100, 300, 500],
    "max_depth":        [3, 6, 9],
    "learning_rate":    [0.01, 0.1],
    "colsample_bytree": [0.8, 1.0],
    "scale_pos_weight": [1, 2],
}

LR_GRID = {
    "C":       [0.1, 1, 10],
    "penalty": ["l1", "l2"],
}

# ---------- Suitability thresholds (Low / Moderate / High) ----------
SUITABILITY_LOW_MAX      = 0.5
SUITABILITY_MODERATE_MAX = 0.7