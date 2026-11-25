"""
GridSearchCV training for all four models.
"""
from sklearn.model_selection import GridSearchCV

from src.config import CV_FOLDS, SCORING, RF_GRID, MLP_GRID, XGB_GRID, LR_GRID
from src.models import rf_base, mlp_base, xgb_base, lr_base


def _run(name, estimator, grid, X, y):
    print(f"GridSearchCV → {name} ...")
    search = GridSearchCV(
        estimator, param_grid=grid, cv=CV_FOLDS,
        scoring=SCORING, n_jobs=-1, verbose=1,
    )
    search.fit(X, y)
    print(f"  best {name} params: {search.best_params_}")
    return search


def train_all(X_train, y_train, X_train_scaled):
    """Returns dict of fitted GridSearchCV objects."""
    return {
        "Random Forest":       _run("Random Forest",       rf_base(),  RF_GRID,  X_train,         y_train),
        "XGBoost":             _run("XGBoost",             xgb_base(), XGB_GRID, X_train,         y_train),
        "MLP":                 _run("MLP",                 mlp_base(), MLP_GRID, X_train_scaled,  y_train),
        "Logistic Regression": _run("Logistic Regression", lr_base(),  LR_GRID,  X_train_scaled,  y_train),
    }