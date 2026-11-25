"""
Model builders (RF / XGB / MLP / LR).
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from src.config import RANDOM_STATE, XGB_RANDOM_STATE


def rf_base():
    return RandomForestClassifier(class_weight="balanced",
                                  random_state=RANDOM_STATE)


def mlp_base():
    return MLPClassifier(
        max_iter=500, random_state=RANDOM_STATE,
        early_stopping=True, n_iter_no_change=10,
        activation="relu", validation_fraction=0.10,
    )


def xgb_base():
    return XGBClassifier(
        use_label_encoder=False, eval_metric="logloss",
        random_state=XGB_RANDOM_STATE,
    )


def lr_base():
    return LogisticRegression(
        class_weight="balanced", solver="saga",
        max_iter=3000, random_state=RANDOM_STATE,
    )