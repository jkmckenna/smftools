"""Legacy unconstrained sklearn builders retained for 2.x compatibility."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None


def build_binary_classifier(model_name: str, random_state: int = 42, **kwargs):
    """Construct a legacy binary sklearn classifier pipeline."""
    if model_name == "bernoulli_nb":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("classifier", BernoulliNB(**kwargs)),
            ]
        )
    if model_name == "random_forest":
        params = {
            "n_estimators": 300,
            "random_state": random_state,
            "class_weight": "balanced",
            "n_jobs": 1,
        }
        params.update(kwargs)
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("classifier", RandomForestClassifier(**params)),
            ]
        )
    if model_name == "xgboost":
        if XGBClassifier is None:
            raise ImportError("xgboost is not available in the current environment")
        params = {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "min_child_weight": 1.0,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "random_state": random_state,
            "n_jobs": 1,
            "scale_pos_weight": 1.0,
        }
        params.update(kwargs)
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("classifier", XGBClassifier(**params)),
            ]
        )
    raise ValueError(f"Unsupported model_name {model_name!r}")


def fit_classifier(estimator, X_train: np.ndarray, y_train: np.ndarray):
    """Fit and return a fresh copy of a legacy estimator."""
    model = deepcopy(estimator)
    model.fit(X_train, y_train)
    return model


def predict_binary_scores(model, X: np.ndarray) -> np.ndarray:
    """Return a continuous positive-class score from a legacy estimator."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim != 2 or proba.shape[1] < 2:
            raise ValueError("predict_proba did not return a 2-class probability matrix")
        return np.asarray(proba[:, 1], dtype=float)
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float)
    raise ValueError("Model does not expose predict_proba or decision_function")


__all__ = ["build_binary_classifier", "fit_classifier", "predict_binary_scores"]
