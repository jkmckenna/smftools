"""
Per-read, per-feature attribution helpers for fitted binary classifiers.

Inputs are fitted sklearn Pipeline(imputer, classifier) objects plus a raw
feature matrix. No AnnData or file I/O occurs here. CNN attribution lives in
``ml_cnn.integrated_gradients_attributions`` since it needs the CNN's
channel-encoded input and trained-model wrapper, not a sklearn Pipeline.
"""

from __future__ import annotations

import numpy as np


def bernoulli_nb_logodds_contributions(pipeline, X: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Exact per-feature log-odds contributions for a fitted BernoulliNB classifier.

    BernoulliNB assumes conditional feature independence, so its log-odds
    output decomposes exactly (no sampling/approximation) as::

        logit(x) = prior_logodds + sum_i contribution_i(x_i)

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Fitted ``Pipeline(imputer, classifier)`` as built by
        ``ml_metrics.build_binary_classifier("bernoulli_nb")``.
    X : np.ndarray
        Raw (pre-imputation) feature matrix, shape ``(n_reads, n_features)``.

    Returns
    -------
    contributions : np.ndarray, shape (n_reads, n_features)
        Per-read, per-feature log-odds contribution. Row sums plus
        ``prior_logodds`` reconstruct the read's overall log-odds exactly.
    prior_logodds : float
        Class-prior log-odds term (constant across reads).
    """
    imputer = pipeline.named_steps["imputer"]
    classifier = pipeline.named_steps["classifier"]
    X_imp = imputer.transform(X)

    classes = list(classifier.classes_)
    idx_pos = classes.index(1)
    idx_neg = classes.index(0)

    feature_log_prob = classifier.feature_log_prob_  # (n_classes, n_features), log P(x_i=1|y)
    neg_log_prob = np.log1p(-np.exp(feature_log_prob))  # log P(x_i=0|y)

    slope = (feature_log_prob[idx_pos] - neg_log_prob[idx_pos]) - (
        feature_log_prob[idx_neg] - neg_log_prob[idx_neg]
    )
    intercept = neg_log_prob[idx_pos] - neg_log_prob[idx_neg]
    prior_logodds = float(classifier.class_log_prior_[idx_pos] - classifier.class_log_prior_[idx_neg])

    contributions = X_imp * slope[np.newaxis, :] + intercept[np.newaxis, :]
    return contributions, prior_logodds


def tree_shap_contributions(pipeline, X: np.ndarray, model_output: str | None = None) -> np.ndarray:
    """
    Positive-class SHAP contribution matrix for a fitted
    Pipeline(imputer, RandomForestClassifier).

    ``RandomForestClassifier`` has no native margin/logit output, so SHAP
    values here are additive in probability space (they sum, with the
    explainer's expected value, to ``predict_proba``). For XGBoost use
    :func:`xgboost_contributions` instead — TreeExplainer's XGBoost loader
    is currently incompatible with recent xgboost model-dump formats (see
    that function's docstring).

    Parameters
    ----------
    model_output : str, optional
        Passed through to ``shap.TreeExplainer``; ``None`` uses the
        explainer's default.

    Returns
    -------
    np.ndarray, shape (n_reads, n_features)
        Positive-class SHAP values.
    """
    import shap

    imputer = pipeline.named_steps["imputer"]
    classifier = pipeline.named_steps["classifier"]
    X_imp = imputer.transform(X)

    kwargs = {} if model_output is None else {"model_output": model_output}
    explainer = shap.TreeExplainer(classifier, **kwargs)
    shap_values = explainer.shap_values(X_imp, check_additivity=False)

    if isinstance(shap_values, list):
        return np.asarray(shap_values[1] if len(shap_values) > 1 else shap_values[0], dtype=float)
    shap_arr = np.asarray(shap_values, dtype=float)
    if shap_arr.ndim == 3 and shap_arr.shape[2] >= 2:
        return shap_arr[:, :, 1]
    if shap_arr.ndim != 2:
        raise ValueError(f"Unexpected SHAP value shape: {shap_arr.shape}")
    return shap_arr


def xgboost_contributions(pipeline, X: np.ndarray) -> np.ndarray:
    """
    Exact per-feature log-odds (margin-space) contributions for a fitted
    Pipeline(imputer, XGBClassifier), via XGBoost's native TreeSHAP
    (``pred_contribs=True``).

    Uses the booster's own contribution prediction rather than
    ``shap.TreeExplainer`` — as of shap 0.49 / xgboost 3.x, TreeExplainer's
    XGBoost model loader fails to parse the ``base_score`` field in recent
    xgboost JSON dumps (``ValueError: could not convert string to float:
    '[5E-1]'``); the booster's native path sidesteps that entirely and
    reproduces the same TreeSHAP algorithm exactly.

    Returns
    -------
    np.ndarray, shape (n_reads, n_features)
        Per-feature log-odds contributions. Row sums plus the booster's
        base margin reconstruct the read's overall log-odds.
    """
    import xgboost as xgb

    imputer = pipeline.named_steps["imputer"]
    classifier = pipeline.named_steps["classifier"]
    X_imp = imputer.transform(X)

    booster = classifier.get_booster()
    contribs = booster.predict(xgb.DMatrix(X_imp), pred_contribs=True)
    return np.asarray(contribs[:, :-1], dtype=float)
