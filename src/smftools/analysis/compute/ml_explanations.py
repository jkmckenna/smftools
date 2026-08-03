"""Deprecated compatibility adapters for fitted-estimator explanations."""

from __future__ import annotations

from smftools.machine_learning.compatibility._warnings import deprecated_ml_alias
from smftools.machine_learning.compatibility.classical_explanations import (
    bernoulli_nb_logodds_contributions as _bernoulli_nb_logodds_contributions,
)
from smftools.machine_learning.compatibility.classical_explanations import (
    tree_shap_contributions as _tree_shap_contributions,
)
from smftools.machine_learning.compatibility.classical_explanations import (
    xgboost_contributions as _xgboost_contributions,
)

_REPLACEMENT = "smftools.machine_learning.interpretability.explain_sklearn_model"

bernoulli_nb_logodds_contributions = deprecated_ml_alias(
    "smftools.analysis.compute.ml_explanations.bernoulli_nb_logodds_contributions",
    _REPLACEMENT,
)(_bernoulli_nb_logodds_contributions)
tree_shap_contributions = deprecated_ml_alias(
    "smftools.analysis.compute.ml_explanations.tree_shap_contributions",
    _REPLACEMENT,
)(_tree_shap_contributions)
xgboost_contributions = deprecated_ml_alias(
    "smftools.analysis.compute.ml_explanations.xgboost_contributions",
    _REPLACEMENT,
)(_xgboost_contributions)

__all__ = [
    "bernoulli_nb_logodds_contributions",
    "tree_shap_contributions",
    "xgboost_contributions",
]
