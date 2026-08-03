"""Deprecated compatibility surface for matrix-level CNN workflows.

Model construction already resolves to the canonical residual CNN family. The
remaining matrix-only helpers are retained through smftools 2.x so existing
projects can migrate to manifest-bound datasets and services.
"""

from __future__ import annotations

from smftools.machine_learning.compatibility._warnings import deprecated_ml_alias
from smftools.machine_learning.compatibility.matrix_cnn import (
    CNNConfig,
    ResidualCNNConfig,
    ResidualDilatedCNN1d,
    TrainedCNNModel,
    build_cnn_model,
    cnn_config_from_dict,
    cnn_config_to_dict,
    default_cnn_config,
)
from smftools.machine_learning.compatibility.matrix_cnn import (
    build_cnn_baseline as _build_cnn_baseline,
)
from smftools.machine_learning.compatibility.matrix_cnn import (
    build_cnn_input as _build_cnn_input,
)
from smftools.machine_learning.compatibility.matrix_cnn import (
    detect_torch_device as _detect_torch_device,
)
from smftools.machine_learning.compatibility.matrix_cnn import (
    fit_simple_cnn as _fit_simple_cnn,
)
from smftools.machine_learning.compatibility.matrix_cnn import (
    integrated_gradients_attributions as _integrated_gradients_attributions,
)
from smftools.machine_learning.compatibility.matrix_cnn import (
    predict_cnn_scores as _predict_cnn_scores,
)
from smftools.machine_learning.compatibility.matrix_cnn import (
    split_train_validation as _split_train_validation,
)

_CANONICAL_TRAINING = "smftools.machine_learning.orchestration.train_partition_model"
_CANONICAL_INFERENCE = "smftools.machine_learning.orchestration.apply_partition_model"
_CANONICAL_EXPLANATION = "smftools.machine_learning.orchestration.explain_partition_model"
_CANONICAL_DATA = "smftools.machine_learning.data.PartitionDataset or MaterializedDataset"

detect_torch_device = deprecated_ml_alias(
    "smftools.analysis.compute.ml_cnn.detect_torch_device",
    "smftools.machine_learning.utils.detect_device",
)(_detect_torch_device)
build_cnn_input = deprecated_ml_alias(
    "smftools.analysis.compute.ml_cnn.build_cnn_input",
    _CANONICAL_DATA,
)(_build_cnn_input)
build_cnn_baseline = deprecated_ml_alias(
    "smftools.analysis.compute.ml_cnn.build_cnn_baseline",
    "a checksummed training BackgroundReference",
)(_build_cnn_baseline)
split_train_validation = deprecated_ml_alias(
    "smftools.analysis.compute.ml_cnn.split_train_validation",
    "smftools.machine_learning.splitting.plan_ml_splits",
)(_split_train_validation)
fit_simple_cnn = deprecated_ml_alias(
    "smftools.analysis.compute.ml_cnn.fit_simple_cnn",
    _CANONICAL_TRAINING,
)(_fit_simple_cnn)
predict_cnn_scores = deprecated_ml_alias(
    "smftools.analysis.compute.ml_cnn.predict_cnn_scores",
    _CANONICAL_INFERENCE,
)(_predict_cnn_scores)
integrated_gradients_attributions = deprecated_ml_alias(
    "smftools.analysis.compute.ml_cnn.integrated_gradients_attributions",
    _CANONICAL_EXPLANATION,
)(_integrated_gradients_attributions)

__all__ = [
    "CNNConfig",
    "ResidualCNNConfig",
    "ResidualDilatedCNN1d",
    "TrainedCNNModel",
    "build_cnn_baseline",
    "build_cnn_input",
    "build_cnn_model",
    "cnn_config_from_dict",
    "cnn_config_to_dict",
    "default_cnn_config",
    "detect_torch_device",
    "fit_simple_cnn",
    "integrated_gradients_attributions",
    "predict_cnn_scores",
    "split_train_validation",
]
