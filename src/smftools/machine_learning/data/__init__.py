from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .balancing import (
    ML_BALANCE_RESOLUTION_VERSION,
    BalanceResolution,
    MLBalanceError,
    balance_counts,
    resolve_evaluation_sensitivity,
    resolve_role_balance,
)
from .materialized_dataset import (
    MaterializedDataset,
    MaterializedDatasetPlan,
    MLDatasetPlanProtocol,
    MLDatasetProtocol,
)
from .partition_dataset import (
    ExperimentPartitionSource,
    MLMaterializedPartitionData,
    MLMemoryBudgetError,
    MLPartitionBatch,
    MLPartitionDataError,
    MLPartitionDataPlan,
    PartitionDataset,
    PartitionReadEntry,
    PartitionReadPolicy,
    build_partition_data_plan,
)
from .preprocessing import random_fill_nans
from .transforms import (
    ML_FEATURE_TRANSFORM_VERSION,
    FeatureTransformSpec,
    FittedFeatureTransform,
    FittedFeatureTransformProtocol,
    ManifestFeatureTransformer,
    MLTransformError,
    TorchFeatureTransform,
    TorchTransformedBatch,
    build_sklearn_preprocessing_pipeline,
    fit_feature_transform,
)

if TYPE_CHECKING:
    from .anndata_data_module import AnnDataModule, build_anndata_loader


_LAZY_EXPORTS = {
    "AnnDataModule": (".anndata_data_module", "AnnDataModule"),
    "build_anndata_loader": (".anndata_data_module", "build_anndata_loader"),
}


def __getattr__(name: str) -> Any:
    """Lazily expose the legacy optional-Lightning AnnData adapter."""
    if name in _LAZY_EXPORTS:
        module_name, attribute = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name, __name__), attribute)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AnnDataModule",
    "BalanceResolution",
    "ExperimentPartitionSource",
    "FeatureTransformSpec",
    "FittedFeatureTransform",
    "FittedFeatureTransformProtocol",
    "MLBalanceError",
    "MLDatasetPlanProtocol",
    "MLDatasetProtocol",
    "MLTransformError",
    "ML_BALANCE_RESOLUTION_VERSION",
    "ML_FEATURE_TRANSFORM_VERSION",
    "MLMaterializedPartitionData",
    "MLMemoryBudgetError",
    "MLPartitionBatch",
    "MLPartitionDataError",
    "MLPartitionDataPlan",
    "ManifestFeatureTransformer",
    "MaterializedDataset",
    "MaterializedDatasetPlan",
    "PartitionDataset",
    "PartitionReadEntry",
    "PartitionReadPolicy",
    "TorchFeatureTransform",
    "TorchTransformedBatch",
    "balance_counts",
    "build_anndata_loader",
    "build_partition_data_plan",
    "build_sklearn_preprocessing_pipeline",
    "fit_feature_transform",
    "random_fill_nans",
    "resolve_evaluation_sensitivity",
    "resolve_role_balance",
]
