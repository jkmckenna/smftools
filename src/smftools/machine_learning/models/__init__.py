from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .base import BaseTorchModel
from .cnn import CNNClassifier
from .mlp import MLPClassifier
from .positional import PositionalEncoding
from .protocols import (
    PredictorError,
    PredictorLoaderProtocol,
    PredictorProtocol,
    SklearnPredictor,
    TorchPredictor,
    adapt_loaded_predictor,
    require_capabilities,
)
from .registry import (
    BUILTIN_MODEL_REGISTRY,
    BUILTIN_MODEL_REGISTRY_VERSION,
    ML_MODEL_RECIPE_VERSION,
    BernoulliNBConfig,
    LogisticRegressionConfig,
    ModelFamilyDefinition,
    ModelRecipe,
    ModelRegistry,
    ModelRegistryError,
    RandomForestConfig,
    ResolvedModelDefinition,
)
from .residual_cnn import (
    AttentionPooling1d,
    ResidualCNNConfig,
    ResidualCNNConfigError,
    ResidualDilatedBlock1d,
    ResidualDilatedCNN1d,
    SqueezeExcite1d,
    build_residual_cnn,
    default_residual_cnn_config,
    residual_cnn_config_from_dict,
    residual_cnn_config_to_dict,
)
from .rnn import RNNClassifier
from .transformer import (
    BaseTransformer,
    DANNTransformerClassifier,
    MaskedTransformerPretrainer,
    TransformerClassifier,
)
from .wrappers import ScaledModel

if TYPE_CHECKING:
    from .lightning_base import TorchClassifierWrapper
    from .sklearn_artifacts import (
        SKLEARN_ARTIFACT_FILENAME,
        SKLEARN_ARTIFACT_SCHEMA_VERSION,
        PublishedSklearnModel,
        SklearnArtifactError,
        load_published_sklearn_model,
        publish_sklearn_model,
    )
    from .sklearn_models import SklearnModelWrapper


_LAZY_EXPORTS = {
    "SKLEARN_ARTIFACT_FILENAME": (".sklearn_artifacts", "SKLEARN_ARTIFACT_FILENAME"),
    "SKLEARN_ARTIFACT_SCHEMA_VERSION": (
        ".sklearn_artifacts",
        "SKLEARN_ARTIFACT_SCHEMA_VERSION",
    ),
    "PublishedSklearnModel": (".sklearn_artifacts", "PublishedSklearnModel"),
    "SklearnArtifactError": (".sklearn_artifacts", "SklearnArtifactError"),
    "SklearnModelWrapper": (".sklearn_models", "SklearnModelWrapper"),
    "TorchClassifierWrapper": (".lightning_base", "TorchClassifierWrapper"),
    "load_published_sklearn_model": (".sklearn_artifacts", "load_published_sklearn_model"),
    "publish_sklearn_model": (".sklearn_artifacts", "publish_sklearn_model"),
}


def __getattr__(name: str) -> Any:
    """Lazily expose wrappers with optional plotting or orchestration imports."""
    if name in _LAZY_EXPORTS:
        module_name, attribute = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name, __name__), attribute)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BUILTIN_MODEL_REGISTRY",
    "BUILTIN_MODEL_REGISTRY_VERSION",
    "BaseTorchModel",
    "BaseTransformer",
    "BernoulliNBConfig",
    "CNNClassifier",
    "DANNTransformerClassifier",
    "LogisticRegressionConfig",
    "ML_MODEL_RECIPE_VERSION",
    "MLPClassifier",
    "MaskedTransformerPretrainer",
    "ModelFamilyDefinition",
    "ModelRecipe",
    "ModelRegistry",
    "ModelRegistryError",
    "PositionalEncoding",
    "PredictorError",
    "PredictorLoaderProtocol",
    "PredictorProtocol",
    "RNNClassifier",
    "RandomForestConfig",
    "ResidualCNNConfig",
    "ResidualCNNConfigError",
    "ResidualDilatedBlock1d",
    "ResidualDilatedCNN1d",
    "ResolvedModelDefinition",
    "SKLEARN_ARTIFACT_FILENAME",
    "SKLEARN_ARTIFACT_SCHEMA_VERSION",
    "ScaledModel",
    "PublishedSklearnModel",
    "SklearnArtifactError",
    "SklearnModelWrapper",
    "SklearnPredictor",
    "TorchClassifierWrapper",
    "TorchPredictor",
    "TransformerClassifier",
    "AttentionPooling1d",
    "SqueezeExcite1d",
    "adapt_loaded_predictor",
    "build_residual_cnn",
    "default_residual_cnn_config",
    "load_published_sklearn_model",
    "publish_sklearn_model",
    "residual_cnn_config_from_dict",
    "residual_cnn_config_to_dict",
    "require_capabilities",
]
