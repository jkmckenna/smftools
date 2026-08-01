"""Explicit built-in model registry and immutable resolved recipe records."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from smftools.optional_imports import require

from ..artifacts import ResolvedDefinition
from ..contracts import (
    ML_CAPABILITY_SCHEMA_VERSION,
    InputSchema,
    PredictorCapabilities,
)
from .residual_cnn import ResidualCNNConfig, build_residual_cnn

ML_MODEL_RECIPE_VERSION = 1
BUILTIN_MODEL_REGISTRY_VERSION = 1
_SUPPORTED_MODALITIES = ("conversion", "deaminase", "direct")


class ModelRegistryError(ValueError):
    """Raised when model definitions, recipes, or resolutions are invalid."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_json(item) for item in value]
    return value


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelRegistryError(f"{path} must be a non-empty string")
    return value.strip()


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ModelRegistryError(f"{path} must be an integer >= {minimum}")
    return value


def _number(value: Any, path: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelRegistryError(f"{path} must be numeric")
    result = float(value)
    if not result == result or result in {float("inf"), float("-inf")}:
        raise ModelRegistryError(f"{path} must be finite")
    if minimum is not None and result < minimum:
        raise ModelRegistryError(f"{path} must be >= {minimum}")
    return result


def _strings(value: Any, path: str, *, allow_wildcard: bool = False) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ModelRegistryError(f"{path} must be a sequence")
    result = tuple(sorted(_string(item, f"{path}[]") for item in value))
    if len(result) != len(set(result)):
        raise ModelRegistryError(f"{path} cannot contain duplicates")
    if "*" in result and (not allow_wildcard or len(result) != 1):
        raise ModelRegistryError(f"{path} wildcard must be the only value")
    return result


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ModelRegistryError(f"{path} must be a mapping")
    if not all(isinstance(key, str) and key for key in value):
        raise ModelRegistryError(f"{path} keys must be non-empty strings")
    try:
        restored = json.loads(_canonical_json(_thaw_json(value)))
    except (TypeError, ValueError) as exc:
        raise ModelRegistryError(f"{path} must contain finite JSON values") from exc
    return _freeze_json(restored)


def _exact_fields(raw: Mapping[str, Any], expected: set[str], path: str) -> None:
    if set(raw) != expected:
        raise ModelRegistryError(f"{path} fields must be exactly {sorted(expected)}")


@dataclass(frozen=True)
class ModelRecipe:
    """Checksummed model defaults plus explicit input-schema compatibility."""

    schema_version: int
    recipe_id: str
    name: str
    version: str
    family: str
    backend: str
    parameters: Mapping[str, Any]
    supported_modalities: tuple[str, ...]
    supported_channel_roles: tuple[str, ...]
    required_channel_roles: tuple[str, ...]
    minimum_channels: int
    maximum_channels: int | None

    def __post_init__(self) -> None:
        if self.schema_version != ML_MODEL_RECIPE_VERSION:
            raise ModelRegistryError(
                f"unsupported recipe version {self.schema_version}; "
                f"expected {ML_MODEL_RECIPE_VERSION}"
            )
        object.__setattr__(self, "name", _string(self.name, "recipe.name"))
        object.__setattr__(self, "version", _string(self.version, "recipe.version"))
        object.__setattr__(self, "family", _string(self.family, "recipe.family"))
        object.__setattr__(self, "backend", _string(self.backend, "recipe.backend"))
        if self.backend not in {"sklearn", "torch"}:
            raise ModelRegistryError("recipe.backend must be 'sklearn' or 'torch'")
        object.__setattr__(self, "parameters", _mapping(self.parameters, "recipe.parameters"))
        modalities = _strings(self.supported_modalities, "recipe.supported_modalities")
        if not modalities:
            raise ModelRegistryError("recipe.supported_modalities cannot be empty")
        unknown_modalities = sorted(set(modalities).difference(_SUPPORTED_MODALITIES))
        if unknown_modalities:
            raise ModelRegistryError(
                f"recipe.supported_modalities contains unknown values: {unknown_modalities}"
            )
        roles = _strings(
            self.supported_channel_roles,
            "recipe.supported_channel_roles",
            allow_wildcard=True,
        )
        if not roles:
            raise ModelRegistryError("recipe.supported_channel_roles cannot be empty")
        required = _strings(
            self.required_channel_roles,
            "recipe.required_channel_roles",
        )
        if "*" not in roles and not set(required).issubset(roles):
            raise ModelRegistryError(
                "recipe.required_channel_roles must be supported channel roles"
            )
        minimum = _integer(self.minimum_channels, "recipe.minimum_channels", minimum=1)
        maximum = self.maximum_channels
        if maximum is not None:
            maximum = _integer(maximum, "recipe.maximum_channels", minimum=minimum)
        object.__setattr__(self, "supported_modalities", modalities)
        object.__setattr__(self, "supported_channel_roles", roles)
        object.__setattr__(self, "required_channel_roles", required)
        object.__setattr__(self, "minimum_channels", minimum)
        object.__setattr__(self, "maximum_channels", maximum)
        if self.recipe_id != _sha256(self._identity_dict()):
            raise ModelRegistryError("recipe.recipe_id does not match its content")

    @classmethod
    def create(
        cls,
        *,
        name: str,
        version: str,
        family: str,
        backend: str,
        parameters: Mapping[str, Any],
        supported_modalities: Sequence[str],
        supported_channel_roles: Sequence[str],
        required_channel_roles: Sequence[str] = (),
        minimum_channels: int = 1,
        maximum_channels: int | None = None,
    ) -> ModelRecipe:
        """Create an immutable content-addressed model recipe."""
        identity = {
            "schema_version": ML_MODEL_RECIPE_VERSION,
            "name": name,
            "version": version,
            "family": family,
            "backend": backend,
            "parameters": _thaw_json(parameters),
            "supported_modalities": sorted(supported_modalities),
            "supported_channel_roles": sorted(supported_channel_roles),
            "required_channel_roles": sorted(required_channel_roles),
            "minimum_channels": minimum_channels,
            "maximum_channels": maximum_channels,
        }
        return cls(recipe_id=_sha256(identity), **identity)

    def _identity_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "version": self.version,
            "family": self.family,
            "backend": self.backend,
            "parameters": _thaw_json(self.parameters),
            "supported_modalities": list(self.supported_modalities),
            "supported_channel_roles": list(self.supported_channel_roles),
            "required_channel_roles": list(self.required_channel_roles),
            "minimum_channels": self.minimum_channels,
            "maximum_channels": self.maximum_channels,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a complete JSON-serializable recipe record."""
        return {"recipe_id": self.recipe_id, **self._identity_dict()}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ModelRecipe:
        """Strictly validate and restore a serialized recipe."""
        raw = _mapping(raw, "recipe")
        expected = {
            "schema_version",
            "recipe_id",
            "name",
            "version",
            "family",
            "backend",
            "parameters",
            "supported_modalities",
            "supported_channel_roles",
            "required_channel_roles",
            "minimum_channels",
            "maximum_channels",
        }
        _exact_fields(raw, expected, "recipe")
        return cls(
            schema_version=_integer(raw["schema_version"], "recipe.schema_version", minimum=1),
            recipe_id=str(raw["recipe_id"]),
            name=str(raw["name"]),
            version=str(raw["version"]),
            family=str(raw["family"]),
            backend=str(raw["backend"]),
            parameters=_mapping(raw["parameters"], "recipe.parameters"),
            supported_modalities=_strings(
                raw["supported_modalities"], "recipe.supported_modalities"
            ),
            supported_channel_roles=_strings(
                raw["supported_channel_roles"],
                "recipe.supported_channel_roles",
                allow_wildcard=True,
            ),
            required_channel_roles=_strings(
                raw["required_channel_roles"], "recipe.required_channel_roles"
            ),
            minimum_channels=_integer(
                raw["minimum_channels"], "recipe.minimum_channels", minimum=1
            ),
            maximum_channels=(
                None
                if raw["maximum_channels"] is None
                else _integer(raw["maximum_channels"], "recipe.maximum_channels", minimum=1)
            ),
        )

    def assert_input_compatible(self, schema: InputSchema) -> None:
        """Reject modality or biological-channel schemas outside recipe support."""
        unsupported_modalities = sorted(
            set(schema.modalities).difference(self.supported_modalities)
        )
        if unsupported_modalities:
            raise ModelRegistryError(
                f"recipe {self.name!r} does not support modalities: {unsupported_modalities}"
            )
        roles = {channel.biological_role for channel in schema.channels}
        if "*" not in self.supported_channel_roles:
            unsupported_roles = sorted(roles.difference(self.supported_channel_roles))
            if unsupported_roles:
                raise ModelRegistryError(
                    f"recipe {self.name!r} does not support channel roles: {unsupported_roles}"
                )
        missing_roles = sorted(set(self.required_channel_roles).difference(roles))
        if missing_roles:
            raise ModelRegistryError(
                f"recipe {self.name!r} requires channel roles: {missing_roles}"
            )
        n_channels = len(schema.channels)
        if n_channels < self.minimum_channels or (
            self.maximum_channels is not None and n_channels > self.maximum_channels
        ):
            raise ModelRegistryError(
                f"recipe {self.name!r} does not support {n_channels} input channels"
            )

    def resolve(self, overrides: Mapping[str, Any] | None = None) -> ResolvedDefinition:
        """Return the fully resolved, checksummed parameter definition."""
        parameters = dict(self.parameters)
        parameters.update({} if overrides is None else dict(overrides))
        return ResolvedDefinition.create(
            name=self.name,
            version=self.version,
            parameters=parameters,
        )


@dataclass(frozen=True)
class BernoulliNBConfig:
    """Validated constructor configuration for sklearn Bernoulli Naive Bayes."""

    alpha: float = 1.0
    binarize: float | None = 0.5
    fit_prior: bool = True
    force_alpha: bool = True

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> BernoulliNBConfig:
        """Validate a fully resolved Bernoulli NB definition."""
        expected = {"alpha", "binarize", "fit_prior", "force_alpha"}
        _exact_fields(raw, expected, "bernoulli_nb")
        if not isinstance(raw["fit_prior"], bool) or not isinstance(raw["force_alpha"], bool):
            raise ModelRegistryError("Bernoulli NB boolean parameters must be boolean")
        return cls(
            alpha=_number(raw["alpha"], "bernoulli_nb.alpha", minimum=0.0),
            binarize=(
                None
                if raw["binarize"] is None
                else _number(raw["binarize"], "bernoulli_nb.binarize")
            ),
            fit_prior=raw["fit_prior"],
            force_alpha=raw["force_alpha"],
        )

    def to_dict(self) -> dict[str, Any]:
        """Return sklearn constructor parameters."""
        return {
            "alpha": self.alpha,
            "binarize": self.binarize,
            "fit_prior": self.fit_prior,
            "force_alpha": self.force_alpha,
        }


@dataclass(frozen=True)
class LogisticRegressionConfig:
    """Validated constructor configuration for sklearn logistic regression."""

    C: float = 1.0
    penalty: str | None = "l2"
    solver: str = "lbfgs"
    max_iter: int = 1000
    class_weight: str | None = None
    random_state: int = 0

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> LogisticRegressionConfig:
        """Validate a fully resolved logistic-regression definition."""
        expected = {"C", "penalty", "solver", "max_iter", "class_weight", "random_state"}
        _exact_fields(raw, expected, "logistic_regression")
        penalty = raw["penalty"]
        if penalty not in {None, "l1", "l2"}:
            raise ModelRegistryError("logistic_regression.penalty is unsupported")
        solver = _string(raw["solver"], "logistic_regression.solver")
        if solver not in {"lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"}:
            raise ModelRegistryError("logistic_regression.solver is unsupported")
        class_weight = raw["class_weight"]
        if class_weight not in {None, "balanced"}:
            raise ModelRegistryError("logistic_regression.class_weight must be null or balanced")
        regularization = _number(raw["C"], "logistic_regression.C", minimum=0.0)
        if regularization == 0:
            raise ModelRegistryError("logistic_regression.C must be > 0")
        if penalty == "l1" and solver not in {"liblinear", "saga"}:
            raise ModelRegistryError("l1 penalty requires the liblinear or saga solver")
        if penalty is None and solver == "liblinear":
            raise ModelRegistryError("null penalty is unsupported by the liblinear solver")
        return cls(
            C=regularization,
            penalty=penalty,
            solver=solver,
            max_iter=_integer(raw["max_iter"], "logistic_regression.max_iter", minimum=1),
            class_weight=class_weight,
            random_state=_integer(
                raw["random_state"], "logistic_regression.random_state", minimum=0
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return sklearn constructor parameters."""
        return {
            "C": self.C,
            "penalty": self.penalty,
            "solver": self.solver,
            "max_iter": self.max_iter,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
        }


@dataclass(frozen=True)
class RandomForestConfig:
    """Validated constructor configuration for sklearn random forests."""

    n_estimators: int = 300
    max_depth: int | None = None
    min_samples_leaf: int = 1
    max_features: str | float | None = "sqrt"
    class_weight: str | None = None
    random_state: int = 0
    n_jobs: int = 1

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> RandomForestConfig:
        """Validate a fully resolved random-forest definition."""
        expected = {
            "n_estimators",
            "max_depth",
            "min_samples_leaf",
            "max_features",
            "class_weight",
            "random_state",
            "n_jobs",
        }
        _exact_fields(raw, expected, "random_forest")
        max_depth = raw["max_depth"]
        if max_depth is not None:
            max_depth = _integer(max_depth, "random_forest.max_depth", minimum=1)
        max_features = raw["max_features"]
        if isinstance(max_features, str):
            if max_features not in {"sqrt", "log2"}:
                raise ModelRegistryError("random_forest.max_features is unsupported")
        elif max_features is not None:
            max_features = _number(max_features, "random_forest.max_features", minimum=0.0)
            if max_features <= 0 or max_features > 1:
                raise ModelRegistryError("random_forest.max_features float must be in (0, 1]")
        class_weight = raw["class_weight"]
        if class_weight not in {None, "balanced", "balanced_subsample"}:
            raise ModelRegistryError("random_forest.class_weight is unsupported")
        n_jobs = raw["n_jobs"]
        if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs == 0:
            raise ModelRegistryError("random_forest.n_jobs must be a non-zero integer")
        return cls(
            n_estimators=_integer(raw["n_estimators"], "random_forest.n_estimators", minimum=1),
            max_depth=max_depth,
            min_samples_leaf=_integer(
                raw["min_samples_leaf"], "random_forest.min_samples_leaf", minimum=1
            ),
            max_features=max_features,
            class_weight=class_weight,
            random_state=_integer(raw["random_state"], "random_forest.random_state", minimum=0),
            n_jobs=n_jobs,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return sklearn constructor parameters."""
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }


@dataclass(frozen=True)
class ModelFamilyDefinition:
    """Runtime-only deterministic association of config, builder, recipes, and capabilities."""

    name: str
    backend: str
    architecture_schema_version: int
    config_type: type[Any]
    builder: Callable[[Any], Any]
    capabilities: PredictorCapabilities
    default_recipe: str

    def __post_init__(self) -> None:
        _string(self.name, "model_definition.name")
        _string(self.backend, "model_definition.backend")
        _integer(
            self.architecture_schema_version,
            "model_definition.architecture_schema_version",
            minimum=1,
        )
        if self.capabilities.backend != self.backend:
            raise ModelRegistryError("model definition and capabilities backends differ")
        if not callable(getattr(self.config_type, "from_dict", None)):
            raise ModelRegistryError("model config type must define from_dict")
        if not callable(self.builder):
            raise ModelRegistryError("model builder must be callable")
        _string(self.default_recipe, "model_definition.default_recipe")


@dataclass(frozen=True)
class ResolvedModelDefinition:
    """Internal runtime resolution with serializable architecture and typed config."""

    family: str
    backend: str
    recipe_id: str
    architecture_schema_version: int
    architecture: ResolvedDefinition
    capabilities: PredictorCapabilities
    config: Any


class ModelRegistry:
    """Immutable explicit registry; definitions never appear through import side effects."""

    def __init__(
        self,
        definitions: Sequence[ModelFamilyDefinition],
        recipes: Sequence[ModelRecipe],
        *,
        schema_version: int = BUILTIN_MODEL_REGISTRY_VERSION,
    ) -> None:
        if schema_version != BUILTIN_MODEL_REGISTRY_VERSION:
            raise ModelRegistryError(
                f"unsupported registry version {schema_version}; "
                f"expected {BUILTIN_MODEL_REGISTRY_VERSION}"
            )
        by_name = {item.name: item for item in definitions}
        by_recipe = {item.name: item for item in recipes}
        if len(by_name) != len(definitions):
            raise ModelRegistryError("model definition names must be unique")
        if len(by_recipe) != len(recipes):
            raise ModelRegistryError("model recipe names must be unique")
        for name, definition in by_name.items():
            if definition.default_recipe not in by_recipe:
                raise ModelRegistryError(
                    f"model definition {name!r} references unknown default recipe"
                )
            recipe = by_recipe[definition.default_recipe]
            if recipe.family != name or recipe.backend != definition.backend:
                raise ModelRegistryError(
                    f"default recipe {recipe.name!r} does not match model definition {name!r}"
                )
        for recipe in recipes:
            if recipe.family not in by_name:
                raise ModelRegistryError(
                    f"recipe {recipe.name!r} references unknown family {recipe.family!r}"
                )
            if recipe.backend != by_name[recipe.family].backend:
                raise ModelRegistryError(f"recipe {recipe.name!r} backend differs from its family")
        self.schema_version = schema_version
        self._definitions = MappingProxyType(dict(sorted(by_name.items())))
        self._recipes = MappingProxyType(dict(sorted(by_recipe.items())))

    @property
    def names(self) -> tuple[str, ...]:
        """Return model family names in deterministic order."""
        return tuple(self._definitions)

    @property
    def recipe_names(self) -> tuple[str, ...]:
        """Return recipe names in deterministic order."""
        return tuple(self._recipes)

    def definition(self, name: str) -> ModelFamilyDefinition:
        """Return one explicit built-in definition."""
        try:
            return self._definitions[name]
        except KeyError as exc:
            raise ModelRegistryError(f"unknown model family {name!r}") from exc

    def recipe(self, name: str) -> ModelRecipe:
        """Return one immutable named recipe."""
        try:
            return self._recipes[name]
        except KeyError as exc:
            raise ModelRegistryError(f"unknown model recipe {name!r}") from exc

    def resolve(
        self,
        family: str,
        *,
        input_schema: InputSchema,
        parameters: Mapping[str, Any] | None = None,
        recipe: str | None = None,
    ) -> ResolvedModelDefinition:
        """Validate compatibility and return typed plus persistable resolved configuration."""
        definition = self.definition(family)
        selected = self.recipe(recipe or definition.default_recipe)
        if selected.family != family:
            raise ModelRegistryError(
                f"recipe {selected.name!r} belongs to family {selected.family!r}, not {family!r}"
            )
        selected.assert_input_compatible(input_schema)
        architecture = selected.resolve(parameters)
        config = definition.config_type.from_dict(architecture.parameters)
        in_channels = getattr(config, "in_channels", None)
        if in_channels is not None and in_channels != len(input_schema.channels):
            raise ModelRegistryError(
                f"model family {family!r} resolves {in_channels} input channels but "
                f"the input schema declares {len(input_schema.channels)}"
            )
        return ResolvedModelDefinition(
            family=family,
            backend=definition.backend,
            recipe_id=selected.recipe_id,
            architecture_schema_version=definition.architecture_schema_version,
            architecture=architecture,
            capabilities=definition.capabilities,
            config=config,
        )

    def build(self, resolved: ResolvedModelDefinition) -> Any:
        """Construct a backend model from an already validated resolution."""
        definition = self.definition(resolved.family)
        if resolved.backend != definition.backend:
            raise ModelRegistryError("resolved model backend differs from registry definition")
        if resolved.architecture_schema_version != definition.architecture_schema_version:
            raise ModelRegistryError("resolved architecture schema version differs from registry")
        recipe = self.recipe(resolved.architecture.name)
        if recipe.recipe_id != resolved.recipe_id or recipe.family != resolved.family:
            raise ModelRegistryError("resolved recipe identity differs from registry")
        if recipe.version != resolved.architecture.version:
            raise ModelRegistryError("resolved recipe version differs from registry")
        if resolved.capabilities != definition.capabilities:
            raise ModelRegistryError("resolved capabilities differ from registry definition")
        expected_config = definition.config_type.from_dict(resolved.architecture.parameters)
        if expected_config != resolved.config:
            raise ModelRegistryError("resolved typed config differs from persisted architecture")
        return definition.builder(resolved.config)


def _capabilities(
    *,
    incremental_fit: bool,
    sample_weights: bool,
) -> PredictorCapabilities:
    return PredictorCapabilities(
        schema_version=ML_CAPABILITY_SCHEMA_VERSION,
        backend="sklearn",
        probability_output=True,
        incremental_fit=incremental_fit,
        sample_weights=sample_weights,
        position_masks=False,
        gradients=False,
        convolutional_layers=False,
        attention_data=False,
        supported_mask_kinds=(),
        required_mask_kinds=(),
    )


def _torch_residual_capabilities() -> PredictorCapabilities:
    return PredictorCapabilities(
        schema_version=ML_CAPABILITY_SCHEMA_VERSION,
        backend="torch",
        probability_output=True,
        incremental_fit=False,
        sample_weights=True,
        position_masks=True,
        gradients=True,
        convolutional_layers=True,
        attention_data=False,
        supported_mask_kinds=("observed", "availability", "design", "padding"),
        required_mask_kinds=(),
    )


def _build_bernoulli_nb(config: BernoulliNBConfig) -> Any:
    sklearn_naive_bayes = require(
        "sklearn.naive_bayes",
        extra="ml-base",
        purpose="Bernoulli Naive Bayes models",
    )
    return sklearn_naive_bayes.BernoulliNB(**config.to_dict())


def _build_logistic_regression(config: LogisticRegressionConfig) -> Any:
    sklearn_linear = require(
        "sklearn.linear_model",
        extra="ml-base",
        purpose="logistic regression models",
    )
    parameters = config.to_dict()
    if parameters["penalty"] == "l2":
        # The sklearn default has represented L2 regularization across all
        # supported versions; omitting it also avoids the 1.8+ deprecation of
        # explicitly passing the legacy ``penalty`` parameter.
        parameters.pop("penalty")
    return sklearn_linear.LogisticRegression(**parameters)


def _build_random_forest(config: RandomForestConfig) -> Any:
    sklearn_ensemble = require(
        "sklearn.ensemble",
        extra="ml-base",
        purpose="random forest models",
    )
    return sklearn_ensemble.RandomForestClassifier(**config.to_dict())


def _build_residual_cnn(config: ResidualCNNConfig) -> Any:
    return build_residual_cnn(config)


def _recipe(name: str, parameters: Mapping[str, Any]) -> ModelRecipe:
    return ModelRecipe.create(
        name=f"{name}_v1",
        version="1",
        family=name,
        backend="sklearn",
        parameters=parameters,
        supported_modalities=_SUPPORTED_MODALITIES,
        supported_channel_roles=("*",),
    )


_BERNOULLI_RECIPE = _recipe("bernoulli_nb", BernoulliNBConfig().to_dict())
_LOGISTIC_RECIPE = _recipe("logistic_regression", LogisticRegressionConfig().to_dict())
_FOREST_RECIPE = _recipe("random_forest", RandomForestConfig().to_dict())
_RESIDUAL_CNN_RECIPE = ModelRecipe.create(
    name="residual_dilated_cnn_v1",
    version="1",
    family="residual_dilated_cnn",
    backend="torch",
    parameters=ResidualCNNConfig(in_channels=1).to_dict(),
    supported_modalities=_SUPPORTED_MODALITIES,
    supported_channel_roles=("*",),
)

BUILTIN_MODEL_REGISTRY = ModelRegistry(
    definitions=(
        ModelFamilyDefinition(
            name="bernoulli_nb",
            backend="sklearn",
            architecture_schema_version=1,
            config_type=BernoulliNBConfig,
            builder=_build_bernoulli_nb,
            capabilities=_capabilities(incremental_fit=True, sample_weights=True),
            default_recipe=_BERNOULLI_RECIPE.name,
        ),
        ModelFamilyDefinition(
            name="logistic_regression",
            backend="sklearn",
            architecture_schema_version=1,
            config_type=LogisticRegressionConfig,
            builder=_build_logistic_regression,
            capabilities=_capabilities(incremental_fit=False, sample_weights=True),
            default_recipe=_LOGISTIC_RECIPE.name,
        ),
        ModelFamilyDefinition(
            name="random_forest",
            backend="sklearn",
            architecture_schema_version=1,
            config_type=RandomForestConfig,
            builder=_build_random_forest,
            capabilities=_capabilities(incremental_fit=False, sample_weights=True),
            default_recipe=_FOREST_RECIPE.name,
        ),
        ModelFamilyDefinition(
            name="residual_dilated_cnn",
            backend="torch",
            architecture_schema_version=1,
            config_type=ResidualCNNConfig,
            builder=_build_residual_cnn,
            capabilities=_torch_residual_capabilities(),
            default_recipe=_RESIDUAL_CNN_RECIPE.name,
        ),
    ),
    recipes=(
        _BERNOULLI_RECIPE,
        _LOGISTIC_RECIPE,
        _FOREST_RECIPE,
        _RESIDUAL_CNN_RECIPE,
    ),
)
