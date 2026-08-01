"""Safe persistence and loading for supported sklearn model artifacts."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any

from smftools.optional_imports import require

from ..artifacts import (
    ArtifactReference,
    EnvironmentRecord,
    ModelLineage,
    ModelManifest,
    PublishedBundle,
    SerializationPolicy,
    file_sha256,
    publish_bundle,
    validate_published_bundle,
)
from ..contracts import InputSchema, LabelSchema
from ..data.transforms import FittedFeatureTransform
from ..workspace import MLWorkspace
from .registry import BUILTIN_MODEL_REGISTRY, ModelRegistry

if TYPE_CHECKING:
    from ..training.sklearn_backend import FittedSklearnModel

SKLEARN_ARTIFACT_SCHEMA_VERSION = 1
SKLEARN_ARTIFACT_FILENAME = "payload/model.skops"
_VERSIONED_PACKAGES = ("numpy", "scipy", "scikit-learn", "skops")


class SklearnArtifactError(ValueError):
    """Raised when a sklearn artifact violates trust or compatibility policy."""


@dataclass(frozen=True)
class PublishedSklearnModel:
    """Published sklearn model manifest and validated immutable bundle."""

    manifest: ModelManifest
    bundle: PublishedBundle


def _qualified_type(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _package_versions() -> dict[str, str]:
    return {name: version(name) for name in _VERSIONED_PACKAGES}


def _payload(model: Any) -> dict[str, Any]:
    return {
        "schema_version": SKLEARN_ARTIFACT_SCHEMA_VERSION,
        "family": model.family,
        "architecture": model.architecture.architecture.to_dict(),
        "input_schema": model.input_schema.to_dict(),
        "label_schema": model.label_schema.to_dict(),
        "dataset_snapshot_id": model.dataset_snapshot_id,
        "split_id": model.split_id,
        "fit_mode": model.fit_mode,
        "transform": model.transform.to_dict(),
        "native_parameters": dict(model.native_parameters),
        "estimator": model.estimator,
    }


def _artifact_path(bundle: PublishedBundle, manifest: ModelManifest) -> Path:
    relative = Path(manifest.artifact.relative_path)
    prefix = Path("models") / manifest.model_id
    try:
        relative = relative.relative_to(prefix)
    except ValueError:
        pass
    return bundle.path / relative


def publish_sklearn_model(
    model: Any,
    workspace: MLWorkspace,
    *,
    model_key: str,
    originating_run_id: str,
    environment: EnvironmentRecord,
    created_at: str,
    lineage: ModelLineage | None = None,
) -> PublishedSklearnModel:
    """Serialize and atomically publish a fitted sklearn inference artifact.

    The serializer inspects every type that ``skops`` considers untrusted and
    refuses the payload unless the only such type is the exact registered
    estimator class. No pickle or joblib fallback is attempted.
    """
    skops_io = require("skops.io", extra="ml-base", purpose="safe sklearn persistence")
    expected_type = _qualified_type(model.estimator)
    expected_estimator = BUILTIN_MODEL_REGISTRY.build(model.architecture)
    if type(model.estimator) is not type(expected_estimator):
        raise SklearnArtifactError("fitted estimator type differs from its registered family")
    with tempfile.TemporaryDirectory(prefix="smftools-sklearn-") as temporary:
        source = Path(temporary) / "model.skops"
        skops_io.dump(_payload(model), source)
        unknown_types = tuple(sorted(skops_io.get_untrusted_types(file=source)))
        unexpected = sorted(set(unknown_types).difference({expected_type}))
        if unexpected:
            raise SklearnArtifactError(f"skops payload contains unreviewed types: {unexpected}")
        reference = ArtifactReference(
            role="model",
            relative_path=SKLEARN_ARTIFACT_FILENAME,
            sha256=file_sha256(source),
            size_bytes=source.stat().st_size,
            media_type="application/vnd.skops+zip",
        )
        manifest = ModelManifest.create(
            model_key=model_key,
            backend="sklearn",
            family=model.family,
            task_type=model.label_schema.task_type,
            originating_run_id=originating_run_id,
            workspace_id=workspace.workspace_id,
            dataset_snapshot_id=model.dataset_snapshot_id,
            split_id=model.split_id,
            input_schema_hash=model.input_schema.schema_hash,
            label_schema_hash=model.label_schema.schema_hash,
            architecture=model.architecture.architecture,
            lineage=lineage
            or ModelLineage(kind="from_scratch", parent_model_ids=(), parent_roles=()),
            artifact=reference,
            serialization=SerializationPolicy(
                format="skops",
                loader="skops.io.load",
                requires_unsafe_load=False,
                allowed_types=tuple(sorted({expected_type, *unknown_types})),
                package_versions=_package_versions(),
            ),
            environment=environment,
            created_at=created_at,
        )
        bundle = publish_bundle(
            workspace,
            manifest,
            sources={reference.relative_path: source},
        )
    return PublishedSklearnModel(manifest=manifest, bundle=bundle)


def load_published_sklearn_model(
    workspace: MLWorkspace,
    model_id: str,
    *,
    registry: ModelRegistry = BUILTIN_MODEL_REGISTRY,
    allow_version_mismatch: bool = False,
) -> FittedSklearnModel:
    """Validate and load one published sklearn artifact under its trust policy."""
    from ..training.sklearn_backend import FittedSklearnModel

    bundle = validate_published_bundle(
        workspace,
        workspace.model_dir(model_id),
        kind="model",
        expected_id=model_id,
    )
    manifest_path = bundle.path / "model_manifest.json"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = ModelManifest.from_dict(json.load(handle))
    policy = manifest.serialization
    if manifest.backend != "sklearn":
        raise SklearnArtifactError("model manifest does not describe an sklearn backend")
    if policy.format != "skops" or policy.loader != "skops.io.load":
        raise SklearnArtifactError("only canonical skops sklearn artifacts are supported")
    if policy.requires_unsafe_load:
        raise SklearnArtifactError("unsafe sklearn artifacts are never loaded implicitly")
    current_versions = _package_versions()
    if not allow_version_mismatch and dict(policy.package_versions) != current_versions:
        raise SklearnArtifactError(
            "sklearn artifact dependency versions differ from the active environment"
        )
    definition = registry.definition(manifest.family)
    config = definition.config_type.from_dict(manifest.architecture.parameters)
    expected_type = _qualified_type(definition.builder(config))
    if set(policy.allowed_types) != {expected_type}:
        raise SklearnArtifactError("serialization policy differs from reviewed model type")
    skops_io = require("skops.io", extra="ml-base", purpose="safe sklearn persistence")
    source = _artifact_path(bundle, manifest)
    unknown_types = tuple(sorted(skops_io.get_untrusted_types(file=source)))
    unexpected = sorted(set(unknown_types).difference({expected_type}))
    if unexpected:
        raise SklearnArtifactError(f"skops payload contains unapproved types: {unexpected}")
    payload = skops_io.load(source, trusted=list(unknown_types))
    expected_fields = {
        "schema_version",
        "family",
        "architecture",
        "input_schema",
        "label_schema",
        "dataset_snapshot_id",
        "split_id",
        "fit_mode",
        "transform",
        "native_parameters",
        "estimator",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise SklearnArtifactError("skops payload has an invalid top-level schema")
    if payload["schema_version"] != SKLEARN_ARTIFACT_SCHEMA_VERSION:
        raise SklearnArtifactError("unsupported sklearn artifact schema version")
    input_schema = InputSchema.from_dict(payload["input_schema"])
    label_schema = LabelSchema.from_dict(payload["label_schema"])
    if input_schema.schema_hash != manifest.input_schema_hash:
        raise SklearnArtifactError("payload input schema differs from model manifest")
    if label_schema.schema_hash != manifest.label_schema_hash:
        raise SklearnArtifactError("payload label schema differs from model manifest")
    if payload["architecture"] != manifest.architecture.to_dict():
        raise SklearnArtifactError("payload architecture differs from model manifest")
    if payload["family"] != manifest.family:
        raise SklearnArtifactError("payload family differs from model manifest")
    if payload["dataset_snapshot_id"] != manifest.dataset_snapshot_id:
        raise SklearnArtifactError("payload dataset snapshot differs from model manifest")
    if payload["split_id"] != manifest.split_id:
        raise SklearnArtifactError("payload split differs from model manifest")
    resolved = registry.resolve(
        manifest.family,
        input_schema=input_schema,
        parameters=manifest.architecture.parameters,
        recipe=manifest.architecture.name,
    )
    estimator = payload["estimator"]
    expected_estimator = registry.build(resolved)
    if type(estimator) is not type(expected_estimator):
        raise SklearnArtifactError("loaded estimator type differs from registered family")
    if _qualified_type(estimator) not in policy.allowed_types:
        raise SklearnArtifactError("loaded estimator type is absent from reviewed allowlist")
    return FittedSklearnModel(
        family=manifest.family,
        architecture=resolved,
        estimator=estimator,
        transform=FittedFeatureTransform.from_dict(payload["transform"]),
        input_schema=input_schema,
        label_schema=label_schema,
        dataset_snapshot_id=manifest.dataset_snapshot_id,
        split_id=manifest.split_id,
        fit_mode=str(payload["fit_mode"]),
        native_parameters=payload["native_parameters"],
    )
