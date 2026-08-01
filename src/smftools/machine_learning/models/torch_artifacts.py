"""Safe publication and loading of plain-Torch inference artifacts."""

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
    from ..training.torch_backend import FittedTorchModel

TORCH_ARTIFACT_SCHEMA_VERSION = 1
TORCH_ARTIFACT_FILENAME = "payload/model.pt"
_VERSIONED_PACKAGES = ("numpy", "torch")


class TorchArtifactError(ValueError):
    """Raised when a Torch artifact violates its schema or trust policy."""


@dataclass(frozen=True)
class PublishedTorchModel:
    """Published Torch model manifest and validated immutable bundle."""

    manifest: ModelManifest
    bundle: PublishedBundle


def _package_versions() -> dict[str, str]:
    return {name: version(name) for name in _VERSIONED_PACKAGES}


def _metadata(model: Any) -> dict[str, Any]:
    return {
        "schema_version": TORCH_ARTIFACT_SCHEMA_VERSION,
        "family": model.family,
        "architecture": model.architecture.architecture.to_dict(),
        "input_schema": model.input_schema.to_dict(),
        "label_schema": model.label_schema.to_dict(),
        "dataset_snapshot_id": model.dataset_snapshot_id,
        "split_id": model.split_id,
        "transform": model.transform.to_dict(),
        "training_config": model.training_config.to_dict(),
        "resolved_device": model.resolved_device,
        "best_epoch": model.best_epoch,
        "history": [row.to_dict() for row in model.history],
        "validation_loss": model.validation_loss,
        "test_loss": model.test_loss,
    }


def _artifact_path(bundle: PublishedBundle, manifest: ModelManifest) -> Path:
    relative = Path(manifest.artifact.relative_path)
    prefix = Path("models") / manifest.model_id
    try:
        relative = relative.relative_to(prefix)
    except ValueError:
        pass
    return bundle.path / relative


def _cpu_state_dict(model: Any) -> dict[str, Any]:
    torch = require("torch", extra="ml-base", purpose="plain Torch model persistence")
    result = {}
    for name, value in model.state_dict().items():
        if not isinstance(name, str) or not torch.is_tensor(value):
            raise TorchArtifactError("Torch state_dict must map string names to tensors")
        result[name] = value.detach().cpu()
    return result


def _assert_state_compatible(model: Any, expected_model: Any) -> None:
    observed = model.state_dict()
    expected = expected_model.state_dict()
    if tuple(observed) != tuple(expected):
        raise TorchArtifactError("fitted state_dict keys differ from the resolved architecture")
    mismatches = [
        name
        for name, value in observed.items()
        if value.shape != expected[name].shape or value.dtype != expected[name].dtype
    ]
    if mismatches:
        raise TorchArtifactError(
            f"fitted state_dict tensors differ from the resolved architecture: {mismatches}"
        )


def publish_torch_model(
    model: Any,
    workspace: MLWorkspace,
    *,
    model_key: str,
    originating_run_id: str,
    environment: EnvironmentRecord,
    created_at: str,
    lineage: ModelLineage | None = None,
) -> PublishedTorchModel:
    """Publish JSON metadata and a plain state dict in one immutable payload."""
    expected_model = BUILTIN_MODEL_REGISTRY.build(model.architecture)
    if type(model.model) is not type(expected_model):
        raise TorchArtifactError("fitted module type differs from its registered family")
    _assert_state_compatible(model.model, expected_model)
    torch = require("torch", extra="ml-base", purpose="plain Torch model persistence")
    with tempfile.TemporaryDirectory(prefix="smftools-torch-") as temporary:
        source = Path(temporary) / "model.pt"
        torch.save(
            {
                "metadata_json": json.dumps(
                    _metadata(model),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ),
                "state_dict": _cpu_state_dict(model.model),
            },
            source,
        )
        reference = ArtifactReference(
            role="model",
            relative_path=TORCH_ARTIFACT_FILENAME,
            sha256=file_sha256(source),
            size_bytes=source.stat().st_size,
            media_type="application/vnd.pytorch.state-dict",
        )
        manifest = ModelManifest.create(
            model_key=model_key,
            backend="torch",
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
                format="torch-state-dict",
                loader=(
                    "smftools.machine_learning.models.torch_artifacts.load_published_torch_model"
                ),
                requires_unsafe_load=False,
                allowed_types=(),
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
    return PublishedTorchModel(manifest=manifest, bundle=bundle)


def _load_payload(path: Path, device: str) -> tuple[dict[str, Any], dict[str, Any]]:
    torch = require("torch", extra="ml-base", purpose="safe plain Torch model loading")
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except Exception as exc:
        raise TorchArtifactError(
            f"Torch state-dict payload could not be loaded safely: {exc}"
        ) from exc
    if not isinstance(payload, dict) or set(payload) != {"metadata_json", "state_dict"}:
        raise TorchArtifactError("Torch payload has an invalid top-level schema")
    if not isinstance(payload["metadata_json"], str):
        raise TorchArtifactError("Torch payload metadata must be canonical JSON text")
    try:
        metadata = json.loads(payload["metadata_json"])
    except (TypeError, ValueError) as exc:
        raise TorchArtifactError("Torch payload metadata is invalid JSON") from exc
    try:
        canonical = json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise TorchArtifactError("Torch payload metadata is not finite canonical JSON") from exc
    if canonical != payload["metadata_json"]:
        raise TorchArtifactError("Torch payload metadata is not canonical JSON")
    state_dict = payload["state_dict"]
    if not isinstance(state_dict, dict) or not all(
        isinstance(name, str) and torch.is_tensor(value) for name, value in state_dict.items()
    ):
        raise TorchArtifactError("Torch payload state_dict must map string names to tensors")
    return metadata, state_dict


def load_published_torch_model(
    workspace: MLWorkspace,
    model_id: str,
    *,
    registry: ModelRegistry = BUILTIN_MODEL_REGISTRY,
    device: str = "cpu",
    allow_version_mismatch: bool = False,
) -> FittedTorchModel:
    """Validate, reconstruct, and safely load one plain-Torch inference model."""
    from ..training.torch_backend import (
        FittedTorchModel,
        TorchEpochRecord,
        TorchTrainingConfig,
    )

    bundle = validate_published_bundle(
        workspace,
        workspace.model_dir(model_id),
        kind="model",
        expected_id=model_id,
    )
    with (bundle.path / "model_manifest.json").open(encoding="utf-8") as handle:
        manifest = ModelManifest.from_dict(json.load(handle))
    policy = manifest.serialization
    expected_loader = "smftools.machine_learning.models.torch_artifacts.load_published_torch_model"
    if manifest.backend != "torch":
        raise TorchArtifactError("model manifest does not describe a Torch backend")
    if policy.format != "torch-state-dict" or policy.loader != expected_loader:
        raise TorchArtifactError("only canonical Torch state-dict artifacts are supported")
    if policy.requires_unsafe_load or policy.allowed_types:
        raise TorchArtifactError("Torch state-dict policy must require no unsafe or custom types")
    if not allow_version_mismatch and dict(policy.package_versions) != _package_versions():
        raise TorchArtifactError("Torch artifact dependency versions differ from the environment")
    metadata, state_dict = _load_payload(_artifact_path(bundle, manifest), device)
    expected_fields = {
        "schema_version",
        "family",
        "architecture",
        "input_schema",
        "label_schema",
        "dataset_snapshot_id",
        "split_id",
        "transform",
        "training_config",
        "resolved_device",
        "best_epoch",
        "history",
        "validation_loss",
        "test_loss",
    }
    if not isinstance(metadata, dict) or set(metadata) != expected_fields:
        raise TorchArtifactError("Torch metadata has an invalid schema")
    if metadata["schema_version"] != TORCH_ARTIFACT_SCHEMA_VERSION:
        raise TorchArtifactError("unsupported Torch artifact schema version")
    if metadata["resolved_device"] not in {"cpu", "cuda", "mps"}:
        raise TorchArtifactError("payload resolved_device is invalid")
    input_schema = InputSchema.from_dict(metadata["input_schema"])
    label_schema = LabelSchema.from_dict(metadata["label_schema"])
    if input_schema.schema_hash != manifest.input_schema_hash:
        raise TorchArtifactError("payload input schema differs from model manifest")
    if label_schema.schema_hash != manifest.label_schema_hash:
        raise TorchArtifactError("payload label schema differs from model manifest")
    if metadata["architecture"] != manifest.architecture.to_dict():
        raise TorchArtifactError("payload architecture differs from model manifest")
    if metadata["family"] != manifest.family:
        raise TorchArtifactError("payload family differs from model manifest")
    if metadata["dataset_snapshot_id"] != manifest.dataset_snapshot_id:
        raise TorchArtifactError("payload dataset snapshot differs from model manifest")
    if metadata["split_id"] != manifest.split_id:
        raise TorchArtifactError("payload split differs from model manifest")
    resolved = registry.resolve(
        manifest.family,
        input_schema=input_schema,
        parameters=manifest.architecture.parameters,
        recipe=manifest.architecture.name,
    )
    model = registry.build(resolved)
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        raise TorchArtifactError(
            f"state_dict differs from the resolved architecture: {exc}"
        ) from exc
    model.to(device)
    return FittedTorchModel(
        family=manifest.family,
        architecture=resolved,
        model=model,
        transform=FittedFeatureTransform.from_dict(metadata["transform"]),
        input_schema=input_schema,
        label_schema=label_schema,
        dataset_snapshot_id=manifest.dataset_snapshot_id,
        split_id=manifest.split_id,
        training_config=TorchTrainingConfig.from_dict(metadata["training_config"]),
        resolved_device=device,
        best_epoch=metadata["best_epoch"],
        history=tuple(TorchEpochRecord.from_dict(row) for row in metadata["history"]),
        validation_loss=metadata["validation_loss"],
        test_loss=metadata["test_loss"],
    )
