"""Prediction and explanation result identity manifests."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..contracts import MASK_KINDS
from ._validation import (
    canonical_json,
    digest,
    fail,
    integer,
    keys,
    mapping,
    optional_string,
    sequence,
    sha256,
    string,
    strings,
    timestamp,
    version,
)
from .common import ArtifactReference, ResolvedDefinition

ML_PREDICTION_MANIFEST_VERSION = 1
ML_EXPLANATION_MANIFEST_VERSION = 1
PREDICTION_SPLIT_ROLES = frozenset({"train", "validation", "test", "inference"})


def _run_id(value: Any, path: str) -> str:
    result = string(value, path)
    try:
        return str(uuid.UUID(result))
    except ValueError:
        fail(path, "must be a UUID")


@dataclass(frozen=True)
class PredictionManifest:
    """Identity and table schema for one immutable prediction cohort."""

    schema_version: int
    prediction_id: str
    run_id: str
    workspace_id: str
    model_id: str
    dataset_snapshot_id: str
    input_schema_hash: str
    label_schema_hash: str | None
    cohort: str
    split_role: str | None
    n_observations: int
    identity_columns: tuple[str, ...]
    prediction_columns: tuple[str, ...]
    table: ArtifactReference
    created_at: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "identity_columns", tuple(self.identity_columns))
        object.__setattr__(self, "prediction_columns", tuple(self.prediction_columns))
        _validate_prediction(self)

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        workspace_id: str,
        model_id: str,
        dataset_snapshot_id: str,
        input_schema_hash: str,
        label_schema_hash: str | None,
        cohort: str,
        split_role: str | None,
        n_observations: int,
        identity_columns: tuple[str, ...],
        prediction_columns: tuple[str, ...],
        table: ArtifactReference,
        created_at: str,
    ) -> PredictionManifest:
        """Create a content-addressed prediction table manifest."""
        payload = {
            "run_id": run_id,
            "workspace_id": workspace_id,
            "model_id": model_id,
            "dataset_snapshot_id": dataset_snapshot_id,
            "input_schema_hash": input_schema_hash,
            "label_schema_hash": label_schema_hash,
            "cohort": cohort,
            "split_role": split_role,
            "n_observations": n_observations,
            "identity_columns": list(identity_columns),
            "prediction_columns": list(prediction_columns),
            "table": table.to_dict(),
        }
        return cls(
            schema_version=ML_PREDICTION_MANIFEST_VERSION,
            prediction_id=sha256(payload),
            run_id=run_id,
            workspace_id=workspace_id,
            model_id=model_id,
            dataset_snapshot_id=dataset_snapshot_id,
            input_schema_hash=input_schema_hash,
            label_schema_hash=label_schema_hash,
            cohort=cohort,
            split_role=split_role,
            n_observations=n_observations,
            identity_columns=identity_columns,
            prediction_columns=prediction_columns,
            table=table,
            created_at=created_at,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the complete JSON-serializable prediction manifest."""
        return {
            "schema_version": self.schema_version,
            "prediction_id": self.prediction_id,
            "run_id": self.run_id,
            "workspace_id": self.workspace_id,
            "model_id": self.model_id,
            "dataset_snapshot_id": self.dataset_snapshot_id,
            "input_schema_hash": self.input_schema_hash,
            "label_schema_hash": self.label_schema_hash,
            "cohort": self.cohort,
            "split_role": self.split_role,
            "n_observations": self.n_observations,
            "identity_columns": list(self.identity_columns),
            "prediction_columns": list(self.prediction_columns),
            "table": self.table.to_dict(),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> PredictionManifest:
        """Validate and restore a version-1 prediction manifest."""
        value = mapping(raw, "prediction")
        fields = {
            "schema_version",
            "prediction_id",
            "run_id",
            "workspace_id",
            "model_id",
            "dataset_snapshot_id",
            "input_schema_hash",
            "label_schema_hash",
            "cohort",
            "split_role",
            "n_observations",
            "identity_columns",
            "prediction_columns",
            "table",
            "created_at",
        }
        keys(value, path="prediction", fields=fields)
        return cls(
            schema_version=version(
                value["schema_version"],
                ML_PREDICTION_MANIFEST_VERSION,
                "prediction.schema_version",
            ),
            prediction_id=digest(value["prediction_id"], "prediction.prediction_id"),
            run_id=_run_id(value["run_id"], "prediction.run_id"),
            workspace_id=digest(value["workspace_id"], "prediction.workspace_id"),
            model_id=digest(value["model_id"], "prediction.model_id"),
            dataset_snapshot_id=digest(
                value["dataset_snapshot_id"], "prediction.dataset_snapshot_id"
            ),
            input_schema_hash=digest(value["input_schema_hash"], "prediction.input_schema_hash"),
            label_schema_hash=(
                None
                if value["label_schema_hash"] is None
                else digest(value["label_schema_hash"], "prediction.label_schema_hash")
            ),
            cohort=string(value["cohort"], "prediction.cohort"),
            split_role=optional_string(value["split_role"], "prediction.split_role"),
            n_observations=integer(
                value["n_observations"],
                "prediction.n_observations",
                minimum=1,
            ),
            identity_columns=strings(
                value["identity_columns"],
                "prediction.identity_columns",
                required=True,
            ),
            prediction_columns=strings(
                value["prediction_columns"],
                "prediction.prediction_columns",
                required=True,
            ),
            table=ArtifactReference.from_dict(mapping(value["table"], "prediction.table")),
            created_at=timestamp(value["created_at"], "prediction.created_at"),
        )

    def canonical_json(self) -> str:
        """Return stable canonical JSON."""
        return canonical_json(self.to_dict())


def _prediction_payload(value: PredictionManifest) -> dict[str, Any]:
    raw = value.to_dict()
    raw.pop("schema_version")
    raw.pop("prediction_id")
    raw.pop("created_at")
    return raw


def _validate_prediction(value: PredictionManifest) -> None:
    version(
        value.schema_version,
        ML_PREDICTION_MANIFEST_VERSION,
        "prediction.schema_version",
    )
    digest(value.prediction_id, "prediction.prediction_id")
    _run_id(value.run_id, "prediction.run_id")
    digest(value.workspace_id, "prediction.workspace_id")
    digest(value.model_id, "prediction.model_id")
    digest(value.dataset_snapshot_id, "prediction.dataset_snapshot_id")
    digest(value.input_schema_hash, "prediction.input_schema_hash")
    if value.label_schema_hash is not None:
        digest(value.label_schema_hash, "prediction.label_schema_hash")
    string(value.cohort, "prediction.cohort")
    if value.split_role is not None and value.split_role not in PREDICTION_SPLIT_ROLES:
        fail(
            "prediction.split_role",
            f"must be one of {sorted(PREDICTION_SPLIT_ROLES)} or null",
        )
    integer(value.n_observations, "prediction.n_observations", minimum=1)
    if len(value.identity_columns) != len(set(value.identity_columns)):
        fail("prediction.identity_columns", "cannot contain duplicates")
    if "molecule_uid" not in value.identity_columns:
        fail("prediction.identity_columns", "must include 'molecule_uid'")
    if not value.prediction_columns:
        fail("prediction.prediction_columns", "must contain at least one column")
    if len(value.prediction_columns) != len(set(value.prediction_columns)):
        fail("prediction.prediction_columns", "cannot contain duplicates")
    overlap = sorted(set(value.identity_columns).intersection(value.prediction_columns))
    if overlap:
        fail("prediction", f"identity and prediction columns overlap: {overlap}")
    if value.table.role != "predictions":
        fail("prediction.table.role", "must be 'predictions'")
    if value.table.size_bytes == 0:
        fail("prediction.table.size_bytes", "prediction table cannot be empty")
    timestamp(value.created_at, "prediction.created_at")
    if value.prediction_id != sha256(_prediction_payload(value)):
        fail("prediction.prediction_id", "does not match prediction content")


@dataclass(frozen=True)
class ExplanationTarget:
    """Declared model output explained by an explanation artifact."""

    output_name: str
    class_id: int | None
    class_name: str | None

    def __post_init__(self) -> None:
        string(self.output_name, "target.output_name")
        if self.class_id is not None:
            integer(self.class_id, "target.class_id")
        if self.class_name is not None:
            string(self.class_name, "target.class_name")
        if (self.class_id is None) != (self.class_name is None):
            fail("target", "class_id and class_name must either both be set or both be null")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable target."""
        return {
            "output_name": self.output_name,
            "class_id": self.class_id,
            "class_name": self.class_name,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ExplanationTarget:
        """Validate and restore an explanation target."""
        value = mapping(raw, "target")
        fields = {"output_name", "class_id", "class_name"}
        keys(value, path="target", fields=fields)
        class_id = value["class_id"]
        if class_id is not None:
            class_id = integer(class_id, "target.class_id")
        return cls(
            output_name=string(value["output_name"], "target.output_name"),
            class_id=class_id,
            class_name=optional_string(value["class_name"], "target.class_name"),
        )


@dataclass(frozen=True)
class ExplanationBaseline:
    """Exact baseline or background cohort used by an explanation method."""

    kind: str
    description: str
    baseline_hash: str
    dataset_snapshot_id: str | None
    cohort: str | None

    def __post_init__(self) -> None:
        string(self.kind, "baseline.kind")
        string(self.description, "baseline.description")
        digest(self.baseline_hash, "baseline.baseline_hash")
        if self.dataset_snapshot_id is not None:
            digest(self.dataset_snapshot_id, "baseline.dataset_snapshot_id")
        if self.cohort is not None:
            string(self.cohort, "baseline.cohort")
            if self.cohort.lower() == "test":
                fail("baseline.cohort", "locked test data cannot define an explanation baseline")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable baseline."""
        return {
            "kind": self.kind,
            "description": self.description,
            "baseline_hash": self.baseline_hash,
            "dataset_snapshot_id": self.dataset_snapshot_id,
            "cohort": self.cohort,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ExplanationBaseline:
        """Validate and restore an explanation baseline."""
        value = mapping(raw, "baseline")
        fields = {
            "kind",
            "description",
            "baseline_hash",
            "dataset_snapshot_id",
            "cohort",
        }
        keys(value, path="baseline", fields=fields)
        dataset_id = value["dataset_snapshot_id"]
        return cls(
            kind=string(value["kind"], "baseline.kind"),
            description=string(value["description"], "baseline.description"),
            baseline_hash=digest(value["baseline_hash"], "baseline.baseline_hash"),
            dataset_snapshot_id=(
                None if dataset_id is None else digest(dataset_id, "baseline.dataset_snapshot_id")
            ),
            cohort=optional_string(value["cohort"], "baseline.cohort"),
        )


@dataclass(frozen=True)
class ExplanationMaskPolicy:
    """Named mask inputs and handling semantics used during explanation."""

    mask_kinds: tuple[str, ...]
    handling: str
    policy_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "mask_kinds", tuple(sorted(self.mask_kinds)))
        if len(self.mask_kinds) != len(set(self.mask_kinds)):
            fail("mask_policy.mask_kinds", "cannot contain duplicates")
        unknown = sorted(set(self.mask_kinds).difference(MASK_KINDS))
        if unknown:
            fail("mask_policy.mask_kinds", f"contains unknown masks: {unknown}")
        string(self.handling, "mask_policy.handling")
        expected = sha256({"mask_kinds": list(self.mask_kinds), "handling": self.handling})
        if self.policy_hash != expected:
            fail("mask_policy.policy_hash", "does not match mask semantics")

    @classmethod
    def create(
        cls,
        *,
        mask_kinds: tuple[str, ...],
        handling: str,
    ) -> ExplanationMaskPolicy:
        """Create a checksummed mask policy."""
        canonical = tuple(sorted(mask_kinds))
        return cls(
            mask_kinds=canonical,
            handling=handling,
            policy_hash=sha256({"mask_kinds": list(canonical), "handling": handling}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mask policy."""
        return {
            "mask_kinds": list(self.mask_kinds),
            "handling": self.handling,
            "policy_hash": self.policy_hash,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ExplanationMaskPolicy:
        """Validate and restore a mask policy."""
        value = mapping(raw, "mask_policy")
        fields = {"mask_kinds", "handling", "policy_hash"}
        keys(value, path="mask_policy", fields=fields)
        return cls(
            mask_kinds=strings(value["mask_kinds"], "mask_policy.mask_kinds"),
            handling=string(value["handling"], "mask_policy.handling"),
            policy_hash=digest(value["policy_hash"], "mask_policy.policy_hash"),
        )


@dataclass(frozen=True)
class ExplanationManifest:
    """Immutable explanation identity with target, baseline, cohort, and masks."""

    schema_version: int
    explanation_id: str
    run_id: str
    workspace_id: str
    model_id: str
    dataset_snapshot_id: str
    cohort: str
    n_observations: int
    method: ResolvedDefinition
    target: ExplanationTarget
    baseline: ExplanationBaseline | None
    mask_policy: ExplanationMaskPolicy
    feature_axes: tuple[str, ...]
    values: ArtifactReference
    summary: ArtifactReference | None
    created_at: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature_axes", tuple(self.feature_axes))
        _validate_explanation(self)

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        workspace_id: str,
        model_id: str,
        dataset_snapshot_id: str,
        cohort: str,
        n_observations: int,
        method: ResolvedDefinition,
        target: ExplanationTarget,
        baseline: ExplanationBaseline | None,
        mask_policy: ExplanationMaskPolicy,
        feature_axes: tuple[str, ...],
        values: ArtifactReference,
        summary: ArtifactReference | None,
        created_at: str,
    ) -> ExplanationManifest:
        """Create a content-addressed explanation manifest."""
        payload = {
            "run_id": run_id,
            "workspace_id": workspace_id,
            "model_id": model_id,
            "dataset_snapshot_id": dataset_snapshot_id,
            "cohort": cohort,
            "n_observations": n_observations,
            "method": method.to_dict(),
            "target": target.to_dict(),
            "baseline": baseline.to_dict() if baseline is not None else None,
            "mask_policy": mask_policy.to_dict(),
            "feature_axes": list(feature_axes),
            "values": values.to_dict(),
            "summary": summary.to_dict() if summary is not None else None,
        }
        return cls(
            schema_version=ML_EXPLANATION_MANIFEST_VERSION,
            explanation_id=sha256(payload),
            run_id=run_id,
            workspace_id=workspace_id,
            model_id=model_id,
            dataset_snapshot_id=dataset_snapshot_id,
            cohort=cohort,
            n_observations=n_observations,
            method=method,
            target=target,
            baseline=baseline,
            mask_policy=mask_policy,
            feature_axes=feature_axes,
            values=values,
            summary=summary,
            created_at=created_at,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the complete JSON-serializable explanation manifest."""
        return {
            "schema_version": self.schema_version,
            "explanation_id": self.explanation_id,
            "run_id": self.run_id,
            "workspace_id": self.workspace_id,
            "model_id": self.model_id,
            "dataset_snapshot_id": self.dataset_snapshot_id,
            "cohort": self.cohort,
            "n_observations": self.n_observations,
            "method": self.method.to_dict(),
            "target": self.target.to_dict(),
            "baseline": self.baseline.to_dict() if self.baseline is not None else None,
            "mask_policy": self.mask_policy.to_dict(),
            "feature_axes": list(self.feature_axes),
            "values": self.values.to_dict(),
            "summary": self.summary.to_dict() if self.summary is not None else None,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ExplanationManifest:
        """Validate and restore a version-1 explanation manifest."""
        value = mapping(raw, "explanation")
        fields = {
            "schema_version",
            "explanation_id",
            "run_id",
            "workspace_id",
            "model_id",
            "dataset_snapshot_id",
            "cohort",
            "n_observations",
            "method",
            "target",
            "baseline",
            "mask_policy",
            "feature_axes",
            "values",
            "summary",
            "created_at",
        }
        keys(value, path="explanation", fields=fields)
        baseline_raw = value["baseline"]
        summary_raw = value["summary"]
        return cls(
            schema_version=version(
                value["schema_version"],
                ML_EXPLANATION_MANIFEST_VERSION,
                "explanation.schema_version",
            ),
            explanation_id=digest(value["explanation_id"], "explanation.explanation_id"),
            run_id=_run_id(value["run_id"], "explanation.run_id"),
            workspace_id=digest(value["workspace_id"], "explanation.workspace_id"),
            model_id=digest(value["model_id"], "explanation.model_id"),
            dataset_snapshot_id=digest(
                value["dataset_snapshot_id"], "explanation.dataset_snapshot_id"
            ),
            cohort=string(value["cohort"], "explanation.cohort"),
            n_observations=integer(
                value["n_observations"],
                "explanation.n_observations",
                minimum=1,
            ),
            method=ResolvedDefinition.from_dict(mapping(value["method"], "explanation.method")),
            target=ExplanationTarget.from_dict(mapping(value["target"], "explanation.target")),
            baseline=(
                None
                if baseline_raw is None
                else ExplanationBaseline.from_dict(mapping(baseline_raw, "explanation.baseline"))
            ),
            mask_policy=ExplanationMaskPolicy.from_dict(
                mapping(value["mask_policy"], "explanation.mask_policy")
            ),
            feature_axes=strings(
                value["feature_axes"],
                "explanation.feature_axes",
                required=True,
            ),
            values=ArtifactReference.from_dict(mapping(value["values"], "explanation.values")),
            summary=(
                None
                if summary_raw is None
                else ArtifactReference.from_dict(mapping(summary_raw, "explanation.summary"))
            ),
            created_at=timestamp(value["created_at"], "explanation.created_at"),
        )

    def canonical_json(self) -> str:
        """Return stable canonical JSON."""
        return canonical_json(self.to_dict())


def _explanation_payload(value: ExplanationManifest) -> dict[str, Any]:
    raw = value.to_dict()
    raw.pop("schema_version")
    raw.pop("explanation_id")
    raw.pop("created_at")
    return raw


def _validate_explanation(value: ExplanationManifest) -> None:
    version(
        value.schema_version,
        ML_EXPLANATION_MANIFEST_VERSION,
        "explanation.schema_version",
    )
    digest(value.explanation_id, "explanation.explanation_id")
    _run_id(value.run_id, "explanation.run_id")
    digest(value.workspace_id, "explanation.workspace_id")
    digest(value.model_id, "explanation.model_id")
    digest(value.dataset_snapshot_id, "explanation.dataset_snapshot_id")
    string(value.cohort, "explanation.cohort")
    integer(value.n_observations, "explanation.n_observations", minimum=1)
    if not value.feature_axes:
        fail("explanation.feature_axes", "must contain at least one axis")
    if len(value.feature_axes) != len(set(value.feature_axes)):
        fail("explanation.feature_axes", "cannot contain duplicates")
    if value.values.role != "explanation_values":
        fail("explanation.values.role", "must be 'explanation_values'")
    if value.values.size_bytes == 0:
        fail("explanation.values.size_bytes", "explanation values cannot be empty")
    if value.summary is not None and value.summary.role != "explanation_summary":
        fail("explanation.summary.role", "must be 'explanation_summary'")
    timestamp(value.created_at, "explanation.created_at")
    if value.explanation_id != sha256(_explanation_payload(value)):
        fail("explanation.explanation_id", "does not match explanation content")
