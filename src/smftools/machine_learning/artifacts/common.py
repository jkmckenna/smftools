"""Common immutable records embedded by ML artifact manifests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from ._validation import (
    MLArtifactManifestError,
    boolean,
    digest,
    freeze_json,
    integer,
    keys,
    mapping,
    portable_path,
    sha256,
    string,
    string_mapping,
    strings,
    thaw_json,
)


@dataclass(frozen=True)
class ArtifactReference:
    """Portable identity and checksum of one artifact payload."""

    role: str
    relative_path: str
    sha256: str
    size_bytes: int
    media_type: str

    def __post_init__(self) -> None:
        string(self.role, "artifact.role")
        portable_path(self.relative_path, "artifact.relative_path")
        digest(self.sha256, "artifact.sha256")
        integer(self.size_bytes, "artifact.size_bytes", minimum=0)
        string(self.media_type, "artifact.media_type")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable artifact reference."""
        return {
            "role": self.role,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "media_type": self.media_type,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ArtifactReference:
        """Validate and restore an artifact reference."""
        value = mapping(raw, "artifact")
        fields = {"role", "relative_path", "sha256", "size_bytes", "media_type"}
        keys(value, path="artifact", fields=fields)
        return cls(
            role=string(value["role"], "artifact.role"),
            relative_path=portable_path(value["relative_path"], "artifact.relative_path"),
            sha256=digest(value["sha256"], "artifact.sha256"),
            size_bytes=integer(value["size_bytes"], "artifact.size_bytes"),
            media_type=string(value["media_type"], "artifact.media_type"),
        )


@dataclass(frozen=True)
class EnvironmentRecord:
    """Self-contained software, code, and platform execution environment."""

    smftools_version: str
    python_version: str
    platform: str
    code_revision: str
    dirty_tree: bool
    dependencies: Mapping[str, str]

    def __post_init__(self) -> None:
        string(self.smftools_version, "environment.smftools_version")
        string(self.python_version, "environment.python_version")
        string(self.platform, "environment.platform")
        string(self.code_revision, "environment.code_revision")
        boolean(self.dirty_tree, "environment.dirty_tree")
        object.__setattr__(
            self,
            "dependencies",
            string_mapping(self.dependencies, "environment.dependencies"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable environment record."""
        return {
            "smftools_version": self.smftools_version,
            "python_version": self.python_version,
            "platform": self.platform,
            "code_revision": self.code_revision,
            "dirty_tree": self.dirty_tree,
            "dependencies": dict(self.dependencies),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> EnvironmentRecord:
        """Validate and restore an environment record."""
        value = mapping(raw, "environment")
        fields = {
            "smftools_version",
            "python_version",
            "platform",
            "code_revision",
            "dirty_tree",
            "dependencies",
        }
        keys(value, path="environment", fields=fields)
        return cls(
            smftools_version=string(value["smftools_version"], "environment.smftools_version"),
            python_version=string(value["python_version"], "environment.python_version"),
            platform=string(value["platform"], "environment.platform"),
            code_revision=string(value["code_revision"], "environment.code_revision"),
            dirty_tree=boolean(value["dirty_tree"], "environment.dirty_tree"),
            dependencies=string_mapping(value["dependencies"], "environment.dependencies"),
        )


@dataclass(frozen=True)
class ResolvedDefinition:
    """Named versioned definition with immutable resolved parameters."""

    name: str
    version: str
    parameters: Mapping[str, Any]
    definition_hash: str

    def __post_init__(self) -> None:
        string(self.name, "definition.name")
        string(self.version, "definition.version")
        object.__setattr__(
            self,
            "parameters",
            freeze_json(self.parameters, "definition.parameters"),
        )
        expected = sha256(
            {
                "name": self.name,
                "version": self.version,
                "parameters": thaw_json(self.parameters),
            }
        )
        if self.definition_hash != expected:
            raise MLArtifactManifestError(
                "definition.definition_hash: does not match resolved definition"
            )

    @classmethod
    def create(
        cls,
        *,
        name: str,
        version: str,
        parameters: Mapping[str, Any],
    ) -> ResolvedDefinition:
        """Create a checksummed resolved definition."""
        frozen = freeze_json(parameters, "definition.parameters")
        payload = {
            "name": string(name, "definition.name"),
            "version": string(version, "definition.version"),
            "parameters": thaw_json(frozen),
        }
        return cls(
            name=payload["name"],
            version=payload["version"],
            parameters=frozen,
            definition_hash=sha256(payload),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable definition."""
        return {
            "name": self.name,
            "version": self.version,
            "parameters": thaw_json(self.parameters),
            "definition_hash": self.definition_hash,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ResolvedDefinition:
        """Validate and restore a resolved definition."""
        value = mapping(raw, "definition")
        fields = {"name", "version", "parameters", "definition_hash"}
        keys(value, path="definition", fields=fields)
        return cls(
            name=string(value["name"], "definition.name"),
            version=string(value["version"], "definition.version"),
            parameters=mapping(value["parameters"], "definition.parameters"),
            definition_hash=digest(value["definition_hash"], "definition.definition_hash"),
        )


@dataclass(frozen=True)
class FailureRecord:
    """Sanitized diagnostic context retained by a failed or cancelled run."""

    error_type: str
    message: str
    phase: str
    traceback_artifact: ArtifactReference | None = None

    def __post_init__(self) -> None:
        string(self.error_type, "failure.error_type")
        string(self.message, "failure.message")
        string(self.phase, "failure.phase")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable failure record."""
        return {
            "error_type": self.error_type,
            "message": self.message,
            "phase": self.phase,
            "traceback_artifact": (
                self.traceback_artifact.to_dict() if self.traceback_artifact is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> FailureRecord:
        """Validate and restore a failure record."""
        value = mapping(raw, "failure")
        fields = {"error_type", "message", "phase", "traceback_artifact"}
        keys(value, path="failure", fields=fields)
        artifact = value["traceback_artifact"]
        return cls(
            error_type=string(value["error_type"], "failure.error_type"),
            message=string(value["message"], "failure.message"),
            phase=string(value["phase"], "failure.phase"),
            traceback_artifact=(
                None
                if artifact is None
                else ArtifactReference.from_dict(mapping(artifact, "failure.traceback_artifact"))
            ),
        )


@dataclass(frozen=True)
class SerializationPolicy:
    """Explicit loader and trust policy for a serialized model payload."""

    format: str
    loader: str
    requires_unsafe_load: bool
    allowed_types: tuple[str, ...]
    package_versions: Mapping[str, str]

    def __post_init__(self) -> None:
        format_name = string(self.format, "serialization.format").lower()
        object.__setattr__(self, "format", format_name)
        string(self.loader, "serialization.loader")
        boolean(
            self.requires_unsafe_load,
            "serialization.requires_unsafe_load",
        )
        object.__setattr__(
            self,
            "allowed_types",
            tuple(sorted(strings(self.allowed_types, "serialization.allowed_types"))),
        )
        object.__setattr__(
            self,
            "package_versions",
            string_mapping(
                self.package_versions,
                "serialization.package_versions",
            ),
        )
        if format_name in {"pickle", "joblib"} and not self.requires_unsafe_load:
            raise MLArtifactManifestError(
                "serialization.requires_unsafe_load: pickle/joblib must be explicitly unsafe"
            )
        if format_name == "skops" and not self.allowed_types:
            raise MLArtifactManifestError(
                "serialization.allowed_types: skops artifacts require a reviewed allowlist"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable serialization policy."""
        return {
            "format": self.format,
            "loader": self.loader,
            "requires_unsafe_load": self.requires_unsafe_load,
            "allowed_types": list(self.allowed_types),
            "package_versions": dict(self.package_versions),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> SerializationPolicy:
        """Validate and restore a serialization policy."""
        value = mapping(raw, "serialization")
        fields = {
            "format",
            "loader",
            "requires_unsafe_load",
            "allowed_types",
            "package_versions",
        }
        keys(value, path="serialization", fields=fields)
        return cls(
            format=string(value["format"], "serialization.format"),
            loader=string(value["loader"], "serialization.loader"),
            requires_unsafe_load=boolean(
                value["requires_unsafe_load"],
                "serialization.requires_unsafe_load",
            ),
            allowed_types=strings(value["allowed_types"], "serialization.allowed_types"),
            package_versions=string_mapping(
                value["package_versions"],
                "serialization.package_versions",
            ),
        )


def unique_artifact_roles(
    artifacts: tuple[ArtifactReference, ...],
    path: str,
) -> tuple[ArtifactReference, ...]:
    """Canonicalize references and reject ambiguous artifact roles."""
    canonical = tuple(sorted(artifacts, key=lambda item: item.role))
    roles = [item.role for item in canonical]
    if len(roles) != len(set(roles)):
        raise MLArtifactManifestError(f"{path}: artifact roles must be unique")
    return canonical
