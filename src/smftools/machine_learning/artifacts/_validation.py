"""Shared strict-validation helpers for ML artifact schemas."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any


class MLArtifactManifestError(ValueError):
    """Raised when an ML artifact manifest is invalid."""


def fail(path: str, message: str) -> None:
    raise MLArtifactManifestError(f"{path}: {message}")


def mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        fail(path, "must be a mapping")
    if not all(isinstance(key, str) for key in value):
        fail(path, "keys must be strings")
    return value


def sequence(value: Any, path: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        fail(path, "must be a sequence")
    return value


def keys(
    value: Mapping[str, Any],
    *,
    path: str,
    fields: set[str],
    optional: set[str] = frozenset(),
) -> None:
    unknown = sorted(set(value).difference(fields))
    if unknown:
        fail(path, f"contains unknown fields: {unknown}")
    missing = sorted(fields.difference(optional).difference(value))
    if missing:
        fail(path, f"is missing required fields: {missing}")


def string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        fail(path, "must be a non-empty string")
    return value.strip()


def optional_string(value: Any, path: str) -> str | None:
    if value is None:
        return None
    return string(value, path)


def strings(value: Any, path: str, *, required: bool = False) -> tuple[str, ...]:
    result = tuple(
        string(item, f"{path}[{index}]") for index, item in enumerate(sequence(value, path))
    )
    if len(result) != len(set(result)):
        fail(path, "cannot contain duplicates")
    if required and not result:
        fail(path, "must contain at least one value")
    return result


def integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        fail(path, f"must be an integer greater than or equal to {minimum}")
    return value


def boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        fail(path, "must be a boolean")
    return value


def version(value: Any, expected: int, path: str) -> int:
    result = integer(value, path, minimum=1)
    if result != expected:
        fail(path, f"unsupported version {result}; supported version is {expected}")
    return result


def digest(value: Any, path: str) -> str:
    result = string(value, path).lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        fail(path, "must be a lowercase SHA-256 digest")
    return result


def timestamp(value: Any, path: str) -> str:
    result = string(value, path)
    try:
        parsed = datetime.fromisoformat(result.replace("Z", "+00:00"))
    except ValueError as exc:
        fail(path, f"must be an ISO-8601 timestamp: {exc}")
    if parsed.tzinfo is None:
        fail(path, "must include a timezone")
    return result


def optional_timestamp(value: Any, path: str) -> str | None:
    if value is None:
        return None
    return timestamp(value, path)


def portable_path(value: Any, path: str) -> str:
    result = string(value, path)
    candidate = PurePosixPath(result)
    if (
        candidate.is_absolute()
        or ".." in candidate.parts
        or "\\" in result
        or "://" in result
        or result == "."
        or any(character in result for character in '<>:"|?*')
    ):
        fail(path, "must be a portable relative POSIX path")
    return result


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise MLArtifactManifestError(f"value is not canonical JSON: {exc}") from exc


def sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def freeze_json(value: Any, path: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            fail(path, "must contain only finite numbers")
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            fail(path, "mapping keys must be strings")
        return MappingProxyType(
            {key: freeze_json(item, f"{path}.{key}") for key, item in sorted(value.items())}
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(freeze_json(item, f"{path}[{index}]") for index, item in enumerate(value))
    fail(path, f"contains unsupported value type {type(value).__name__}")


def thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


def string_mapping(value: Any, path: str) -> Mapping[str, str]:
    raw = mapping(value, path)
    return MappingProxyType(
        {
            string(key, f"{path}.key"): string(item, f"{path}.{key}")
            for key, item in sorted(raw.items())
        }
    )
