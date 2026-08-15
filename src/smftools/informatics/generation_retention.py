"""Mutable retention metadata for immutable generation directories.

Retention decisions are deliberately stored beside ``current.json`` rather
than inside ``generation_manifest.json``. Published manifests are immutable,
and several generation kinds bind them to ``current.json`` with a checksum;
editing one to add a publication pin would both violate that contract and make
the selected generation unreadable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..readwrite import atomic_write_json
from .generation import GenerationError, resolve_stage_generation

RETENTION_FILENAME = "retention.json"
RETENTION_SCHEMA_VERSION = 1


class GenerationRetentionError(RuntimeError):
    """Raised when generation retention metadata is unsafe or inconsistent."""


@dataclass(frozen=True)
class RetentionReason:
    """One durable explanation for retaining a generation."""

    reason: str
    recorded_at: str

    def to_dict(self) -> dict[str, str]:
        """Return the stable JSON representation."""
        return {"reason": self.reason, "recorded_at": self.recorded_at}


@dataclass(frozen=True)
class GenerationRetention:
    """All active retention reasons for one generation."""

    generation_id: str
    reasons: tuple[RetentionReason, ...]

    @property
    def pinned(self) -> bool:
        """Whether this entry protects its generation from pruning."""
        return bool(self.reasons)

    def to_dict(self) -> dict[str, Any]:
        """Return the registry value for this generation."""
        return {
            "pinned": True,
            "reasons": [reason.to_dict() for reason in self.reasons],
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_generation_id(generation_id: str) -> str:
    normalized = str(generation_id).strip()
    relative = Path(normalized)
    if (
        not normalized
        or relative.is_absolute()
        or ".." in relative.parts
        or len(relative.parts) != 1
    ):
        raise GenerationRetentionError("generation ID is not portable")
    return normalized


def _parse_registry(payload: Any) -> dict[str, GenerationRetention]:
    if not isinstance(payload, dict):
        raise GenerationRetentionError("retention registry is not a JSON object")
    if payload.get("schema_version") != RETENTION_SCHEMA_VERSION:
        raise GenerationRetentionError("retention registry schema is incompatible")
    raw_generations = payload.get("generations")
    if not isinstance(raw_generations, dict):
        raise GenerationRetentionError("retention registry generation map is missing")

    entries: dict[str, GenerationRetention] = {}
    for raw_generation_id, raw_entry in raw_generations.items():
        generation_id = _validate_generation_id(str(raw_generation_id))
        if not isinstance(raw_entry, dict) or raw_entry.get("pinned") is not True:
            raise GenerationRetentionError(f"retention entry for {generation_id!r} is not pinned")
        raw_reasons = raw_entry.get("reasons")
        if not isinstance(raw_reasons, list) or not raw_reasons:
            raise GenerationRetentionError(f"retention entry for {generation_id!r} has no reasons")
        reasons: list[RetentionReason] = []
        seen: set[str] = set()
        for raw_reason in raw_reasons:
            if not isinstance(raw_reason, dict):
                raise GenerationRetentionError(f"retention reason for {generation_id!r} is invalid")
            reason = str(raw_reason.get("reason", "")).strip()
            recorded_at = str(raw_reason.get("recorded_at", "")).strip()
            if not reason or not recorded_at:
                raise GenerationRetentionError(
                    f"retention reason for {generation_id!r} is incomplete"
                )
            if reason in seen:
                raise GenerationRetentionError(
                    f"retention entry for {generation_id!r} repeats a reason"
                )
            seen.add(reason)
            reasons.append(RetentionReason(reason=reason, recorded_at=recorded_at))
        entries[generation_id] = GenerationRetention(
            generation_id=generation_id,
            reasons=tuple(reasons),
        )
    return entries


def read_generation_retention(
    container: str | Path,
) -> dict[str, GenerationRetention]:
    """Read and strictly validate one generation container's retention registry."""
    path = Path(container) / RETENTION_FILENAME
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GenerationRetentionError("retention registry is unreadable") from exc
    return _parse_registry(payload)


def read_generation_retention_lenient(
    container: str | Path,
) -> tuple[dict[str, GenerationRetention], str | None]:
    """Read retention metadata for inventory reporting without hiding generations."""
    try:
        return read_generation_retention(container), None
    except GenerationRetentionError as exc:
        return {}, str(exc)


def _write_generation_retention(
    container: Path,
    entries: dict[str, GenerationRetention],
) -> Path:
    path = container / RETENTION_FILENAME
    atomic_write_json(
        path,
        {
            "schema_version": RETENTION_SCHEMA_VERSION,
            "generations": {
                generation_id: entry.to_dict() for generation_id, entry in sorted(entries.items())
            },
        },
    )
    return path


def pin_generation(
    container: str | Path,
    generation_id: str,
    *,
    reason: str,
) -> GenerationRetention:
    """Add one retention reason without modifying the generation manifest."""
    container = Path(container)
    generation_id = _validate_generation_id(generation_id)
    normalized_reason = str(reason).strip()
    if not normalized_reason:
        raise GenerationRetentionError("retention reason must not be empty")
    try:
        resolved = resolve_stage_generation(container, lineage=generation_id)
    except GenerationError as exc:
        raise GenerationRetentionError(str(exc)) from exc
    if resolved is None:  # Defensive: a lineage request never resolves to None.
        raise GenerationRetentionError(f"generation {generation_id!r} does not exist")

    entries = read_generation_retention(container)
    existing = entries.get(generation_id)
    if existing is not None and normalized_reason in {item.reason for item in existing.reasons}:
        return existing
    reasons = list(existing.reasons if existing is not None else ())
    reasons.append(RetentionReason(reason=normalized_reason, recorded_at=_now()))
    updated = GenerationRetention(generation_id=generation_id, reasons=tuple(reasons))
    entries[generation_id] = updated
    _write_generation_retention(container, entries)
    return updated


def unpin_generation(
    container: str | Path,
    generation_id: str,
    *,
    reason: str | None = None,
) -> GenerationRetention | None:
    """Remove one reason, or every reason when ``reason`` is omitted."""
    container = Path(container)
    generation_id = _validate_generation_id(generation_id)
    entries = read_generation_retention(container)
    existing = entries.get(generation_id)
    if existing is None:
        raise GenerationRetentionError(f"generation {generation_id!r} is not pinned")

    if reason is None:
        del entries[generation_id]
        _write_generation_retention(container, entries)
        return None

    normalized_reason = str(reason).strip()
    if not normalized_reason:
        raise GenerationRetentionError("retention reason must not be empty")
    remaining = tuple(item for item in existing.reasons if item.reason != normalized_reason)
    if len(remaining) == len(existing.reasons):
        raise GenerationRetentionError(
            f"generation {generation_id!r} has no matching retention reason"
        )
    if not remaining:
        del entries[generation_id]
        _write_generation_retention(container, entries)
        return None

    updated = GenerationRetention(generation_id=generation_id, reasons=remaining)
    entries[generation_id] = updated
    _write_generation_retention(container, entries)
    return updated
