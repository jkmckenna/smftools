"""Resolve immutable POD5 manifest rows to checksum-validated source paths."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .input_manifest import InputManifestError, InputManifestRow, checksum_input_source


@dataclass(frozen=True)
class Pod5SourceCandidate:
    """One explicitly identified candidate location for a POD5 source."""

    path: Path
    evidence: str
    source_id: str | None = None
    sha256: str | None = None


@dataclass(frozen=True)
class Pod5SourceResolutionRow:
    """One path-independent resolution decision for a POD5 manifest row."""

    source_id: str
    sha256: str
    size_bytes: int
    status: str
    evidence: str
    resolved_path: Path | None = None
    candidate_count: int = 0
    valid_candidate_count: int = 0
    rejected_candidate_count: int = 0
    observed_sha256s: tuple[str, ...] = ()

    def semantic_payload(self) -> dict[str, object]:
        """Return the relocation-invariant identity of this decision."""
        payload: dict[str, object] = {
            "source_id": self.source_id,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "status": self.status,
        }
        if self.status != "resolved":
            payload["observed_sha256s"] = list(self.observed_sha256s)
        return payload

    def to_dict(self, *, include_path: bool = True) -> dict[str, object]:
        """Return a stable JSON-compatible resolution record."""
        payload = {
            **self.semantic_payload(),
            "evidence": self.evidence,
            "candidate_count": self.candidate_count,
            "valid_candidate_count": self.valid_candidate_count,
            "rejected_candidate_count": self.rejected_candidate_count,
        }
        if include_path:
            payload["resolved_path"] = (
                None if self.resolved_path is None else self.resolved_path.as_posix()
            )
        return payload


@dataclass(frozen=True)
class Pod5SourceResolution:
    """Deterministic checksum resolution for an authoritative POD5 manifest."""

    rows: tuple[Pod5SourceResolutionRow, ...]
    unmatched_candidate_count: int = 0

    @property
    def resolved_count(self) -> int:
        """Return the number of source rows resolved to exact bytes."""
        return sum(row.status == "resolved" for row in self.rows)

    @property
    def missing_count(self) -> int:
        """Return the number of source rows with no available candidate path."""
        return sum(row.status == "missing" for row in self.rows)

    @property
    def checksum_mismatch_count(self) -> int:
        """Return the number of source rows whose available candidates differ."""
        return sum(row.status == "checksum_mismatch" for row in self.rows)

    @property
    def unreadable_count(self) -> int:
        """Return the number of source rows whose candidates could not be hashed."""
        return sum(row.status == "unreadable" for row in self.rows)

    @property
    def recorded_path_count(self) -> int:
        """Return the number resolved at their manifest-recorded paths."""
        return sum(
            row.status == "resolved" and row.evidence == "recorded_path" for row in self.rows
        )

    @property
    def relocated_path_count(self) -> int:
        """Return the number resolved through explicit candidate locations."""
        return sum(
            row.status == "resolved" and row.evidence != "recorded_path" for row in self.rows
        )

    @property
    def rejected_candidate_count(self) -> int:
        """Return available or declared candidates that did not validate."""
        return sum(row.rejected_candidate_count for row in self.rows)

    @property
    def duplicate_valid_candidate_count(self) -> int:
        """Return redundant exact-byte locations beyond each selected location."""
        return sum(max(0, row.valid_candidate_count - 1) for row in self.rows)

    @property
    def complete(self) -> bool:
        """Return whether every manifest row has one selected exact-byte path."""
        return self.resolved_count == len(self.rows)

    @property
    def evidence_counts(self) -> dict[str, int]:
        """Return stable counts for the selected resolution evidence."""
        return dict(
            sorted(Counter(row.evidence for row in self.rows if row.status == "resolved").items())
        )

    @property
    def digest(self) -> str:
        """Return a relocation-invariant digest over all source decisions."""
        payload = [row.semantic_payload() for row in self.rows]
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def resolved_sources(self) -> tuple[tuple[str, Path], ...]:
        """Return deterministic ``(source_id, path)`` pairs for resolved rows."""
        return tuple(
            (row.source_id, row.resolved_path)
            for row in self.rows
            if row.status == "resolved" and row.resolved_path is not None
        )


@dataclass(frozen=True)
class _CandidateCheck:
    path: Path
    evidence: str
    status: str
    observed_sha256: str | None = None


def _normalized_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _candidate_matches(candidate: Pod5SourceCandidate, row: InputManifestRow) -> bool:
    return bool(
        (candidate.source_id is None or candidate.source_id == row.source_id)
        and (candidate.sha256 is None or candidate.sha256.lower() == row.sha256.lower())
    )


def _check_candidate(
    candidate: Pod5SourceCandidate,
    row: InputManifestRow,
    *,
    checksum_reader: Callable[[Path], tuple[str, int]],
) -> _CandidateCheck:
    path = _normalized_path(candidate.path)
    if not path.is_file():
        return _CandidateCheck(path, candidate.evidence, "missing")
    try:
        observed_sha256, size_bytes = checksum_reader(path)
    except (InputManifestError, OSError, ValueError):
        return _CandidateCheck(path, candidate.evidence, "unreadable")
    normalized_digest = str(observed_sha256).lower()
    if size_bytes != row.size_bytes or normalized_digest != row.sha256.lower():
        return _CandidateCheck(
            path,
            candidate.evidence,
            "checksum_mismatch",
            normalized_digest,
        )
    return _CandidateCheck(path, candidate.evidence, "valid", normalized_digest)


def resolve_pod5_sources(
    rows: Iterable[InputManifestRow],
    *,
    candidates: Iterable[Pod5SourceCandidate] = (),
    checksum_reader: Callable[[Path], tuple[str, int]] = checksum_input_source,
) -> Pod5SourceResolution:
    """Resolve POD5 rows to exact bytes at original or explicit locations.

    Manifest paths are always considered first. Explicit candidates must name a
    unique manifest row by source ID, checksum, or both. Every available path is
    size- and SHA-256-validated before it can be selected.

    Args:
        rows: Authoritative POD5/raw-signal manifest rows.
        candidates: Additional source locations with identity selectors.
        checksum_reader: Mutation-safe bounded checksum implementation.

    Returns:
        Complete deterministic source decisions and unmatched-candidate count.

    Raises:
        ValueError: If rows are duplicated or are not POD5 raw-signal sources.
    """
    ordered_rows = tuple(sorted(rows, key=lambda row: row.source_id))
    if len({row.source_id for row in ordered_rows}) != len(ordered_rows):
        raise ValueError("POD5 source resolution requires unique source IDs")
    if any(row.source_kind != "pod5" or row.source_role != "raw_signal" for row in ordered_rows):
        raise ValueError("POD5 source resolution requires only POD5 raw-signal rows")

    external = tuple(candidates)
    matched_by_source: dict[str, list[Pod5SourceCandidate]] = {
        row.source_id: [] for row in ordered_rows
    }
    unmatched = 0
    for candidate in external:
        matches = [row for row in ordered_rows if _candidate_matches(candidate, row)]
        if len(matches) != 1:
            unmatched += 1
            continue
        matched_by_source[matches[0].source_id].append(candidate)

    decisions: list[Pod5SourceResolutionRow] = []
    for row in ordered_rows:
        source_candidates = [
            Pod5SourceCandidate(
                path=Path(row.path),
                evidence="recorded_path",
                source_id=row.source_id,
                sha256=row.sha256,
            ),
            *matched_by_source[row.source_id],
        ]
        by_path: dict[Path, Pod5SourceCandidate] = {}
        for candidate in source_candidates:
            path = _normalized_path(candidate.path)
            current = by_path.get(path)
            if current is None or (
                current.evidence != "recorded_path" and candidate.evidence == "recorded_path"
            ):
                by_path[path] = candidate
        checks = [
            _check_candidate(candidate, row, checksum_reader=checksum_reader)
            for _, candidate in sorted(by_path.items(), key=lambda item: item[0].as_posix())
        ]
        valid = sorted(
            (check for check in checks if check.status == "valid"),
            key=lambda check: (
                0 if check.evidence == "recorded_path" else 1,
                check.evidence,
                check.path.as_posix(),
            ),
        )
        observed = tuple(
            sorted(
                {
                    check.observed_sha256
                    for check in checks
                    if check.status == "checksum_mismatch" and check.observed_sha256 is not None
                }
            )
        )
        if valid:
            selected = valid[0]
            status = "resolved"
            evidence = selected.evidence
            resolved_path = selected.path
        elif any(check.status == "checksum_mismatch" for check in checks):
            status = "checksum_mismatch"
            evidence = "no_checksum_match"
            resolved_path = None
        elif any(check.status == "unreadable" for check in checks):
            status = "unreadable"
            evidence = "no_readable_candidate"
            resolved_path = None
        else:
            status = "missing"
            evidence = "no_available_candidate"
            resolved_path = None
        decisions.append(
            Pod5SourceResolutionRow(
                source_id=row.source_id,
                sha256=row.sha256.lower(),
                size_bytes=row.size_bytes,
                status=status,
                evidence=evidence,
                resolved_path=resolved_path,
                candidate_count=len(checks),
                valid_candidate_count=len(valid),
                rejected_candidate_count=sum(check.status != "valid" for check in checks),
                observed_sha256s=observed,
            )
        )
    return Pod5SourceResolution(tuple(decisions), unmatched_candidate_count=unmatched)
