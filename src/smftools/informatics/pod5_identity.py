"""Resolve historical raw observations to authoritative POD5 read UUIDs."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from .molecule_identity import (
    BASECALL_PARENT_READ_ID_COLUMN,
    BASECALL_READ_ID_COLUMN,
    POD5_READ_ID_COLUMN,
    READ_ID_COLUMN,
)

SOURCE_READ_ID_COLUMN = "source_read_id"
MOLECULE_UID_COLUMN = "molecule_uid"
_DIRECT_COLUMNS = (POD5_READ_ID_COLUMN, BASECALL_PARENT_READ_ID_COLUMN)
_LEGACY_COLUMNS = (SOURCE_READ_ID_COLUMN, BASECALL_READ_ID_COLUMN, READ_ID_COLUMN)


@dataclass(frozen=True)
class Pod5DatasetIndex:
    """Deterministic mapping from POD5 read UUIDs to source-manifest IDs."""

    sources_by_read_id: Mapping[str, tuple[str, ...]]

    @property
    def unique_read_count(self) -> int:
        """Return the number of distinct indexed POD5 read UUIDs."""
        return len(self.sources_by_read_id)

    @property
    def duplicate_read_id_count(self) -> int:
        """Return the number of UUIDs occurring in more than one source location."""
        return sum(len(sources) > 1 for sources in self.sources_by_read_id.values())

    def sources_for(self, read_id: object) -> tuple[str, ...]:
        """Return source occurrences for one normalized read UUID."""
        return self.sources_by_read_id.get(str(read_id), ())


@dataclass(frozen=True)
class Pod5IdentityResolutionRow:
    """One deterministic selected-observation identity decision."""

    observation_id: str
    molecule_uid: str | None
    pod5_read_id: str | None
    status: str
    evidence: str
    source_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible representation."""
        return {
            "observation_id": self.observation_id,
            "molecule_uid": self.molecule_uid,
            "pod5_read_id": self.pod5_read_id,
            "status": self.status,
            "evidence": self.evidence,
            "source_ids": list(self.source_ids),
        }


@dataclass(frozen=True)
class Pod5IdentityResolution:
    """Bounded-summary source for a selected historical molecule cohort."""

    rows: tuple[Pod5IdentityResolutionRow, ...]

    @property
    def resolved_count(self) -> int:
        """Return the number of unambiguously resolved observations."""
        return sum(row.status == "resolved" for row in self.rows)

    @property
    def unresolved_count(self) -> int:
        """Return the number of observations with no supported POD5 candidate."""
        return sum(row.status == "unresolved" for row in self.rows)

    @property
    def ambiguous_count(self) -> int:
        """Return the number of observations mapping to non-unique source signal."""
        return sum(row.status == "ambiguous" for row in self.rows)

    @property
    def unique_pod5_read_count(self) -> int:
        """Return the distinct resolved POD5 UUID count."""
        return len({row.pod5_read_id for row in self.rows if row.status == "resolved"})

    @property
    def duplicate_parent_reference_count(self) -> int:
        """Return extra selected rows that share an unambiguous POD5 parent."""
        return self.resolved_count - self.unique_pod5_read_count

    @property
    def evidence_counts(self) -> dict[str, int]:
        """Return stable evidence counts."""
        return dict(sorted(Counter(row.evidence for row in self.rows).items()))

    @property
    def digest(self) -> str:
        """Return a stable digest over every resolution decision."""
        payload = [row.to_dict() for row in self.rows]
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def build_pod5_dataset_index(
    sources: Iterable[tuple[str, str | Path]],
) -> Pod5DatasetIndex:
    """Index POD5 UUIDs without loading signal arrays.

    Args:
        sources: Ordered ``(source_id, path)`` pairs from an input manifest.

    Returns:
        A deterministic index retaining every source occurrence of each UUID.
    """
    from ..optional_imports import require

    pod5 = require("pod5", extra="ont", purpose="POD5 identity resolution")
    occurrences: defaultdict[str, list[str]] = defaultdict(list)
    for source_id, path in sources:
        with pod5.Reader(Path(path)) as reader:
            for read in reader.reads():
                occurrences[str(read.read_id)].append(str(source_id))
    return Pod5DatasetIndex(
        {read_id: tuple(sorted(source_ids)) for read_id, source_ids in sorted(occurrences.items())}
    )


def _normalized(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    normalized = str(value).strip()
    return normalized or None


def resolve_pod5_identities(
    observations: pd.DataFrame,
    index: Pod5DatasetIndex,
    *,
    bam_parent_by_observation: Mapping[str, object] | None = None,
) -> Pod5IdentityResolution:
    """Resolve selected raw rows to indexed POD5 UUIDs in evidence order.

    Durable POD5 and basecall-parent fields are authoritative. Historical source
    and basecall read names are considered next, followed by a retained BAM
    ``pi`` lookup supplied by the caller. A UUID occurring in multiple manifest
    sources is ambiguous, while several observations sharing one unique UUID are
    valid split-child references.

    Args:
        observations: Selected raw observation rows containing unique ``read_id``.
        index: Authoritative POD5 dataset index.
        bam_parent_by_observation: Optional mapping from observation ID to retained
            BAM ``pi`` parent UUID.

    Returns:
        One stable identity decision per selected observation.
    """
    if READ_ID_COLUMN not in observations.columns:
        raise ValueError("selected observations require a read_id column")
    observation_ids = observations[READ_ID_COLUMN].astype(str)
    if observation_ids.duplicated().any():
        raise ValueError("selected observations require unique read_id values")

    bam_parents = bam_parent_by_observation or {}
    decisions: list[Pod5IdentityResolutionRow] = []
    for _, row in observations.assign(**{READ_ID_COLUMN: observation_ids}).iterrows():
        observation_id = str(row[READ_ID_COLUMN])
        molecule_uid = _normalized(row.get(MOLECULE_UID_COLUMN))

        authoritative: tuple[str, str] | None = None
        for column in _DIRECT_COLUMNS:
            candidate = _normalized(row.get(column))
            if candidate is not None:
                authoritative = (column, candidate)
                break
        if authoritative is not None:
            evidence, candidate = authoritative
            sources = index.sources_for(candidate)
            if len(sources) == 1:
                decisions.append(
                    Pod5IdentityResolutionRow(
                        observation_id,
                        molecule_uid,
                        candidate,
                        "resolved",
                        evidence,
                        sources,
                    )
                )
            else:
                decisions.append(
                    Pod5IdentityResolutionRow(
                        observation_id,
                        molecule_uid,
                        candidate,
                        "ambiguous" if sources else "unresolved",
                        f"{evidence}_{'duplicate_source' if sources else 'not_in_index'}",
                        sources,
                    )
                )
            continue

        matched = False
        for column in _LEGACY_COLUMNS:
            candidate = _normalized(row.get(column))
            if candidate is None:
                continue
            sources = index.sources_for(candidate)
            if not sources:
                continue
            decisions.append(
                Pod5IdentityResolutionRow(
                    observation_id,
                    molecule_uid,
                    candidate,
                    "resolved" if len(sources) == 1 else "ambiguous",
                    column if len(sources) == 1 else f"{column}_duplicate_source",
                    sources,
                )
            )
            matched = True
            break
        if matched:
            continue

        candidate = _normalized(bam_parents.get(observation_id))
        sources = index.sources_for(candidate) if candidate is not None else ()
        if candidate is not None and sources:
            decisions.append(
                Pod5IdentityResolutionRow(
                    observation_id,
                    molecule_uid,
                    candidate,
                    "resolved" if len(sources) == 1 else "ambiguous",
                    "bam_pi" if len(sources) == 1 else "bam_pi_duplicate_source",
                    sources,
                )
            )
        else:
            decisions.append(
                Pod5IdentityResolutionRow(
                    observation_id,
                    molecule_uid,
                    candidate,
                    "unresolved",
                    "bam_pi_not_in_index" if candidate is not None else "no_supported_identity",
                    sources,
                )
            )

    return Pod5IdentityResolution(tuple(sorted(decisions, key=lambda item: item.observation_id)))
