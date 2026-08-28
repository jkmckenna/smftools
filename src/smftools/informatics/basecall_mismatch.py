"""Classify whether a basecall generation still covers the signal beside it (`BCS-11`).

`BCS-07` made a basecall generation record its input as one identity per
source (`"source:<source_id>:<sha256>"` entries in `input_artifact_ids`, plus
one `"input-manifest:<digest>"` summary entry) instead of a single digest over
the whole input path. That per-source shape is what lets a later comparison
tell a pruned archive apart from an incomplete one, rather than only telling
"changed" from "unchanged".

"Mismatch" is not one condition -- it has three shapes with opposite correct
responses (see the plan's own table): a basecall generation with `source_id`s
POD5s no longer present is the pruned-archive end state this program exists
to reach and should be reused silently; a basecall generation missing
`source_id`s that are present now under-covers the signal and must refuse
unless the manifest itself records that the gap was deliberate
(`max_basecall_reads`); anything else -- overlapping but neither a pure
subset nor a pure superset, or no overlap at all -- must refuse outright.
"""

from __future__ import annotations

from enum import Enum
from typing import Iterable, Sequence

_SOURCE_PREFIX = "source:"


class BasecallSourceShape(str, Enum):
    """The relationship between a basecall generation's recorded sources and the signal now."""

    IDENTICAL = "identical"
    #: basecalls superset signal: every source basecalled is still recorded,
    #: some may since be gone from disk -- reuse silently.
    SIGNAL_PRUNED = "signal_pruned"
    #: basecalls subset signal: sources are present now that this generation
    #: never basecalled -- refuse unless recorded as a deliberate subsample.
    SIGNAL_EXPANDED = "signal_expanded"
    #: neither a pure subset nor a pure superset, or no overlap -- refuse.
    DISJOINT = "disjoint"


def source_identity_ids(input_artifact_ids: Iterable[str]) -> frozenset[str]:
    """Return the per-source `source_id`s an `input_artifact_ids` list carries.

    Drops the leading `"input-manifest:<digest>"` summary entry and any
    non-source entry (e.g. `raw_input_artifact_ids`' alignment-reference-bundle
    one) -- comparison is over sources only.
    """
    return frozenset(
        entry[len(_SOURCE_PREFIX) :].split(":", 1)[0]
        for entry in input_artifact_ids
        if entry.startswith(_SOURCE_PREFIX)
    )


def classify_basecall_source_shape(
    recorded_input_artifact_ids: Sequence[str],
    current_input_artifact_ids: Sequence[str],
) -> BasecallSourceShape:
    """Classify a basecall generation's recorded sources against the signal present now."""
    recorded = source_identity_ids(recorded_input_artifact_ids)
    current = source_identity_ids(current_input_artifact_ids)
    if recorded == current:
        return BasecallSourceShape.IDENTICAL
    if current < recorded:
        return BasecallSourceShape.SIGNAL_PRUNED
    if recorded < current:
        return BasecallSourceShape.SIGNAL_EXPANDED
    return BasecallSourceShape.DISJOINT
