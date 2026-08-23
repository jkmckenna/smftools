"""Agreement between an assigned barcode and a re-derived one (`EGL-29b`).

When FASTQs arrive already demultiplexed, the barcode is carried by the
directory tree and is authoritative. The single- vs double-ended status is not
carried at all, so it has to be re-derived by scanning the read sequences --
which produces a *second*, independent barcode call as a by-product.

Keeping both and reporting where they differ is the point. Silently preferring
either one throws away a real signal: systematic disagreement means the
directory assignment, the barcode kit, or the extraction parameters are wrong,
and none of those announce themselves. `F12` and `F13` were both cases where a
silently-empty or silently-wrong assignment looked like a valid result until
something much further downstream became impossible to explain.

Disagreement is reported, never resolved. The assigned barcode stays
authoritative -- this module adds a column and a summary and changes nothing
else.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from smftools.logging_utils import get_logger

logger = get_logger(__name__)

AGREEMENT_COLUMN = "barcode_agreement"

AGREE = "agree"
DISAGREE = "disagree"
REDERIVED_UNCLASSIFIED = "rederived_unclassified"
ASSIGNED_UNCLASSIFIED = "assigned_unclassified"
NOT_COMPARED = "not_compared"

_UNCLASSIFIED = frozenset({"unclassified", "unassigned", "", "nan", "none", "unknown"})


def _normalize(value: Any) -> str:
    return str(value).strip().lower()


def _is_unclassified(value: str) -> bool:
    return value in _UNCLASSIFIED


def classify_agreement(assigned: pd.Series, rederived: pd.Series) -> pd.Series:
    """Label each read's assigned-vs-re-derived barcode relationship.

    The four outcomes are kept distinct rather than collapsed to a boolean
    because they mean different things operationally: a re-derivation that
    found nothing is a sensitivity problem, whereas one that found a *different*
    barcode is a correctness problem, and only the second calls the assignment
    into question.
    """
    left = assigned.map(_normalize)
    right = rederived.map(_normalize)

    result = pd.Series(NOT_COMPARED, index=assigned.index, dtype=object)
    left_missing = left.map(_is_unclassified)
    right_missing = right.map(_is_unclassified)

    result[~left_missing & right_missing] = REDERIVED_UNCLASSIFIED
    result[left_missing & ~right_missing] = ASSIGNED_UNCLASSIFIED
    comparable = ~left_missing & ~right_missing
    result[comparable & (left == right)] = AGREE
    result[comparable & (left != right)] = DISAGREE
    return result


def summarize_agreement(agreement: pd.Series, assigned: pd.Series, rederived: pd.Series) -> dict:
    """Counts, the disagreement rate, and the most common confusions.

    The confusion pairs matter more than the rate: a rate says something is
    wrong, a concentrated pair says *what*. Two barcodes that differ by one base
    swapping into each other reads very differently from disagreement spread
    evenly, which would point at the extraction parameters instead.
    """
    counts = agreement.value_counts().to_dict()
    comparable = int(counts.get(AGREE, 0) + counts.get(DISAGREE, 0))
    disagreements = int(counts.get(DISAGREE, 0))
    pairs: list[dict[str, Any]] = []
    if disagreements:
        mask = agreement == DISAGREE
        confusion = (
            pd.DataFrame({"assigned": assigned[mask], "rederived": rederived[mask]})
            .value_counts()
            .head(10)
        )
        pairs = [
            {"assigned": str(index[0]), "rederived": str(index[1]), "reads": int(value)}
            for index, value in confusion.items()
        ]
    return {
        "reads": int(len(agreement)),
        "comparable": comparable,
        "agree": int(counts.get(AGREE, 0)),
        "disagree": disagreements,
        "rederived_unclassified": int(counts.get(REDERIVED_UNCLASSIFIED, 0)),
        "assigned_unclassified": int(counts.get(ASSIGNED_UNCLASSIFIED, 0)),
        "not_compared": int(counts.get(NOT_COMPARED, 0)),
        "disagreement_rate": (disagreements / comparable) if comparable else 0.0,
        "top_confusions": pairs,
    }


def report_barcode_agreement(
    obs: pd.DataFrame,
    *,
    assigned_column: str = "barcode",
    rederived_column: str = "BC",
    warn_above: float = 0.01,
) -> Mapping[str, Any] | None:
    """Add the agreement column to ``obs`` and log a summary.

    Returns the summary, or ``None`` when there is no second assignment to
    compare against -- which is the normal case for inputs that were never
    re-demultiplexed, and is not a problem.
    """
    if assigned_column not in obs.columns or rederived_column not in obs.columns:
        logger.debug(
            "Barcode agreement not computed: need both %r and %r in obs.",
            assigned_column,
            rederived_column,
        )
        return None

    agreement = classify_agreement(obs[assigned_column], obs[rederived_column])
    obs[AGREEMENT_COLUMN] = pd.Categorical(agreement)
    summary = summarize_agreement(agreement, obs[assigned_column], obs[rederived_column])

    logger.info(
        "Barcode agreement: %d/%d comparable reads agree (%.2f%% disagree); "
        "%d re-derivation found nothing, %d assigned unclassified",
        summary["agree"],
        summary["comparable"],
        100 * summary["disagreement_rate"],
        summary["rederived_unclassified"],
        summary["assigned_unclassified"],
    )
    if summary["disagreement_rate"] > warn_above:
        # Loud, because a high rate means one of the two assignments is wrong
        # and every downstream per-sample number inherits it.
        logger.warning(
            "%.2f%% of comparable reads disagree between %r and %r (%d reads). "
            "One of the two assignments is wrong; the assigned barcode is being kept. "
            "Most common confusions: %s",
            100 * summary["disagreement_rate"],
            assigned_column,
            rederived_column,
            summary["disagree"],
            ", ".join(
                f"{pair['assigned']}->{pair['rederived']} ({pair['reads']})"
                for pair in summary["top_confusions"][:5]
            )
            or "none",
        )
    return summary


def should_derive_demux_status(cfg, demux_backend: str, *, dorado_supports: bool = False) -> bool:
    """Whether to derive `demux_type` from the `BM` tag for this run (`EGL-29a`).

    The case this exists for: input that is *already demultiplexed* but whose
    barcode assignment carries no end reason. `input_already_demuxed` means "do
    not demux" and remains true there, so the decision needs its own flag rather
    than an overload -- otherwise "keep my barcodes" and "re-scan sequences for
    their end status" cannot both be expressed, which is exactly the
    combination an already-demuxed FASTQ tree needs.
    """
    backend = str(demux_backend or "").strip().lower()
    if not getattr(cfg, "barcode_kit", None):
        return False
    if backend == "smftools":
        return not getattr(cfg, "input_already_demuxed", False) or bool(
            getattr(cfg, "derive_demux_status_from_sequence", False)
        )
    if backend == "dorado":
        # Dorado only emits the per-end tag this reads from recent versions, and
        # it cannot re-derive on already-demuxed input the way the sequence
        # scanner can -- there is no second pass to attach it to.
        return dorado_supports and not getattr(cfg, "input_already_demuxed", False)
    return False


#: `BM` values that mean a barcode was found at both ends.
_BM_DOUBLE = frozenset({"both"})
#: `BM` values that mean exactly one end carried a recognizable barcode.
_BM_SINGLE = frozenset({"left_only", "right_only", "read_start_only", "read_end_only"})

BM_SOURCE = "bm_tag"


def derive_demux_type_from_bm(frame: pd.DataFrame, *, bm_column: str = "BM") -> int:
    """Write `demux_type` (+ provenance) onto a ragged read frame from `BM`.

    The equivalent dense-path helper (`add_demux_type_from_bm_tag`) operates on
    an AnnData and is only reachable from a branch the partitioned pipeline
    never takes, so `demux_type` never reached the partitioned obs at all --
    leaving the single- vs double-ended status that `EGL-29` exists to recover
    unavailable to every consumer, and silently disabling
    `duplicate_detection_demux_types_to_use`, which is guarded on the column
    being present (`F31`).

    Returns the number of rows labelled. The mapping matches the dense helper
    exactly so the two paths cannot drift.
    """
    if bm_column not in frame.columns:
        logger.debug("No %r column; demux_type not derived on this frame.", bm_column)
        return 0

    values = frame[bm_column].astype(str).str.strip().str.lower()
    demux_type = np.where(
        values.isin(_BM_DOUBLE),
        "double",
        np.where(values.isin(_BM_SINGLE), "single", "unclassified"),
    )
    frame["demux_type"] = demux_type
    frame["demux_type_source"] = BM_SOURCE
    # `BM` is a classifier assertion rather than a score, so confidence is 1.0
    # where it made a call and 0.0 where it did not -- keeping the column shape
    # identical to the sequencing-summary route, which reports a real fraction.
    frame["demux_type_confidence"] = np.where(demux_type == "unclassified", 0.0, 1.0)

    counts = pd.Series(demux_type).value_counts().to_dict()
    logger.info(
        "Derived demux_type from %s: double=%d, single=%d, unclassified=%d",
        bm_column,
        counts.get("double", 0),
        counts.get("single", 0),
        counts.get("unclassified", 0),
    )
    return int(len(frame))
