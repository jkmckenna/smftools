"""Barcode contamination QC from an unbarcoded spike-in (`EGL-31`).

Barcodes are ligated to read ends before libraries are pooled, so a barcode from
one library can ligate to a molecule from another. The resulting read is
misassigned, and nothing about it looks wrong downstream.

An unbarcoded spike-in amplicon makes that rate measurable. Pooled without a
barcode of its own, its true assignment is known for **every** read, so any
barcode observed on a spike-in read is a mis-barcoding event by construction --
a direct count of known errors rather than a rate inferred from a model. It also
scavenges free adapter that would otherwise land on real molecules, which is why
the same amplicon serves both roles.

Three things are reported, and they fail independently:

1. **Per-barcode contamination.** How often each barcode appears on the spike-in,
   against how large that barcode's library is. Contamination scales with
   library abundance, so the informative quantity is the enrichment over that
   expectation, not the raw count.
2. **Single- vs double-ended mis-barcoding.** Every spike-in read carrying a
   barcode is an error, so splitting them by `demux_type` measures directly how
   much trust a double-ended assignment earns.
3. **End disagreement**, which needs no spike-in at all. `barcode_front` and
   `barcode_rear` are independent per-end calls (`F35`); a read whose two ends
   name *different* barcodes is a mis-ligation caught in the act, and that
   applies to the whole run rather than to 0.1% of it.

Rates are reported with Poisson intervals. A per-barcode cell can hold only tens
of reads, where a ratio of 1.3 against 1.0 is noise; presenting those as point
estimates would invite exactly the over-reading this QC exists to prevent.

Nothing here filters or corrects anything. It measures, and the assigned barcode
stays authoritative.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from smftools.logging_utils import get_logger

logger = get_logger(__name__)

CONTAMINATION_QC_SUBDIR = "barcode_contamination_qc"
PER_BARCODE_FILENAME = "contamination_by_barcode.parquet"
SUMMARY_FILENAME = "contamination_summary.json"
QC_SCHEMA_VERSION = 2
#: Library reads a barcode needs before its enrichment is treated as a finding.
#:
#: Enrichment divides the spike-in share by the library share, so a barcode with
#: a handful of library reads has a near-zero denominator and the ratio
#: explodes. On a real run, barcodes with 1 and 5 library reads produced
#: enrichments of 2,606x and 1,042x and were flagged significant -- sitting at
#: the top of the table where they are most misleading. The Poisson interval
#: cannot catch this: it is computed on the spike-in *count* and knows nothing
#: about uncertainty in the denominator (`F50`).
MIN_LIBRARY_READS_FOR_ENRICHMENT = 1000

#: Spellings that mean "no barcode was assigned" rather than naming one.
UNASSIGNED = frozenset({"", "unclassified", "unassigned", "unknown", "none", "nan", "null", "na"})

REFERENCE_COLUMN = "Reference_strand"
_STRAND_SUFFIXES = ("_top", "_bottom")


def _is_assigned(values: pd.Series) -> pd.Series:
    """True where a barcode column names an actual barcode."""
    return ~values.astype(str).str.strip().str.lower().isin(UNASSIGNED)


def amplicon_names(references: pd.Series) -> pd.Series:
    """Strip the strand suffix so `ctcf_mNanog_top` matches a configured `ctcf_mNanog`.

    References are stored per strand, but a spike-in is named as an amplicon.
    Requiring the user to list both strands would be a trap that fails silently
    on the half they forgot.
    """
    text = references.astype(str)
    for suffix in _STRAND_SUFFIXES:
        text = text.str.removesuffix(suffix)
    return text


def spike_in_mask(obs: pd.DataFrame, spike_in_references: Iterable[str]) -> pd.Series:
    """Rows belonging to a configured spike-in amplicon."""
    wanted = {str(name).strip() for name in spike_in_references if str(name).strip()}
    if not wanted or REFERENCE_COLUMN not in obs.columns:
        return pd.Series(False, index=obs.index)
    return amplicon_names(obs[REFERENCE_COLUMN]).isin(wanted)


def poisson_interval(count: int, *, confidence: float = 0.95) -> tuple[float, float]:
    """Byar's approximation to the exact Poisson interval for a count.

    Chosen over a normal approximation because the per-barcode counts here are
    small -- tens of reads -- where a Wald interval goes negative and stops
    meaning anything. Byar's is accurate to well under a percent for counts of
    one or more and needs no extra dependency.
    """
    if count < 0:
        raise ValueError("count must be non-negative")
    z = 1.959963984540054 if abs(confidence - 0.95) < 1e-9 else _normal_quantile(confidence)
    if count == 0:
        # The lower bound is exactly zero; the upper is the one-sided limit.
        return 0.0, -math.log1p(-confidence)
    lower = count * (1.0 - 1.0 / (9.0 * count) - z / (3.0 * math.sqrt(count))) ** 3
    upper = (count + 1) * (1.0 - 1.0 / (9.0 * (count + 1)) + z / (3.0 * math.sqrt(count + 1))) ** 3
    return max(0.0, lower), upper


def _normal_quantile(confidence: float) -> float:
    """Two-sided standard-normal quantile, via the inverse error function."""
    from statistics import NormalDist

    return NormalDist().inv_cdf(0.5 + confidence / 2.0)


class ContaminationQCError(RuntimeError):
    """Raised when obs cannot support an honest contamination measurement."""


def _require_columns(obs: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in obs.columns]
    if missing:
        raise ContaminationQCError(
            "barcode contamination QC needs "
            + ", ".join(repr(column) for column in missing)
            + " in obs. These are published by raw identity schema 2 (`F35`); a store "
            "written before it collapsed the directory assignment and the sequence "
            "re-derivation into one column, which cannot be compared against itself. "
            "Rebuild the raw stage rather than substituting the collapsed column."
        )


def contamination_by_barcode(
    obs: pd.DataFrame,
    *,
    spike_in_references: Iterable[str],
    barcode_column: str = "barcode_assigned",
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Per-barcode contamination of the spike-in, with Poisson intervals.

    ``enrichment`` is the share of contaminated spike-in reads carrying a barcode
    divided by that barcode's share of the library. Free adapter scales with
    library size, so 1.0 is the null expectation and the interesting barcodes are
    those whose interval excludes it -- not those with the largest raw count.
    """
    _require_columns(obs, [barcode_column])
    spike = spike_in_mask(obs, spike_in_references)
    if not spike.any():
        raise ContaminationQCError(
            "no reads matched the configured spike_in_references; nothing to measure against"
        )

    barcodes = obs[barcode_column].astype(str)
    assigned = _is_assigned(barcodes)
    spike_assigned = obs.loc[spike & assigned, barcode_column].astype(str)
    library = obs.loc[~spike & assigned, barcode_column].astype(str)

    spike_counts = spike_assigned.value_counts()
    library_counts = library.value_counts()
    total_spike_assigned = int(spike_counts.sum())
    total_library = int(library_counts.sum())

    frame = pd.DataFrame(
        {
            "spike_in_reads": spike_counts,
            "library_reads": library_counts,
        }
    ).fillna(0)
    frame.index.name = "barcode"
    frame = frame.astype({"spike_in_reads": int, "library_reads": int})

    frame["spike_in_share"] = _safe_divide(frame["spike_in_reads"], total_spike_assigned)
    frame["library_share"] = _safe_divide(frame["library_reads"], total_library)
    frame["enrichment"] = _safe_divide(frame["spike_in_share"], frame["library_share"])

    # Enrichment is linear in the count, so a Poisson interval on the count maps
    # straight onto one for the enrichment. This also gives a zero-count barcode
    # a meaningful upper bound -- the most contamination its data could hide --
    # instead of a bare NaN.
    counts = frame["spike_in_reads"].to_numpy()
    per_count = _safe_divide(
        1.0 / max(total_spike_assigned, 1),
        frame["library_share"].to_numpy(),
    )
    intervals = [poisson_interval(int(count), confidence=confidence) for count in counts]
    frame["enrichment_low"] = np.array([low for low, _ in intervals]) * per_count
    frame["enrichment_high"] = np.array([high for _, high in intervals]) * per_count

    # A barcode with almost no library reads has no meaningful share to divide
    # by, so it is reported but never ranked as a finding.
    frame["library_reads_sufficient"] = frame["library_reads"] >= MIN_LIBRARY_READS_FOR_ENRICHMENT
    frame["significant"] = (frame["enrichment_low"] > 1.0) & frame["library_reads_sufficient"]
    return frame.sort_values("spike_in_reads", ascending=False)


def _safe_divide(numerator, denominator):
    """Elementwise division that yields NaN rather than raising on a zero denominator."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return numerator / np.where(np.asarray(denominator) == 0, np.nan, denominator)


def mislabeling_by_demux_type(
    obs: pd.DataFrame,
    *,
    spike_in_references: Iterable[str],
    barcode_column: str = "barcode_assigned",
    demux_column: str = "demux_type",
    confidence: float = 0.95,
) -> dict[str, Any]:
    """How often each demux type mislabels, measured on known-negative reads.

    Every spike-in read carrying a barcode is an error, so this is a direct
    per-read probability of spurious assignment rather than an inferred one.

    The denominator is *all* spike-in reads, including those correctly left
    unassigned -- which is why the unclassified population has to be ingested.
    Without it only the composition of the errors is knowable, not their rate.

    Every rate divides by the **whole** spike-in population, never by the subset
    carrying that demux call. `demux_type` is an *outcome* of the mis-ligation,
    not a stratum that exists beforehand: a read with barcodes at both ends will
    be assigned, so "100% of double-ended spike-in reads are contaminated" is
    near-tautological and says nothing about reliability. Dividing within the
    stratum reported a discrimination of 0.88x on a run whose true figure is
    ~560x -- it inverted the conclusion (`F43`).

    ``discrimination`` is therefore the ratio of two per-molecule exposures: how
    much more likely a spike-in molecule is to acquire a spurious single-ended
    barcode than a spurious double-ended one, which is what requiring
    double-ended assignment actually buys.
    """
    _require_columns(obs, [barcode_column, demux_column])
    spike = spike_in_mask(obs, spike_in_references)
    if not spike.any():
        raise ContaminationQCError(
            "no reads matched the configured spike_in_references; nothing to measure against"
        )

    spike_obs = obs.loc[spike]
    assigned = _is_assigned(spike_obs[barcode_column])
    demux = spike_obs[demux_column].astype(str)

    total = int(len(spike_obs))
    errors_total = int(assigned.sum())
    by_type: dict[str, dict[str, Any]] = {}
    for demux_type in sorted(set(demux) | {"single", "double"}):
        in_type = demux == demux_type
        errors = int((in_type & assigned).sum())
        low, high = poisson_interval(errors, confidence=confidence)
        by_type[demux_type] = {
            # Population carrying this call, for context only -- never the
            # denominator below.
            "spike_in_reads_of_type": int(in_type.sum()),
            "spurious_assignments": errors,
            "rate_per_spike_in_read": (errors / total) if total else None,
            "rate_low": (low / total) if total else None,
            "rate_high": (high / total) if total else None,
        }

    single = by_type.get("single", {}).get("rate_per_spike_in_read")
    double = by_type.get("double", {}).get("rate_per_spike_in_read")
    discrimination = (single / double) if single and double else None
    return {
        "spike_in_reads": total,
        "mislabeled_reads": errors_total,
        "contamination_rate": (errors_total / total) if total else None,
        "by_demux_type": by_type,
        "single_over_double_discrimination": discrimination,
    }


def end_disagreement(
    obs: pd.DataFrame,
    *,
    front_column: str = "barcode_front",
    rear_column: str = "barcode_rear",
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Reads whose two ends name different barcodes, across the whole run.

    This needs no spike-in. A molecule carrying one barcode at its start and a
    different one at its end was mis-ligated, and both ends observed it -- so
    unlike the spike-in estimate, which rests on 0.1% of the data, this applies
    to every read where both ends were called.

    It is a floor rather than the whole rate: two ends that mis-ligate the *same*
    wrong barcode agree with each other and are invisible here. Read together
    with the spike-in numbers, which do see that case.
    """
    _require_columns(obs, [front_column, rear_column])
    front = obs[front_column].astype(str).str.strip()
    rear = obs[rear_column].astype(str).str.strip()
    both_called = _is_assigned(front) & _is_assigned(rear)
    comparable = int(both_called.sum())
    if not comparable:
        return {
            "comparable_reads": 0,
            "disagreeing_reads": 0,
            "disagreement_rate": None,
            "rate_low": None,
            "rate_high": None,
            "top_pairs": [],
        }
    disagree = both_called & (front.str.casefold() != rear.str.casefold())
    count = int(disagree.sum())
    low, high = poisson_interval(count, confidence=confidence)
    pairs = pd.DataFrame({"front": front[disagree], "rear": rear[disagree]}).value_counts().head(10)
    return {
        "comparable_reads": comparable,
        "disagreeing_reads": count,
        "disagreement_rate": count / comparable,
        "rate_low": low / comparable,
        "rate_high": high / comparable,
        "top_pairs": [
            {"front": str(index[0]), "rear": str(index[1]), "reads": int(value)}
            for index, value in pairs.items()
        ],
    }


def barcode_contamination_report(
    obs: pd.DataFrame,
    *,
    spike_in_references: Iterable[str],
    barcode_column: str = "barcode_assigned",
    confidence: float = 0.95,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run every contamination measure obs can support.

    End disagreement is computed even when the spike-in sections cannot run, and
    vice versa: they answer the same question from independent evidence, and a
    run without a spike-in should still get the part that does not need one.
    """
    per_barcode = pd.DataFrame()
    summary: dict[str, Any] = {
        "schema_version": QC_SCHEMA_VERSION,
        "spike_in_references": sorted({str(name) for name in spike_in_references if str(name)}),
        "confidence": confidence,
        "total_reads": int(len(obs)),
    }

    try:
        per_barcode = contamination_by_barcode(
            obs,
            spike_in_references=spike_in_references,
            barcode_column=barcode_column,
            confidence=confidence,
        )
        summary["spike_in"] = mislabeling_by_demux_type(
            obs,
            spike_in_references=spike_in_references,
            barcode_column=barcode_column,
            confidence=confidence,
        )
        thin = per_barcode.loc[~per_barcode["library_reads_sufficient"]]
        summary["barcodes_below_library_floor"] = [
            {"barcode": str(index), "library_reads": int(row["library_reads"])}
            for index, row in thin.iterrows()
        ]
        summary["min_library_reads_for_enrichment"] = MIN_LIBRARY_READS_FOR_ENRICHMENT
        significant = per_barcode.loc[per_barcode["significant"]]
        summary["barcodes_enriched_above_library_share"] = [
            {"barcode": str(index), "enrichment": float(row["enrichment"])}
            for index, row in significant.iterrows()
        ]
    except ContaminationQCError as error:
        # Not an error for a run with no spike-in, and not a reason to lose the
        # end-disagreement measure below.
        logger.info("Spike-in contamination QC skipped: %s", error)
        summary["spike_in"] = None
        summary["spike_in_skipped_reason"] = str(error)

    try:
        summary["end_disagreement"] = end_disagreement(obs, confidence=confidence)
    except ContaminationQCError as error:
        logger.info("End-disagreement QC skipped: %s", error)
        summary["end_disagreement"] = None
        summary["end_disagreement_skipped_reason"] = str(error)

    return per_barcode, summary


def write_barcode_contamination_qc(
    obs: pd.DataFrame,
    output_dir: str | Path,
    *,
    spike_in_references: Iterable[str],
    barcode_column: str = "barcode_assigned",
    confidence: float = 0.95,
) -> dict[str, Path]:
    """Write the per-barcode table and the summary beside a preprocess generation."""
    output_dir = Path(output_dir) / CONTAMINATION_QC_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    per_barcode, summary = barcode_contamination_report(
        obs,
        spike_in_references=spike_in_references,
        barcode_column=barcode_column,
        confidence=confidence,
    )
    per_barcode_path = output_dir / PER_BARCODE_FILENAME
    per_barcode.reset_index().to_parquet(per_barcode_path, index=False)
    summary_path = output_dir / SUMMARY_FILENAME
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _log_summary(summary)
    return {"per_barcode": per_barcode_path, "summary": summary_path}


def _log_summary(summary: Mapping[str, Any]) -> None:
    """Put the headline numbers in the log, where a silent QC file is easy to miss."""
    spike = summary.get("spike_in")
    if spike:
        rate = spike.get("contamination_rate")
        logger.info(
            "Spike-in contamination: %d of %d reads carry a barcode (%.3f%%)",
            spike.get("mislabeled_reads", 0),
            spike.get("spike_in_reads", 0),
            100.0 * rate if rate is not None else float("nan"),
        )
        discrimination = spike.get("single_over_double_discrimination")
        if discrimination:
            logger.info(
                "Single-ended assignments are %.0fx more likely to be spurious than double-ended",
                discrimination,
            )
    ends = summary.get("end_disagreement")
    if ends and ends.get("disagreement_rate") is not None:
        logger.info(
            "Barcode ends disagree on %d of %d reads with both ends called (%.3f%%)",
            ends["disagreeing_reads"],
            ends["comparable_reads"],
            100.0 * ends["disagreement_rate"],
        )
