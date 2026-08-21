"""Demux status from dorado/MinKNOW's `sequencing_summary.txt` (`EGL-29c`).

When the basecaller leaves a sequencing summary beside the FASTQs it is per
read and free to read, so it is worth using before falling back to re-scanning
sequences. It is *not* always present, which is why the sequence scanner
(`EGL-29a`) remains the primary route rather than this one.

**This answers a subtly different question than the `BM` tag does, and the two
must not silently share a provenance.** `BM` is a classifier assertion -- the
extractor found this barcode at this end. The summary gives per-end *scores*,
and the status has to come from thresholding them. On the run that motivated
this lane (`260820_Enh_del_DAFseq`) MinKNOW classified single-ended:
`barcode_score == max(front, rear)`, and 374,648 of 374,649 classified reads
have a found sequence at *both* ends, so presence discriminates nothing.

The thresholding is well-founded rather than arbitrary. Across all 1.74M
classified reads of that run the rear score is cleanly bimodal -- a mode at
30-50 (spurious partial matches), a valley at 60-65 holding 1.0% of reads, and
a mode at 80-100 (genuine rear barcodes). The choice is also insensitive: 55
gives 68.6% double, 70 gives 64.6%, about four points across a wide range.

Each read carries a confidence derived from how far its scores sit from the
threshold, so reads in the valley -- the ones the threshold cannot really
separate -- are identifiable without recomputing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from smftools.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_END_SCORE_THRESHOLD = 62.0
SOURCE = "sequencing_summary"

REQUIRED_COLUMNS = ("read_id", "barcode_front_score", "barcode_rear_score")
_OPTIONAL_COLUMNS = ("barcode_arrangement",)


def find_sequencing_summary(search_root: str | Path) -> Path | None:
    """Locate a `sequencing_summary*.txt` at or near ``search_root``.

    Searches the directory itself and its parents before descending, because
    the natural thing to point `input_data_path` at is `fastq_pass/`, while the
    summary sits beside it in the run directory.
    """
    root = Path(search_root)
    if root.is_file():
        root = root.parent
    candidates: list[Path] = []
    for base in (root, *root.parents[:2]):
        candidates.extend(sorted(base.glob("sequencing_summary*.txt")))
        if candidates:
            break
    if not candidates:
        candidates = sorted(root.rglob("sequencing_summary*.txt"))
    if not candidates:
        return None
    if len(candidates) > 1:
        logger.warning(
            "Multiple sequencing summaries found near %s; using %s. The others are ignored, "
            "which is wrong if this directory holds more than one run.",
            root,
            candidates[0],
        )
    return candidates[0]


def classify_end_status(
    front_score: pd.Series,
    rear_score: pd.Series,
    *,
    threshold: float = DEFAULT_END_SCORE_THRESHOLD,
) -> tuple[pd.Series, pd.Series]:
    """Label reads single/double from per-end scores; return status and confidence.

    A read is *double* when both ends clear the threshold and *single* when
    exactly one does -- rather than thresholding the rear alone, even though on
    the motivating run the front score is almost always high. Requiring both
    states the actual criterion, so a run where the front end is unreliable is
    described correctly instead of silently counted as double.

    Confidence is the distance of the deciding score from the threshold,
    normalized so a score at the threshold scores 0 and one at either extreme
    approaches 1. Reads in the valley are the ones the threshold cannot really
    separate, and this is what makes them findable afterwards.
    """
    front = pd.to_numeric(front_score, errors="coerce")
    rear = pd.to_numeric(rear_score, errors="coerce")
    front_ok = front >= threshold
    rear_ok = rear >= threshold

    status = pd.Series("unclassified", index=front.index, dtype=object)
    status[front_ok & rear_ok] = "double"
    status[front_ok ^ rear_ok] = "single"

    # For a double the weaker end decides; for a single the stronger one does.
    deciding = np.where(front_ok & rear_ok, np.minimum(front, rear), np.maximum(front, rear))
    # Normalize against the room available on the *relevant* side of the
    # threshold, not a single span: with a threshold of 62 a score can run 38
    # points above it but 62 below, so one span would cap a perfect 100 at 0.61
    # and make decisive reads look marginal.
    above = np.maximum(100.0 - threshold, 1e-9)
    below = np.maximum(threshold, 1e-9)
    distance = deciding - threshold
    confidence = pd.Series(
        np.clip(np.where(distance >= 0, distance / above, -distance / below), 0.0, 1.0),
        index=front.index,
    )
    # An unreadable score is treated as "not barcoded at that end" rather than
    # unknown, which is the conservative direction: it can only downgrade a read
    # from double to single, never invent a double. The confidence is zeroed so
    # such reads stay findable.
    unreadable = front.isna() | rear.isna()
    confidence[unreadable] = 0.0
    return status, confidence.astype(float)


def read_demux_status(
    summary_path: str | Path,
    *,
    threshold: float = DEFAULT_END_SCORE_THRESHOLD,
    chunk_size: int = 500_000,
) -> pd.DataFrame:
    """Per-read demux status from a sequencing summary.

    Read in chunks: these files run to millions of rows (1.88M on the
    motivating run) and only five columns are needed, so loading the whole
    table would cost far more memory than the result.
    """
    summary_path = Path(summary_path)
    usecols = [*REQUIRED_COLUMNS, *_OPTIONAL_COLUMNS]
    header = pd.read_csv(summary_path, sep="\t", nrows=0)
    missing = [column for column in REQUIRED_COLUMNS if column not in header.columns]
    if missing:
        raise ValueError(
            f"sequencing summary {summary_path} lacks required column(s) {missing}; "
            "per-end barcode scores are needed to derive demux status."
        )
    available = [column for column in usecols if column in header.columns]

    frames = []
    for chunk in pd.read_csv(
        summary_path, sep="\t", usecols=available, chunksize=chunk_size, low_memory=False
    ):
        status, confidence = classify_end_status(
            chunk["barcode_front_score"], chunk["barcode_rear_score"], threshold=threshold
        )
        frames.append(
            pd.DataFrame(
                {
                    "read_id": chunk["read_id"].astype(str),
                    "demux_type": status.to_numpy(),
                    "demux_type_source": SOURCE,
                    "demux_type_confidence": confidence.to_numpy(),
                }
            )
        )
    if not frames:
        return pd.DataFrame(
            columns=["read_id", "demux_type", "demux_type_source", "demux_type_confidence"]
        )
    result = pd.concat(frames, ignore_index=True)
    logger.info(
        "Read demux status for %d read(s) from %s (threshold %.1f): %s",
        len(result),
        summary_path.name,
        threshold,
        result["demux_type"].value_counts().to_dict(),
    )
    return result


def attach_demux_status(
    obs: pd.DataFrame,
    status: pd.DataFrame,
    *,
    overwrite: bool = False,
) -> int:
    """Attach summary-derived status to ``obs``, returning the rows filled.

    Does not overwrite an existing `demux_type` unless asked. The `BM` route is
    a classifier assertion and this one is a score threshold, so where both
    exist the assertion is the better evidence -- but the provenance column
    records which produced each value either way, so a mixed column is still
    interpretable rather than ambiguous.
    """
    indexed = status.drop_duplicates("read_id").set_index("read_id")
    aligned = indexed.reindex(obs.index.astype(str))

    if "demux_type" in obs.columns and not overwrite:
        existing = obs["demux_type"].astype(str)
        fill = existing.isin(["", "nan", "None", "unknown"]) | existing.isna()
        if not fill.any():
            logger.info(
                "demux_type already populated for every read; keeping it and not applying "
                "the sequencing-summary values."
            )
            return 0
    else:
        fill = pd.Series(True, index=obs.index)

    fill = fill & aligned["demux_type"].notna().to_numpy()
    if not fill.any():
        return 0
    for column in ("demux_type", "demux_type_source", "demux_type_confidence"):
        if column not in obs.columns:
            obs[column] = np.nan if column.endswith("confidence") else ""
        obs.loc[fill, column] = aligned.loc[fill, column].to_numpy()
    return int(fill.sum())
