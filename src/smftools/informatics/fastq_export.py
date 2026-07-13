"""Write per-barcode FASTQ files of reads directly from the raw ragged store.

Reads sequence and quality straight from the ragged parquet shards (the literal
as-sequenced read in query coordinates, not a reference-padded reconstruction),
so this needs only the raw spine's ``obs`` (with its ``ragged_shard`` pointer
column) -- no BAM re-parsing or dense materialization.
"""

from __future__ import annotations

import gzip
import re
from pathlib import Path
from typing import Mapping

import pandas as pd

from smftools.logging_utils import get_logger

from .ragged_store import QUALITY, READ_ID, SEQUENCE, read_ragged_parquet
from .sequence_encoding import decode_int_sequence, phred_to_fastq_quality_string

logger = get_logger(__name__)

_UNSAFE_FILENAME_CHARS = re.compile(r"[^\w.-]+")


def _sanitize_filename(name: str) -> str:
    sanitized = _UNSAFE_FILENAME_CHARS.sub("_", str(name)).strip("_")
    return sanitized or "unknown"


def write_fastq_per_barcode(
    raw_obs: pd.DataFrame,
    base_dir: str | Path,
    outdir: str | Path,
    *,
    read_ids: set[str] | None = None,
    group_labels: pd.Series | str = "Barcode",
    gzip_output: bool = True,
) -> dict[str, dict[str, object]]:
    """Write one FASTQ per barcode/group from the raw ragged store.

    Reads are grouped by their ``ragged_shard`` pointer so every shard is read
    at most once, regardless of how many barcode groups its reads belong to;
    decoded records then stream to one open file handle per barcode.

    Args:
        raw_obs: The raw spine's ``obs``, indexed by read_id, with a
            ``ragged_shard`` column (see ``informatics.raw_store.write_raw_store``).
        base_dir: Raw spine's parent directory, used to resolve ``ragged_shard``
            relative paths.
        outdir: Directory to write ``<barcode>.fastq[.gz]`` into.
        read_ids: Read IDs to include (e.g. a QC-passed set). ``None`` writes
            every read in ``raw_obs``.
        group_labels: Either a column name in ``raw_obs``, or an externally
            supplied ``Series`` (indexed by read_id) of per-read group/barcode
            labels -- e.g. sample-sheet-enriched names from a preprocessed AnnData.
        gzip_output: Whether to gzip-compress the FASTQ output.

    Returns:
        dict[str, dict]: ``{barcode: {"path": Path, "n_reads": int}}`` for every
        barcode that had at least one read written.

    Raises:
        KeyError: If ``raw_obs`` lacks a ``ragged_shard`` column, or
            ``group_labels`` names a missing column.
    """
    if "ragged_shard" not in raw_obs.columns:
        raise KeyError("raw_obs is missing 'ragged_shard'; pass the raw spine's obs")

    base_dir = Path(base_dir)
    outdir = Path(outdir)

    subset = raw_obs if read_ids is None else raw_obs.loc[raw_obs.index.isin(read_ids)]
    if subset.empty:
        logger.warning("write_fastq_per_barcode: no reads matched; nothing written.")
        return {}

    if isinstance(group_labels, str):
        if group_labels not in subset.columns:
            raise KeyError(f"raw_obs is missing group-by column {group_labels!r}")
        labels = subset[group_labels].astype(str)
    else:
        labels = group_labels.reindex(subset.index)
        missing = labels.isna()
        if missing.any():
            logger.warning(
                "%d read(s) have no group label; falling back to 'unknown'.",
                int(missing.sum()),
            )
        labels = labels.astype(str).where(~missing, "unknown")

    outdir.mkdir(parents=True, exist_ok=True)
    suffix = ".fastq.gz" if gzip_output else ".fastq"

    def _open(path: Path):
        return gzip.open(path, "wt") if gzip_output else open(path, "w")

    handles: dict[str, object] = {}
    paths: dict[str, Path] = {}
    counts: dict[str, int] = {}

    def _handle_for(barcode: str):
        handle = handles.get(barcode)
        if handle is None:
            path = outdir / f"{_sanitize_filename(barcode)}{suffix}"
            handle = _open(path)
            handles[barcode] = handle
            paths[barcode] = path
            counts[barcode] = 0
        return handle

    try:
        for shard_rel, shard_group in subset.groupby("ragged_shard", sort=False, observed=True):
            shard_path = base_dir / str(shard_rel)
            shard_ids = set(shard_group.index.astype(str))
            frame = read_ragged_parquet(shard_path, read_ids=shard_ids).set_index(
                READ_ID, drop=False
            )
            for read_id in shard_group.index.astype(str):
                row = frame.loc[read_id]
                barcode = labels.loc[read_id]
                sequence = "".join(decode_int_sequence(row[SEQUENCE]))
                quality = phred_to_fastq_quality_string(row[QUALITY])
                handle = _handle_for(barcode)
                handle.write(f"@{read_id}\n{sequence}\n+\n{quality}\n")
                counts[barcode] += 1
    finally:
        for handle in handles.values():
            handle.close()

    logger.info(
        "Wrote %d FASTQ file(s) covering %d read(s) to %s",
        len(handles),
        sum(counts.values()),
        outdir,
    )
    return {
        barcode: {"path": paths[barcode], "n_reads": counts[barcode]} for barcode in handles
    }


def write_fastq_manifest(outdir: str | Path, manifest: Mapping[str, Mapping[str, object]]) -> Path:
    """Write a ``fastq_manifest.csv`` summarizing per-barcode read counts and paths."""
    rows = [
        {"barcode": barcode, "n_reads": int(info["n_reads"]), "path": str(info["path"])}
        for barcode, info in sorted(manifest.items())
    ]
    path = Path(outdir) / "fastq_manifest.csv"
    pd.DataFrame(rows, columns=["barcode", "n_reads", "path"]).to_csv(path, index=False)
    return path
