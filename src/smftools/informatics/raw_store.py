"""Write the ragged source-of-truth store and its thin molecule spine."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Mapping
from urllib.parse import quote

import anndata as ad
import pandas as pd

from smftools.constants import (
    BARCODE,
    DATASET,
    READ_MAPPING_DIRECTION,
    REFERENCE,
    REFERENCE_STRAND,
    SAMPLE,
    STRAND,
)
from smftools.logging_utils import get_logger

from ..readwrite import safe_write_h5ad
from .partition_read import relative_uns_path
from .ragged_store import (
    RAGGED_ARRAY_COLUMNS,
    READ_ID,
    validate_ragged_frame,
    write_ragged_parquet,
)
from .sidecar_manifest import register_sidecar, sidecar_manifest_path
from .storage_planner import plan_references

logger = get_logger(__name__)

RAW_SUBDIR = "raw"
RAW_SHARD_TEMPLATE = "part-{index:05d}.parquet"
INTERVAL_CATALOG_FILENAME = "interval_catalog.parquet"
SPINE_FILENAME = "spine.h5ad"
RAW_SCHEMA_VERSION = 2

_OBS_COLUMN_ALIASES = {
    REFERENCE: (REFERENCE, "reference"),
    REFERENCE_STRAND: (REFERENCE_STRAND, "reference_strand"),
    SAMPLE: (SAMPLE, "sample"),
    BARCODE: (BARCODE, "barcode"),
    STRAND: (STRAND, "strand"),
    DATASET: (DATASET, "dataset"),
    READ_MAPPING_DIRECTION: (READ_MAPPING_DIRECTION, "mapping_direction"),
}

_RAW_SCALAR_OBS_COLUMNS = (
    "BC",
    "U1",
    "U2",
    "Experiment_name",
    "Experiment_name_and_barcode",
    "Read_mismatch_trend",
    "ct_event_count",
    "ga_event_count",
    "strand_segment_purity",
    "strand_switch_position",
    "read_length",
    "read_quality",
    "reference_length",
    "mapped_length",
    "mapping_quality",
    "read_length_to_reference_length_ratio",
    "mapped_length_to_reference_length_ratio",
    "mapped_length_to_read_length_ratio",
    "max_insertion_length",
    "max_deletion_length",
)


def _build_raw_spine_obs(
    frame: pd.DataFrame,
    shard_by_read: Mapping[str, str],
    row_by_read: Mapping[str, int],
) -> pd.DataFrame:
    """Build canonical spine obs columns and ragged row pointers."""
    obs = pd.DataFrame(index=pd.Index(frame[READ_ID].astype(str), name=None))
    for canonical, candidates in _OBS_COLUMN_ALIASES.items():
        source = next((column for column in candidates if column in frame.columns), None)
        if source is not None:
            obs[canonical] = frame[source].to_numpy()

    # Scalar BAM/sidecar metrics belong on the molecule spine. Variable-length
    # arrays remain exclusively in parquet.
    excluded = set(RAGGED_ARRAY_COLUMNS) | {READ_ID}
    for column in _RAW_SCALAR_OBS_COLUMNS:
        if column not in frame or column in excluded or column in obs.columns:
            continue
        non_null = frame[column].dropna()
        if not non_null.empty and isinstance(non_null.iloc[0], (list, tuple)):
            continue
        obs[column] = frame[column].to_numpy()

    # Carry POD5 sequencing/signal metadata (scalar ``pod5_*`` columns). Ragged
    # arrays such as ``pod5_current_pa`` are skipped here and stay in the parquet.
    for column in frame.columns:
        column = str(column)
        if not column.startswith("pod5_") or column in obs.columns:
            continue
        non_null = frame[column].dropna()
        if not non_null.empty and isinstance(non_null.iloc[0], (list, tuple)):
            continue
        obs[column] = frame[column].to_numpy()

    obs["reference_start"] = frame["reference_start"].to_numpy(dtype="int64")
    obs["reference_end"] = (frame["reference_start"] + frame["aligned_length"]).to_numpy(
        dtype="int64"
    )
    obs["aligned_length"] = frame["aligned_length"].to_numpy(dtype="int64")
    obs["ragged_shard"] = [shard_by_read[read_id] for read_id in obs.index]
    obs["ragged_row"] = [row_by_read[read_id] for read_id in obs.index]
    return obs


def _reference_path_component(reference: str) -> str:
    """Return a reversible filesystem-safe reference component."""
    return quote(str(reference), safe="._-")


def write_raw_store(
    frame: pd.DataFrame,
    output_dir: str | Path,
    *,
    reference_lengths: Mapping[str, int],
    shard_size: int = 100_000,
    start_bin_size: int = 1_000_000,
    analysis_mode: str = "auto",
    load_cache_mode: str = "auto",
    max_full_matrix_gb: float = 8.0,
    genome_tile_size: int = 10_000,
    genome_tile_halo: int = 1_000,
    bam_path: str | Path | None = None,
    extra_uns: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Write ragged parquet shards, a thin spine, and manifest registration.

    Args:
        frame: One row per physical read using the ragged schema.
        output_dir: Experiment load output directory.
        reference_lengths: Exact lengths keyed by ``Reference_strand``.
        shard_size: Maximum reads per parquet shard.
        start_bin_size: Width used to group reads by alignment start.
        bam_path: Optional aligned BAM pointer copied to every spine row.
        extra_uns: Additional self-describing metadata for the spine.

    Returns:
        Dictionary containing ``spine``, ``ragged_store``, and ``manifest`` paths.
    """
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    if start_bin_size <= 0:
        raise ValueError("start_bin_size must be positive")
    t0 = perf_counter()
    normalized = validate_ragged_frame(frame).reset_index(drop=True)
    if normalized.empty:
        raise ValueError("cannot write an empty raw store")

    output_dir = Path(output_dir)
    raw_dir = output_dir / RAW_SUBDIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    normalized["reference_end"] = (
        normalized["reference_start"] + normalized["aligned_length"]
    ).astype("int64")
    normalized["start_bin"] = (normalized["reference_start"] // start_bin_size).astype("int64")
    logger.info(
        "Writing raw store: %d reads, %d reference strand(s), shard_size=%d, start_bin_size=%d",
        len(normalized),
        normalized[REFERENCE_STRAND].astype(str).nunique(),
        shard_size,
        start_bin_size,
    )

    shard_paths: list[Path] = []
    catalog_rows: list[dict[str, object]] = []
    shard_by_read: dict[str, str] = {}
    row_by_read: dict[str, int] = {}
    grouped = normalized.sort_values(
        [REFERENCE_STRAND, "start_bin", "reference_start", READ_ID], kind="stable"
    ).groupby([REFERENCE_STRAND, "start_bin"], sort=True, observed=True)
    for (reference, start_bin), group in grouped:
        group_t0 = perf_counter()
        reference_dir = raw_dir / f"reference={_reference_path_component(str(reference))}"
        bin_dir = reference_dir / f"start_bin={int(start_bin):012d}"
        logger.debug(
            "Writing raw shard group reference=%s start_bin=%d (%d reads)",
            reference,
            int(start_bin),
            len(group),
        )
        for shard_index, start in enumerate(range(0, len(group), shard_size)):
            shard = group.iloc[start : start + shard_size]
            path = bin_dir / RAW_SHARD_TEMPLATE.format(index=shard_index)
            relative_path = path.relative_to(output_dir).as_posix()
            write_ragged_parquet(shard.drop(columns=["reference_end", "start_bin"]), path)
            shard_paths.append(path)
            for row_number, read_id in enumerate(shard[READ_ID].astype(str)):
                shard_by_read[read_id] = relative_path
                row_by_read[read_id] = row_number
            catalog_rows.append(
                {
                    "reference": str(reference),
                    "start_bin": int(start_bin),
                    "min_start": int(shard["reference_start"].min()),
                    "max_end": int(shard["reference_end"].max()),
                    "n_reads": len(shard),
                    "aligned_bases": int(shard["aligned_length"].sum()),
                    "group_path": relative_path,
                }
            )
        logger.debug(
            "Finished raw shard group reference=%s start_bin=%d in %.2fs",
            reference,
            int(start_bin),
            perf_counter() - group_t0,
        )

    catalog_path = output_dir / INTERVAL_CATALOG_FILENAME
    pd.DataFrame(catalog_rows).to_parquet(catalog_path, index=False)

    obs = _build_raw_spine_obs(normalized, shard_by_read, row_by_read)
    if bam_path is not None:
        # Relative to the run's output_directory (not output_dir/raw_dir), not
        # absolute -- the aligned BAM lives under this same raw store (bam_outputs/),
        # and this column is copied unchanged into every downstream stage's spine
        # via spine.copy(), so it needs the same stage-independent anchor as the
        # uns cross-stage pointers. See _run_root_from_spine_path / relative_uns_path.
        obs["bam_path"] = relative_uns_path(bam_path, output_dir.parent)
    spine = ad.AnnData(obs=obs)
    plans = plan_references(
        normalized,
        reference_lengths,
        analysis_mode=analysis_mode,
        load_cache_mode=load_cache_mode,
        max_full_matrix_gb=max_full_matrix_gb,
        tile_size=genome_tile_size,
        tile_halo=genome_tile_halo,
    )
    spine.uns.update(
        {
            "is_spine": True,
            "raw_schema_version": RAW_SCHEMA_VERSION,
            "ragged_store": [path.relative_to(output_dir).as_posix() for path in shard_paths],
            "interval_catalog": catalog_path.relative_to(output_dir).as_posix(),
            "reference_plans": {plan.reference: plan.to_dict() for plan in plans},
            "reference_lengths": {
                str(reference): int(length) for reference, length in reference_lengths.items()
            },
        }
    )
    if extra_uns:
        spine.uns.update(dict(extra_uns))

    spine_path = output_dir / SPINE_FILENAME
    safe_write_h5ad(spine, spine_path, backup=False, verbose=False)
    manifest_path = sidecar_manifest_path(output_dir)
    register_sidecar(
        manifest_path,
        "ragged_store",
        raw_dir,
        metadata={"schema_version": RAW_SCHEMA_VERSION, "shards": len(shard_paths)},
    )
    register_sidecar(manifest_path, "spine", spine_path)
    register_sidecar(manifest_path, "interval_catalog", catalog_path)
    logger.info(
        "Wrote raw store with %d reads in %d shard(s) in %.2fs",
        len(normalized),
        len(shard_paths),
        perf_counter() - t0,
    )
    return {
        "spine": spine_path,
        "ragged_store": shard_paths,
        "interval_catalog": catalog_path,
        "manifest": manifest_path,
    }
