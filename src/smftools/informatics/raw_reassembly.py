"""Rebuild a raw generation's obs from its existing shards (`F34`).

The raw stage has two phases with wildly different costs. Extraction reads the
aligned BAM and sidecars and writes the ragged parquet shards -- roughly an hour
on a real run. Assembly turns those shards into the molecule spine: obs, the
catalogs, `spine.h5ad`. The two are fused inside `write_raw_store_streaming`, so
a change that touches only the second one still pays for the first.

`F31` was exactly that change. It added `demux_type` and `barcode_agreement` to
obs and nothing else, yet recovering those columns meant re-extracting shards
that were already correct and already on disk.

They *are* recoverable, because the shards keep every scalar column alongside
the four ragged arrays -- `BM` and `barcode` included. Parquet is columnar, so
projecting the arrays away is nearly free: reading all scalar columns for 1.28M
reads across 124 shards measures at **0.4 seconds** against ~60 minutes to
re-extract them.

This module replays the assembly phase against shards that already exist. It
reuses `_build_segment_obs` and `_collapse_segment_obs` rather than
reimplementing them, so a rebuilt obs is the same function of the same inputs
that the live path computes -- the only intended difference is the annotation,
which is deliberately re-run.

What it does *not* do is re-derive anything that needs the BAM. Annotation that
reads sidecars, POD5 metadata, or BAM tags absent from the shards cannot be
replayed here, and `reassemble_obs` will not pretend otherwise: pass an
``annotate`` callable that only touches shard-resident columns.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterator

import anndata as ad
import pandas as pd
import pyarrow.parquet as pq

from smftools.logging_utils import get_logger

from ..readwrite import safe_read_h5ad, safe_write_h5ad
from .demux_agreement import annotate_demux_obs
from .molecule_identity import MOLECULE_UID_COLUMN
from .ragged_store import RAGGED_ARRAY_COLUMNS, READ_ID
from .raw_store import (
    SEGMENTS_FILENAME,
    SPINE_FILENAME,
    _build_segment_obs,
    _collapse_segment_obs,
)
from .stage_obs import write_stage_obs

logger = get_logger(__name__)

Annotator = Callable[[pd.DataFrame], pd.DataFrame]


def shard_relative_paths(generation_dir: str | Path) -> list[str]:
    """Shard paths for ``generation_dir``, in the order the spine records them.

    Read from ``spine.uns['ragged_store']`` rather than by globbing: the spine's
    order is the one `ragged_row` pointers were assigned against, and a glob
    would also pick up shards belonging to a partially-written sibling.
    """
    generation_dir = Path(generation_dir)
    spine, _ = safe_read_h5ad(generation_dir / SPINE_FILENAME, verbose=False)
    store = spine.uns.get("ragged_store")
    if store is None:
        raise ValueError(f"{generation_dir} has no 'ragged_store' pointer in spine.uns")
    return [str(entry) for entry in store]


def iter_shard_scalars(
    generation_dir: str | Path,
    *,
    relative_paths: list[str] | None = None,
) -> Iterator[tuple[str, pd.DataFrame]]:
    """Yield ``(relative_path, scalar_frame)`` for each shard, arrays projected out.

    Shards do not share one schema -- `reference` is dictionary-encoded in some
    and a plain string in others -- so each file is opened with its own schema
    instead of through a merged dataset, which would raise on the mismatch.
    """
    generation_dir = Path(generation_dir)
    if relative_paths is None:
        relative_paths = shard_relative_paths(generation_dir)
    for relative_path in relative_paths:
        handle = pq.ParquetFile(generation_dir / relative_path)
        columns = [name for name in handle.schema_arrow.names if name not in RAGGED_ARRAY_COLUMNS]
        yield relative_path, handle.read(columns=columns).to_pandas()


def reassemble_obs(
    generation_dir: str | Path,
    *,
    annotate: Annotator | None = annotate_demux_obs,
) -> pd.DataFrame:
    """Rebuild molecule-level obs from ``generation_dir``'s shards.

    With ``annotate=None`` this reproduces the stored obs exactly, which is how
    the reconstruction is tested against ground truth: any difference then is a
    defect in this module rather than an intended re-annotation.
    """
    generation_dir = Path(generation_dir)
    started = perf_counter()
    relative_paths = shard_relative_paths(generation_dir)

    segment_frames: list[pd.DataFrame] = []
    for relative_path, frame in iter_shard_scalars(generation_dir, relative_paths=relative_paths):
        if annotate is not None:
            frame = annotate(frame)
        read_ids = frame[READ_ID].astype(str)
        # Rows are written to a shard in order, so position within the file is
        # the `ragged_row` the pointer was originally assigned.
        shard_by_read = dict.fromkeys(read_ids, relative_path)
        row_by_read = {read_id: row for row, read_id in enumerate(read_ids)}
        segment_frames.append(_build_segment_obs(frame, shard_by_read, row_by_read))

    if not segment_frames:
        raise ValueError(f"{generation_dir} has no shards to reassemble from")

    segment_obs = pd.concat(segment_frames)
    molecule_obs = _collapse_segment_obs(segment_obs)
    molecule_obs = _restore_canonical_order(molecule_obs, generation_dir)
    # `write_raw_store_streaming` clears this before writing the spine; leaving
    # it set makes anndata refuse the frame when a `read_id` column is also
    # present, which it always is.
    molecule_obs.index.name = None

    logger.info(
        "Reassembled obs for %d molecules from %d shard(s) in %.2fs",
        len(molecule_obs),
        len(relative_paths),
        perf_counter() - started,
    )
    return molecule_obs


def _restore_canonical_order(
    molecule_obs: pd.DataFrame,
    generation_dir: Path,
) -> pd.DataFrame:
    """Apply the stored `canonical_row` ordering to a rebuilt obs.

    The live path derives this from the in-memory segment rows; here it is read
    back from the persisted segment catalog so a rebuild lands molecules in the
    same order the generation already published them.
    """
    segments_path = _resolve_segments_path(generation_dir)
    if segments_path is None:
        return molecule_obs
    segments = pd.read_parquet(segments_path, columns=[MOLECULE_UID_COLUMN, "canonical_row"])
    first_rows = segments.groupby(MOLECULE_UID_COLUMN, sort=False, observed=True)[
        "canonical_row"
    ].min()
    molecule_obs = molecule_obs.copy()
    molecule_obs["canonical_row"] = molecule_obs[MOLECULE_UID_COLUMN].map(first_rows).astype(int)
    # The index stays the read id rather than being reset: `write_stage_obs`
    # requires obs to be indexed by the same ids its `read_id` column carries,
    # and the live path preserves the index `_collapse_segment_obs` produced.
    return molecule_obs.sort_values("canonical_row", kind="stable")


def _resolve_segments_path(generation_dir: Path) -> Path | None:
    """Locate the segment catalog for a generation, or ``None`` if absent."""
    for candidate in (
        generation_dir / SEGMENTS_FILENAME,
        generation_dir.parent / SEGMENTS_FILENAME,
    ):
        if candidate.exists():
            return candidate
    return None


def reassemble_raw_generation(
    run_root: str | Path,
    *,
    annotate: Annotator | None = annotate_demux_obs,
    generation_dir: str | Path | None = None,
    select_current: bool = True,
) -> dict[str, object]:
    """Publish a new raw generation whose obs is rebuilt from an existing one's shards.

    Generations are immutable, so this never edits the source in place: it
    publishes a sibling through the ordinary raw publication path, which
    hardlinks every artifact whose checksum is unchanged. The 7.9 GB of shards
    are shared with the parent rather than copied -- only obs, the molecule
    catalog, and the spine are genuinely rewritten.

    ``config_hash`` and ``input_artifact_ids`` are carried over from the source
    manifest: the inputs and the configuration really are the same, and the
    whole point is that only the assembly step differs.
    """
    from .generation import resolve_current_generation
    from .raw_generation import RAW_GENERATION_ARTIFACT_PATHS, publish_raw_generation

    run_root = Path(run_root)
    raw_output_dir = run_root / "raw_outputs"

    if generation_dir is None:
        resolved = resolve_current_generation(raw_output_dir)
        if resolved is None:
            raise ValueError(f"{raw_output_dir} has no published generation to reassemble")
        generation_dir, manifest = resolved
    else:
        generation_dir = Path(generation_dir)
        manifest = json.loads(
            (generation_dir / "generation_manifest.json").read_text(encoding="utf-8")
        )

    obs = reassemble_obs(generation_dir, annotate=annotate)
    obs = _carry_bam_path(obs, generation_dir)

    with tempfile.TemporaryDirectory(prefix="smftools-reassemble-") as tmp:
        staging = Path(tmp)
        molecules_path = staging / "molecules.parquet"
        obs.drop(columns=["bam_path"], errors="ignore").to_parquet(molecules_path, index=False)
        obs_path = write_stage_obs(staging, obs)
        spine_path = _write_rebuilt_spine(generation_dir, staging, obs)

        sources: dict[str, Path] = {
            key: generation_dir / relative
            for key, relative in RAW_GENERATION_ARTIFACT_PATHS.items()
        }
        barcode_index = generation_dir / "barcode_index.parquet"
        if barcode_index.exists():
            sources["barcode_index"] = barcode_index
        sources["obs"] = obs_path
        sources["molecules"] = molecules_path
        sources["spine"] = spine_path

        return publish_raw_generation(
            run_root,
            sources,
            config_hash=str(manifest["config_hash"]),
            input_artifact_ids=list(manifest.get("input_artifact_ids") or []),
            reuse_generation=generation_dir,
            select_current=select_current,
        )


def _carry_bam_path(obs: pd.DataFrame, generation_dir: Path) -> pd.DataFrame:
    """Restore the ``bam_path`` column the live path appends after collapsing.

    It is a stage-level constant rather than anything derived per read, so it
    cannot come back from the shards and is copied from the source obs instead.
    """
    stored_obs = generation_dir / "obs.parquet"
    if not stored_obs.exists():
        return obs
    columns = pq.ParquetFile(stored_obs).schema_arrow.names
    if "bam_path" not in columns:
        return obs
    values = pd.read_parquet(stored_obs, columns=["bam_path"])["bam_path"].unique()
    if len(values) != 1:
        logger.warning(
            "bam_path is not constant across the source obs (%d distinct values); "
            "leaving it off the rebuilt obs.",
            len(values),
        )
        return obs
    obs = obs.copy()
    obs["bam_path"] = values[0]
    return obs


def _write_rebuilt_spine(generation_dir: Path, staging: Path, obs: pd.DataFrame) -> Path:
    """Write a spine carrying the rebuilt obs and the source spine's ``uns``.

    Everything in ``uns`` -- the shard list, catalog pointers, reference plans --
    describes artifacts this rebuild reuses unchanged, so it is preserved rather
    than recomputed. The publication path rebinds the generation-relative
    pointers afterwards.
    """
    source_spine, _ = safe_read_h5ad(generation_dir / SPINE_FILENAME, verbose=False)
    spine = ad.AnnData(obs=obs)
    spine.uns.update(dict(source_spine.uns))
    spine_path = staging / SPINE_FILENAME
    safe_write_h5ad(spine, spine_path, backup=False, verbose=False)
    return spine_path
