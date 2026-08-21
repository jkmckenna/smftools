"""Variant segment clustermaps for the partitioned pipeline.

The pre-partition `variant` CLI rendered these; the partitioned pipeline never
did, because the renderer reads two dense layers (``*_variant_segments`` and
``*_variant_call``) and the partitioned store keeps variant evidence sparse --
one row per read per informative site, plus one row per segment and per
breakpoint -- precisely so it never materializes a dense per-read matrix.

Both layers are reconstructible from that sparse evidence, so this module
rasterizes them for one reference at a time and hands them to the existing
renderer rather than reimplementing it.

Segments are *read back* from the stored ``events.parquet`` rather than
recomputed. Recomputation would have to re-derive each read's span and aligned
member and re-run the segmenter, and any drift between that and what the
pipeline actually recorded would show up as a picture disagreeing with the
flags it is meant to explain.

Note the segment geometry is deliberately the raw interpolation, including
single-site slivers of the other reference. The chimera *flag*
(`variant_chimera_min_adjacent_sites`, `F14`) gates interpretation, not
geometry, so the mismatch-type row strip reflects the gated call while the
heatmap still shows every switch the caller saw. That contrast is the point of
the plot: it is how you check the threshold is doing the right thing.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from smftools.constants import DEFAULT_CLUSTERMAP_MAX_READS_PER_PLOT, REFERENCE_STRAND
from smftools.logging_utils import get_logger

logger = get_logger(__name__)

NO_COVERAGE_STATE = 0


def _read_shards(variant_dir: Path, filename: str, columns: list[str]) -> pd.DataFrame:
    """Concatenate one kind of task-store shard across every task."""
    pattern = str(variant_dir / "task_store" / "**" / filename)
    shards = sorted(glob.glob(pattern, recursive=True))
    if not shards:
        return pd.DataFrame(columns=columns)
    frames = [pd.read_parquet(shard, columns=columns) for shard in shards]
    frame = pd.concat(frames, ignore_index=True)
    frame["read_id"] = frame["read_id"].astype(str)
    return frame


def _select_reads(
    obs: pd.DataFrame,
    sample_column: str,
    max_reads_per_panel: int,
    *,
    seed: int = 0,
) -> pd.DataFrame:
    """Cap reads per (reference, sample) panel by seeded random sampling.

    Reproducibility comes from the *seed*, not from taking the first N. The
    original version took the head of a read-id sort, reasoning that a plot
    whose composition changes between runs over unchanged data is not a
    diagnostic. The reasoning holds; the implementation was the wrong way to get
    it. Read ids are not randomly ordered with respect to anything, so first-N
    is a biased slice of the population, while a seeded sample is both
    reproducible and representative -- which is what `select_plot_reads` and
    `subsample_reads_for_plot` already do elsewhere (`EGL-27`).
    """
    if not max_reads_per_panel or max_reads_per_panel <= 0:
        return obs
    cap = int(max_reads_per_panel)
    # Explicit loop rather than `groupby.apply`: the latter also operates on the
    # grouping columns, which pandas deprecates, and the per-group short-circuit
    # is clearer here than inside a lambda.
    parts = [
        group if len(group) <= cap else group.sample(n=cap, random_state=seed)
        for _key, group in obs.groupby([REFERENCE_STRAND, sample_column], observed=True, sort=False)
    ]
    if not parts:
        return obs
    sampled = pd.concat(parts)
    return sampled.sort_values([REFERENCE_STRAND, sample_column, "read_id"], kind="stable")


def build_variant_segment_layers(
    read_ids: list[str],
    positions: np.ndarray,
    segments: pd.DataFrame,
    calls: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rasterize sparse segments and calls onto a (read x position) grid.

    Returns the segment layer (0 no coverage / 1 seq1 / 2 seq2 / 3 transition),
    the variant call layer, and a read-span mask built from the segments
    themselves -- a position is spanned exactly when some segment covers it,
    which is what the renderer's column filter needs.
    """
    row_of = {read_id: row for row, read_id in enumerate(read_ids)}
    n_obs, n_vars = len(read_ids), positions.size
    segment_layer = np.zeros((n_obs, n_vars), dtype=np.int8)
    call_layer = np.zeros((n_obs, n_vars), dtype=np.int8)

    if not segments.empty:
        rows = segments["read_id"].map(row_of)
        keep = rows.notna()
        starts = np.searchsorted(positions, segments.loc[keep, "start"].to_numpy(dtype=np.int64))
        ends = np.searchsorted(positions, segments.loc[keep, "end"].to_numpy(dtype=np.int64))
        states = segments.loc[keep, "state"].to_numpy(dtype=np.int64)
        for row, start, end, state in zip(
            rows[keep].to_numpy(dtype=np.int64), starts, ends, states
        ):
            if end > start:
                segment_layer[row, start:end] = np.int8(state)

    if not calls.empty:
        rows = calls["read_id"].map(row_of)
        keep = rows.notna() & calls["call"].isin([1, 2])
        if bool(keep.any()):
            columns = np.searchsorted(positions, calls.loc[keep, "position"].to_numpy(np.int64))
            inside = (columns >= 0) & (columns < n_vars)
            call_layer[rows[keep].to_numpy(dtype=np.int64)[inside], columns[inside]] = calls.loc[
                keep, "call"
            ].to_numpy(dtype=np.int8)[inside]

    span_mask = (segment_layer != NO_COVERAGE_STATE).astype(np.int8)
    return segment_layer, call_layer, span_mask


def generate_variant_segment_plots(
    variant_dir,
    obs_path,
    plot_layout,
    *,
    cfg=None,
    category: str = "variant_segments",
) -> list[dict]:
    """Render variant segment clustermaps for every reference in a generation."""
    import anndata as ad

    from smftools.cli.stage_artifacts import register_plot_artifact
    from smftools.plotting import plot_variant_segment_clustermaps

    from .reindex_references_adata import reindex_references_adata

    variant_dir = Path(variant_dir)
    events = _read_shards(
        variant_dir, "events.parquet", ["read_id", "event_type", "start", "end", "state"]
    )
    calls = _read_shards(variant_dir, "calls.parquet", ["read_id", "position", "call"])
    if events.empty:
        logger.info("No variant segment events recorded; skipping variant segment clustermaps.")
        return []
    segments = events[events["event_type"] == "segment"].dropna(subset=["start", "end", "state"])
    if segments.empty:
        logger.info("No variant segments recorded; skipping variant segment clustermaps.")
        return []

    obs = pd.read_parquet(obs_path)
    obs["read_id"] = obs["read_id"].astype(str)
    if REFERENCE_STRAND not in obs.columns:
        logger.warning("obs lacks %s; skipping variant segment clustermaps.", REFERENCE_STRAND)
        return []

    sample_column = str(getattr(cfg, "sample_name_col_for_plotting", "Sample"))
    if sample_column not in obs.columns:
        sample_column = "Sample" if "Sample" in obs.columns else "Barcode"

    # Plot the analysed population: dedup-passing where dedup has run, else
    # QC-passing. Matches the `deduplicated/` variant clustermaps the
    # pre-partition CLI produced, and keeps duplicate stacks from dominating a
    # panel. Chimeric reads are deliberately *kept* -- they are the subject of
    # the plot -- but QC failures and duplicates are not.
    for gate in ("passes_dedup", "passes_qc"):
        if gate in obs.columns:
            obs = obs[obs[gate].fillna(False).astype(bool)]
            break
    obs = obs[obs["read_id"].isin(set(segments["read_id"]))]
    if obs.empty:
        logger.info("No QC-passing reads carry variant segments; skipping.")
        return []

    max_reads = int(
        getattr(cfg, "clustermap_max_reads_per_plot", DEFAULT_CLUSTERMAP_MAX_READS_PER_PLOT)
        or DEFAULT_CLUSTERMAP_MAX_READS_PER_PLOT
    )
    obs = _select_reads(
        obs, sample_column, max_reads, seed=int(getattr(cfg, "plot_subsample_seed", 0))
    )

    references = list(getattr(cfg, "references_to_align_for_variant_annotation", []) or [])
    seq1_column = str(references[0]) if len(references) > 0 and references[0] else "member_1"
    seq2_column = str(references[1]) if len(references) > 1 and references[1] else "member_2"
    prefix = f"{seq1_column}__{seq2_column}"

    results: list[dict] = []
    save_root = Path(plot_layout.categories[category])
    for reference, reference_obs in obs.groupby(REFERENCE_STRAND, observed=True, sort=True):
        read_ids = list(reference_obs["read_id"])
        selected = set(read_ids)
        reference_segments = segments[segments["read_id"].isin(selected)]
        if reference_segments.empty:
            continue
        low = int(reference_segments["start"].min())
        high = int(reference_segments["end"].max())
        if high <= low:
            continue
        positions = np.arange(low, high, dtype=np.int64)

        segment_layer, call_layer, span_mask = build_variant_segment_layers(
            read_ids,
            positions,
            reference_segments,
            calls[calls["read_id"].isin(selected)],
        )

        adata = ad.AnnData(
            X=np.zeros((len(read_ids), positions.size), dtype=np.float32),
            obs=reference_obs.reset_index(drop=True),
            var=pd.DataFrame(index=[str(position) for position in positions]),
        )
        adata.obs_names = read_ids
        adata.layers[f"{prefix}_variant_segments"] = segment_layer
        adata.layers[f"{prefix}_variant_call"] = call_layer
        adata.layers["read_span_mask"] = span_mask
        for column in (REFERENCE_STRAND, sample_column):
            adata.obs[column] = adata.obs[column].astype(str).astype("category")

        # Project onto the configured display coordinate system, exactly as the
        # HMM and spatial clustermaps do. Without this these panels plot raw
        # `var_names` while those plot reindexed coordinates, so two
        # position-dependent figures in one generation disagree about the axis
        # -- and under `reindexing_invert` they run in opposite directions.
        # Purely additive: a new var column, never a reordering of the data.
        index_suffix = str(getattr(cfg, "reindexed_var_suffix", None) or "") or None
        reindexing_offsets = getattr(cfg, "reindexing_offsets", None)
        reindexing_invert = getattr(cfg, "reindexing_invert", None)
        if index_suffix and (reindexing_offsets or reindexing_invert):
            reindex_references_adata(
                adata,
                reference_col=REFERENCE_STRAND,
                offsets=reindexing_offsets,
                new_col=index_suffix,
                invert=reindexing_invert,
            )
        if index_suffix and f"{reference}_{index_suffix}" not in adata.var:
            index_suffix = None

        rendered = plot_variant_segment_clustermaps(
            adata,
            seq1_column=seq1_column,
            seq2_column=seq2_column,
            sample_col=sample_column,
            reference_col=REFERENCE_STRAND,
            save_path=save_root,
            mismatch_type_obs_col=(
                "chimeric_variant_sites_type"
                if "chimeric_variant_sites_type" in adata.obs.columns
                else None
            ),
            marker_size=float(getattr(cfg, "variant_overlay_marker_size", 4.0)),
            show_position_axis=True,
            index_col_suffix=index_suffix,
            max_reads=max_reads,
            n_jobs=max(1, int(getattr(cfg, "threads", 1) or 1)),
        )
        results.extend(rendered)
        # Register into the stage plot catalog like every other category. A plot
        # that exists on disk but not in the catalog is invisible to anything
        # that discovers plots through it, which is most things.
        for record in rendered:
            path = record.get("output_path") if isinstance(record, dict) else None
            if not path:
                continue
            register_plot_artifact(
                plot_layout,
                path,
                stage="preprocess",
                category=category,
                plot_type="variant_segment_clustermap",
                reference=str(reference),
                sample=str(record.get("sample")) if record.get("sample") else None,
            )
        logger.info(
            "Variant segment clustermaps for %s: %d read(s) over %d position(s)",
            reference,
            adata.n_obs,
            adata.n_vars,
        )
    return results
