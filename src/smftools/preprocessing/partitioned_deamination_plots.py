"""Deamination segment clustermaps for the partitioned pipeline (`EGL-21`).

The `EGL-17` pattern applied to strand chemistry: rasterize the sparse
deamination evidence onto a read x position grid and hand it to the existing
variant-segment renderer, rather than writing a second one.

Reuse works because the state encodings coincide. The renderer draws
``0`` no coverage, ``1`` first member, ``2`` second member, ``3`` transition;
deamination supplies ``1`` top-strand chemistry, ``2`` bottom-strand, and a
synthesized ``3`` across the gap between segments of differing strand -- which
is both semantically right (the switch happened somewhere in there) and what
makes the renderer draw the breakpoint markers.

What the panel shows: each row is a molecule, coloured by which strand's
chemistry the change-point model assigned along its length. A pure molecule is
one colour end to end; a PCR chimera shows a block of each with a breakpoint
between. The overlay marks individual deamination events, so the density of
evidence behind each segment is visible rather than implied.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from smftools.constants import REFERENCE_STRAND
from smftools.logging_utils import get_logger

logger = get_logger(__name__)

NO_COVERAGE_STATE = 0
TOP_STATE = 1
BOTTOM_STATE = 2
TRANSITION_STATE = 3
_STRAND_STATE = {"top": TOP_STATE, "bottom": BOTTOM_STATE}


def _read_shards(deamination_dir: Path, filename: str) -> pd.DataFrame:
    pattern = str(deamination_dir / "task_store" / "**" / filename)
    shards = sorted(glob.glob(pattern, recursive=True))
    if not shards:
        return pd.DataFrame()
    frame = pd.concat([pd.read_parquet(shard) for shard in shards], ignore_index=True)
    if "read_id" in frame.columns:
        frame["read_id"] = frame["read_id"].astype(str)
    return frame


def build_deamination_layers(
    read_ids: list[str],
    positions: np.ndarray,
    segments: pd.DataFrame,
    events: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rasterize segments and events onto a (read x position) grid.

    Returns the segment layer, an event overlay encoded like the renderer's
    call layer, and a span mask. Transitions are synthesized between adjacent
    segments of differing strand; that gap is genuinely unassigned -- the model
    locates the switch between two supported segments, not at a base -- so
    filling it with the transition state is more honest than extending either
    neighbour to meet the other.
    """
    row_of = {read_id: row for row, read_id in enumerate(read_ids)}
    n_obs, n_vars = len(read_ids), positions.size
    segment_layer = np.zeros((n_obs, n_vars), dtype=np.int8)
    event_layer = np.zeros((n_obs, n_vars), dtype=np.int8)

    if not segments.empty:
        ordered = segments.sort_values(["read_id", "start"], kind="stable")
        previous_read: str | None = None
        previous_end: int | None = None
        previous_state: int | None = None
        for row in ordered.itertuples(index=False):
            read_row = row_of.get(str(row.read_id))
            if read_row is None:
                previous_read = None
                continue
            state = _STRAND_STATE.get(str(row.strand), NO_COVERAGE_STATE)
            low = int(np.searchsorted(positions, int(row.start)))
            high = int(np.searchsorted(positions, int(row.end), side="right"))
            if high > low:
                segment_layer[read_row, low:high] = np.int8(state)
            if (
                previous_read == str(row.read_id)
                and previous_end is not None
                and previous_state is not None
                and previous_state != state
            ):
                gap_low = int(np.searchsorted(positions, previous_end, side="right"))
                if low > gap_low:
                    segment_layer[read_row, gap_low:low] = np.int8(TRANSITION_STATE)
            previous_read, previous_end, previous_state = str(row.read_id), int(row.end), state

    if not events.empty and "converted" in events.columns:
        fired = events[events["converted"].fillna(False).astype(bool)]
        if not fired.empty:
            rows = fired["read_id"].map(row_of)
            keep = rows.notna()
            if bool(keep.any()):
                columns = np.searchsorted(positions, fired.loc[keep, "position"].to_numpy(np.int64))
                inside = (columns >= 0) & (columns < n_vars)
                states = fired.loc[keep, "strand"].map(_STRAND_STATE).fillna(0).to_numpy(np.int8)
                event_layer[rows[keep].to_numpy(dtype=np.int64)[inside], columns[inside]] = states[
                    inside
                ]

    span_mask = (segment_layer != NO_COVERAGE_STATE).astype(np.int8)
    return segment_layer, event_layer, span_mask


def generate_deamination_segment_plots(
    deamination_dir,
    obs_path,
    plot_layout,
    *,
    cfg=None,
    category: str = "deamination_segments",
) -> list[dict]:
    """Render one deamination segment clustermap per reference and sample."""
    import anndata as ad

    from smftools.cli.stage_artifacts import register_plot_artifact
    from smftools.plotting import plot_variant_segment_clustermaps

    from .reindex_references_adata import reindex_references_adata

    deamination_dir = Path(deamination_dir)
    segments = _read_shards(deamination_dir, "segments.parquet")
    if segments.empty:
        logger.info("No deamination segments recorded; skipping segment clustermaps.")
        return []
    events = _read_shards(deamination_dir, "events.parquet")

    obs = pd.read_parquet(obs_path)
    obs["read_id"] = obs["read_id"].astype(str)
    if REFERENCE_STRAND not in obs.columns:
        logger.warning("obs lacks %s; skipping deamination segment clustermaps.", REFERENCE_STRAND)
        return []

    sample_column = str(getattr(cfg, "sample_name_col_for_plotting", "Sample"))
    if sample_column not in obs.columns:
        sample_column = "Sample" if "Sample" in obs.columns else "Barcode"

    # The analysed population, matching every other per-read panel in a
    # generation (`EGL-17`, `EGL-26`) so rows can be compared across them.
    for gate in ("passes_dedup", "passes_qc"):
        if gate in obs.columns:
            obs = obs[obs[gate].fillna(False).astype(bool)]
            break
    obs = obs[obs["read_id"].isin(set(segments["read_id"]))]
    if obs.empty:
        logger.info("No analysed reads carry deamination segments; skipping.")
        return []

    max_reads = int(getattr(cfg, "clustermap_max_reads_per_plot", 5000) or 5000)
    seed = int(getattr(cfg, "plot_subsample_seed", 0))

    results: list[dict] = []
    save_root = Path(plot_layout.categories[category])
    for reference, reference_obs in obs.groupby(REFERENCE_STRAND, observed=True, sort=True):
        # Seeded random rather than first-N: reproducible without being a
        # biased slice of the population (`EGL-27`).
        if max_reads > 0 and len(reference_obs) > max_reads:
            reference_obs = reference_obs.sample(n=max_reads, random_state=seed).sort_values(
                "read_id", kind="stable"
            )
        read_ids = list(reference_obs["read_id"])
        selected = set(read_ids)
        reference_segments = segments[segments["read_id"].isin(selected)]
        if reference_segments.empty:
            continue
        low = int(reference_segments["start"].min())
        high = int(reference_segments["end"].max())
        if high <= low:
            continue
        positions = np.arange(low, high + 1, dtype=np.int64)

        segment_layer, event_layer, span_mask = build_deamination_layers(
            read_ids,
            positions,
            reference_segments,
            events[events["read_id"].isin(selected)] if not events.empty else events,
        )

        adata = ad.AnnData(
            X=np.zeros((len(read_ids), positions.size), dtype=np.float32),
            obs=reference_obs.reset_index(drop=True),
            var=pd.DataFrame(index=[str(position) for position in positions]),
        )
        adata.obs_names = read_ids
        prefix = "top__bottom"
        adata.layers[f"{prefix}_variant_segments"] = segment_layer
        adata.layers[f"{prefix}_variant_call"] = event_layer
        adata.layers["read_span_mask"] = span_mask
        for column in (REFERENCE_STRAND, sample_column):
            adata.obs[column] = adata.obs[column].astype(str).astype("category")

        index_suffix = str(getattr(cfg, "reindexed_var_suffix", None) or "") or None
        offsets = getattr(cfg, "reindexing_offsets", None)
        invert = getattr(cfg, "reindexing_invert", None)
        if index_suffix and (offsets or invert):
            reindex_references_adata(
                adata,
                reference_col=REFERENCE_STRAND,
                offsets=offsets,
                new_col=index_suffix,
                invert=invert,
            )
        if index_suffix and f"{reference}_{index_suffix}" not in adata.var:
            index_suffix = None

        rendered = plot_variant_segment_clustermaps(
            adata,
            seq1_column="top",
            seq2_column="bottom",
            sample_col=sample_column,
            reference_col=REFERENCE_STRAND,
            save_path=save_root,
            mismatch_type_obs_col=(
                "deamination_strands_present"
                if "deamination_strands_present" in adata.obs.columns
                else None
            ),
            mismatch_type_legend_prefix="Deamination",
            marker_size=float(getattr(cfg, "variant_overlay_marker_size", 4.0)),
            show_position_axis=True,
            index_col_suffix=index_suffix,
            filename_suffix="deamination_segments",
            max_reads=max_reads,
            n_jobs=max(1, int(getattr(cfg, "threads", 1) or 1)),
        )
        results.extend(rendered)
        for record in rendered:
            path = record.get("output_path") if isinstance(record, dict) else None
            if not path:
                continue
            register_plot_artifact(
                plot_layout,
                path,
                stage="preprocess",
                category=category,
                plot_type="deamination_segment_clustermap",
                reference=str(reference),
                sample=str(record.get("sample")) if record.get("sample") else None,
            )
        logger.info(
            "Deamination segment clustermaps for %s: %d read(s) over %d position(s)",
            reference,
            adata.n_obs,
            adata.n_vars,
        )
    return results
