"""Latent-ordered clustermaps of projected layers (`EGL-28c`).

Three views of the same molecules, side by side in one figure, with rows in the
order the latent clustering determined: raw accessibility, footprint-feature
lengths, and accessible-feature lengths. Sharing one row order across the three
panels is the point -- three separate figures cannot be read against each other
molecule by molecule, which is why this departs from the reference analysis's
directory-per-layer output.

**The layers live in two different stages.** The latent store carries the
embeddings and cluster labels but only `nan_half` and
`sequence_integer_encoding`; the feature-length layers are published by the HMM
stage, prefixed by the modality's target base (`C_` for deaminase, `GpC_` for
conversion), so the prefix is resolved rather than hard-coded. The raw
accessibility layer comes from `materialize`, since the HMM store's `X` is
empty and it keeps layers only.

The two stages also partition differently -- the HMM stage splits by barcode
(32 tasks on the DAF pilot) where latent splits by reference (2) -- so the
length layers are gathered across HMM shards and joined by read id. Checked on
both pilots: every latent molecule is present in the HMM generation, so the
join is total rather than a silent inner-join that would drop rows without
saying so. `missing_length_rows` in the returned record reports it if that ever
stops being true.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from smftools.logging_utils import get_logger

logger = get_logger(__name__)

PLOT_TYPE = "latent_ordered_clustermap"
COMPOSITION_PLOT_TYPE = "leiden_composition"
RAW_PANEL = "raw accessibility"


def resolve_feature_prefix(cfg) -> str:
    """The HMM layer prefix for this modality's target base.

    Deaminase publishes `C_*`, conversion `GpC_*`. Derived from
    `mod_target_bases` rather than the modality name so a config that targets
    something else still resolves, and falls back to the modality only when the
    bases are absent.
    """
    bases = [str(base) for base in (getattr(cfg, "mod_target_bases", None) or [])]
    for candidate in ("GpC", "C", "A"):
        if candidate in bases:
            return f"{candidate}_"
    modality = str(getattr(cfg, "smf_modality", "") or "").strip().lower()
    return "GpC_" if modality == "conversion" else "C_"


def length_layer_names(prefix: str) -> tuple[str, str]:
    """The two feature-length layers this figure projects back."""
    return (
        f"{prefix}all_footprint_features_lengths",
        f"{prefix}all_accessible_features_lengths",
    )


def _generation_path(store_root: Path) -> Path | None:
    pointer = Path(store_root) / "current.json"
    if not pointer.is_file():
        return None
    return pointer.parent / json.loads(pointer.read_text())["generation_path"]


def resolve_hmm_generation(spine, run_root) -> Path | None:
    """The HMM generation *this* latent lineage was built from.

    Read from the spine's `hmm_catalog` pointer rather than the store's
    `current.json`: current can have advanced since this latent generation was
    published, and pairing these labels with a different HMM generation would
    silently draw feature lengths for a different analysis beside them. The
    whole point of the generation pointers is that a lineage stays internally
    consistent.
    """
    from smftools.informatics.partition_read import resolve_relative_path

    pointer = getattr(spine, "uns", {}).get("hmm_catalog")
    if not pointer:
        return None
    path = resolve_relative_path(pointer, run_root)
    if path is None or not Path(path).exists():
        logger.warning(
            "Spine claims an HMM catalog at %r but it does not resolve; the length "
            "panels will be omitted.",
            pointer,
        )
        return None
    return Path(path).parent


def gather_hmm_length_layers(
    hmm_generation: Path,
    *,
    reference: str,
    read_ids: list[str],
    layer_names: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """Collect per-molecule length layers across the HMM stage's shards.

    Rows are returned in ``read_ids`` order, with molecules absent from the HMM
    generation left as NaN rather than dropped -- dropping would silently
    change which molecules the figure shows relative to the panels beside it.
    """
    import anndata as ad

    catalog_path = Path(hmm_generation) / "task_catalog.parquet"
    if not catalog_path.is_file():
        return {}
    catalog = pd.read_parquet(catalog_path)
    catalog = catalog[catalog["reference"].astype(str) == str(reference)]
    if catalog.empty:
        return {}

    wanted = {read_id: row for row, read_id in enumerate(read_ids)}
    grids: dict[str, np.ndarray] = {}
    for _, task in catalog.iterrows():
        group = Path(hmm_generation) / str(task["group_path"])
        try:
            shard = ad.read_zarr(group)
        except Exception as exc:
            logger.warning("Could not read HMM shard %s: %s", group, exc)
            continue
        shard_names = [str(name) for name in shard.obs_names]
        target_rows = np.array(
            [wanted[name] for name in shard_names if name in wanted], dtype=np.int64
        )
        if target_rows.size == 0:
            continue
        source_rows = np.array(
            [index for index, name in enumerate(shard_names) if name in wanted], dtype=np.int64
        )
        for name in layer_names:
            if name not in shard.layers:
                continue
            if name not in grids:
                grids[name] = np.full((len(read_ids), shard.n_vars), np.nan, dtype=np.float32)
            values = np.asarray(shard.layers[name], dtype=np.float32)
            grids[name][target_rows] = values[source_rows]
    return grids


def _panel_specs(raw_panel, grids, layer_names, positions):
    """Assemble the drawable panels, skipping any layer the stage never wrote."""
    raw, raw_positions = raw_panel
    panels = [
        {
            "name": f"{RAW_PANEL} (mod sites, n={raw.shape[1]})",
            "matrix": raw,
            "cmap": "coolwarm",
            "positions": raw_positions,
        }
    ]
    for name in layer_names:
        values = grids.get(name)
        if values is None:
            logger.info("Layer %s absent from the HMM generation; omitting its panel.", name)
            continue
        finite = values[np.isfinite(values)]
        # Length layers are heavy-tailed -- a single 881 bp feature would flatten
        # everything else against a max-scaled colourbar, so cap at a high
        # percentile and let the outliers saturate.
        vmax = float(np.percentile(finite, 99)) if finite.size else None
        panels.append(
            {
                "name": name.split("_", 1)[-1].replace("_", " "),
                "matrix": values,
                "cmap": "viridis",
                "vmin": 0.0,
                "vmax": vmax,
                "positions": positions,
            }
        )
    return panels


def resolve_composition_groups(obs, cfg) -> list[str]:
    """Which obs columns to break composition down by.

    Configurable rather than fixed because there is no biorep column in the
    store today -- neither the obs nor the config has any notion of one, so
    hard-coding `biorep` would produce a plot that never renders. The default
    is the plotting sample column; adding a biorep column to obs (or naming an
    existing one here) makes the per-biorep breakdown appear with no code
    change.

    Columns with a single value across the unit are dropped: a stacked bar
    chart with one bar is the cluster strip again, with more ink.
    """
    requested = [
        str(name) for name in (getattr(cfg, "latent_composition_group_columns", None) or [])
    ]
    if not requested:
        fallback = str(getattr(cfg, "sample_name_col_for_plotting", "Barcode"))
        requested = [fallback]
    usable = []
    for column in requested:
        if column not in obs.columns:
            logger.info("Composition column %s absent from obs; skipping it.", column)
            continue
        if obs[column].astype(str).nunique() < 2:
            logger.info(
                "Composition column %s has a single value in this unit; skipping it.", column
            )
            continue
        usable.append(column)
    return usable


def render_unit_composition(
    unit,
    *,
    reference: str,
    start: int,
    end: int,
    plot_layout,
    cfg,
    category: str = "clusters",
    model_id: str = "",
    model_checksum: str = "",
) -> list[dict]:
    """Stacked cluster-composition barplots per group, per embedding."""
    from smftools.cli.stage_artifacts import register_plot_artifact
    from smftools.plotting import cluster_color_map, plot_leiden_composition

    from .latent_clustering import embedding_keys, parse_embedding_key

    selected = [str(name) for name in (getattr(cfg, "latent_clustermap_strategies", None) or [])]
    group_columns = resolve_composition_groups(unit.obs, cfg)
    if not group_columns:
        logger.info("No usable composition grouping for %s; skipping barplots.", reference)
        return []
    min_group_size = int(getattr(cfg, "latent_composition_min_group_size", 1))
    save_root = Path(plot_layout.categories[category])

    results: list[dict] = []
    for key in embedding_keys(unit):
        strategy, suffix = parse_embedding_key(key)
        if selected and strategy not in selected:
            continue
        label_key = f"leiden_{strategy}_{suffix}"
        if label_key not in unit.obs:
            continue
        labels = unit.obs[label_key].astype(str).to_numpy()
        # One colour map per embedding, shared across its groupings *and* with
        # the clustermap, so cluster 3 is the same colour everywhere it appears.
        colors = cluster_color_map(labels)
        for column in group_columns:
            filename = f"{reference}__{start}-{end}__{key}__by_{column}.png".replace("/", "_")
            rendered = plot_leiden_composition(
                labels,
                unit.obs[column].astype(str).to_numpy(),
                group_name=column,
                title=f"{reference} {start}-{end} \u2014 {key} by {column}",
                save_path=save_root / filename,
                min_group_size=min_group_size,
                color_map=colors,
            )
            if rendered is None:
                logger.info(
                    "No %s group reached %d molecule(s) for %s; no barplot drawn.",
                    column,
                    min_group_size,
                    reference,
                )
                continue
            rendered.update({"reference": reference, "embedding": key, "group_column": column})
            results.append(rendered)
            register_plot_artifact(
                plot_layout,
                rendered["output_path"],
                stage="latent",
                category=category,
                plot_type=COMPOSITION_PLOT_TYPE,
                reference=reference,
                core_start=start,
                core_end=end,
                model_id=model_id,
                model_checksum=model_checksum,
            )
            logger.info(
                "Composition barplot for %s %s by %s: %d group(s), %d cluster(s)",
                reference,
                key,
                column,
                rendered["n_groups"],
                rendered["n_clusters"],
            )
    return results


def render_unit_clustermaps(
    unit,
    *,
    reference: str,
    start: int,
    end: int,
    raw,
    positions,
    grids: dict[str, np.ndarray],
    layer_names: tuple[str, ...],
    plot_layout,
    cfg,
    category: str = "clusters",
    model_id: str = "",
    model_checksum: str = "",
) -> list[dict]:
    """Render every selected embedding of one already-clustered unit.

    Takes the unit in memory rather than a path: the latent stage holds it
    right after clustering, so re-reading it from disk would both cost a read
    and risk clustering a *different* population than the one published, since
    the labels depend on which molecules were in the fit subset.
    """
    from smftools.cli.stage_artifacts import register_plot_artifact
    from smftools.plotting import plot_latent_ordered_clustermap

    from .latent_clustering import embedding_keys, parse_embedding_key
    from .latent_ordering import latent_row_order

    selected = [str(name) for name in (getattr(cfg, "latent_clustermap_strategies", None) or [])]
    missing_length_rows = {
        name: int(np.isnan(values).all(axis=1).sum()) for name, values in grids.items()
    }
    for name, count in missing_length_rows.items():
        if count:
            # Not fatal -- the row is drawn blank rather than dropped, so it
            # stays aligned with the other panels -- but a silent hole in one
            # panel of three reads as biology, so say it.
            logger.warning(
                "%d molecule(s) have no %s in the HMM generation for %s; "
                "their rows are blank in that panel.",
                count,
                name,
                reference,
            )
    panels = _panel_specs(raw, grids, layer_names, positions)
    save_root = Path(plot_layout.categories[category])

    results: list[dict] = []
    for key in embedding_keys(unit):
        strategy, suffix = parse_embedding_key(key)
        if selected and strategy not in selected:
            continue
        label_key = f"leiden_{strategy}_{suffix}"
        if label_key not in unit.obs:
            # Generations published before `EGL-28a` carry `leiden_<suffix>`
            # only. Skipping is right -- borrowing that column would reinstate
            # exactly the shared-Leiden coupling `28a` removed -- but say so,
            # because an empty plot directory otherwise looks like a bug.
            logger.info(
                "Unit %s has no %s (generation predates per-embedding clustering); "
                "skipping that embedding.",
                reference,
                label_key,
            )
            continue
        labels = unit.obs[label_key].astype(str).to_numpy()
        row_order, blocks = latent_row_order(np.asarray(unit.obsm[key], dtype=float), labels)
        source_key = f"{label_key}_label_source"
        label_source = (
            unit.obs[source_key].astype(str).to_numpy() if source_key in unit.obs else None
        )
        filename = f"{reference}__{start}-{end}__{key}.png".replace("/", "_")
        rendered = plot_latent_ordered_clustermap(
            panels,
            row_order=row_order,
            blocks=blocks,
            labels=labels,
            label_source=label_source,
            position_labels=positions,
            title=f"{reference} {start}-{end} \u2014 {key} ({len(blocks)} clusters)",
            save_path=save_root / filename,
        )
        if rendered is None:
            continue
        rendered.update(
            {
                "reference": reference,
                "embedding": key,
                "missing_length_rows": missing_length_rows,
            }
        )
        results.append(rendered)
        register_plot_artifact(
            plot_layout,
            rendered["output_path"],
            stage="latent",
            category=category,
            plot_type=PLOT_TYPE,
            reference=reference,
            core_start=start,
            core_end=end,
            model_id=model_id,
            model_checksum=model_checksum,
        )
        logger.info(
            "Latent clustermap for %s %s: %d molecule(s), %d cluster(s), %d panel(s)",
            reference,
            key,
            rendered["n_molecules"],
            rendered["n_clusters"],
            len(panels),
        )
    return results


def mod_site_mask(var_frame, reference: str, mod_target_bases) -> np.ndarray | None:
    """Columns where this reference has a modification site.

    Accessibility is only *defined* at target-base positions; the rest of the
    layer is a constant fill. Drawing all 4,690 positions of a locus renders
    the informative columns sub-pixel and the panel reads as vertical noise,
    which is what the first version of this figure did. Restricting to the
    sites is what makes the raw panel comparable to the two beside it.
    """
    bases = [str(base) for base in (mod_target_bases or [])]
    columns = [f"{reference}_{base}_site" for base in bases]
    available = [column for column in columns if column in var_frame.columns]
    if not available:
        return None
    mask = np.logical_or.reduce(
        [np.asarray(var_frame[column].values, dtype=bool) for column in available]
    )
    membership = f"position_in_{reference}"
    if membership in var_frame.columns:
        mask &= np.asarray(var_frame[membership].values, dtype=bool)
    return mask if mask.any() else None


def display_positions(var_frame, reference: str, cfg, fallback) -> np.ndarray:
    """Position labels in the configured display coordinate system.

    Without this the axis shows raw `var_names` while the HMM, spatial and
    segment panels show reindexed coordinates, so two position-dependent
    figures in one run disagree about the axis -- and under `reindexing_invert`
    they run in opposite directions (`EGL-23`).
    """
    suffix = str(getattr(cfg, "reindexed_var_suffix", None) or "") or None
    if suffix:
        column = f"{reference}_{suffix}"
        if column in var_frame.columns:
            return np.asarray(var_frame[column].values)
    return np.asarray(fallback)


def load_unit_panels(
    *,
    reference: str,
    start: int,
    end: int,
    read_ids: list[str],
    spine_path,
    hmm_generation,
    cfg,
) -> tuple[np.ndarray, object, dict[str, np.ndarray], tuple[str, ...]] | None:
    """Fetch the raw layer and the HMM length layers for one unit's molecules."""
    from smftools.informatics.partition_read import materialize
    from smftools.preprocessing.reindex_references_adata import reindex_references_adata

    raw_layer = str(getattr(cfg, "layer_for_clustermap_plotting", "nan0_0minus1"))
    layer_names = length_layer_names(resolve_feature_prefix(cfg))
    try:
        slice_ = materialize(
            spine_path,
            references=reference,
            read_ids=read_ids,
            start=start,
            end=end,
            layers=[raw_layer],
        )
    except Exception:
        logger.exception("Could not materialize %s for %s; skipping unit", raw_layer, reference)
        return None
    # Project onto the display coordinate system before anything reads the
    # positions, so every consumer below sees the same axis.
    suffix = str(getattr(cfg, "reindexed_var_suffix", None) or "") or None
    offsets = getattr(cfg, "reindexing_offsets", None)
    invert = getattr(cfg, "reindexing_invert", None)
    if suffix and (offsets or invert):
        try:
            reindex_references_adata(
                slice_,
                reference_col="Reference_strand",
                offsets=offsets,
                new_col=suffix,
                invert=invert,
            )
        except Exception:
            logger.exception(
                "Reindexing failed for %s; falling back to stored coordinates", reference
            )

    order = {str(name): index for index, name in enumerate(slice_.obs_names)}
    present = [index for index, name in enumerate(read_ids) if name in order]
    source_rows = np.array([order[read_ids[index]] for index in present], dtype=np.int64)
    raw = np.full((len(read_ids), slice_.n_vars), np.nan, dtype=np.float32)
    if present:
        raw[np.asarray(present, dtype=np.int64)] = np.asarray(
            slice_.layers[raw_layer], dtype=np.float32
        )[source_rows]
    positions = display_positions(
        slice_.var, reference, cfg, np.asarray(slice_.var_names, dtype=np.int64)
    )
    site_mask = mod_site_mask(slice_.var, reference, getattr(cfg, "mod_target_bases", None))
    if site_mask is None:
        logger.warning(
            "No modification-site columns for %s; drawing the raw panel over every "
            "position, where accessibility is undefined between sites.",
            reference,
        )
    else:
        raw = raw[:, site_mask]
        raw_positions = positions[site_mask]
    if site_mask is None:
        raw_positions = positions
    grids = (
        gather_hmm_length_layers(
            hmm_generation, reference=reference, read_ids=read_ids, layer_names=layer_names
        )
        if hmm_generation is not None
        else {}
    )
    # The raw panel is restricted to sites while the length layers span every
    # position, so each panel carries its own axis. Both are labelled in the
    # same coordinate system, so they stay comparable by value.
    return (raw, raw_positions), positions, grids, layer_names


def generate_latent_clustermaps(
    latent_generation,
    plot_layout,
    *,
    cfg,
    spine_path=None,
    hmm_store_root=None,
    category: str = "clusters",
) -> list[dict]:
    """Render latent-ordered clustermaps for a published generation on disk."""
    import anndata as ad

    latent_generation = Path(latent_generation)
    catalog_path = latent_generation / "task_catalog.parquet"
    if not catalog_path.is_file():
        logger.info("No latent task catalog; skipping latent clustermaps.")
        return []
    if spine_path is None:
        logger.info("No spine available; skipping latent clustermaps.")
        return []
    catalog = pd.read_parquet(catalog_path)
    hmm_generation = _generation_path(Path(hmm_store_root)) if hmm_store_root else None
    if hmm_generation is None:
        logger.warning(
            "No HMM generation available; latent clustermaps will show the raw layer only."
        )

    results: list[dict] = []
    for _, task in catalog.iterrows():
        reference = str(task["reference"])
        start, end = int(task["core_start"]), int(task["core_end"])
        try:
            unit = ad.read_zarr(latent_generation / str(task["group_path"]))
        except Exception as exc:
            logger.warning("Could not read latent unit %s: %s", task["group_path"], exc)
            continue
        read_ids = [str(name) for name in unit.obs_names]
        loaded = load_unit_panels(
            reference=reference,
            start=start,
            end=end,
            read_ids=read_ids,
            spine_path=spine_path,
            hmm_generation=hmm_generation,
            cfg=cfg,
        )
        if loaded is None:
            continue
        raw, positions, grids, layer_names = loaded
        results.extend(
            render_unit_clustermaps(
                unit,
                reference=reference,
                start=start,
                end=end,
                raw=raw,
                positions=positions,
                grids=grids,
                layer_names=layer_names,
                plot_layout=plot_layout,
                cfg=cfg,
                category=category,
            )
        )
    return results
