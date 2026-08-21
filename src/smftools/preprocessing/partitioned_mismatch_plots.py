"""Per-read mismatch clustermaps for the partitioned preprocess stage.

The pre-partition `variant` CLI rendered these (`cli/variant_adata.py`, into
``03_mismatch_integer_encoding_clustermaps_no_mod_sites``); the partitioned
pipeline never did. The renderer already exists and is reused unchanged --
only the caller was missing, the same situation `EGL-17` found for variant
segment clustermaps.

Unlike `EGL-17` and `EGL-21`, nothing here needs rasterizing. Those lanes had
to rebuild dense grids because variant and deamination evidence is stored
sparsely by design. Sequence and mismatch are different: they live in the raw
ragged store as per-read arrays, and ``materialize`` already scatters them onto
a reference grid through the shared CIGAR path, emitting
``sequence_integer_encoding`` and ``mismatch_integer_encoding`` directly. So
this module selects reads and a window, materializes, and plots.

**Excluding modification sites is not cosmetic.** In a conversion or deaminase
experiment every converted C *is* a mismatch against the reference, so without
the exclusion these panels show chemistry rather than sequence error -- which
is the opposite of what they are for. The renderer builds the mask from
``<reference>_<base>_site`` var columns AND ``position_in_<reference>``; both
are attached by ``materialize``'s preprocess-var overlay, so the mask is only
as good as that overlay (`F12`). A reference whose site columns are missing
yields no mask and is skipped rather than plotted misleadingly.
"""

from __future__ import annotations

from pathlib import Path

from smftools.constants import DEFAULT_CLUSTERMAP_MAX_READS_PER_PLOT
from smftools.logging_utils import get_logger

logger = get_logger(__name__)

PLOT_TYPE = "mismatch_integer_encoding_clustermap"
DEFAULT_DEMUX_TYPES = ("single", "double", "already")


def _mod_site_columns(reference: str, mod_site_bases) -> list[str]:
    """Var columns the renderer's mod-site mask requires for one reference.

    Only the named bases are required. The renderer additionally folds in
    ``<reference>_ambiguous_GpC_CpG_site`` when it is present, but treats it as
    optional, so requiring it here would skip references the renderer can
    handle.
    """
    return [f"{reference}_{base}_site" for base in mod_site_bases]


def generate_mismatch_clustermaps(
    spine_path,
    plot_layout,
    task_catalog,
    read_index,
    *,
    cfg=None,
    category: str = "mismatch_clustermaps",
) -> list[dict]:
    """Render per-read mismatch clustermaps for every reference in a generation."""
    from smftools.cli.stage_artifacts import register_plot_artifact, write_plot_source_manifest
    from smftools.constants import (
        MISMATCH_INTEGER_ENCODING,
        MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT,
        READ_SPAN_MASK,
        REFERENCE_STRAND,
        SEQUENCE_INTEGER_ENCODING,
    )
    from smftools.informatics.partition_read import load_spine, materialize
    from smftools.informatics.plot_region_stitching import (
        mask_unanalyzed_gaps,
        resolve_plot_region_plans,
        select_plot_reads,
    )
    from smftools.plotting import plot_sequence_integer_encoding_clustermaps

    from .reindex_references_adata import reindex_references_adata

    mod_site_bases = list(getattr(cfg, "mod_target_bases", []) or [])
    if not mod_site_bases:
        logger.info("No mod_target_bases configured; skipping mismatch clustermaps.")
        return []

    spine = load_spine(spine_path, verbose=False)
    plans = resolve_plot_region_plans(
        spine,
        task_catalog,
        spine_path=spine_path,
        allow_gaps=bool(getattr(cfg, "plot_allow_unanalyzed_gaps", False)),
    )
    if not plans:
        logger.info("No plot regions resolved; skipping mismatch clustermaps.")
        return []

    # The analysed population: dedup-passing where dedup has run, else
    # QC-passing. Decided 2026-08-20 -- matches the variant segment clustermaps
    # and the `deduplicated/` output these replace, so every per-read panel in a
    # generation shows the same molecules and can be read against its neighbours.
    filter_column = next(
        (column for column in ("passes_dedup", "passes_qc") if column in spine.obs), None
    )
    eligible_read_ids = (
        spine.obs.index[spine.obs[filter_column].astype(bool)] if filter_column else None
    )

    max_reads = int(
        getattr(cfg, "clustermap_max_reads_per_plot", DEFAULT_CLUSTERMAP_MAX_READS_PER_PLOT)
        or DEFAULT_CLUSTERMAP_MAX_READS_PER_PLOT
    )
    seed = int(getattr(cfg, "plot_subsample_seed", 0))
    save_root = Path(plot_layout.categories[category])

    results: list[dict] = []
    for plan in plans:
        selection = select_plot_reads(
            read_index,
            plan,
            max_reads_per_barcode=max_reads,
            seed=seed,
            eligible_read_ids=eligible_read_ids,
        )
        read_ids = list(selection.read_ids)
        if not read_ids:
            continue

        adata = materialize(
            spine_path,
            references=plan.reference,
            read_ids=read_ids,
            start=plan.start,
            end=plan.end,
            layers=[SEQUENCE_INTEGER_ENCODING, MISMATCH_INTEGER_ENCODING, READ_SPAN_MASK],
        )
        # A region can be stitched from tasks that do not tile it completely;
        # unanalysed columns must read as absent rather than as agreement with
        # the reference, which is what an unmasked gap would look like here.
        mask_unanalyzed_gaps(adata, plan.gaps)
        if adata.n_obs == 0 or adata.n_vars == 0:
            continue

        reference = str(plan.reference)
        required = _mod_site_columns(reference, mod_site_bases)
        missing = [column for column in required if column not in adata.var.columns]
        if missing:
            # Without the mask every converted base reads as a mismatch, so the
            # panel would show chemistry, not sequence error. Skipping is the
            # honest outcome; rendering it anyway is worse than no plot.
            logger.warning(
                "Reference %s lacks mod-site var columns %s; skipping its mismatch "
                "clustermaps rather than plotting conversion chemistry as mismatch.",
                reference,
                missing,
            )
            continue

        # The renderer decodes integers to bases through these maps. The ragged
        # store encodes both layers with the same table, so both point at it.
        adata.uns["sequence_integer_encoding_map"] = dict(MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT)
        adata.uns["mismatch_integer_encoding_map"] = dict(MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT)

        sample_column = str(getattr(cfg, "sample_name_col_for_plotting", "Sample"))
        if sample_column not in adata.obs.columns:
            sample_column = "Sample" if "Sample" in adata.obs.columns else "Barcode"
        for column in (REFERENCE_STRAND, sample_column):
            adata.obs[column] = adata.obs[column].astype(str).astype("category")

        # Project onto the configured display coordinate system so this panel
        # agrees with the HMM, spatial and segment panels beside it. `EGL-23`
        # had to fix exactly this omission, and under `reindexing_invert` a
        # panel that skips it runs backwards relative to its neighbours.
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

        rendered = plot_sequence_integer_encoding_clustermaps(
            adata,
            sample_col=sample_column,
            reference_col=REFERENCE_STRAND,
            layer=SEQUENCE_INTEGER_ENCODING,
            mismatch_layer=MISMATCH_INTEGER_ENCODING,
            exclude_mod_sites=True,
            mod_site_bases=mod_site_bases,
            # The population is already gated above; re-filtering here on read
            # metrics would silently drop reads the other panels keep.
            min_quality=None,
            min_length=None,
            min_mapped_length_to_reference_length_ratio=None,
            # Falling back to an empty sequence here would filter out every
            # read rather than none of them -- the renderer keeps reads whose
            # `demux_type` is *in* this set.
            demux_types=tuple(
                getattr(cfg, "clustermap_demux_types_to_plot", None) or DEFAULT_DEMUX_TYPES
            ),
            sort_by="none",
            max_unknown_fraction=0.5,
            save_path=save_root,
            show_position_axis=True,
            max_reads=max_reads,
            index_col_suffix=index_suffix,
            # The renderer names files after the plotted layer by default,
            # which would put `..._sequence_integer_encoding.png` into a
            # `mismatch_clustermaps` category. The old CLI disambiguated the
            # two variants by directory; here the category is the directory,
            # so the name has to carry it (same fix as `EGL-21`).
            filename_suffix="mismatch_no_mod_sites",
            n_jobs=max(1, int(getattr(cfg, "threads", 1) or 1)),
        )
        results.extend(rendered)

        for record in rendered:
            path = record.get("output_path") if isinstance(record, dict) else None
            if not path:
                continue
            source_manifest = write_plot_source_manifest(
                plot_layout,
                path,
                stage="preprocess",
                plot_type=PLOT_TYPE,
                region=plan.source_manifest(),
                layers=[SEQUENCE_INTEGER_ENCODING, MISMATCH_INTEGER_ENCODING],
                selection_seed=selection.seed,
                selection_sha256=selection.selection_sha256,
                selected_molecule_uids=selection.molecule_uids,
            )
            # Register into the stage catalog like every other category. `EGL-17`
            # shipped 30 plots that contributed zero catalog rows and were
            # invisible to anything discovering plots through it.
            register_plot_artifact(
                plot_layout,
                path,
                stage="preprocess",
                category=category,
                plot_type=PLOT_TYPE,
                reference=reference,
                sample=str(record.get("sample")) if record.get("sample") else None,
                core_start=plan.start,
                core_end=plan.end,
                source_manifest=source_manifest,
            )
        logger.info(
            "Mismatch clustermaps for %s: %d read(s) over %d position(s)",
            reference,
            adata.n_obs,
            adata.n_vars,
        )
    return results
