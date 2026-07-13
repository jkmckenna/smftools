"""Execute bounded preprocessing work units into derived partition stores."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
from urllib.parse import quote

import anndata as ad
import numpy as np
import pandas as pd

from smftools.constants import REFERENCE_STRAND
from smftools.logging_utils import get_logger

from ..informatics.partition_read import load_spine, materialize, relative_uns_path
from ..informatics.sidecar_manifest import register_sidecar, sidecar_manifest_path
from ..readwrite import safe_write_h5ad, safe_write_zarr
from .dispatch_plan import PreprocessTask, plan_preprocess_tasks, write_preprocess_task_catalog
from .partitioned_plots import generate_preprocess_summary_plots

logger = get_logger(__name__)

PREPROCESS_STORE_SUBDIR = "store"
PREPROCESS_SPINE_FILENAME = "spine.h5ad"
PREPROCESS_TASK_CATALOG = "task_catalog.parquet"
PREPROCESS_PARTITION_CATALOG = "catalog.parquet"
PREPROCESS_VAR_CATALOG = "var.parquet"
PREPROCESS_OBS_SIDECAR = "obs.parquet"
YOUDEN_FIT_SUBDIR = "02B_Position_wide_Youden_threshold_performance"

DERIVED_LAYER_ABSENT_FILL = {
    "nan0_0minus1": 0.0,
    "nan1_12": 1.0,
    "nan_minus_1": -1.0,
    "nan_half": 0.5,
}


def _component(value: str) -> str:
    return quote(str(value), safe="._-")


def _task_path(output_dir: Path, task: PreprocessTask) -> Path:
    return (
        output_dir
        / PREPROCESS_STORE_SUBDIR
        / f"reference={_component(task.reference)}"
        / f"core={task.core_start:012d}-{task.core_end:012d}"
        / f"barcode={_component(task.barcode)}"
        / f"chunk={task.chunk_index:05d}"
    )


def fit_direct_modality_youden_thresholds(
    spine_path: str | Path,
    cfg,
    references: Iterable[str],
    output_dir: str | Path,
) -> dict[str, pd.DataFrame]:
    """Fit per-position Youden methylation thresholds once per reference.

    Threshold fitting (``calculate_position_Youden``) needs the whole control-sample
    population covering a position at once -- it can't run per bounded task the way
    the rest of partitioned preprocessing does. This runs as a pre-pass before task
    dispatch, reusing the same per-reference windowing (``reference_windows``) that
    task planning uses: for locus references that's a single window spanning the
    whole reference (small, so no tiling needed); for genome-mode references it's
    the same haloed tiles the rest of partitioned preprocessing already bounds
    memory with, since a read only ever covers a small local span of a
    chromosome-scale reference. Each window's fit keeps only its own core range
    (the halo is covered by an adjacent window) before results from all windows
    for a reference are concatenated into one lookup table.

    Returns:
        Mapping of reference -> DataFrame indexed by genomic position (int) with
        columns ``threshold``, ``j_statistic``, ``passed_qc``.
    """
    from .calculate_position_Youden import calculate_position_Youden
    from .dispatch_plan import reference_windows

    ref_column = str(cfg.reference_column)
    roc_dir = Path(output_dir) / YOUDEN_FIT_SUBDIR
    roc_dir.mkdir(parents=True, exist_ok=True)
    # Load once and reuse across references/windows rather than re-reading
    # spine.h5ad repeatedly. materialize() only infers base_dir from a *path*,
    # and the raw spine deliberately doesn't carry its own source_base_dir
    # (derived spines inherit its uns wholesale and need to fall back to
    # their own on-disk location, not the original raw store's), so it must
    # be passed explicitly whenever an in-memory spine object is handed in.
    spine_path = Path(spine_path)
    spine = load_spine(spine_path, verbose=False)
    base_dir = Path(spine.uns.get("source_base_dir", spine_path.parent))

    positive_sample = cfg.positive_control_sample_methylation_fitting
    negative_sample = cfg.negative_control_sample_methylation_fitting
    # When explicit control samples are configured, restrict materialization
    # to just those samples -- narrower than calculate_position_Youden's own
    # post-hoc filtering, which matters once a window covers many multiplexed
    # samples. Percentile-based inference needs every read's signal in the
    # window to compute its cutoffs, so it can't take this shortcut.
    control_samples = (
        [positive_sample, negative_sample] if positive_sample and negative_sample else None
    )

    thresholds: dict[str, pd.DataFrame] = {}
    for reference in references:
        windows = reference_windows(spine, reference)
        reference_obs = spine.obs.loc[spine.obs[REFERENCE_STRAND].astype(str) == str(reference)]
        window_tables: list[pd.DataFrame] = []
        for window_index, (core_start, core_end, load_start, load_end) in enumerate(windows):
            # Skip windows whose core owns no reads at all -- same ownership
            # rule plan_preprocess_tasks uses, and avoids materialize()
            # raising on a genuinely empty selection for a sparsely-covered
            # genome-mode reference.
            overlapping = reference_obs.loc[
                (reference_obs["reference_start"].astype("int64") < core_end)
                & (reference_obs["reference_end"].astype("int64") > core_start)
            ]
            if overlapping.empty:
                continue
            ref_adata = materialize(
                spine,
                references=reference,
                samples=control_samples,
                base_dir=base_dir,
                start=load_start,
                end=load_end,
            )
            if ref_column in ref_adata.obs and hasattr(ref_adata.obs[ref_column], "cat"):
                ref_adata.obs[ref_column] = ref_adata.obs[ref_column].cat.remove_unused_categories()
            window_roc_dir = roc_dir if len(windows) == 1 else roc_dir / f"window_{window_index:05d}"
            if len(windows) != 1:
                window_roc_dir.mkdir(parents=True, exist_ok=True)
            calculate_position_Youden(
                ref_adata,
                positive_control_sample=positive_sample,
                negative_control_sample=negative_sample,
                J_threshold=cfg.fit_j_threshold,
                ref_column=ref_column,
                sample_column=cfg.sample_column,
                infer_on_percentile=cfg.infer_on_percentile_sample_methylation_fitting,
                inference_variable=cfg.inference_variable_sample_methylation_fitting,
                save=True,
                output_directory=window_roc_dir,
            )
            stats_col = f"{reference}_position_methylation_thresholding_Youden_stats"
            qc_col = f"{reference}_position_passed_Youden_thresholding_QC"
            stats = ref_adata.var[stats_col].to_numpy()
            table = pd.DataFrame(
                {
                    "threshold": [
                        t[0] if isinstance(t, (tuple, list)) else np.nan for t in stats
                    ],
                    "j_statistic": [
                        t[1] if isinstance(t, (tuple, list)) else np.nan for t in stats
                    ],
                    "passed_qc": ref_adata.var[qc_col].to_numpy(dtype=bool),
                },
                index=pd.Index(np.asarray(ref_adata.var_names, dtype=np.int64), name="position"),
            )
            # Keep only this window's core range: halo positions are owned by
            # (and already fit within) an adjacent window.
            core_mask = (table.index >= core_start) & (table.index < core_end)
            window_tables.append(table.loc[core_mask])
            del ref_adata
        thresholds[reference] = (
            pd.concat(window_tables).sort_index()
            if window_tables
            else pd.DataFrame(
                columns=["threshold", "j_statistic", "passed_qc"],
                index=pd.Index([], dtype=np.int64, name="position"),
            )
        )
    del spine
    return thresholds


def _apply_youden_thresholds(
    adata: ad.AnnData,
    cfg,
    reference: str,
    thresholds: pd.DataFrame,
    target_layer: str,
) -> None:
    """Inject one reference's fitted thresholds into a task's window, then binarize."""
    from .binarize_on_Youden import binarize_on_Youden

    positions = np.asarray(adata.var_names, dtype=np.int64)
    aligned = thresholds.reindex(positions)
    stats_col = f"{reference}_position_methylation_thresholding_Youden_stats"
    qc_col = f"{reference}_position_passed_Youden_thresholding_QC"
    adata.var[stats_col] = list(
        zip(aligned["threshold"].to_numpy(), aligned["j_statistic"].to_numpy())
    )
    adata.var[qc_col] = aligned["passed_qc"].fillna(False).to_numpy(dtype=bool)
    binarize_on_Youden(
        adata,
        ref_column=str(cfg.reference_column),
        output_layer_name=target_layer,
    )


def _run_local_transforms(
    adata: ad.AnnData,
    cfg,
    youden_thresholds: dict[str, pd.DataFrame] | None = None,
) -> tuple[list[str], list[str]]:
    """Apply transforms whose outputs are valid independently per read/window."""
    from .append_base_context import append_base_context
    from .binarize import binarize_adata
    from .clean_NaN import clean_NaN

    before_layers = set(adata.layers)
    modality = str(cfg.smf_modality)
    target_layer = str(getattr(cfg, "output_binary_layer_name", "binarized_methylation"))
    if modality == "direct":
        if getattr(cfg, "fit_position_methylation_thresholds", False):
            reference = str(adata.obs[REFERENCE_STRAND].astype(str).iloc[0])
            if not youden_thresholds or reference not in youden_thresholds:
                raise ValueError(
                    f"missing fitted Youden thresholds for reference {reference!r}; "
                    "fit_direct_modality_youden_thresholds must run before task dispatch"
                )
            _apply_youden_thresholds(
                adata, cfg, reference, youden_thresholds[reference], target_layer
            )
        else:
            binarize_adata(
                adata,
                source="X",
                target_layer=target_layer,
                threshold=float(cfg.binarize_on_fixed_methlyation_threshold),
            )
        nan_source = target_layer
    else:
        nan_source = None

    clean_NaN(
        adata,
        layer=nan_source,
        bypass=bool(cfg.bypass_clean_nan),
        force_redo=True,
        layers_to_build=getattr(cfg, "clean_nan_layers", None),
    )
    append_base_context(
        adata,
        ref_column=str(cfg.reference_column),
        use_consensus=False,
        native=modality == "direct",
        mod_target_bases=list(cfg.mod_target_bases),
        bypass=bool(cfg.bypass_append_base_context),
        force_redo=True,
    )

    derived_layers = sorted(set(adata.layers).difference(before_layers))
    derived_var = sorted(
        column
        for column in adata.var.columns
        if column.startswith(f"{adata.obs[REFERENCE_STRAND].astype(str).iloc[0]}_")
    )
    return derived_layers, derived_var


def _core_result(
    adata: ad.AnnData,
    task: PreprocessTask,
    derived_layers: Iterable[str],
    derived_var: Iterable[str],
    cfg,
) -> ad.AnnData:
    positions = np.asarray(adata.var_names, dtype=np.int64)
    core_mask = (positions >= task.core_start) & (positions < task.core_end)
    core = adata[:, core_mask]
    var = core.var[list(derived_var)].copy()
    modality = str(cfg.smf_modality)
    target_layer = str(getattr(cfg, "output_binary_layer_name", "binarized_methylation"))
    coverage_matrix = core.layers[target_layer] if modality == "direct" else core.X
    var["valid_count_partial"] = np.sum(~np.isnan(coverage_matrix), axis=0).astype(np.int64)
    var["n_reads_partial"] = core.n_obs
    obs = core.obs.copy()
    obs["Raw_modification_signal_partial"] = np.nansum(core.X, axis=1)
    reference = task.reference
    for column in derived_var:
        prefix = f"{reference}_"
        if not column.startswith(prefix) or not column.endswith("_site"):
            continue
        site_type = column.removeprefix(prefix)
        site_mask = core.var[column].to_numpy(dtype=bool)
        site_values = np.asarray(coverage_matrix)[:, site_mask]
        obs[f"Modified_{site_type}_count_partial"] = np.nansum(site_values, axis=1)
        obs[f"Total_{site_type}_in_read_partial"] = np.sum(~np.isnan(site_values), axis=1)
    result = ad.AnnData(
        obs=obs,
        var=var,
        layers={name: np.asarray(core.layers[name]) for name in derived_layers},
    )
    result.uns.update(
        {
            "task_id": task.task_id,
            "reference": task.reference,
            "core_start": task.core_start,
            "core_end": task.core_end,
            "load_start": task.load_start,
            "load_end": task.load_end,
        }
    )
    return result


def execute_preprocess_task(
    spine_path: str | Path,
    task: PreprocessTask,
    cfg,
    output_dir: str | Path,
    *,
    youden_thresholds: dict[str, pd.DataFrame] | None = None,
) -> dict[str, object]:
    """Materialize, transform, core-crop, and write one preprocessing task."""
    output_dir = Path(output_dir)
    adata = materialize(
        spine_path,
        references=task.reference,
        read_ids=task.read_ids,
        start=task.load_start,
        end=task.load_end,
    )
    if REFERENCE_STRAND in adata.obs and hasattr(adata.obs[REFERENCE_STRAND], "cat"):
        adata.obs[REFERENCE_STRAND] = adata.obs[REFERENCE_STRAND].cat.remove_unused_categories()
    derived_layers, derived_var = _run_local_transforms(adata, cfg, youden_thresholds)
    result = _core_result(adata, task, derived_layers, derived_var, cfg)
    path = _task_path(output_dir, task)
    chunks = (max(1, min(result.n_obs, 10_000)), max(1, result.n_vars))
    safe_write_zarr(result, path, backup=True, verbose=False, zarr_format=3, chunks=chunks)
    return {
        **task.to_dict(include_read_ids=False),
        "group_path": path.relative_to(output_dir).as_posix(),
        "n_positions": result.n_vars,
        "layers": derived_layers,
    }


def reduce_partial_coverage(
    catalog_path: str | Path,
    spine,
    output_path: str | Path,
    *,
    minimum_valid_fraction: float,
) -> Path:
    """Reduce task-local coverage counts into one row per genomic position."""
    from ..readwrite import safe_read_zarr

    if not 0 <= minimum_valid_fraction <= 1:
        raise ValueError("minimum_valid_fraction must be between zero and one")
    catalog_path = Path(catalog_path)
    output_path = Path(output_path)
    catalog = pd.read_parquet(catalog_path)
    plans = spine.uns.get("reference_plans", {})
    partials: list[pd.DataFrame] = []
    context_partials: list[pd.DataFrame] = []
    for record in catalog.to_dict("records"):
        group_path = catalog_path.parent / str(record["group_path"])
        result, _ = safe_read_zarr(group_path)
        partials.append(
            pd.DataFrame(
                {
                    "reference": str(record["reference"]),
                    "position": np.asarray(result.var_names, dtype=np.int64),
                    "valid_count_partial": result.var["valid_count_partial"].to_numpy(
                        dtype=np.int64
                    ),
                }
            )
        )
        context_columns = [
            column
            for column in result.var.columns
            if column not in {"valid_count_partial", "n_reads_partial"}
        ]
        if context_columns:
            context = result.var[context_columns].copy()
            context.insert(0, "position", np.asarray(result.var_names, dtype=np.int64))
            context.insert(0, "reference", str(record["reference"]))
            context_partials.append(context)
    columns = ["reference", "position", "valid_count", "valid_fraction", "position_valid"]
    if not partials:
        reduced = pd.DataFrame(columns=columns)
    else:
        reduced = (
            pd.concat(partials, ignore_index=True)
            .groupby(["reference", "position"], as_index=False, sort=True)["valid_count_partial"]
            .sum()
            .rename(columns={"valid_count_partial": "valid_count"})
        )
        denominators = {
            str(reference): int(dict(plan)["n_reads"]) for reference, plan in dict(plans).items()
        }
        reduced["valid_fraction"] = reduced["valid_count"] / reduced["reference"].map(denominators)
        reduced["position_valid"] = reduced["valid_fraction"] >= minimum_valid_fraction
        reduced = reduced[columns]
        if context_partials:
            context = pd.concat(context_partials, ignore_index=True).drop_duplicates(
                ["reference", "position"]
            )
            reduced = reduced.merge(
                context, on=["reference", "position"], how="left", validate="one_to_one"
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reduced.to_parquet(output_path, index=False)
    return output_path


def write_read_qc_sidecar(spine, cfg, output_path: str | Path) -> Path:
    """Compute read-level QC as an aligned mask without deleting spine rows."""
    from .filter_reads_on_cigar_indels import filter_reads_on_cigar_indels
    from .filter_reads_on_length_quality_mapping import (
        filter_reads_on_length_quality_mapping,
    )

    qc_input = ad.AnnData(obs=spine.obs.copy())
    filtered = filter_reads_on_length_quality_mapping(
        qc_input,
        filter_on_coordinates=False,
        read_length=cfg.read_len_filter_thresholds,
        mapped_length=cfg.mapped_len_filter_thresholds,
        length_ratio=cfg.read_len_to_ref_ratio_filter_thresholds,
        mapped_length_ratio=cfg.mapped_len_to_ref_ratio_filter_thresholds,
        mapped_to_read_ratio=cfg.mapped_len_to_read_len_ratio_filter_thresholds,
        read_quality=cfg.read_quality_filter_thresholds,
        mapping_quality=cfg.read_mapping_quality_filter_thresholds,
        bypass=bool(getattr(cfg, "bypass_filter_reads_on_length_quality_mapping", False)),
        force_redo=True,
    )
    filtered = filter_reads_on_cigar_indels(
        filtered,
        max_insertion_length=getattr(cfg, "max_internal_insertion_length", 10),
        max_deletion_length=getattr(cfg, "max_internal_deletion_length", 10),
        bypass=bool(getattr(cfg, "bypass_filter_reads_on_cigar_indels", False)),
        force_redo=True,
    )
    output = spine.obs.copy()
    if "mapped_length_to_reference_length_ratio" not in output and {
        "mapped_length",
        "reference_length",
    }.issubset(output.columns):
        mapped = pd.to_numeric(output["mapped_length"], errors="coerce")
        reference = pd.to_numeric(output["reference_length"], errors="coerce")
        output["mapped_length_to_reference_length_ratio"] = np.divide(
            mapped,
            reference,
            out=np.full(len(output), np.nan, dtype=float),
            where=reference.to_numpy() != 0,
        )
    output.insert(0, "read_id", output.index.astype(str))
    output["passes_read_qc"] = output["read_id"].isin(filtered.obs_names)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False)
    return output_path


def reduce_read_modification_stats(
    catalog_path: str | Path,
    var_path: str | Path,
    obs_path: str | Path,
) -> Path:
    """Reduce non-overlapping core partials into legacy-compatible read metrics."""
    from ..readwrite import safe_read_zarr

    catalog_path = Path(catalog_path)
    var_frame = pd.read_parquet(var_path)
    partials: list[pd.DataFrame] = []
    for record in pd.read_parquet(catalog_path).to_dict("records"):
        result, _ = safe_read_zarr(catalog_path.parent / str(record["group_path"]), verbose=False)
        columns = [column for column in result.obs if column.endswith("_partial")]
        partial = result.obs[columns].copy()
        partial.insert(0, "reference", str(record["reference"]))
        partial.insert(0, "read_id", partial.index.astype(str))
        partials.append(partial)
    obs = pd.read_parquet(obs_path)
    if not partials:
        return Path(obs_path)
    reduced = pd.concat(partials, ignore_index=True)
    partial_columns = [column for column in reduced if column.endswith("_partial")]
    reduced = reduced.groupby(["read_id", "reference"], as_index=False)[partial_columns].sum()
    reduced = reduced.rename(
        columns={
            "Raw_modification_signal_partial": "Raw_modification_signal",
            **{
                column: column.removesuffix("_partial")
                for column in partial_columns
                if column != "Raw_modification_signal_partial"
            },
        }
    )
    count_columns = [
        column for column in reduced if column.startswith("Total_") and column.endswith("_in_read")
    ]
    for count_column in count_columns:
        site_type = count_column.removeprefix("Total_").removesuffix("_in_read")
        modified_column = f"Modified_{site_type}_count"
        fraction_column = f"Fraction_{site_type}_modified"
        reduced[fraction_column] = np.divide(
            reduced[modified_column],
            reduced[count_column],
            out=np.full(len(reduced), np.nan, dtype=float),
            where=reduced[count_column].to_numpy() != 0,
        )
        context_column = reduced["reference"] + f"_{site_type}"
        totals = {
            (reference, column): int(
                var_frame.loc[var_frame["reference"] == reference, column]
                .astype("boolean")
                .fillna(False)
                .sum()
            )
            for reference, column in zip(reduced["reference"], context_column)
            if column in var_frame
        }
        total_reference_column = f"Total_{site_type}_in_reference"
        reduced[total_reference_column] = [
            totals.get((reference, column), 0)
            for reference, column in zip(reduced["reference"], context_column)
        ]
        reduced[f"Valid_{site_type}_in_read_vs_reference"] = np.divide(
            reduced[count_column],
            reduced[total_reference_column],
            out=np.full(len(reduced), np.nan, dtype=float),
            where=reduced[total_reference_column].to_numpy() != 0,
        )
    for foreground in ("GpC", "CpG"):
        foreground_column = f"Fraction_{foreground}_site_modified"
        background_column = "Fraction_other_C_site_modified"
        if foreground_column in reduced and background_column in reduced:
            reduced[f"{foreground}_to_other_C_mod_ratio"] = np.divide(
                reduced[foreground_column],
                reduced[background_column],
                out=np.full(len(reduced), np.nan, dtype=float),
                where=reduced[background_column].to_numpy() != 0,
            )
    obs = obs.merge(reduced.drop(columns="reference"), on="read_id", how="left")
    obs.to_parquet(obs_path, index=False)
    return Path(obs_path)


def append_modification_qc_mask(obs_path: str | Path, cfg) -> Path:
    """Apply legacy modification thresholds to reduced metrics as a mask."""
    from .filter_reads_on_modification_thresholds import (
        filter_reads_on_modification_thresholds,
    )

    obs_path = Path(obs_path)
    obs = pd.read_parquet(obs_path).set_index("read_id", drop=False)
    input_adata = ad.AnnData(obs=obs.drop(columns="read_id"))
    filtered = filter_reads_on_modification_thresholds(
        input_adata,
        smf_modality=str(cfg.smf_modality),
        mod_target_bases=list(cfg.mod_target_bases),
        gpc_thresholds=cfg.read_mod_filtering_gpc_thresholds,
        cpg_thresholds=cfg.read_mod_filtering_cpg_thresholds,
        any_c_thresholds=cfg.read_mod_filtering_c_thresholds,
        a_thresholds=cfg.read_mod_filtering_a_thresholds,
        use_other_c_as_background=bool(cfg.read_mod_filtering_use_other_c_as_background),
        min_valid_fraction_positions_in_read_vs_ref=(
            cfg.min_valid_fraction_positions_in_read_vs_ref
        ),
        bypass=bool(cfg.bypass_filter_reads_on_modification_thresholds),
        force_redo=True,
        reference_column=str(cfg.reference_column),
        compute_obs_if_missing=False,
    )
    obs["passes_modification_qc"] = obs.index.isin(filtered.obs_names)
    obs["passes_qc"] = obs["passes_read_qc"] & obs["passes_modification_qc"]
    obs.reset_index(drop=True).to_parquet(obs_path, index=False)
    return obs_path


def append_deaminase_chimera_label(obs_path: str | Path, cfg) -> Path:
    """Label deaminase PCR chimeras on the obs sidecar from strand-switch metrics."""
    if str(getattr(cfg, "smf_modality", "conversion")) != "deaminase":
        return Path(obs_path)

    from .label_deaminase_pcr_chimeras import label_deaminase_pcr_chimeras

    obs_path = Path(obs_path)
    obs = pd.read_parquet(obs_path).set_index("read_id", drop=False)
    input_adata = ad.AnnData(obs=obs.drop(columns="read_id"))
    labeled = label_deaminase_pcr_chimeras(
        input_adata,
        min_events_per_span=cfg.deaminase_chimera_min_events_per_span,
        min_segment_purity=cfg.deaminase_chimera_min_segment_purity,
        max_single_strand_fraction=cfg.deaminase_chimera_max_single_strand_fraction,
        bypass=bool(getattr(cfg, "bypass_label_deaminase_pcr_chimeras", False)),
    )
    if "deaminase_PCR_chimera" not in labeled.obs:  # bypassed
        return obs_path
    obs["deaminase_PCR_chimera"] = labeled.obs["deaminase_PCR_chimera"].to_numpy()
    obs.reset_index(drop=True).to_parquet(obs_path, index=False)
    return obs_path


def reduce_duplicate_reads(
    preprocess_spine_path: str | Path,
    obs_path: str | Path,
    cfg,
) -> Path:
    """Run bounded duplicate comparisons and reconcile clusters across cores."""
    from .flag_duplicate_reads import UnionFind, _process_group

    obs_path = Path(obs_path)
    obs = pd.read_parquet(obs_path).set_index("read_id", drop=False)
    if bool(cfg.bypass_flag_duplicate_reads):
        obs["is_duplicate"] = False
        obs["duplicate_cluster_id"] = -1
        obs["duplicate_cluster_size"] = 1
        obs["passes_dedup"] = obs["passes_qc"]
        obs.reset_index(drop=True).to_parquet(obs_path, index=False)
        return obs_path

    spine = load_spine(preprocess_spine_path)
    read_ids = list(map(str, obs.index))
    read_position = {read_id: index for index, read_id in enumerate(read_ids)}
    union_find = UnionFind(len(read_ids))
    hamming_columns = (
        "fwd_hamming_to_next",
        "rev_hamming_to_prev",
        "sequence__hier_hamming_to_pair",
        "sequence__min_hamming_to_pair",
    )
    hamming_minima = {
        column: np.full(len(read_ids), np.nan, dtype=float) for column in hamming_columns
    }
    plans = dict(spine.uns.get("reference_plans", {}))
    reference_lengths = dict(spine.uns.get("reference_lengths", {}))
    sample_column = str(getattr(cfg, "sample_name_col_for_plotting", "Sample"))
    if sample_column not in obs:
        sample_column = "Sample" if "Sample" in obs else "Barcode"
    max_reads = int(getattr(cfg, "duplicate_detection_max_reads_per_window", 50_000))

    for reference, raw_plan in sorted(plans.items()):
        plan = dict(raw_plan)
        reference_length = int(reference_lengths.get(reference, plan["reference_length"]))
        tile_size = (
            reference_length if plan.get("analysis_mode") == "locus" else int(plan["tile_size"])
        )
        reference_obs = obs.loc[
            (obs[REFERENCE_STRAND].astype(str) == str(reference)) & obs["passes_qc"].astype(bool)
        ]
        for sample, sample_obs in reference_obs.groupby(sample_column, sort=True, observed=True):
            for core_start in range(0, reference_length, tile_size):
                core_end = min(core_start + tile_size, reference_length)
                core_obs = sample_obs.loc[
                    (sample_obs["reference_start"].astype("int64") < core_end)
                    & (sample_obs["reference_end"].astype("int64") > core_start)
                ]
                if len(core_obs) < 2:
                    continue
                if len(core_obs) > max_reads:
                    raise MemoryError(
                        f"duplicate window {reference}:{core_start}-{core_end} sample "
                        f"{sample!r} has {len(core_obs)} reads, exceeding "
                        f"duplicate_detection_max_reads_per_window={max_reads}"
                    )
                load_start = max(0, int(core_obs["reference_start"].min()))
                load_end = min(reference_length, int(core_obs["reference_end"].max()))
                window = materialize(
                    preprocess_spine_path,
                    references=reference,
                    read_ids=core_obs.index,
                    start=load_start,
                    end=load_end,
                    layers=[],
                )
                context_mask = np.zeros(window.n_vars, dtype=bool)
                for site_type in cfg.duplicate_detection_site_types:
                    column = f"{reference}_{site_type}_site"
                    if column in window.var:
                        context_mask |= (
                            window.var[column].astype("boolean").fillna(False).to_numpy(dtype=bool)
                        )
                valid_column = f"position_in_{reference}"
                if valid_column in window.var:
                    context_mask &= (
                        window.var[valid_column]
                        .astype("boolean")
                        .fillna(False)
                        .to_numpy(dtype=bool)
                    )
                if not context_mask.any():
                    continue
                local_obs = obs.loc[window.obs_names]
                result = _process_group(
                    {
                        "X_sub": np.asarray(window.X)[:, context_mask].astype(float),
                        "obs_df": local_obs,
                        "obs_index": list(map(str, window.obs_names)),
                        "sample": sample,
                        "ref": reference,
                        "distance_threshold": float(cfg.duplicate_detection_distance_threshold),
                        "window_size": int(
                            cfg.duplicate_detection_window_size_for_hamming_neighbors
                        ),
                        "min_overlap_positions": int(
                            cfg.duplicate_detection_min_overlapping_positions
                        ),
                        "keep_best_metric": cfg.duplicate_detection_keep_best_metric,
                        "keep_best_higher": True,
                        "do_hierarchical": bool(cfg.duplicate_detection_do_hierarchical),
                        "hierarchical_linkage": cfg.duplicate_detection_hierarchical_linkage,
                        "hierarchical_metric": "euclidean",
                        "hierarchical_window": int(
                            cfg.duplicate_detection_window_size_for_hamming_neighbors
                        ),
                        "do_pca": bool(cfg.duplicate_detection_do_pca),
                        "pca_n_components": 50,
                        "pca_center": True,
                        "random_state": 0,
                        "demux_col": "demux_type",
                        "demux_types": list(cfg.duplicate_detection_demux_types_to_use),
                    }
                )
                if result is None:
                    continue
                local_ids = list(map(str, result["obs_index"]))
                cluster_ids = np.asarray(result["sequence__merged_cluster_id"])
                cluster_sizes = np.asarray(result["sequence__cluster_size"])
                for cluster_id in np.unique(cluster_ids[cluster_sizes > 1]):
                    members = np.where(cluster_ids == cluster_id)[0]
                    anchor = read_position[local_ids[int(members[0])]]
                    for member in members[1:]:
                        union_find.union(anchor, read_position[local_ids[int(member)]])
                for column in hamming_columns:
                    local_values = np.asarray(result[column], dtype=float)
                    global_values = hamming_minima[column]
                    for local_index, read_id in enumerate(local_ids):
                        value = local_values[local_index]
                        global_index = read_position[read_id]
                        if not np.isnan(value) and (
                            np.isnan(global_values[global_index])
                            or value < global_values[global_index]
                        ):
                            global_values[global_index] = value

    clusters: dict[int, list[str]] = {}
    for read_id, index in read_position.items():
        if not bool(obs.at[read_id, "passes_qc"]):
            continue
        clusters.setdefault(union_find.find(index), []).append(read_id)
    duplicate = pd.Series(False, index=obs.index)
    cluster_id_series = pd.Series(-1, index=obs.index, dtype="int64")
    cluster_size_series = pd.Series(1, index=obs.index, dtype="int64")
    preferred_demux = set(map(str, cfg.duplicate_detection_demux_types_to_use))
    metric = str(cfg.duplicate_detection_keep_best_metric)
    ordered_clusters = sorted(clusters.values(), key=lambda members: min(members))
    for cluster_id, members in enumerate(ordered_clusters):
        cluster_id_series.loc[members] = cluster_id
        cluster_size_series.loc[members] = len(members)
        if len(members) == 1:
            continue
        candidates = members
        if "demux_type" in obs and preferred_demux:
            preferred = [
                member for member in members if str(obs.at[member, "demux_type"]) in preferred_demux
            ]
            if preferred:
                candidates = preferred
        if metric in obs:
            values = pd.to_numeric(obs.loc[candidates, metric], errors="coerce")
            keeper = values.idxmax() if values.notna().any() else min(candidates)
        else:
            keeper = min(candidates)
        duplicate.loc[[member for member in members if member != keeper]] = True

    for column, values in hamming_minima.items():
        obs[column] = values
    obs["duplicate_cluster_id"] = cluster_id_series
    obs["duplicate_cluster_size"] = cluster_size_series
    obs["is_duplicate"] = duplicate
    obs["is_duplicate_reason"] = np.where(duplicate, "sequence_cluster", "")
    obs["passes_dedup"] = obs["passes_qc"].astype(bool) & ~duplicate
    obs.reset_index(drop=True).to_parquet(obs_path, index=False)
    return obs_path


def _generate_preprocess_summary_plots_legacy(obs_path, var_path, plot_layout) -> None:
    """Generate bounded summary figures from reduced preprocessing sidecars."""
    import matplotlib.pyplot as plt

    from ..cli.stage_artifacts import register_plot_artifact

    obs = pd.read_parquet(obs_path)
    var = pd.read_parquet(var_path)

    if "read_length" in obs:
        figure, axis = plt.subplots(figsize=(7, 4))
        for label, mask, color in (
            ("pass", obs["passes_read_qc"].astype(bool), "#197278"),
            ("fail", ~obs["passes_read_qc"].astype(bool), "#d95d39"),
        ):
            values = pd.to_numeric(obs.loc[mask, "read_length"], errors="coerce").dropna()
            if not values.empty:
                axis.hist(values, bins=50, alpha=0.65, label=label, color=color)
        axis.set(xlabel="Read length", ylabel="Reads", title="Read-length QC")
        axis.legend(frameon=False)
        figure.tight_layout()
        path = plot_layout.categories["read_qc"] / "read_length_qc.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        register_plot_artifact(
            plot_layout,
            path,
            stage="preprocess",
            category="read_qc",
            plot_type="read_length_histogram",
        )

    fraction_columns = [
        column for column in obs if column.startswith("Fraction_") and column.endswith("_modified")
    ]
    if fraction_columns:
        figure, axis = plt.subplots(figsize=(7, 4))
        for column in fraction_columns:
            values = pd.to_numeric(obs[column], errors="coerce").dropna()
            if not values.empty:
                axis.hist(values, bins=40, histtype="step", linewidth=1.5, label=column)
        axis.set(xlabel="Modified fraction", ylabel="Reads", title="Modification QC")
        axis.legend(frameon=False, fontsize=7)
        figure.tight_layout()
        path = plot_layout.categories["modification_qc"] / "modification_fractions.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        register_plot_artifact(
            plot_layout,
            path,
            stage="preprocess",
            category="modification_qc",
            plot_type="modification_fraction_histograms",
        )

    if "duplicate_cluster_size" in obs:
        sizes = pd.to_numeric(obs["duplicate_cluster_size"], errors="coerce").dropna()
        sizes = sizes.loc[sizes >= 1]
        if not sizes.empty:
            figure, axis = plt.subplots(figsize=(7, 4))
            axis.hist(sizes, bins=min(50, max(1, int(sizes.max()))), color="#6c5b7b")
            axis.set(
                xlabel="Duplicate cluster size",
                ylabel="Reads",
                title="Duplicate-cluster distribution",
                yscale="log",
            )
            figure.tight_layout()
            path = plot_layout.categories["duplicate_qc"] / "duplicate_cluster_sizes.png"
            figure.savefig(path, dpi=150)
            plt.close(figure)
            register_plot_artifact(
                plot_layout,
                path,
                stage="preprocess",
                category="duplicate_qc",
                plot_type="duplicate_cluster_size_histogram",
            )

    for reference, reference_var in var.groupby("reference", sort=True, observed=True):
        figure, axis = plt.subplots(figsize=(8, 3.5))
        axis.plot(
            reference_var["position"],
            reference_var["valid_fraction"],
            color="#31572c",
            linewidth=0.8,
        )
        axis.set(
            xlabel="Reference position",
            ylabel="Valid read fraction",
            title=f"Coverage: {reference}",
            ylim=(0, 1.02),
        )
        figure.tight_layout()
        path = plot_layout.categories["coverage"] / f"{_component(reference)}_coverage.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        register_plot_artifact(
            plot_layout,
            path,
            stage="preprocess",
            category="coverage",
            plot_type="valid_fraction_by_position",
            reference=str(reference),
        )


def execute_partitioned_preprocessing(
    spine_path: str | Path,
    cfg,
    output_dir: str | Path,
    *,
    tasks: Iterable[PreprocessTask] | None = None,
) -> dict[str, Path]:
    """Execute planned tasks sequentially and publish catalogs plus a derived spine."""
    spine_path = Path(spine_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    from ..cli.stage_artifacts import prepare_analysis_plot_layout

    plot_layout = prepare_analysis_plot_layout(
        output_dir,
        stage="preprocess",
        source_spine=spine_path,
    )
    spine = load_spine(spine_path)
    task_list = (
        list(tasks)
        if tasks is not None
        else plan_preprocess_tasks(
            spine,
            target_task_memory_mb=int(getattr(cfg, "target_task_memory_mb", 512)),
        )
    )
    task_catalog = write_preprocess_task_catalog(task_list, output_dir / PREPROCESS_TASK_CATALOG)
    obs_sidecar = write_read_qc_sidecar(spine, cfg, output_dir / PREPROCESS_OBS_SIDECAR)

    youden_thresholds: dict[str, pd.DataFrame] | None = None
    if str(cfg.smf_modality) == "direct" and getattr(
        cfg, "fit_position_methylation_thresholds", False
    ):
        fit_references = sorted({task.reference for task in task_list})
        logger.info(
            "Fitting Youden position thresholds for %d reference(s) before task dispatch",
            len(fit_references),
        )
        youden_thresholds = fit_direct_modality_youden_thresholds(
            spine_path, cfg, fit_references, output_dir
        )

    records = [
        execute_preprocess_task(
            spine_path, task, cfg, output_dir, youden_thresholds=youden_thresholds
        )
        for task in task_list
    ]
    catalog_path = output_dir / PREPROCESS_PARTITION_CATALOG
    pd.DataFrame(records).to_parquet(catalog_path, index=False)
    var_catalog = reduce_partial_coverage(
        catalog_path,
        spine,
        output_dir / PREPROCESS_VAR_CATALOG,
        minimum_valid_fraction=1 - float(cfg.position_max_nan_threshold),
    )
    obs_sidecar = reduce_read_modification_stats(catalog_path, var_catalog, obs_sidecar)
    obs_sidecar = append_modification_qc_mask(obs_sidecar, cfg)
    obs_sidecar = append_deaminase_chimera_label(obs_sidecar, cfg)

    derived_spine = spine.copy()
    # Stored relative to the run's output_directory (not output_dir itself), so
    # these pointers stay correct even after being copied unchanged into a later
    # stage's spine (spatial/hmm), which lives in a sibling directory -- see
    # informatics.partition_read._run_root_from_spine_path.
    run_root = output_dir.parent
    derived_spine.uns["preprocess_store"] = relative_uns_path(
        output_dir / PREPROCESS_STORE_SUBDIR, run_root
    )
    derived_spine.uns["preprocess_catalog"] = relative_uns_path(catalog_path, run_root)
    derived_spine.uns["preprocess_source_spine"] = relative_uns_path(spine_path, run_root)
    derived_spine.uns["source_base_dir"] = relative_uns_path(spine_path.parent, run_root)
    derived_spine.uns["preprocess_var"] = relative_uns_path(var_catalog, run_root)
    derived_spine.uns["preprocess_obs"] = relative_uns_path(obs_sidecar, run_root)
    published_layers = sorted({layer for record in records for layer in record.get("layers", [])})
    derived_spine.uns["preprocess_layer_absent_fill"] = {
        layer: float(DERIVED_LAYER_ABSENT_FILL.get(layer, np.nan)) for layer in published_layers
    }
    output_spine = output_dir / PREPROCESS_SPINE_FILENAME
    # Duplicate reduction reads bounded raw + derived slices through this spine.
    safe_write_h5ad(derived_spine, output_spine, backup=False, verbose=False)
    obs_sidecar = reduce_duplicate_reads(output_spine, obs_sidecar, cfg)
    derived_obs = pd.read_parquet(obs_sidecar).set_index("read_id")
    for column in derived_obs.columns:
        if column not in derived_spine.obs:
            derived_spine.obs[column] = derived_obs[column].reindex(derived_spine.obs_names)
    safe_write_h5ad(derived_spine, output_spine, backup=False, verbose=False)
    if bool(getattr(cfg, "emit_automated_plots", True)):
        generate_preprocess_summary_plots(
            obs_sidecar,
            var_catalog,
            plot_layout,
            cfg=cfg,
            spine_path=output_spine,
        )

    manifest = sidecar_manifest_path(output_dir)
    register_sidecar(manifest, "preprocess_store", output_dir / PREPROCESS_STORE_SUBDIR)
    register_sidecar(manifest, "preprocess_catalog", catalog_path)
    register_sidecar(manifest, "preprocess_task_catalog", task_catalog)
    register_sidecar(manifest, "preprocess_var", var_catalog)
    register_sidecar(manifest, "preprocess_obs", obs_sidecar)
    register_sidecar(manifest, "preprocess_spine", output_spine)
    register_sidecar(manifest, "preprocess_plots", plot_layout.root)
    register_sidecar(manifest, "preprocess_plot_catalog", plot_layout.catalog)
    logger.info("Wrote %d partitioned preprocessing task result(s)", len(records))
    return {
        "store": output_dir / PREPROCESS_STORE_SUBDIR,
        "spine": output_spine,
        "catalog": catalog_path,
        "task_catalog": task_catalog,
        "var": var_catalog,
        "obs": obs_sidecar,
        "plots": plot_layout.root,
        "plot_catalog": plot_layout.catalog,
        "manifest": manifest,
    }
