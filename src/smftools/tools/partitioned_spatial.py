"""Bounded spatial analysis over partitioned preprocessing outputs."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd

from smftools.constants import REFERENCE_STRAND
from smftools.logging_utils import get_logger

from ..cli.stage_artifacts import (
    PLOT_CATALOG_COLUMNS,
    prepare_analysis_plot_layout,
    register_plot_artifact,
    write_plot_source_manifest,
)
from ..informatics.analysis_region_plan import has_analysis_catalog
from ..informatics.experiment_spine import write_experiment_spine
from ..informatics.partition_read import load_spine, materialize, relative_uns_path
from ..informatics.plot_region_stitching import (
    mask_unanalyzed_gaps,
    resolve_plot_region_plans,
    select_plot_reads,
)
from ..informatics.sidecar_manifest import register_sidecar, sidecar_manifest_path
from ..preprocessing.dispatch_plan import (
    BYTES_PER_WORKING_POSITION,
    PreprocessTask,
    plan_preprocess_tasks,
)
from ..readwrite import safe_read_zarr, safe_write_h5ad, safe_write_zarr
from .spatial_autocorrelation import (
    binary_autocorrelation_with_spacing,
    weighted_mean_autocorr,
)

logger = get_logger(__name__)

SPATIAL_SPINE_FILENAME = "spine.h5ad"
SPATIAL_TASK_CATALOG = "task_catalog.parquet"
SPATIAL_METRICS = "metrics.parquet"
SPATIAL_AUTOCORRELATION = "autocorrelation.parquet"
SPATIAL_PARTIAL_SUBDIR = "store"
SPATIAL_REGION_CATALOG = "regions.parquet"
SPATIAL_MATRIX_SUBDIR = "matrices"
SPATIAL_READ_AUTOCORRELATION_AXIS = "read_autocorrelation_lags.parquet"
SPATIAL_READ_PERIODOGRAM_AXIS = "read_periodogram_axis.parquet"
SPATIAL_REGION_CATALOG_DTYPES = {
    "reference": "string",
    "start": "int64",
    "end": "int64",
    "name": "string",
    "source": "string",
}


def _component(value: object) -> str:
    return quote(str(value), safe="._-")


def _task_directory(output_dir: Path, task) -> Path:
    return (
        output_dir
        / SPATIAL_PARTIAL_SUBDIR
        / f"reference={_component(task.reference)}"
        / f"core={task.core_start:012d}-{task.core_end:012d}"
        / f"barcode={_component(task.barcode)}"
        / f"chunk={task.chunk_index:05d}"
    )


def _config_range(cfg, name: str, default: tuple[float, float]) -> tuple[float, float]:
    values = list(getattr(cfg, name, default))
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    lower, upper = map(float, values)
    if lower <= 0 or upper <= lower:
        raise ValueError(f"{name} must be a positive increasing range")
    return lower, upper


def _compute_read_spatial_statistics(
    values: np.ndarray,
    site_positions: np.ndarray,
    cfg,
) -> dict[str, object]:
    """Compute read-level ACF and direct-signal Lomb-Scargle results."""
    from ..analysis.compute.ls_periodicity import analyze_ls_periodicity_direct

    max_lag = int(getattr(cfg, "autocorr_max_lag", 800))
    normalization = str(getattr(cfg, "autocorr_normalization_method", "pearson"))
    compute_periodogram = bool(getattr(cfg, "spatial_compute_read_lomb_scargle", True))
    period_range = _config_range(cfg, "spatial_lomb_scargle_period_range_bp", (80.0, 400.0))
    peak_range = _config_range(cfg, "spatial_lomb_scargle_peak_range_bp", (150.0, 250.0))
    poly_degree = int(getattr(cfg, "spatial_lomb_scargle_poly_degree", 2))
    min_sites = int(getattr(cfg, "spatial_lomb_scargle_min_sites", 40))
    n_sites = np.sum(np.isfinite(values), axis=1).astype(np.int64)
    periods = np.arange(period_range[1], period_range[0] - 1, -1, dtype=float)
    periodogram_power = np.full((values.shape[0], len(periods)), np.nan, dtype=np.float32)
    # Preallocated to match periodogram_power's pattern -- binary_autocorrelation_with_spacing
    # always returns a fixed-length (max_lag + 1) pair, so growing Python lists of
    # per-read arrays (then np.asarray()-ing them at the end) only added per-object
    # overhead and a second, momentarily-doubled copy at conversion time.
    autocorrelation_matrix = np.full((values.shape[0], max_lag + 1), np.nan, dtype=np.float32)
    pair_counts_matrix = np.zeros((values.shape[0], max_lag + 1), dtype=np.int64)
    scalar_names = (
        "ls_nrl_bp",
        "ls_snr",
        "ls_peak_power",
        "ls_peak_power_raw",
        "ls_fwhm_bp",
    )
    scalars = {name: np.full(values.shape[0], np.nan, dtype=float) for name in scalar_names}
    status = np.full(values.shape[0], "not_requested", dtype=object)

    for read_index, row in enumerate(values):
        autocorrelation, counts = binary_autocorrelation_with_spacing(
            row,
            site_positions,
            max_lag=max_lag,
            normalize=normalization,
            return_counts=True,
        )
        autocorrelation_matrix[read_index] = autocorrelation
        pair_counts_matrix[read_index] = counts
        if compute_periodogram:
            result = analyze_ls_periodicity_direct(
                site_positions,
                row,
                nrl_search_bp=peak_range,
                period_range_bp=period_range,
                poly_degree=poly_degree,
                min_sites=min_sites,
            )
            status[read_index] = "ok" if result is not None else "insufficient_sites_or_signal"
            if result is not None:
                for name in scalar_names:
                    scalars[name][read_index] = float(result[name])
                periodogram_power[read_index] = np.asarray(result["ls_power"], dtype=np.float32)
    return {
        "autocorrelation": autocorrelation_matrix,
        "pair_counts": pair_counts_matrix,
        "n_sites": n_sites,
        "status": status,
        "periodogram_power": periodogram_power,
        "periods": periods,
        "frequencies": 1.0 / periods,
        **scalars,
    }


def _site_column(adata, reference: str, site_type: str) -> str | None:
    normalized = str(site_type).removesuffix("_site")
    candidates = (f"{reference}_{normalized}_site", f"{normalized}_site")
    return next((column for column in candidates if column in adata.var), None)


def _native_reference(reference: str) -> str:
    for suffix in ("_top", "_bottom"):
        if reference.endswith(suffix):
            return reference.removesuffix(suffix)
    return reference


def _read_spatial_regions(spine, bed_path: str | Path | None) -> pd.DataFrame:
    """Resolve BED intervals to exact reference-strand coordinates."""
    columns = ["reference", "start", "end", "name", "bed_chrom"]
    if bed_path is None:
        return pd.DataFrame(columns=columns)
    bed_path = Path(bed_path)
    if not bed_path.exists():
        raise FileNotFoundError(f"spatial_regions_bed not found: {bed_path}")
    references = sorted(map(str, spine.uns.get("reference_plans", {}).keys()))
    lengths = {str(key): int(value) for key, value in spine.uns["reference_lengths"].items()}
    records = []
    with bed_path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith(("#", "track", "browser")):
                continue
            fields = line.split("\t")
            if len(fields) < 3:
                raise ValueError(f"BED line {line_number} has fewer than three columns")
            chrom = fields[0]
            try:
                start, end = int(fields[1]), int(fields[2])
            except ValueError as exc:
                raise ValueError(f"BED line {line_number} has non-integer coordinates") from exc
            if start < 0 or end <= start:
                raise ValueError(f"BED line {line_number} is not a valid half-open interval")
            matched = [
                reference
                for reference in references
                if reference == chrom or _native_reference(reference) == chrom
            ]
            for reference in matched:
                bounded_start = min(start, lengths[reference])
                bounded_end = min(end, lengths[reference])
                if bounded_end <= bounded_start:
                    continue
                records.append(
                    {
                        "reference": reference,
                        "start": bounded_start,
                        "end": bounded_end,
                        "name": fields[3] if len(fields) > 3 else f"region_{line_number}",
                        "bed_chrom": chrom,
                    }
                )
    regions = pd.DataFrame(records, columns=columns).drop_duplicates(["reference", "start", "end"])
    if regions.empty:
        raise ValueError("spatial_regions_bed contains no intervals matching stored references")
    return regions.sort_values(["reference", "start", "end"]).reset_index(drop=True)


def _region_tasks(
    spine,
    cfg,
    filter_mask: str | None,
    *,
    spine_path: str | Path | None = None,
) -> tuple[list[PreprocessTask], pd.DataFrame]:
    """Plan shared analysis tasks or preserve the legacy spatial-only BED scope."""
    standard_tasks = plan_preprocess_tasks(
        spine,
        target_task_memory_mb=int(getattr(cfg, "target_task_memory_mb", 512)),
        partition_by_barcode=True,
        filter_mask=filter_mask,
        spine_path=spine_path,
        minimum_halo=int(getattr(cfg, "autocorr_max_lag", 800)),
    )
    if has_analysis_catalog(spine):
        regions = pd.DataFrame(
            [
                {
                    "reference": task.reference,
                    "start": task.core_start,
                    "end": task.core_end,
                    "name": task.analysis_core_id,
                    "bed_chrom": task.original_reference,
                    "source": "analysis",
                }
                for task in standard_tasks
                if task.analysis_mode == "genome"
            ],
            columns=["reference", "start", "end", "name", "bed_chrom", "source"],
        ).drop_duplicates(["reference", "start", "end"])
        return standard_tasks, regions

    bed_path = getattr(cfg, "spatial_regions_bed", None)
    regions = _read_spatial_regions(spine, bed_path)
    if regions.empty:
        return standard_tasks, regions

    locus_tasks = [task for task in standard_tasks if task.analysis_mode == "locus"]
    obs = spine.obs
    if filter_mask is not None:
        obs = obs.loc[obs[filter_mask].astype(bool)]
    barcode_column = "Barcode" if "Barcode" in obs else "Sample"
    plans = {str(key): dict(value) for key, value in spine.uns["reference_plans"].items()}
    memory_budget = int(getattr(cfg, "target_task_memory_mb", 512)) * 1024**2
    genome_tasks = []
    for region_index, region in regions.iterrows():
        reference = str(region["reference"])
        plan = plans[reference]
        if str(plan["analysis_mode"]) != "genome":
            continue
        core_start, core_end = int(region["start"]), int(region["end"])
        halo = max(int(plan["tile_halo"]), int(getattr(cfg, "autocorr_max_lag", 800)))
        load_start = max(0, core_start - halo)
        load_end = min(int(plan["reference_length"]), core_end + halo)
        overlapping = obs.loc[
            (obs[REFERENCE_STRAND].astype(str) == reference)
            & (obs["reference_start"].astype("int64") < core_end)
            & (obs["reference_end"].astype("int64") > core_start)
        ]
        loaded_width = load_end - load_start
        reads_per_chunk = max(
            1,
            memory_budget // max(1, loaded_width * BYTES_PER_WORKING_POSITION),
        )
        for barcode, barcode_obs in overlapping.groupby(barcode_column, sort=True, observed=True):
            read_ids = sorted(map(str, barcode_obs.index))
            for chunk_index, start_index in enumerate(range(0, len(read_ids), reads_per_chunk)):
                chunk = read_ids[start_index : start_index + reads_per_chunk]
                genome_tasks.append(
                    PreprocessTask(
                        task_id=(
                            f"{reference}|{barcode}|bed{region_index}:"
                            f"{core_start}-{core_end}|{chunk_index:05d}"
                        ),
                        reference=reference,
                        barcode=str(barcode),
                        analysis_mode="genome",
                        chunk_index=chunk_index,
                        core_start=core_start,
                        core_end=core_end,
                        load_start=load_start,
                        load_end=load_end,
                        n_reads=len(chunk),
                        estimated_memory_bytes=len(chunk)
                        * loaded_width
                        * BYTES_PER_WORKING_POSITION,
                        read_ids=tuple(chunk),
                    )
                )
    return [*locus_tasks, *genome_tasks], regions


def execute_spatial_task(spine_path, task, cfg, output_dir) -> dict[str, object]:
    """Compute one barcode/reference/window spatial partial."""
    output_dir = Path(output_dir)
    adata = materialize(
        spine_path,
        references=task.reference,
        read_ids=task.read_ids,
        start=task.load_start,
        end=task.load_end,
        layers=[],
    )
    positions = np.asarray(adata.var_names, dtype=np.int64)
    core_mask = (positions >= task.core_start) & (positions < task.core_end)
    core = adata[:, core_mask]
    positions = positions[core_mask]

    metric_rows = []
    autocorr_rows = []
    position_rows = []
    identity_columns = [
        column for column in ("read_id", "experiment_uid", "molecule_uid") if column in core.obs
    ]
    read_obs = core.obs[identity_columns].copy()
    read_obsm = {}
    read_uns = {
        "task_id": task.task_id,
        "reference": task.reference,
        "barcode": task.barcode,
        "core_start": task.core_start,
        "core_end": task.core_end,
        "load_start": task.load_start,
        "load_end": task.load_end,
        "analysis_core_id": task.analysis_core_id,
        "analysis_region_ids": list(task.analysis_region_ids),
        "analysis_planner_version": task.analysis_planner_version,
    }
    for requested_site in getattr(cfg, "autocorr_site_types", ["GpC"]):
        site_type = str(requested_site).removesuffix("_site")
        column = _site_column(core, task.reference, site_type)
        if column is None:
            continue
        site_mask = core.var[column].astype("boolean").fillna(False).to_numpy(dtype=bool)
        if not site_mask.any():
            continue
        site_positions = positions[site_mask]
        # core.X is already float32 binary/NaN data; slice to site columns
        # *before* touching dtype (not np.asarray(core.X, dtype=float)[:, mask],
        # which upcasts the whole locus-width window to float64 first). Nothing
        # downstream needs float64 here -- binary_autocorrelation_with_spacing
        # and analyze_ls_periodicity_direct each upcast per-read, on the much
        # smaller already-masked row, when they actually need it.
        values = np.asarray(core.X[:, site_mask])
        valid = np.isfinite(values)
        valid_count = valid.sum(axis=0).astype(np.int64)
        modified_sum = np.nansum(values, axis=0)
        for position, count, modified in zip(site_positions, valid_count, modified_sum):
            if count:
                position_rows.append(
                    {
                        "task_id": task.task_id,
                        "reference": task.reference,
                        "barcode": task.barcode,
                        "core_start": task.core_start,
                        "core_end": task.core_end,
                        "site_type": site_type,
                        "position": int(position),
                        "n_valid": int(count),
                        "modified_sum": float(modified),
                    }
                )

        read_statistics = _compute_read_spatial_statistics(values, site_positions, cfg)
        autocorrelation_matrix = read_statistics["autocorrelation"]
        counts_matrix = read_statistics["pair_counts"]
        read_obs[f"{site_type}_n_sites"] = read_statistics["n_sites"]
        if bool(getattr(cfg, "spatial_save_read_autocorrelation", True)):
            read_obsm[f"{site_type}_spatial_autocorr"] = autocorrelation_matrix
            read_obsm[f"{site_type}_spatial_autocorr_counts"] = counts_matrix
            read_uns[f"{site_type}_spatial_autocorr_lags"] = np.arange(
                autocorrelation_matrix.shape[1], dtype=np.int64
            ).tolist()
        if bool(getattr(cfg, "spatial_compute_read_lomb_scargle", True)):
            read_obsm[f"{site_type}_lomb_scargle_power"] = read_statistics["periodogram_power"]
            read_uns[f"{site_type}_lomb_scargle_frequencies"] = read_statistics[
                "frequencies"
            ].tolist()
            read_uns[f"{site_type}_lomb_scargle_periods_bp"] = read_statistics["periods"].tolist()
            read_obs[f"{site_type}_ls_status"] = read_statistics["status"]
            for name in (
                "ls_nrl_bp",
                "ls_snr",
                "ls_peak_power",
                "ls_peak_power_raw",
                "ls_fwhm_bp",
            ):
                read_obs[f"{site_type}_{name}"] = read_statistics[name]
        mean_autocorrelation, total_pair_counts = weighted_mean_autocorr(
            autocorrelation_matrix,
            counts_matrix,
            min_count=1,
        )
        for lag, (autocorrelation, count) in enumerate(
            zip(mean_autocorrelation, total_pair_counts)
        ):
            if count:
                autocorr_rows.append(
                    {
                        "task_id": task.task_id,
                        "reference": task.reference,
                        "barcode": task.barcode,
                        "core_start": task.core_start,
                        "core_end": task.core_end,
                        "site_type": site_type,
                        "lag": lag,
                        "autocorrelation": float(autocorrelation),
                        "pair_count": int(count),
                    }
                )
        total_valid = int(valid.sum())
        metric_rows.append(
            {
                "task_id": task.task_id,
                "reference": task.reference,
                "barcode": task.barcode,
                "core_start": task.core_start,
                "core_end": task.core_end,
                "site_type": site_type,
                "n_reads": int(core.n_obs),
                "n_site_positions": int(site_mask.sum()),
                "n_valid_calls": total_valid,
                "modified_sum": float(np.nansum(values)),
                "mean_modification": (
                    float(np.nansum(values) / total_valid) if total_valid else np.nan
                ),
            }
        )

    task_dir = _task_directory(output_dir, task)
    task_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = task_dir / "metrics.parquet"
    autocorr_path = task_dir / "autocorrelation.parquet"
    positions_path = task_dir / "positions.parquet"
    pd.DataFrame(metric_rows).to_parquet(metrics_path, index=False)
    pd.DataFrame(autocorr_rows).to_parquet(autocorr_path, index=False)
    pd.DataFrame(position_rows).to_parquet(positions_path, index=False)
    obsm_keys = sorted(read_obsm)
    group_path = None
    if read_obsm:
        # Obsm entries (autocorrelation/pair-count/Lomb-Scargle matrices, one
        # set per site type) are streamed to disk one at a time and freed
        # immediately, instead of being handed to one ``ad.AnnData(obsm=...)``
        # + ``safe_write_zarr`` call that holds all of them (plus whatever
        # copies zarr serialization makes) in memory together -- same
        # rationale as ``tools.partitioned_hmm.execute_hmm_task``.
        import anndata as ad

        from ..informatics.incremental_zarr import (
            append_zarr_obsm,
            consolidate_zarr_store,
        )
        from ..informatics.physical_layout import portable_matrix_chunks

        read_metrics_path = task_dir / "read_metrics.zarr"
        skeleton = ad.AnnData(obs=read_obs, uns=read_uns)
        safe_write_zarr(skeleton, read_metrics_path, backup=False, verbose=False, zarr_format=3)
        for index, name in enumerate(obsm_keys):
            array = read_obsm.pop(name)
            append_zarr_obsm(
                read_metrics_path,
                name,
                array,
                chunks=portable_matrix_chunks(array.shape, array.dtype),
                consolidate=(index == len(obsm_keys) - 1),
            )
            del array
        group_path = read_metrics_path.relative_to(output_dir).as_posix()
    from ..informatics.derived_read_index import write_derived_read_index

    read_index_path = write_derived_read_index(
        output_dir,
        stage="spatial",
        task=task,
        obs=core.obs,
        group_path=group_path,
        stage_schema_version=3,
    )
    return {
        **task.to_dict(include_read_ids=False),
        "metrics_path": metrics_path.relative_to(output_dir).as_posix(),
        "autocorrelation_path": autocorr_path.relative_to(output_dir).as_posix(),
        "positions_path": positions_path.relative_to(output_dir).as_posix(),
        # group_path/obsm_keys: same store-per-task addressing convention as
        # preprocess/hmm's catalogs (informatics.experiment_storage_schema doc's
        # "Store organization" section) -- group_path is None and obsm_keys is
        # empty when this task computed no read-level outputs at all.
        "group_path": group_path,
        "obsm_keys": obsm_keys,
        "site_types": sorted({row["site_type"] for row in metric_rows}),
        "read_index_path": read_index_path.relative_to(output_dir).as_posix(),
    }


def _reduce_metrics(records, output_dir: Path) -> tuple[Path, Path]:
    metric_keys = ["reference", "barcode", "core_start", "core_end", "site_type"]
    metric_totals: dict[tuple, dict[str, float]] = {}
    for record in records:
        frame = pd.read_parquet(output_dir / record["metrics_path"])
        for row in frame.itertuples(index=False):
            key = tuple(getattr(row, column) for column in metric_keys)
            total = metric_totals.setdefault(
                key,
                {"n_reads": 0, "n_site_positions": 0, "n_valid_calls": 0, "modified_sum": 0.0},
            )
            total["n_reads"] += int(row.n_reads)
            total["n_site_positions"] = max(total["n_site_positions"], int(row.n_site_positions))
            total["n_valid_calls"] += int(row.n_valid_calls)
            modified_sum = float(row.modified_sum)
            total["modified_sum"] += modified_sum if np.isfinite(modified_sum) else 0.0
    metrics = pd.DataFrame(
        (
            {**dict(zip(metric_keys, key, strict=True)), **values}
            for key, values in sorted(metric_totals.items())
        )
    )
    if not metrics.empty:
        metrics["mean_modification"] = metrics["modified_sum"].div(
            metrics["n_valid_calls"].replace(0, np.nan)
        )
    metrics_path = output_dir / SPATIAL_METRICS
    metrics.to_parquet(metrics_path, index=False)

    autocorr_keys = ["reference", "barcode", "core_start", "core_end", "site_type", "lag"]
    autocorr_totals: dict[tuple, tuple[float, int]] = {}
    for record in records:
        frame = pd.read_parquet(output_dir / record["autocorrelation_path"])
        for row in frame.itertuples(index=False):
            key = tuple(getattr(row, column) for column in autocorr_keys)
            weighted, count = autocorr_totals.get(key, (0.0, 0))
            pair_count = int(row.pair_count)
            autocorrelation = float(row.autocorrelation)
            autocorr_totals[key] = (
                weighted + (autocorrelation * pair_count if np.isfinite(autocorrelation) else 0.0),
                count + pair_count,
            )
    autocorrelation = pd.DataFrame(
        (
            {
                **dict(zip(autocorr_keys, key, strict=True)),
                "weighted_autocorrelation": weighted,
                "pair_count": count,
            }
            for key, (weighted, count) in sorted(autocorr_totals.items())
        )
    )
    if not autocorrelation.empty:
        autocorrelation["autocorrelation"] = autocorrelation["weighted_autocorrelation"].div(
            autocorrelation["pair_count"].replace(0, np.nan)
        )
        autocorrelation = autocorrelation.drop(columns="weighted_autocorrelation")
    autocorrelation_path = output_dir / SPATIAL_AUTOCORRELATION
    autocorrelation.to_parquet(autocorrelation_path, index=False)
    return metrics_path, autocorrelation_path


def _plot_autocorrelation(autocorrelation, layout) -> None:
    import matplotlib.pyplot as plt

    if autocorrelation.empty:
        return
    keys = ["reference", "core_start", "core_end", "site_type"]
    for (reference, core_start, core_end, site_type), frame in autocorrelation.groupby(
        keys, sort=True, observed=True
    ):
        weighted = (
            frame.assign(weighted=frame["autocorrelation"] * frame["pair_count"])
            .groupby("lag", sort=True, observed=True)[["weighted", "pair_count"]]
            .sum()
        )
        weighted["mean"] = weighted["weighted"].div(weighted["pair_count"].replace(0, np.nan))
        pivot = frame.pivot_table(
            index="barcode",
            columns="lag",
            values="autocorrelation",
            aggfunc="first",
            observed=True,
        ).sort_index()
        figure = plt.figure(figsize=(10, max(6, 0.22 * len(pivot) + 3)))
        grid = figure.add_gridspec(
            2,
            2,
            height_ratios=(2, max(2, 0.16 * len(pivot))),
            width_ratios=(30, 1),
            hspace=0.08,
            wspace=0.08,
        )
        bulk_axis = figure.add_subplot(grid[0, 0])
        heatmap_axis = figure.add_subplot(grid[1, 0], sharex=bulk_axis)
        colorbar_axis = figure.add_subplot(grid[1, 1])
        lag_min = float(pivot.columns.min())
        lag_max = float(pivot.columns.max())
        bulk_axis.plot(weighted.index, weighted["mean"], color="#264653", linewidth=1)
        bulk_axis.axhline(0, color="#444444", linewidth=0.6)
        bulk_axis.set(ylabel="Weighted mean", title="All barcodes", xlim=(lag_min, lag_max))
        bulk_axis.tick_params(axis="x", labelbottom=False)
        image = heatmap_axis.imshow(
            pivot.to_numpy(),
            aspect="auto",
            interpolation="nearest",
            cmap="coolwarm",
            vmin=-0.5,
            vmax=0.5,
            extent=(lag_min, lag_max, len(pivot), 0),
        )
        heatmap_axis.set_yticks(
            np.arange(len(pivot)) + 0.5,
            pivot.index.astype(str),
            fontsize=6,
        )
        heatmap_axis.set(xlabel="Lag (bp)", ylabel="Barcode", xlim=(lag_min, lag_max))
        figure.colorbar(image, cax=colorbar_axis, label="Autocorrelation")
        figure.suptitle(f"{reference}:{core_start}-{core_end} / {site_type}")
        figure.subplots_adjust(top=0.9, left=0.1, right=0.92, bottom=0.1)
        path = layout.categories["autocorrelation"] / (
            f"{_component(reference)}__{core_start}_{core_end}__{_component(site_type)}.png"
        )
        figure.savefig(path, dpi=160)
        plt.close(figure)
        register_plot_artifact(
            layout,
            path,
            stage="spatial",
            category="autocorrelation",
            plot_type="barcode_autocorrelation",
            reference=str(reference),
            core_start=int(core_start),
            core_end=int(core_end),
        )


def _cap_clustermap_rows(matrix: np.ndarray, max_rows: int | None, *, seed: int = 0) -> np.ndarray:
    """Reproducibly subsample a matrix's rows to at most ``max_rows``.

    ``_clustered_row_order`` runs scipy's hierarchical ``linkage`` on every row
    it's given -- O(n^2) memory and worse-than-quadratic time in row count.
    Uncapped, a busy barcode with tens of thousands of reads (e.g. a
    genome-mode reference like ``6B6_top`` at 650k+ reads total) turned one
    read-metric clustermap PNG into a multi-minute single-threaded stall.
    Same reproducible convention as ``subsample_reads_for_plot``/
    ``subsample_read_ids`` in ``plotting_utils`` (fixed seed, sorted indices),
    but applied to an already-vstacked numpy matrix -- read identity isn't
    tracked this far into the reduce phase. Only the heatmap/clustering input
    is capped; callers pass the *uncapped* matrix when computing the mean
    profile line, so the true full-population mean is unaffected.
    """
    n_rows = matrix.shape[0]
    if max_rows is None or max_rows <= 0 or n_rows <= int(max_rows):
        return matrix
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(n_rows, size=int(max_rows), replace=False))
    return matrix[chosen]


def _clustered_row_order(values: np.ndarray) -> np.ndarray:
    """Return a deterministic hierarchical row order for a NaN-aware matrix."""
    if values.shape[0] < 2:
        return np.arange(values.shape[0])
    from scipy.cluster.hierarchy import leaves_list, linkage

    finite = np.isfinite(values)
    denominator = finite.sum(axis=0)
    column_means = np.zeros(values.shape[1], dtype=float)
    np.divide(
        np.nansum(values, axis=0),
        denominator,
        out=column_means,
        where=denominator > 0,
    )
    filled = np.where(finite, values, column_means)
    row_mean = filled.mean(axis=1, keepdims=True)
    row_std = filled.std(axis=1, keepdims=True)
    normalized = np.divide(
        filled - row_mean,
        row_std,
        out=np.zeros_like(filled),
        where=row_std > 0,
    )
    if not np.any(normalized):
        return np.arange(values.shape[0])
    try:
        return leaves_list(linkage(normalized, method="average", metric="euclidean"))
    except ValueError:
        return np.arange(values.shape[0])


def _plot_read_profile_clustermap(
    values: np.ndarray,
    x: np.ndarray,
    mean_profile: np.ndarray,
    *,
    path: Path,
    title: str,
    xlabel: str,
    colorbar_label: str,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    highlight_range: tuple[float, float] | None = None,
) -> None:
    """Render an aligned mean profile and clustered read heatmap."""
    import matplotlib.pyplot as plt

    order = _clustered_row_order(values)
    ordered = values[order]
    figure = plt.figure(figsize=(10, max(5, min(12, 3 + 0.012 * len(ordered)))))
    figure.patch.set_facecolor("white")
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=(2, max(2, min(8, 0.01 * len(ordered)))),
        width_ratios=(30, 1),
        hspace=0.08,
        wspace=0.08,
    )
    mean_axis = figure.add_subplot(grid[0, 0])
    heatmap_axis = figure.add_subplot(grid[1, 0], sharex=mean_axis)
    colorbar_axis = figure.add_subplot(grid[1, 1])
    mean_axis.plot(x, mean_profile, color="#264653", linewidth=1)
    mean_axis.set(ylabel="Mean", title=f"{len(ordered):,} reads", xlim=(x.min(), x.max()))
    mean_axis.tick_params(axis="x", labelbottom=False)
    image = heatmap_axis.imshow(
        ordered,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=(x.min(), x.max(), len(ordered), 0),
        rasterized=True,
    )
    if highlight_range is not None:
        for axis in (mean_axis, heatmap_axis):
            axis.axvspan(*highlight_range, color="#e9c46a", alpha=0.15)
    heatmap_axis.set(xlabel=xlabel, ylabel="Clustered reads", xlim=(x.min(), x.max()))
    heatmap_axis.set_yticks([])
    figure.colorbar(image, cax=colorbar_axis, label=colorbar_label)
    figure.suptitle(title)
    figure.subplots_adjust(top=0.9, left=0.1, right=0.92, bottom=0.1)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, facecolor="white")
    plt.close(figure)


def _plot_read_metric_clustermaps(records, output_dir: Path, layout, cfg) -> None:
    """Plot per-read ACF and Lomb-Scargle heatmaps for each barcode region."""
    if not bool(getattr(cfg, "spatial_plot_read_metric_clustermaps", True)):
        return
    from ..informatics.partition_query import query_derived_index, read_zarr_subset

    max_reads_per_plot = getattr(cfg, "clustermap_max_reads_per_plot", 5000)
    selection_seed = int(getattr(cfg, "plot_subsample_seed", 0))
    indexed = query_derived_index(output_dir / "read_index")
    selected_rows: dict[str, list[int]] = {}
    if not indexed.empty:
        group_columns = ["reference", "core_start", "core_end", "barcode"]
        identity = "molecule_uid" if "molecule_uid" in indexed else "read_id"
        for _, group in indexed.groupby(group_columns, sort=True, observed=True):
            group = group.sort_values(identity, kind="stable").drop_duplicates(identity)
            if (
                max_reads_per_plot is not None
                and int(max_reads_per_plot) > 0
                and len(group) > int(max_reads_per_plot)
            ):
                rng = np.random.default_rng(selection_seed)
                group = group.iloc[
                    np.sort(rng.choice(len(group), size=int(max_reads_per_plot), replace=False))
                ]
            for group_path, rows in group.groupby("group_path", sort=True, observed=True):
                selected_rows[str(group_path)] = rows["group_row"].astype(int).tolist()
    groups = {}
    for record in records:
        relative_path = record.get("group_path")
        rows = selected_rows.get(str(relative_path), [])
        if not relative_path or not rows:
            continue
        read_metrics = read_zarr_subset(output_dir / relative_path, row_indices=rows)
        for key in read_metrics.obsm:
            suffix = "_spatial_autocorr"
            if not str(key).endswith(suffix):
                continue
            site_type = str(key)[: -len(suffix)]
            group_key = (
                str(record["reference"]),
                int(record["core_start"]),
                int(record["core_end"]),
                str(record["barcode"]),
                site_type,
            )
            bucket = groups.setdefault(
                group_key,
                {"acf": [], "counts": [], "power": [], "lags": None, "periods": None},
            )
            bucket["acf"].append(np.asarray(read_metrics.obsm[key], dtype=float))
            bucket["counts"].append(
                np.asarray(read_metrics.obsm[f"{site_type}_spatial_autocorr_counts"], dtype=float)
            )
            bucket["lags"] = np.asarray(
                read_metrics.uns[f"{site_type}_spatial_autocorr_lags"], dtype=float
            )
            power_key = f"{site_type}_lomb_scargle_power"
            if power_key in read_metrics.obsm:
                bucket["power"].append(np.asarray(read_metrics.obsm[power_key], dtype=float))
                bucket["periods"] = np.asarray(
                    read_metrics.uns[f"{site_type}_lomb_scargle_periods_bp"], dtype=float
                )

    peak_range = _config_range(cfg, "spatial_lomb_scargle_peak_range_bp", (150.0, 250.0))
    for (reference, core_start, core_end, barcode, site_type), bucket in sorted(groups.items()):
        region_label = f"{_component(reference)}__{core_start}_{core_end}__{_component(site_type)}"
        acf = np.vstack(bucket["acf"])
        counts = np.vstack(bucket["counts"])
        denominator = counts.sum(axis=0)
        mean_acf = np.full(acf.shape[1], np.nan, dtype=float)
        np.divide(
            np.nansum(np.where(np.isfinite(acf), acf, 0.0) * counts, axis=0),
            denominator,
            out=mean_acf,
            where=denominator > 0,
        )
        acf_path = (
            layout.categories["autocorrelation"]
            / "read_clustermaps"
            / region_label
            / f"{_component(barcode)}.png"
        )
        _plot_read_profile_clustermap(
            _cap_clustermap_rows(acf, max_reads_per_plot),
            np.asarray(bucket["lags"]),
            mean_acf,
            path=acf_path,
            title=f"{barcode} | {reference}:{core_start}-{core_end} | {site_type} ACF",
            xlabel="Lag (bp)",
            colorbar_label="Autocorrelation",
            cmap="coolwarm",
            vmin=-0.5,
            vmax=0.5,
        )
        register_plot_artifact(
            layout,
            acf_path,
            stage="spatial",
            category="autocorrelation",
            plot_type="read_autocorrelation_clustermap",
            reference=reference,
            sample=barcode,
            core_start=core_start,
            core_end=core_end,
        )

        if not bucket["power"]:
            continue
        power = np.vstack(bucket["power"])
        if not np.isfinite(power).any():
            continue
        periods = np.asarray(bucket["periods"])
        period_order = np.argsort(periods)
        power = power[:, period_order]
        periods = periods[period_order]
        finite = np.isfinite(power)
        denominator = finite.sum(axis=0)
        mean_power = np.full(power.shape[1], np.nan, dtype=float)
        np.divide(
            np.nansum(power, axis=0),
            denominator,
            out=mean_power,
            where=denominator > 0,
        )
        finite_power = power[np.isfinite(power)]
        vmax = float(np.nanpercentile(finite_power, 99)) if finite_power.size else 1.0
        periodogram_path = (
            layout.categories["periodicity"]
            / "read_clustermaps"
            / region_label
            / f"{_component(barcode)}.png"
        )
        _plot_read_profile_clustermap(
            _cap_clustermap_rows(power, max_reads_per_plot),
            periods,
            mean_power,
            path=periodogram_path,
            title=f"{barcode} | {reference}:{core_start}-{core_end} | {site_type} periodogram",
            xlabel="Period (bp)",
            colorbar_label="Normalized power",
            cmap="magma",
            vmin=0.0,
            vmax=max(vmax, np.finfo(float).eps),
            highlight_range=peak_range,
        )
        register_plot_artifact(
            layout,
            periodogram_path,
            stage="spatial",
            category="periodicity",
            plot_type="read_lomb_scargle_periodogram_clustermap",
            reference=reference,
            sample=barcode,
            core_start=core_start,
            core_end=core_end,
        )


def _plot_read_periodicity(records, output_dir: Path, layout, cfg) -> None:
    """Plot read-level Lomb-Scargle summaries from task-local AnnData stores."""
    import matplotlib.pyplot as plt

    if not bool(getattr(cfg, "spatial_compute_read_lomb_scargle", True)) or not bool(
        getattr(cfg, "spatial_plot_read_lomb_scargle", True)
    ):
        return
    from ..informatics.partition_query import query_derived_index, read_zarr_subset

    max_reads = getattr(cfg, "clustermap_max_reads_per_plot", 5000)
    selection_seed = int(getattr(cfg, "plot_subsample_seed", 0))
    indexed = query_derived_index(output_dir / "read_index")
    selected_rows: dict[str, list[int]] = {}
    if not indexed.empty:
        group_columns = ["reference", "core_start", "core_end", "barcode"]
        identity = "molecule_uid" if "molecule_uid" in indexed else "read_id"
        for _, group in indexed.groupby(group_columns, sort=True, observed=True):
            group = group.sort_values(identity, kind="stable").drop_duplicates(identity)
            if max_reads is not None and int(max_reads) > 0 and len(group) > int(max_reads):
                rng = np.random.default_rng(selection_seed)
                group = group.iloc[
                    np.sort(rng.choice(len(group), size=int(max_reads), replace=False))
                ]
            for group_path, rows in group.groupby("group_path", sort=True, observed=True):
                selected_rows[str(group_path)] = rows["group_row"].astype(int).tolist()
    windows = {}
    for record in records:
        relative_path = record.get("group_path")
        rows = selected_rows.get(str(relative_path), [])
        if not relative_path or not rows:
            continue
        read_metrics = read_zarr_subset(output_dir / relative_path, row_indices=rows)
        for key in read_metrics.obsm:
            suffix = "_lomb_scargle_power"
            if not str(key).endswith(suffix):
                continue
            site_type = str(key)[: -len(suffix)]
            window_key = (
                str(record["reference"]),
                int(record["core_start"]),
                int(record["core_end"]),
                site_type,
            )
            bucket = windows.setdefault(
                window_key,
                {"metrics": [], "power": [], "periods": None},
            )
            columns = [
                f"{site_type}_ls_status",
                f"{site_type}_ls_nrl_bp",
                f"{site_type}_ls_peak_power",
                f"{site_type}_ls_snr",
                f"{site_type}_ls_fwhm_bp",
                f"{site_type}_n_sites",
            ]
            frame = read_metrics.obs.loc[:, columns].copy()
            frame.columns = [column.removeprefix(f"{site_type}_") for column in columns]
            frame["read_id"] = frame.index.astype(str)
            frame["barcode"] = str(record["barcode"])
            bucket["metrics"].append(frame.reset_index(drop=True))
            bucket["power"].append(
                (str(record["barcode"]), np.asarray(read_metrics.obsm[key], dtype=float))
            )
            bucket["periods"] = np.asarray(
                read_metrics.uns[f"{site_type}_lomb_scargle_periods_bp"], dtype=float
            )

    peak_min, peak_max = _config_range(cfg, "spatial_lomb_scargle_peak_range_bp", (150.0, 250.0))
    swarm_rng = np.random.default_rng(0)
    swarm_max_points = 4000
    for (reference, core_start, core_end, site_type), bucket in sorted(windows.items()):
        metrics = pd.concat(bucket["metrics"], ignore_index=True)
        metrics["scored"] = metrics["ls_status"].astype(str).eq("ok")
        barcodes = sorted(metrics["barcode"].astype(str).unique())
        valid = metrics.loc[metrics["scored"]].copy()

        figure, axes = plt.subplots(2, 2, figsize=(13, 8))
        for axis, column, ylabel in (
            (axes[0, 0], "ls_nrl_bp", "Peak period (bp)"),
            (axes[0, 1], "ls_peak_power", "Peak power"),
        ):
            distributions = [
                valid.loc[valid["barcode"].astype(str) == barcode, column].dropna().to_numpy()
                for barcode in barcodes
            ]
            if any(len(values) for values in distributions):
                # A KDE (what violinplot draws) is undefined for <2 points or
                # zero-variance data, so those positions are dropped from the
                # violin call and rely on the swarm overlay alone.
                violin_positions = [
                    position
                    for position, values in enumerate(distributions, start=1)
                    if len(values) >= 2 and np.std(values) > 0
                ]
                violin_distributions = [
                    distributions[position - 1] for position in violin_positions
                ]
                if violin_distributions:
                    parts = axis.violinplot(
                        violin_distributions,
                        positions=violin_positions,
                        showmeans=False,
                        showmedians=True,
                        showextrema=False,
                        widths=0.7,
                    )
                    for body in parts["bodies"]:
                        body.set_facecolor("#9ec5ab")
                        body.set_edgecolor("#264653")
                        body.set_alpha(0.7)
                        body.set_zorder(2)
                    parts["cmedians"].set_color("#264653")
                    parts["cmedians"].set_linewidth(1.0)
                    parts["cmedians"].set_zorder(2.5)
                # Jittered per-read points on top of each violin -- see
                # preprocessing/partitioned_plots.py::_barcode_distribution_plots
                # for why this uses jitter+alpha instead of a true swarmplot.
                for position, values in enumerate(distributions, start=1):
                    if len(values) == 0:
                        continue
                    if len(values) > swarm_max_points:
                        values = swarm_rng.choice(values, swarm_max_points, replace=False)
                    jitter = swarm_rng.uniform(-0.2, 0.2, size=len(values))
                    axis.scatter(
                        np.full(len(values), position) + jitter,
                        values,
                        s=3,
                        alpha=0.3,
                        color="#264653",
                        edgecolors="none",
                        rasterized=True,
                        zorder=3,
                    )
            axis.set_xticks(range(1, len(barcodes) + 1), barcodes, rotation=90, fontsize=6)
            axis.set_ylabel(ylabel)
            axis.grid(axis="y", color="#dddddd", linewidth=0.5)

        axes[1, 0].scatter(
            valid["ls_nrl_bp"],
            valid["ls_peak_power"],
            s=5,
            alpha=0.15,
            color="#52796f",
            rasterized=True,
        )
        medians = valid.groupby("barcode", observed=True)[["ls_nrl_bp", "ls_peak_power"]].median()
        axes[1, 0].scatter(medians["ls_nrl_bp"], medians["ls_peak_power"], s=24, color="#e76f51")
        for barcode, row in medians.iterrows():
            axes[1, 0].annotate(
                str(barcode),
                (row["ls_nrl_bp"], row["ls_peak_power"]),
                xytext=(3, 2),
                textcoords="offset points",
                fontsize=6,
            )
        axes[1, 0].axvspan(peak_min, peak_max, color="#e9c46a", alpha=0.12)
        axes[1, 0].set(xlabel="Peak period (bp)", ylabel="Peak power")

        scored_fraction = (
            metrics.groupby("barcode", observed=True)["scored"].mean().reindex(barcodes)
        )
        axes[1, 1].bar(
            np.arange(len(barcodes)), scored_fraction.to_numpy(), color="#2a9d8f", width=0.8
        )
        axes[1, 1].set_xticks(np.arange(len(barcodes)), barcodes, rotation=90, fontsize=6)
        axes[1, 1].set(ylabel="Scored read fraction", ylim=(0, 1.05))
        axes[1, 1].grid(axis="y", color="#dddddd", linewidth=0.5)
        figure.suptitle(f"{reference}:{core_start}-{core_end} / {site_type} periodicity")
        figure.tight_layout()
        stem = f"{_component(reference)}__{core_start}_{core_end}__{_component(site_type)}"
        metric_path = layout.categories["periodicity"] / f"{stem}__read_metrics.png"
        figure.savefig(metric_path, dpi=160)
        plt.close(figure)
        register_plot_artifact(
            layout,
            metric_path,
            stage="spatial",
            category="periodicity",
            plot_type="read_lomb_scargle_metrics",
            reference=reference,
            core_start=core_start,
            core_end=core_end,
        )

        periods = np.asarray(bucket["periods"], dtype=float)
        order = np.argsort(periods)
        barcode_means = []
        all_power = []
        for barcode in barcodes:
            arrays = [power for name, power in bucket["power"] if name == barcode]
            if not arrays:
                continue
            power = np.vstack(arrays)
            all_power.append(power)
            finite = np.isfinite(power)
            denominator = finite.sum(axis=0)
            mean_power = np.full(power.shape[1], np.nan, dtype=float)
            np.divide(
                np.nansum(power, axis=0),
                denominator,
                out=mean_power,
                where=denominator > 0,
            )
            barcode_means.append(mean_power)
        if not barcode_means or not all_power:
            continue
        mean_matrix = np.vstack(barcode_means)[:, order]
        combined = np.vstack(all_power)
        if not np.isfinite(combined).any():
            continue
        combined_finite = np.isfinite(combined)
        denominator = combined_finite.sum(axis=0)
        bulk = np.full(combined.shape[1], np.nan, dtype=float)
        np.divide(
            np.nansum(combined, axis=0),
            denominator,
            out=bulk,
            where=denominator > 0,
        )
        bulk = bulk[order]
        x = periods[order]
        figure = plt.figure(figsize=(10, max(6, 0.22 * len(barcodes) + 3)))
        grid = figure.add_gridspec(
            2,
            2,
            height_ratios=(2, max(2, 0.16 * len(barcodes))),
            width_ratios=(30, 1),
            hspace=0.08,
            wspace=0.08,
        )
        bulk_axis = figure.add_subplot(grid[0, 0])
        heatmap_axis = figure.add_subplot(grid[1, 0], sharex=bulk_axis)
        colorbar_axis = figure.add_subplot(grid[1, 1])
        bulk_axis.plot(x, bulk, color="#264653", linewidth=1)
        bulk_axis.axvspan(peak_min, peak_max, color="#e9c46a", alpha=0.2)
        bulk_axis.set(ylabel="Mean power", title="All reads", xlim=(x.min(), x.max()))
        bulk_axis.tick_params(axis="x", labelbottom=False)
        image = heatmap_axis.imshow(
            mean_matrix,
            aspect="auto",
            interpolation="nearest",
            cmap="magma",
            extent=(x.min(), x.max(), len(barcodes), 0),
        )
        heatmap_axis.axvspan(peak_min, peak_max, color="white", alpha=0.08)
        heatmap_axis.set_yticks(
            np.arange(len(barcodes)) + 0.5,
            barcodes,
            fontsize=6,
        )
        heatmap_axis.set(
            xlabel="Period (bp)",
            ylabel="Barcode",
            xlim=(x.min(), x.max()),
        )
        figure.colorbar(image, cax=colorbar_axis, label="Mean normalized power")
        figure.suptitle(f"{reference}:{core_start}-{core_end} / {site_type}")
        figure.subplots_adjust(top=0.9, left=0.1, right=0.92, bottom=0.1)
        periodogram_path = layout.categories["periodicity"] / f"{stem}__periodogram.png"
        figure.savefig(periodogram_path, dpi=160)
        plt.close(figure)
        register_plot_artifact(
            layout,
            periodogram_path,
            stage="spatial",
            category="periodicity",
            plot_type="barcode_lomb_scargle_periodogram",
            reference=reference,
            core_start=core_start,
            core_end=core_end,
        )


def _plot_position_profiles(records, output_dir: Path, layout) -> None:
    import matplotlib.pyplot as plt

    record_frame = pd.DataFrame(records)
    window_keys = ["reference", "core_start", "core_end"]
    for (reference, core_start, core_end), window_records in record_frame.groupby(
        window_keys, sort=True, observed=True
    ):
        frames = [pd.read_parquet(output_dir / path) for path in window_records["positions_path"]]
        positions = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if positions.empty:
            continue
        for site_type, frame in positions.groupby("site_type", sort=True, observed=True):
            grouped = (
                frame.groupby(["barcode", "position"], sort=True, observed=True)[
                    ["n_valid", "modified_sum"]
                ]
                .sum()
                .reset_index()
            )
            barcode_support = grouped.groupby("barcode", observed=True)["n_valid"].sum()
            selected = set(barcode_support.nlargest(12).index.astype(str))
            figure, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
            overall = grouped.groupby("position", observed=True)[["n_valid", "modified_sum"]].sum()
            axes[0].plot(
                overall.index,
                overall["modified_sum"].div(overall["n_valid"].replace(0, np.nan)),
                color="#264653",
                linewidth=0.8,
            )
            axes[0].set(ylabel="Modified fraction", ylim=(0, 1), title="All barcodes")
            for barcode, barcode_frame in grouped.groupby("barcode", sort=True, observed=True):
                if str(barcode) not in selected:
                    continue
                axes[1].plot(
                    barcode_frame["position"],
                    barcode_frame["modified_sum"].div(barcode_frame["n_valid"].replace(0, np.nan)),
                    linewidth=0.7,
                    alpha=0.75,
                    label=str(barcode),
                )
            axes[1].set(xlabel="Reference position", ylabel="Modified fraction", ylim=(0, 1))
            axes[1].legend(frameon=False, fontsize=7, ncol=4)
            figure.suptitle(f"{reference}:{core_start}-{core_end} / {site_type}")
            figure.tight_layout()
            path = layout.categories["position_correlation"] / (
                f"{_component(reference)}__{core_start}_{core_end}__{_component(site_type)}.png"
            )
            figure.savefig(path, dpi=160)
            plt.close(figure)
            register_plot_artifact(
                layout,
                path,
                stage="spatial",
                category="position_correlation",
                plot_type="barcode_position_modification_profile",
                reference=str(reference),
                core_start=int(core_start),
                core_end=int(core_end),
            )


def _dense_product_regions(spine, bed_regions: pd.DataFrame) -> pd.DataFrame:
    """Build the portable catalog of regions eligible for dense products."""
    plans = {str(key): dict(value) for key, value in spine.uns["reference_plans"].items()}
    records = [
        {
            "reference": reference,
            "start": 0,
            "end": int(plan["reference_length"]),
            "name": reference,
            "source": "locus",
        }
        for reference, plan in plans.items()
        if str(plan["analysis_mode"]) == "locus"
    ]
    for region in bed_regions.to_dict("records"):
        reference = str(region["reference"])
        if str(plans[reference]["analysis_mode"]) != "genome":
            continue
        records.append(
            {
                "reference": reference,
                "start": int(region["start"]),
                "end": int(region["end"]),
                "name": str(region["name"]),
                "source": str(region.get("source", "bed")),
            }
        )
    regions = pd.DataFrame(
        {
            column: pd.Series(
                (record[column] for record in records),
                dtype=dtype,
            )
            for column, dtype in SPATIAL_REGION_CATALOG_DTYPES.items()
        }
    )
    return regions.sort_values(["reference", "start", "end"]).reset_index(drop=True)


def _position_matrix_estimated_bytes(width: int, *, n_methods: int, n_barcodes: int) -> int:
    """Estimate retained result and transient buffers for dense P-by-P products."""
    return int(
        max(1, width) ** 2
        * np.dtype(np.float64).itemsize
        * max(1, n_methods)
        * max(1, n_barcodes)
        * 3
    )


def _validate_position_matrix_budget(
    reference: str,
    start: int,
    end: int,
    *,
    n_methods: int,
    n_barcodes: int,
    max_width: int,
    max_bytes: int,
) -> int:
    """Return the dense-product estimate or reject an indivisible oversized region."""
    width = int(end) - int(start)
    estimated_bytes = _position_matrix_estimated_bytes(
        width, n_methods=n_methods, n_barcodes=n_barcodes
    )
    if width > int(max_width):
        raise ValueError(
            f"position matrix region {reference}:{start}-{end} has width {width:,}, "
            f"exceeding spatial_position_matrix_max_width={int(max_width):,}; "
            "narrow the plot region or disable spatial_generate_position_matrices"
        )
    if estimated_bytes > int(max_bytes):
        raise ValueError(
            f"position matrices for {reference}:{start}-{end} are estimated at "
            f"{estimated_bytes / 1024**2:.1f} MiB, exceeding "
            f"spatial_position_matrix_max_mb={int(max_bytes) / 1024**2:.1f}; "
            "narrow the plot region, request fewer methods/barcodes, or disable "
            "spatial_generate_position_matrices"
        )
    return estimated_bytes


def _write_position_matrix_sidecars(adata, output_dir: Path, region, output_key: str) -> list[Path]:
    paths = []
    reference = str(region["reference"])
    region_dir = (
        output_dir
        / SPATIAL_MATRIX_SUBDIR
        / f"reference={_component(reference)}"
        / f"region={int(region['start']):012d}-{int(region['end']):012d}"
    )
    for method, matrices in adata.uns.get(output_key, {}).items():
        for key, matrix in matrices.items():
            if not isinstance(matrix, pd.DataFrame) or matrix.empty:
                continue
            barcode = str(key[0] if isinstance(key, tuple) else key)
            path = region_dir / f"method={_component(method)}" / f"{_component(barcode)}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            matrix.rename_axis(index="position", columns=None).to_parquet(path)
            paths.append(path)
    return paths


def _plot_position_matrix_sidecars(
    adata, output_key: str, region, plot_dir: Path, cfg
) -> list[tuple[Path, str]]:
    """Render each stored barcode matrix independently for one bounded region."""
    import matplotlib.pyplot as plt

    reference = str(region["reference"])
    start, end = int(region["start"]), int(region["end"])
    methods = list(getattr(cfg, "correlation_matrix_types", ["pearson"]))
    cmaps = list(getattr(cfg, "correlation_matrix_cmaps", ["seismic"])) or ["seismic"]
    flip_axes = bool(getattr(cfg, "correlation_matrix_flip_axes", True))
    n_ticks = max(2, int(getattr(cfg, "correlation_matrix_n_ticks", 10)))
    tick_fontsize = int(getattr(cfg, "correlation_matrix_tick_fontsize", 7))
    tick_rotation = float(getattr(cfg, "correlation_matrix_tick_rotation", 90))
    plotted = []

    for method_index, method in enumerate(methods):
        method_name = str(method).lower()
        method_store = adata.uns.get(output_key, {}).get(method_name, {})
        cmap = cmaps[method_index % len(cmaps)]
        vmin, vmax = (-1.0, 1.0) if method_name == "pearson" else (0.0, 1.0)
        for key, matrix in method_store.items():
            if not isinstance(matrix, pd.DataFrame) or matrix.empty:
                continue
            barcode = str(key[0] if isinstance(key, tuple) else key)
            values = matrix.to_numpy(dtype=float)
            figure, axis = plt.subplots(figsize=(7, 6), dpi=150)
            image = axis.imshow(
                values,
                origin="upper" if flip_axes else "lower",
                aspect="auto",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            positions = np.asarray(matrix.columns)
            tick_indices = (
                np.arange(len(positions))
                if len(positions) <= n_ticks
                else np.unique(np.round(np.linspace(0, len(positions) - 1, n_ticks)).astype(int))
            )
            tick_labels = []
            for value in positions[tick_indices]:
                try:
                    numeric = float(value)
                    tick_labels.append(str(int(numeric)) if numeric.is_integer() else str(value))
                except (TypeError, ValueError):
                    tick_labels.append(str(value))
            axis.set_xticks(
                tick_indices, tick_labels, rotation=tick_rotation, fontsize=tick_fontsize
            )
            axis.set_yticks(tick_indices, tick_labels, fontsize=tick_fontsize)
            axis.set_xlabel("Reference position")
            axis.set_ylabel("Reference position")
            axis.set_title(f"{barcode} | {reference}:{start}-{end} | {method_name}")
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
            figure.tight_layout()

            path = (
                plot_dir
                / f"reference={_component(reference)}"
                / f"region={start:012d}-{end:012d}"
                / f"method={_component(method_name)}"
                / f"{_component(barcode)}.png"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, bbox_inches="tight")
            plt.close(figure)
            plotted.append((path, barcode))
    return plotted


def _generate_dense_region_products(
    spine_path, spine, plans, cfg, output_dir, layout
) -> list[Path]:
    """Generate full-locus or BED-bounded clustermaps and position matrices."""
    from ..plotting import combined_raw_clustermap

    # Imported directly from the submodule -- see the identical note in
    # tools/partitioned_hmm.py's _plot_feature_clustermaps for why the lazy
    # `from ..preprocessing import reindex_references_adata` package
    # attribute is order-dependent-fragile.
    from ..preprocessing.reindex_references_adata import reindex_references_adata
    from .position_stats import compute_positionwise_statistics

    if not plans:
        return []
    filter_mask = next(
        (column for column in ("passes_dedup", "passes_qc") if column in spine.obs),
        None,
    )
    matrix_paths = []
    sample_column = "Barcode" if "Barcode" in spine.obs else "Sample"
    minimum_reads = int(getattr(cfg, "spatial_matrix_min_reads", 2))
    requested_layer = str(getattr(cfg, "layer_for_clustermap_plotting", "nan0_0minus1"))
    selection_seed = int(getattr(cfg, "plot_subsample_seed", 0))
    matrix_max_width = int(getattr(cfg, "spatial_position_matrix_max_width", 5000))
    matrix_max_bytes = int(getattr(cfg, "spatial_position_matrix_max_mb", 1024)) * 1024**2
    for plan in plans:
        reference, start, end = plan.reference, plan.start, plan.end
        region = {
            "reference": reference,
            "start": start,
            "end": end,
            "name": plan.name,
            "source": plan.source,
        }
        region_obs = spine.obs.loc[
            (spine.obs[REFERENCE_STRAND].astype(str) == reference)
            & (spine.obs["reference_start"].astype("int64") < end)
            & (spine.obs["reference_end"].astype("int64") > start)
        ]
        if filter_mask is not None:
            region_obs = region_obs.loc[region_obs[filter_mask].astype(bool)]
        counts = region_obs[sample_column].value_counts()
        keep_samples = set(map(str, counts.loc[counts >= minimum_reads].index))
        region_obs = region_obs.loc[region_obs[sample_column].astype(str).isin(keep_samples)]
        if region_obs.empty:
            continue
        selection = select_plot_reads(
            output_dir / "read_index",
            plan,
            max_reads_per_barcode=getattr(cfg, "clustermap_max_reads_per_plot", 5000),
            seed=selection_seed,
            eligible_read_ids=region_obs.index,
        )
        if not selection.read_ids:
            continue
        if bool(getattr(cfg, "spatial_generate_position_matrices", True)):
            methods = list(getattr(cfg, "correlation_matrix_types", ["pearson"]))
            # compute_positionwise_statistics retains every method/barcode DataFrame
            # in adata.uns until sidecars are published. Count three float64-sized
            # buffers per matrix for the result plus correlation/covariance work arrays.
            estimated_matrix_bytes = _validate_position_matrix_budget(
                reference,
                start,
                end,
                n_methods=len(methods),
                n_barcodes=len(keep_samples),
                max_width=matrix_max_width,
                max_bytes=matrix_max_bytes,
            )
            from ..memory_guard import require_memory_headroom

            require_memory_headroom(
                cfg,
                estimated_memory_mb=estimated_matrix_bytes / 1024**2,
                operation_label=f"spatial position matrices {reference}:{start}-{end}",
                estimator="spatial_position_matrix_peak",
            )
        layers = sorted({"nan0_0minus1", requested_layer})
        adata = materialize(
            spine_path,
            references=reference,
            read_ids=selection.read_ids,
            start=start,
            end=end,
            layers=layers,
        )
        mask_unanalyzed_gaps(adata, plan.gaps)
        # Materialization preserves global category levels from the spine; dense
        # region routines must only iterate groups actually present in this slice.
        adata.obs[sample_column] = adata.obs[sample_column].astype(str).astype("category")
        adata.obs[REFERENCE_STRAND] = adata.obs[REFERENCE_STRAND].astype(str).astype("category")
        region_label = f"{_component(reference)}__{start}_{end}"

        # Ported from the legacy (non-partitioned) pipeline, where this ran
        # once on the whole-experiment adata (cli/spatial_adata.py). Purely
        # additive (writes a new var column, never touches X/layers), so it's
        # safe to run per region materialization here. Computed once,
        # unconditionally, since both the clustermap and position-matrix
        # blocks below consume index_suffix independently of one another.
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

        if bool(getattr(cfg, "spatial_generate_clustermaps", True)):
            clustermap_dir = layout.categories["clustermaps"] / region_label
            combined_raw_clustermap(
                adata,
                sample_col=sample_column,
                reference_col=REFERENCE_STRAND,
                mod_target_bases=list(getattr(cfg, "mod_target_bases", ["GpC", "CpG"])),
                layer_c=requested_layer,
                layer_gpc=requested_layer,
                layer_cpg=requested_layer,
                layer_a=requested_layer,
                cmap_c=str(getattr(cfg, "clustermap_cmap_c", "coolwarm")),
                cmap_gpc=str(getattr(cfg, "clustermap_cmap_gpc", "coolwarm")),
                cmap_cpg=str(getattr(cfg, "clustermap_cmap_cpg", "viridis")),
                cmap_a=str(getattr(cfg, "clustermap_cmap_a", "coolwarm")),
                min_quality=None,
                min_length=None,
                min_mapped_length_to_reference_length_ratio=None,
                min_position_valid_fraction=None,
                demux_types=None,
                save_path=clustermap_dir,
                sort_by=str(getattr(cfg, "spatial_clustermap_sortby", "gpc")),
                deaminase=str(getattr(cfg, "smf_modality", "conversion")) != "conversion",
                n_jobs=max(1, int(getattr(cfg, "threads", 1) or 1)),
                restrict_to_read_span=bool(
                    getattr(cfg, "spatial_clustermap_restrict_to_read_span", False)
                ),
                max_reads_per_plot=getattr(cfg, "clustermap_max_reads_per_plot", 5000),
                index_col_suffix=index_suffix,
                cfg=cfg,
            )
            for path in sorted(clustermap_dir.glob("*.png")):
                source_manifest = write_plot_source_manifest(
                    layout,
                    path,
                    stage="spatial",
                    plot_type="barcode_region_clustermap",
                    region=plan.source_manifest(),
                    layers=layers,
                    selection_seed=selection.seed,
                    selection_sha256=selection.selection_sha256,
                    selected_molecule_uids=selection.molecule_uids,
                )
                register_plot_artifact(
                    layout,
                    path,
                    stage="spatial",
                    category="clustermaps",
                    plot_type="barcode_region_clustermap",
                    reference=reference,
                    core_start=start,
                    core_end=end,
                    source_manifest=source_manifest,
                )

        if bool(getattr(cfg, "spatial_generate_position_matrices", True)):
            output_key = f"positionwise_{reference}_{start}_{end}"
            methods = list(getattr(cfg, "correlation_matrix_types", ["pearson"]))
            compute_positionwise_statistics(
                adata,
                layer="nan0_0minus1",
                methods=methods,
                sample_col=sample_column,
                ref_col=REFERENCE_STRAND,
                output_key=output_key,
                site_types=list(getattr(cfg, "correlation_matrix_site_types", ["GpC_site"])),
                encoding="signed",
                max_threads=max(1, int(getattr(cfg, "threads", 1) or 1)),
                min_count_for_pairwise=10,
                min_position_valid_fraction=None,
                index_col_suffix=index_suffix,
            )
            matrix_paths.extend(
                _write_position_matrix_sidecars(adata, output_dir, region, output_key)
            )
            matrix_plots = _plot_position_matrix_sidecars(
                adata,
                output_key,
                region,
                layout.categories["position_correlation"],
                cfg,
            )
            for path, barcode in matrix_plots:
                source_manifest = write_plot_source_manifest(
                    layout,
                    path,
                    stage="spatial",
                    plot_type="barcode_region_position_matrix",
                    region=plan.source_manifest(),
                    layers=["nan0_0minus1"],
                    selection_seed=selection.seed,
                    selection_sha256=selection.selection_sha256,
                    selected_molecule_uids=selection.molecule_uids,
                )
                register_plot_artifact(
                    layout,
                    path,
                    stage="spatial",
                    category="position_correlation",
                    plot_type="barcode_region_position_matrix",
                    reference=reference,
                    sample=barcode,
                    core_start=start,
                    core_end=end,
                    source_manifest=source_manifest,
                )
    return matrix_paths


def _write_read_metric_axes(output_dir: Path, cfg) -> tuple[Path | None, Path | None]:
    """Write stage-wide coordinate axes shared by all read-level array sidecars."""
    autocorrelation_axis = None
    if bool(getattr(cfg, "spatial_save_read_autocorrelation", True)):
        autocorrelation_axis = output_dir / SPATIAL_READ_AUTOCORRELATION_AXIS
        pd.DataFrame(
            {"lag_bp": np.arange(int(getattr(cfg, "autocorr_max_lag", 800)) + 1)}
        ).to_parquet(autocorrelation_axis, index=False)

    periodogram_axis = None
    if bool(getattr(cfg, "spatial_compute_read_lomb_scargle", True)):
        period_min, period_max = _config_range(
            cfg, "spatial_lomb_scargle_period_range_bp", (80.0, 400.0)
        )
        periods = np.arange(period_max, period_min - 1, -1, dtype=float)
        periodogram_axis = output_dir / SPATIAL_READ_PERIODOGRAM_AXIS
        pd.DataFrame(
            {
                "frequency_per_bp": 1.0 / periods,
                "period_bp": periods,
            }
        ).to_parquet(periodogram_axis, index=False)
    return autocorrelation_axis, periodogram_axis


def execute_partitioned_spatial(spine_path, cfg, output_dir) -> dict[str, Path]:
    """Run bounded spatial analysis and publish a linked thin spine."""
    spine_path = Path(spine_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    from ..informatics.derived_read_index import prepare_derived_read_index

    prepare_derived_read_index(output_dir)
    spine = load_spine(spine_path)
    filter_mask = next(
        (column for column in ("passes_dedup", "passes_qc") if column in spine.obs),
        None,
    )
    tasks, bed_regions = _region_tasks(
        spine,
        cfg,
        filter_mask,
        spine_path=spine_path,
    )
    if not tasks:
        raise RuntimeError("partitioned spatial analysis has no non-empty tasks")
    from ..memory_guard import require_memory_headroom, run_tasks_parallel

    records = run_tasks_parallel(
        execute_spatial_task,
        [(spine_path, task, cfg, output_dir) for task in tasks],
        cfg=cfg,
        pool_label=f"spatial task pool ({len(tasks)} tasks)",
        per_item_memory_mb=max(task.estimated_memory_bytes for task in tasks) / (1024**2),
        estimator="spatial_task_plan_peak",
    )
    task_catalog = output_dir / SPATIAL_TASK_CATALOG
    record_frame = pd.DataFrame(records)
    record_frame.to_parquet(task_catalog, index=False)
    max_lag = int(getattr(cfg, "autocorr_max_lag", 800)) + 1
    max_periods = int(
        _config_range(cfg, "spatial_lomb_scargle_period_range_bp", (80.0, 400.0))[1]
        - _config_range(cfg, "spatial_lomb_scargle_period_range_bp", (80.0, 400.0))[0]
        + 1
    )
    site_type_count = max((len(record.get("site_types", [])) for record in records), default=1)
    plot_group_columns = ["reference", "core_start", "core_end", "barcode"]
    plot_group_count = (
        len(record_frame.drop_duplicates(plot_group_columns)) if not record_frame.empty else 0
    )
    estimated_reducer_bytes = len(records) * site_type_count * max_lag * 128
    require_memory_headroom(
        cfg,
        estimated_memory_mb=max(1, estimated_reducer_bytes) / 1024**2,
        operation_label="spatial reducers",
        estimator="spatial_reducer_peak",
    )
    read_autocorrelation_axis, read_periodogram_axis = _write_read_metric_axes(output_dir, cfg)
    metrics_path, autocorrelation_path = _reduce_metrics(records, output_dir)
    legacy_dense_regions = _dense_product_regions(spine, bed_regions)
    plot_plans = resolve_plot_region_plans(
        spine,
        record_frame,
        spine_path=spine_path,
        allow_gaps=bool(getattr(cfg, "plot_allow_unanalyzed_gaps", False)),
        fallback_regions=legacy_dense_regions,
    )
    dense_region_records = [
        {
            "reference": plan.reference,
            "start": plan.start,
            "end": plan.end,
            "name": plan.name,
            "source": plan.source,
        }
        for plan in plot_plans
    ]
    dense_regions = pd.DataFrame(
        {
            column: pd.Series((record[column] for record in dense_region_records), dtype=dtype)
            for column, dtype in SPATIAL_REGION_CATALOG_DTYPES.items()
        }
    )
    region_catalog = output_dir / SPATIAL_REGION_CATALOG
    dense_regions.to_parquet(region_catalog, index=False)

    layout = prepare_analysis_plot_layout(
        output_dir,
        stage="spatial",
        source_spine=spine_path,
    )
    pd.DataFrame(columns=PLOT_CATALOG_COLUMNS).to_parquet(layout.catalog, index=False)
    require_memory_headroom(
        cfg,
        estimated_memory_mb=(
            max(1, int(getattr(cfg, "clustermap_max_reads_per_plot", 5000) or 5000))
            * max(1, plot_group_count)
            * site_type_count
            * (max_lag + max_periods)
            * np.dtype(np.float64).itemsize
            * 2
            / 1024**2
        ),
        operation_label="spatial plots",
        estimator="spatial_plot_peak",
    )
    autocorrelation = pd.read_parquet(autocorrelation_path)
    _plot_autocorrelation(autocorrelation, layout)
    _plot_read_periodicity(records, output_dir, layout, cfg)
    _plot_read_metric_clustermaps(records, output_dir, layout, cfg)
    _plot_position_profiles(records, output_dir, layout)
    matrix_paths = _generate_dense_region_products(
        spine_path,
        spine,
        plot_plans,
        cfg,
        output_dir,
        layout,
    )

    spatial_spine = spine.copy()
    # Relative to the run's output_directory, not output_dir -- see
    # informatics.partition_read._run_root_from_spine_path.
    run_root = output_dir.parent
    spatial_spine.uns["spatial_source_spine"] = relative_uns_path(spine_path, run_root)
    spatial_spine.uns["spatial_task_catalog"] = relative_uns_path(task_catalog, run_root)
    spatial_spine.uns["spatial_metrics"] = relative_uns_path(metrics_path, run_root)
    spatial_spine.uns["spatial_autocorrelation"] = relative_uns_path(autocorrelation_path, run_root)
    spatial_spine.uns["spatial_position_store"] = relative_uns_path(
        output_dir / SPATIAL_PARTIAL_SUBDIR, run_root
    )
    spatial_spine.uns["spatial_task_store"] = relative_uns_path(
        output_dir / SPATIAL_PARTIAL_SUBDIR, run_root
    )
    spatial_spine.uns["spatial_region_catalog"] = relative_uns_path(region_catalog, run_root)
    spatial_spine.uns["spatial_matrix_store"] = relative_uns_path(
        output_dir / SPATIAL_MATRIX_SUBDIR, run_root
    )
    if read_autocorrelation_axis is not None:
        spatial_spine.uns["spatial_read_autocorrelation_axis"] = relative_uns_path(
            read_autocorrelation_axis, run_root
        )
    if read_periodogram_axis is not None:
        spatial_spine.uns["spatial_read_periodogram_axis"] = relative_uns_path(
            read_periodogram_axis, run_root
        )
    spatial_spine.uns["spatial_filter_mask"] = filter_mask or ""
    from ..informatics.derived_read_index import DERIVED_READ_INDEX_DIRNAME

    read_index_dir = output_dir / DERIVED_READ_INDEX_DIRNAME
    spatial_spine.uns["spatial_read_index"] = relative_uns_path(read_index_dir, run_root)
    spatial_spine.uns["spatial_schema_version"] = 3
    output_spine = output_dir / SPATIAL_SPINE_FILENAME
    safe_write_h5ad(spatial_spine, output_spine, backup=False, verbose=False)
    write_experiment_spine(run_root)

    manifest = sidecar_manifest_path(output_dir)
    register_sidecar(manifest, "spatial_spine", output_spine)
    register_sidecar(manifest, "spatial_source_spine", spine_path)
    register_sidecar(manifest, "spatial_task_catalog", task_catalog)
    register_sidecar(manifest, "spatial_read_index", read_index_dir)
    register_sidecar(manifest, "spatial_metrics", metrics_path)
    register_sidecar(manifest, "spatial_autocorrelation", autocorrelation_path)
    register_sidecar(manifest, "spatial_position_store", output_dir / SPATIAL_PARTIAL_SUBDIR)
    register_sidecar(
        manifest,
        "spatial_task_store",
        output_dir / SPATIAL_PARTIAL_SUBDIR,
        metadata={
            "format": "partitioned_parquet",
            "products": [
                "metrics",
                "autocorrelation",
                "positions",
                "read_metrics.zarr",
            ],
            "task_catalog": str(task_catalog),
        },
    )
    register_sidecar(manifest, "spatial_region_catalog", region_catalog)
    register_sidecar(
        manifest,
        "spatial_matrix_store",
        output_dir / SPATIAL_MATRIX_SUBDIR,
        metadata={"matrices": len(matrix_paths)},
    )
    if read_autocorrelation_axis is not None:
        register_sidecar(manifest, "spatial_read_autocorrelation_axis", read_autocorrelation_axis)
    if read_periodogram_axis is not None:
        register_sidecar(manifest, "spatial_read_periodogram_axis", read_periodogram_axis)
    register_sidecar(manifest, "spatial_plot_catalog", layout.catalog)
    logger.info("Wrote partitioned spatial stage with %d task(s)", len(tasks))
    outputs = {
        "spine": output_spine,
        "task_catalog": task_catalog,
        "read_index": read_index_dir,
        "metrics": metrics_path,
        "autocorrelation": autocorrelation_path,
        "task_store": output_dir / SPATIAL_PARTIAL_SUBDIR,
        "position_store": output_dir / SPATIAL_PARTIAL_SUBDIR,
        "region_catalog": region_catalog,
        "matrix_store": output_dir / SPATIAL_MATRIX_SUBDIR,
        "plots": layout.root,
        "plot_catalog": layout.catalog,
        "manifest": manifest,
    }
    if read_autocorrelation_axis is not None:
        outputs["read_autocorrelation_axis"] = read_autocorrelation_axis
    if read_periodogram_axis is not None:
        outputs["read_periodogram_axis"] = read_periodogram_axis
    return outputs
