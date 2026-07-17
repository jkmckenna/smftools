"""Automated, bounded diagnostics for partitioned preprocessing outputs."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from smftools.constants import BASE_QUALITY_SCORES, READ_SPAN_MASK, REFERENCE_STRAND


def _register(layout, path, category, plot_type, **metadata):
    from ..cli.stage_artifacts import register_plot_artifact

    register_plot_artifact(
        layout,
        path,
        stage="preprocess",
        category=category,
        plot_type=plot_type,
        **metadata,
    )


def _sample_column(obs: pd.DataFrame) -> str:
    for column in ("Experiment_name_and_barcode", "Sample", "Barcode"):
        if column in obs:
            return column
    raise KeyError("preprocessing obs lacks a sample/barcode column")


def _barcode_column(obs: pd.DataFrame) -> str:
    for column in ("Barcode", "barcode", "Sample", "Experiment_name_and_barcode"):
        if column in obs:
            return column
    raise KeyError("preprocessing obs lacks a barcode/sample column")


def _metric_dashboard(obs, columns, pass_column, path, title):
    import matplotlib.pyplot as plt

    columns = [column for column in columns if column in obs]
    if not columns:
        return False
    n_columns = min(3, len(columns))
    n_rows = math.ceil(len(columns) / n_columns)
    figure, axes = plt.subplots(n_rows, n_columns, figsize=(5 * n_columns, 3.4 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    passed = obs[pass_column].astype(bool)
    for axis, column in zip(axes, columns):
        for label, mask, color in (
            ("pass", passed, "#197278"),
            ("fail", ~passed, "#d95d39"),
        ):
            values = pd.to_numeric(obs.loc[mask, column], errors="coerce").dropna()
            if not values.empty:
                axis.hist(values, bins=45, alpha=0.6, label=label, color=color)
        axis.set(title=column, ylabel="Reads")
    for axis in axes[len(columns) :]:
        axis.set_visible(False)
    axes[0].legend(frameon=False)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return True


def _barcode_reference_summary(
    obs: pd.DataFrame,
    barcode_column: str,
    metric_columns: list[str],
    complexity_results: dict | None,
) -> pd.DataFrame:
    """Return one native-unit summary row per barcode and mapped reference."""
    complexity = {}
    for key, result in (complexity_results or {}).items():
        if not isinstance(key, tuple) or len(key) != 2:
            continue
        complexity[(str(key[0]), str(key[1]))] = result

    records = []
    grouped = obs.groupby([barcode_column, REFERENCE_STRAND], sort=True, observed=True)
    for (barcode, reference), group in grouped:
        record = {
            "barcode": str(barcode),
            "reference": str(reference),
            "n_reads": int(len(group)),
        }
        for column in (
            "passes_read_qc",
            "passes_modification_qc",
            "passes_qc",
            "passes_dedup",
        ):
            if column in group:
                values = group[column].astype(bool)
                label = column.removeprefix("passes_")
                record[f"n_{label}_pass"] = int(values.sum())
                record[f"{label}_pass_fraction"] = float(values.mean())

        qc_mask = (
            group["passes_qc"].astype(bool)
            if "passes_qc" in group
            else pd.Series(True, index=group.index)
        )
        duplicate_values = (
            group.loc[qc_mask, "is_duplicate"].astype(bool)
            if "is_duplicate" in group
            else pd.Series(dtype=bool)
        )
        record["n_qc_reads_for_duplicate_detection"] = int(qc_mask.sum())
        record["n_duplicate_reads"] = int(duplicate_values.sum())
        record["duplicate_fraction"] = (
            float(duplicate_values.mean()) if not duplicate_values.empty else np.nan
        )

        for column in metric_columns:
            if column not in group:
                continue
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            if values.empty:
                continue
            record[f"{column}_q25"] = float(values.quantile(0.25))
            record[f"{column}_median"] = float(values.median())
            record[f"{column}_q75"] = float(values.quantile(0.75))

        result = complexity.get((str(barcode), str(reference)))
        if result is not None:
            record["library_complexity_estimate"] = float(result["C0"])
            record["n_unique_molecules_observed"] = int(result["n_unique"])
            record["complexity_reads_used"] = int(result["n_reads"])
        records.append(record)
    return pd.DataFrame.from_records(records).sort_values(["reference", "barcode"])


def _barcode_reference_overviews(summary, layout):
    import matplotlib.pyplot as plt

    for reference, frame in summary.groupby("reference", sort=True, observed=True):
        frame = frame.sort_values("barcode").reset_index(drop=True)
        barcodes = frame["barcode"].astype(str)
        y = np.arange(len(frame))
        figure, axes = plt.subplots(
            3,
            2,
            figsize=(15, max(9, 0.38 * len(frame))),
            sharey=True,
        )

        count_columns = [
            ("n_reads", "input", "#264653"),
            ("n_qc_pass", "QC pass", "#2a9d8f"),
            ("n_dedup_pass", "deduplicated", "#e9c46a"),
        ]
        available_counts = [item for item in count_columns if item[0] in frame]
        height = 0.75 / max(1, len(available_counts))
        for index, (column, label, color) in enumerate(available_counts):
            axes[0, 0].barh(
                y + (index - (len(available_counts) - 1) / 2) * height,
                frame[column].fillna(0),
                height=height,
                label=label,
                color=color,
            )
        axes[0, 0].set(title="Read counts", xlabel="Reads")
        axes[0, 0].legend(frameon=False, fontsize=8)

        retention_columns = [
            ("read_qc_pass_fraction", "read QC"),
            ("modification_qc_pass_fraction", "modification QC"),
            ("qc_pass_fraction", "combined QC"),
            ("dedup_pass_fraction", "after deduplication"),
        ]
        for column, label in retention_columns:
            if column in frame:
                axes[0, 1].scatter(frame[column], y, s=20, label=label)
        axes[0, 1].set(title="Retention", xlabel="Fraction of input reads", xlim=(0, 1.02))
        axes[0, 1].legend(frameon=False, fontsize=8)

        for column, label, color in (
            ("read_quality_median", "read quality", "#277da1"),
            ("mapping_quality_median", "mapping quality", "#f9844a"),
        ):
            if column in frame:
                axes[1, 0].scatter(frame[column], y, s=20, label=label, color=color)
        axes[1, 0].set(title="Median quality scores", xlabel="Phred score")
        axes[1, 0].legend(frameon=False, fontsize=8)

        for column, label, color in (
            ("read_length_median", "read length", "#577590"),
            ("mapped_length_median", "mapped length", "#90be6d"),
        ):
            if column in frame:
                axes[1, 1].scatter(frame[column], y, s=20, label=label, color=color)
        axes[1, 1].set(title="Median molecule lengths", xlabel="Bases")
        axes[1, 1].legend(frameon=False, fontsize=8)

        axes[2, 0].barh(y, frame["duplicate_fraction"].fillna(0), color="#e76f51")
        axes[2, 0].set(
            title="Duplicate fraction among QC-passing reads",
            xlabel="Duplicate fraction",
            xlim=(0, 1.02),
        )

        complexity_columns = [
            ("n_unique_molecules_observed", "observed unique", "#43aa8b"),
            ("library_complexity_estimate", "fitted complexity", "#f9c74f"),
        ]
        available_complexity = [item for item in complexity_columns if item[0] in frame]
        height = 0.75 / max(1, len(available_complexity))
        for index, (column, label, color) in enumerate(available_complexity):
            axes[2, 1].barh(
                y + (index - (len(available_complexity) - 1) / 2) * height,
                frame[column].fillna(0),
                height=height,
                label=label,
                color=color,
            )
        axes[2, 1].set(title="Library complexity", xlabel="Molecules")
        if available_complexity:
            axes[2, 1].legend(frameon=False, fontsize=8)

        for axis in axes[:, 0]:
            axis.set_yticks(y, barcodes, fontsize=7)
        figure.suptitle(f"Barcode summary / {reference}")
        figure.tight_layout()
        safe_reference = str(reference).replace("/", "_")
        path = layout.categories["barcode_summary"] / f"{safe_reference}__overview.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        _register(
            layout,
            path,
            "barcode_summary",
            "barcode_reference_overview",
            reference=str(reference),
        )


def _barcode_distribution_plots(
    obs, barcode_column, metrics, layout, *, category, plot_type, swarm_max_points=4000
):
    import matplotlib.pyplot as plt

    metrics = [column for column in metrics if column in obs]
    if not metrics:
        return
    rng = np.random.default_rng(0)
    for reference, reference_obs in obs.groupby(REFERENCE_STRAND, sort=True, observed=True):
        barcodes = sorted(reference_obs[barcode_column].dropna().astype(str).unique())
        n_columns = min(2, len(metrics))
        n_rows = math.ceil(len(metrics) / n_columns)
        figure, axes = plt.subplots(
            n_rows,
            n_columns,
            figsize=(max(13, 0.45 * len(barcodes)), 4 * n_rows),
            squeeze=False,
        )
        axes = axes.ravel()
        for axis, metric in zip(axes, metrics):
            distributions = [
                pd.to_numeric(
                    reference_obs.loc[reference_obs[barcode_column].astype(str) == barcode, metric],
                    errors="coerce",
                ).dropna()
                for barcode in barcodes
            ]
            # A KDE (what violinplot draws) is undefined for <2 points or
            # zero-variance data, so those positions are dropped from the
            # violin call and rely on the swarm overlay alone -- xticks are
            # still set for every barcode below so positions stay aligned.
            violin_positions = [
                position
                for position, values in enumerate(distributions, start=1)
                if len(values) >= 2 and values.std() > 0
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
                    body.set_facecolor("#a8dadc")
                    body.set_edgecolor("#457b9d")
                    body.set_alpha(0.7)
                    body.set_zorder(2)
                parts["cmedians"].set_color("#c1121f")
                parts["cmedians"].set_linewidth(1.2)
                parts["cmedians"].set_zorder(2.5)
            # Jittered per-read points on top of each violin -- a true swarmplot
            # (non-overlapping points) doesn't scale to the read counts seen
            # here, so this uses random x-jitter with low alpha instead,
            # rasterized so the saved PNG stays a reasonable size at high
            # point counts. Subsampled per box past swarm_max_points so a
            # single deeply-sequenced barcode can't blow up render time.
            for position, values in enumerate(distributions, start=1):
                if len(values) == 0:
                    continue
                if len(values) > swarm_max_points:
                    values = values.sample(swarm_max_points, random_state=0)
                jitter = rng.uniform(-0.2, 0.2, size=len(values))
                axis.scatter(
                    np.full(len(values), position) + jitter,
                    values,
                    s=3,
                    alpha=0.3,
                    color="#1d3557",
                    edgecolors="none",
                    rasterized=True,
                    zorder=3,
                )
            axis.set_xticks(range(1, len(barcodes) + 1))
            axis.set_xticklabels(barcodes)
            axis.tick_params(axis="x", labelrotation=90, labelsize=6)
            axis.set(title=metric, ylabel=metric)
            if metric.startswith("Fraction_") or metric.endswith("_ratio"):
                axis.set_ylim(0, 1.05)
        for axis in axes[len(metrics) :]:
            axis.set_visible(False)
        figure.suptitle(f"Barcode distributions / {reference}")
        figure.tight_layout()
        safe_reference = str(reference).replace("/", "_")
        path = layout.categories[category] / f"{safe_reference}__barcode_distributions.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        _register(
            layout,
            path,
            category,
            plot_type,
            reference=str(reference),
        )


def _duplicate_diagnostics(obs, sample_column, layout, threshold):
    import matplotlib.pyplot as plt

    if "passes_qc" in obs:
        obs = obs.loc[obs["passes_qc"].astype(bool)].copy()
    hamming = "sequence__min_hamming_to_pair"
    distance_columns = {
        "minimum": hamming,
        "forward neighbor": "fwd_hamming_to_next",
        "reverse neighbor": "rev_hamming_to_prev",
        "hierarchical neighbor": "sequence__hier_hamming_to_pair",
    }
    colors = {
        "minimum": "#6d597a",
        "forward neighbor": "#2a9d8f",
        "reverse neighbor": "#e9c46a",
        "hierarchical neighbor": "#e76f51",
    }
    references = sorted(obs[REFERENCE_STRAND].astype(str).unique())
    if not references:
        return False
    if hamming in obs:
        n_columns = min(3, len(references))
        n_rows = math.ceil(len(references) / n_columns)
        figure, axes = plt.subplots(n_rows, n_columns, figsize=(5 * n_columns, 3.4 * n_rows))
        axes = np.atleast_1d(axes).ravel()
        for axis, reference in zip(axes, references):
            reference_obs = obs.loc[obs[REFERENCE_STRAND].astype(str) == reference]
            distance_values = {}
            for label, column in distance_columns.items():
                if column not in reference_obs:
                    continue
                values = pd.to_numeric(reference_obs[column], errors="coerce").dropna()
                if not values.empty:
                    distance_values[label] = values
            maximum = max(
                (float(values.max()) for values in distance_values.values()),
                default=threshold,
            )
            axis_maximum = min(1.0, max(0.1, maximum * 1.05, threshold * 1.2))
            bins = np.linspace(0, axis_maximum, 46)
            for label, values in distance_values.items():
                axis.hist(
                    values,
                    bins=bins,
                    histtype="step",
                    linewidth=1.4,
                    color=colors[label],
                    label=label,
                )
            axis.axvline(threshold, color="#c1121f", linestyle="--", linewidth=1)
            axis.set(title=reference, xlabel="Hamming distance", ylabel="Reads")
            axis.set_xlim(0, axis_maximum)
            axis.legend(frameon=False, fontsize=7)
        for axis in axes[len(references) :]:
            axis.set_visible(False)
        figure.suptitle("Nearest-read Hamming distance by reference")
        figure.tight_layout()
        path = layout.categories["duplicate_qc"] / "hamming_distance_by_reference.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        _register(layout, path, "duplicate_qc", "hamming_distance_by_reference")

        metric = next(
            (
                column
                for column in ("Fraction_C_site_modified", "Raw_modification_signal")
                if column in obs
            ),
            None,
        )
        if metric:
            frame = obs[[hamming, metric, "is_duplicate"]].dropna().copy()
            if len(frame) > 20_000:
                frame = frame.sample(20_000, random_state=0)
            figure, axis = plt.subplots(figsize=(7, 5))
            for duplicate, color, label in (
                (False, "#457b9d", "unique"),
                (True, "#e63946", "duplicate"),
            ):
                subset = frame.loc[frame["is_duplicate"].astype(bool) == duplicate]
                axis.scatter(
                    subset[hamming], subset[metric], s=8, alpha=0.35, color=color, label=label
                )
            axis.axvline(threshold, color="#c1121f", linestyle="--", linewidth=1)
            axis.set(
                xlabel="Minimum Hamming distance",
                ylabel=metric,
                title="Hamming distance vs read metric",
            )
            axis.legend(frameon=False)
            figure.tight_layout()
            path = layout.categories["duplicate_qc"] / "hamming_vs_read_metric.png"
            figure.savefig(path, dpi=150)
            plt.close(figure)
            _register(layout, path, "duplicate_qc", "hamming_vs_read_metric")

    if {"duplicate_cluster_id", "duplicate_cluster_size"}.issubset(obs):
        clusters = obs.loc[obs["duplicate_cluster_id"] >= 0].drop_duplicates("duplicate_cluster_id")
        sizes = pd.to_numeric(clusters["duplicate_cluster_size"], errors="coerce").dropna()
        if not sizes.empty:
            max_size = max(1, int(sizes.max()))
            bins = np.arange(0.5, max_size + 1.5)
            figure, axis = plt.subplots(figsize=(7, 5))
            axis.hist(sizes, bins=bins, color="#457b9d", edgecolor="white")
            axis.set(
                xlabel="Reads per duplicate cluster",
                ylabel="Clusters",
                title="Duplicate cluster-size distribution",
                yscale="log" if len(sizes) > 10 else "linear",
            )
            figure.tight_layout()
            path = layout.categories["duplicate_qc"] / "duplicate_cluster_sizes.png"
            figure.savefig(path, dpi=150)
            plt.close(figure)
            _register(layout, path, "duplicate_qc", "duplicate_cluster_size_histogram")

    rates = obs.pivot_table(
        index=sample_column,
        columns=REFERENCE_STRAND,
        values="is_duplicate",
        aggfunc="mean",
        observed=True,
    )
    figure, axis = plt.subplots(figsize=(9, max(5, 0.22 * len(rates))))
    image = axis.imshow(rates.fillna(0).to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="magma")
    axis.set_xticks(range(len(rates.columns)), rates.columns.astype(str), rotation=35, ha="right")
    axis.set_yticks(range(len(rates)), rates.index.astype(str), fontsize=6)
    axis.set_title("Duplicate fraction by sample and reference")
    figure.colorbar(image, ax=axis, label="Duplicate fraction", shrink=0.8)
    figure.tight_layout()
    path = layout.categories["duplicate_qc"] / "duplicate_rate_by_sample_reference.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    _register(layout, path, "duplicate_qc", "duplicate_rate_by_sample_reference")


def _complexity_plots(obs, sample_column, layout, cfg):
    import anndata as ad

    from .calculate_complexity_II import calculate_complexity_II

    output = layout.categories["library_complexity"]
    output.mkdir(parents=True, exist_ok=True)
    if "passes_qc" in obs:
        obs = obs.loc[obs["passes_qc"].astype(bool)].copy()
    adata = ad.AnnData(obs=obs.set_index("read_id", drop=False))
    results = calculate_complexity_II(
        adata,
        output_directory=output,
        sample_col=sample_column,
        ref_col=REFERENCE_STRAND,
        cluster_col="duplicate_cluster_id",
        plot=True,
        save_plot=True,
        n_boot=int(getattr(cfg, "preprocess_plot_complexity_bootstraps", 20)),
        n_depths=10,
        random_state=0,
        csv_summary=True,
        force_redo=True,
        bypass=bool(getattr(cfg, "bypass_complexity_analysis", False)),
    )
    if not results:
        return results
    for key in results:
        group_label = "__".join(map(str, key if isinstance(key, tuple) else (key,)))
        safe_label = "".join(
            character if character.isalnum() or character in "-._" else "_"
            for character in group_label
        )
        path = output / f"complexity_{safe_label}.png"
        if path.exists():
            _register(layout, path, "library_complexity", "library_complexity_curve")
    return results


def _read_span_quality_plots(spine_path, layout, max_reads=300, max_positions=1800):
    import matplotlib.pyplot as plt

    from ..cli.stage_input import iter_stage_slices

    for stage_slice in iter_stage_slices(
        spine_path,
        layers=[READ_SPAN_MASK, BASE_QUALITY_SCORES],
    ):
        core = stage_slice.core()
        sample_column = _sample_column(core.obs)
        for sample, sample_obs in core.obs.groupby(sample_column, sort=True, observed=True):
            read_ids = list(map(str, sample_obs.index))
            if len(read_ids) > max_reads:
                indices = np.linspace(0, len(read_ids) - 1, max_reads, dtype=int)
                read_ids = [read_ids[index] for index in indices]
            subset = core[read_ids]
            column_step = max(1, math.ceil(subset.n_vars / max_positions))
            positions = np.asarray(subset.var_names, dtype=np.int64)[::column_step]
            span = np.asarray(subset.layers[READ_SPAN_MASK])[:, ::column_step]
            quality = np.asarray(subset.layers[BASE_QUALITY_SCORES], dtype=float)[:, ::column_step]
            quality[(span == 0) | (quality < 0)] = np.nan
            quality_count = np.sum(~np.isnan(quality), axis=0)
            quality_mean = np.divide(
                np.nansum(quality, axis=0),
                quality_count,
                out=np.full(quality.shape[1], np.nan, dtype=float),
                where=quality_count != 0,
            )
            figure, axes = plt.subplots(2, 2, figsize=(13, 7), height_ratios=(1, 4))
            axes[0, 0].plot(positions, span.mean(axis=0), color="#2a9d8f", linewidth=0.8)
            axes[0, 0].set(title="Mean read span", ylim=(0, 1.02))
            axes[0, 1].plot(positions, quality_mean, color="#3a86ff", linewidth=0.8)
            axes[0, 1].set(title="Mean base quality")
            axes[1, 0].imshow(
                span, aspect="auto", interpolation="nearest", cmap="Greens", vmin=0, vmax=1
            )
            image = axes[1, 1].imshow(
                quality, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0, vmax=50
            )
            axes[1, 0].set(
                xlabel="Reference position", ylabel="Sampled reads", title="Read-span mask"
            )
            axes[1, 1].set(xlabel="Reference position", title="Base-quality scores")
            figure.colorbar(image, ax=axes[1, 1], label="Phred score", shrink=0.8)
            figure.suptitle(f"{sample} / {stage_slice.reference} ({len(sample_obs)} reads)")
            figure.tight_layout()
            core_suffix = (
                ""
                if stage_slice.analysis_mode == "locus"
                else f"__{stage_slice.core_start}_{stage_slice.core_end}"
            )
            path = layout.categories["read_span_quality"] / (
                f"{str(stage_slice.reference).replace('/', '_')}__"
                f"{str(sample).replace('/', '_')}{core_suffix}.png"
            )
            figure.savefig(path, dpi=140)
            plt.close(figure)
            _register(
                layout,
                path,
                "read_span_quality",
                "read_span_base_quality",
                reference=stage_slice.reference,
                sample=str(sample),
                core_start=stage_slice.core_start,
                core_end=stage_slice.core_end,
            )


def generate_preprocess_summary_plots(
    obs_path,
    var_path,
    plot_layout,
    *,
    cfg=None,
    spine_path=None,
) -> None:
    """Generate global, sample-stratified, and bounded matrix diagnostics."""
    import matplotlib.pyplot as plt

    obs = pd.read_parquet(obs_path)
    var = pd.read_parquet(var_path)
    barcode_column = _barcode_column(obs)
    read_metrics = [
        "read_length",
        "mapped_length",
        "read_quality",
        "mapping_quality",
        "mapped_length_to_reference_length_ratio",
        "mapped_length_to_read_length_ratio",
    ]
    modification_metrics = [
        "Raw_modification_signal",
        *[
            column
            for column in obs
            if column.startswith("Fraction_") and column.endswith("_modified")
        ],
    ]

    path = plot_layout.categories["read_qc"] / "read_qc_metric_dashboard.png"
    if _metric_dashboard(obs, read_metrics, "passes_read_qc", path, "Read QC: pass vs fail"):
        _register(plot_layout, path, "read_qc", "read_qc_metric_dashboard")
    path = plot_layout.categories["modification_qc"] / "modification_qc_dashboard.png"
    if _metric_dashboard(
        obs,
        modification_metrics,
        "passes_modification_qc",
        path,
        "Modification QC: pass vs fail",
    ):
        _register(plot_layout, path, "modification_qc", "modification_qc_metric_dashboard")
    threshold = float(getattr(cfg, "duplicate_detection_distance_threshold", 0.07))
    _duplicate_diagnostics(obs, barcode_column, plot_layout, threshold)
    complexity_results = None
    if cfg is not None and "duplicate_cluster_id" in obs:
        complexity_results = _complexity_plots(obs, barcode_column, plot_layout, cfg)

    summary_metrics = list(dict.fromkeys([*read_metrics, *modification_metrics]))
    summary = _barcode_reference_summary(
        obs,
        barcode_column,
        summary_metrics,
        complexity_results,
    )
    summary_dir = plot_layout.categories["barcode_summary"]
    summary.to_parquet(summary_dir / "barcode_reference_summary.parquet", index=False)
    summary.to_csv(summary_dir / "barcode_reference_summary.csv", index=False)
    _barcode_reference_overviews(summary, plot_layout)
    _barcode_distribution_plots(
        obs,
        barcode_column,
        read_metrics,
        plot_layout,
        category="read_qc",
        plot_type="barcode_reference_read_metric_distributions",
    )
    _barcode_distribution_plots(
        obs,
        barcode_column,
        modification_metrics,
        plot_layout,
        category="modification_qc",
        plot_type="barcode_reference_modification_distributions",
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
        path = (
            plot_layout.categories["coverage"] / f"{str(reference).replace('/', '_')}_coverage.png"
        )
        figure.savefig(path, dpi=150)
        plt.close(figure)
        _register(
            plot_layout, path, "coverage", "valid_fraction_by_position", reference=str(reference)
        )

    if spine_path is not None:
        _read_span_quality_plots(
            spine_path,
            plot_layout,
            max_reads=int(getattr(cfg, "preprocess_plot_max_heatmap_reads", 300)),
            max_positions=int(getattr(cfg, "preprocess_plot_max_heatmap_positions", 1800)),
        )
