"""Durable named-cohort metrics for preprocess variant evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ..constants import REFERENCE_STRAND, VARIANT_QC_METRICS_SCHEMA_VERSION
from ..readwrite import atomic_write_json

VARIANT_QC_METRICS = "variant_qc_metrics.parquet"
VARIANT_QC_SUMMARY_JSON = "variant_qc_summary.json"
VARIANT_QC_SUMMARY_TSV = "variant_qc_summary.tsv"
VARIANT_QC_METRICS_ANALYSIS_VERSION = "1"

COHORTS = (
    "all_aligned",
    "pre_dedup_nonvariant_qc",
    "post_dedup_nonvariant_qc",
    "pre_dedup_final_qc",
    "post_dedup_final_qc",
)

_METRIC_COLUMNS = (
    "schema_version",
    "analysis_version",
    "source_generation_id",
    "variant_reference_set_id",
    "cohort",
    "grouping",
    "reference",
    "sample",
    "level",
    "measure",
    "numerator",
    "denominator",
    "value",
)


def _sample_column(obs: pd.DataFrame) -> str | None:
    for column in ("Experiment_name_and_barcode", "Sample", "Barcode", "sample", "barcode"):
        if column in obs:
            return column
    return None


def _cohort_masks(obs: pd.DataFrame) -> dict[str, pd.Series]:
    index = obs.index

    def mask(column: str, default: bool) -> pd.Series:
        if column not in obs:
            return pd.Series(default, index=index, dtype=bool)
        return obs[column].astype("boolean").fillna(False).astype(bool)

    nonvariant = mask("passes_nonvariant_qc", True)
    final = mask("passes_qc", True)
    duplicate = mask("is_duplicate", False)
    dedup = mask("passes_dedup", True)
    return {
        "all_aligned": pd.Series(True, index=index, dtype=bool),
        "pre_dedup_nonvariant_qc": nonvariant,
        "post_dedup_nonvariant_qc": nonvariant & ~duplicate,
        "pre_dedup_final_qc": final,
        "post_dedup_final_qc": dedup,
    }


def _cluster_keys(frame: pd.DataFrame) -> pd.Series:
    cluster_ids = pd.to_numeric(
        frame.get("duplicate_cluster_id", pd.Series(-1, index=frame.index)),
        errors="coerce",
    ).fillna(-1)
    molecule_ids = frame.get("molecule_uid", frame["read_id"]).astype(str)
    return pd.Series(
        np.where(
            cluster_ids.to_numpy() >= 0,
            "cluster:" + cluster_ids.astype("int64").astype(str),
            "molecule:" + molecule_ids,
        ),
        index=frame.index,
        dtype="string",
    )


def _rate_record(
    *,
    base: dict[str, object],
    level: str,
    measure: str,
    numerator: int,
    denominator: int,
) -> dict[str, object]:
    return {
        **base,
        "level": level,
        "measure": measure,
        "numerator": int(numerator),
        "denominator": int(denominator),
        "value": float(numerator / denominator) if denominator else np.nan,
    }


def _count_record(
    *,
    base: dict[str, object],
    level: str,
    measure: str,
    count: int,
) -> dict[str, object]:
    return {
        **base,
        "level": level,
        "measure": measure,
        "numerator": int(count),
        "denominator": 1,
        "value": float(count),
    }


def _group_slices(
    frame: pd.DataFrame,
    sample_column: str | None,
) -> Iterable[tuple[str, str | None, str | None, pd.DataFrame]]:
    yield "overall", None, None, frame
    if REFERENCE_STRAND in frame:
        for reference, group in frame.groupby(REFERENCE_STRAND, sort=True, observed=True):
            yield "reference", str(reference), None, group
    if sample_column is not None:
        for sample, group in frame.groupby(sample_column, sort=True, observed=True):
            yield "sample", None, str(sample), group
    if sample_column is not None and REFERENCE_STRAND in frame:
        for (reference, sample), group in frame.groupby(
            [REFERENCE_STRAND, sample_column],
            sort=True,
            observed=True,
        ):
            yield "reference_sample", str(reference), str(sample), group


def calculate_variant_qc_metrics(
    obs: pd.DataFrame,
    *,
    source_generation_id: str,
) -> pd.DataFrame:
    """Calculate read/cluster metrics for every durable variant QC cohort.

    Callable denominators include only rows with complete evidence and at least
    one callable informative site. Noncallable rates use every row in the named
    cohort, including incomplete evidence and zero-callable-site reads.
    """
    required = {
        "read_id",
        "variant_reference_set_id",
        "variant_evidence_status",
        "variant_callable_site_count",
        "chimeric_variant_sites",
        "variant_has_breakpoint",
    }
    missing = required.difference(obs.columns)
    if missing:
        raise ValueError(f"preprocess obs lacks variant metric columns: {sorted(missing)}")

    records: list[dict[str, object]] = []
    sample_column = _sample_column(obs)
    cohort_masks = _cohort_masks(obs)
    reference_set_ids = sorted(obs["variant_reference_set_id"].dropna().astype(str).unique())
    for reference_set_id in reference_set_ids:
        set_mask = (
            obs["variant_reference_set_id"].astype("string").eq(reference_set_id).fillna(False)
        )
        set_obs = obs.loc[set_mask].copy()
        set_masks = {
            name: mask.reindex(set_obs.index).fillna(False) for name, mask in cohort_masks.items()
        }
        for cohort in COHORTS:
            cohort_obs = set_obs.loc[set_masks[cohort]]
            for grouping, reference, sample, group in _group_slices(
                cohort_obs,
                sample_column,
            ):
                base = {
                    "schema_version": VARIANT_QC_METRICS_SCHEMA_VERSION,
                    "analysis_version": VARIANT_QC_METRICS_ANALYSIS_VERSION,
                    "source_generation_id": str(source_generation_id),
                    "variant_reference_set_id": reference_set_id,
                    "cohort": cohort,
                    "grouping": grouping,
                    "reference": reference,
                    "sample": sample,
                }
                callable_reads = group["variant_evidence_status"].astype(str).eq(
                    "complete"
                ) & pd.to_numeric(
                    group["variant_callable_site_count"],
                    errors="coerce",
                ).fillna(0).gt(0)
                broad_event = group["chimeric_variant_sites"].astype("boolean").fillna(False)
                breakpoint = group["variant_has_breakpoint"].astype("boolean").fillna(False)
                read_count = len(group)
                callable_count = int(callable_reads.sum())
                records.extend(
                    (
                        _count_record(
                            base=base,
                            level="read",
                            measure="read_count",
                            count=read_count,
                        ),
                        _rate_record(
                            base=base,
                            level="read",
                            measure="callable_read_rate",
                            numerator=callable_count,
                            denominator=read_count,
                        ),
                        _rate_record(
                            base=base,
                            level="read",
                            measure="noncallable_read_rate",
                            numerator=read_count - callable_count,
                            denominator=read_count,
                        ),
                        _rate_record(
                            base=base,
                            level="read",
                            measure="broad_other_reference_read_rate",
                            numerator=int((broad_event & callable_reads).sum()),
                            denominator=callable_count,
                        ),
                        _rate_record(
                            base=base,
                            level="read",
                            measure="breakpoint_read_rate",
                            numerator=int((breakpoint & callable_reads).sum()),
                            denominator=callable_count,
                        ),
                    )
                )

                cluster_frame = pd.DataFrame(
                    {
                        "cluster_key": _cluster_keys(group),
                        "callable": callable_reads,
                        "broad_event": broad_event,
                        "breakpoint": breakpoint,
                        "retained": group.get(
                            "passes_dedup",
                            pd.Series(True, index=group.index),
                        )
                        .astype("boolean")
                        .fillna(False),
                    },
                    index=group.index,
                )
                cluster_rows = []
                for _, members in cluster_frame.groupby("cluster_key", sort=True):
                    callable_members = members.loc[members["callable"]]
                    broad_positive = bool(callable_members["broad_event"].any())
                    retained_positive = bool(
                        (callable_members["broad_event"] & callable_members["retained"]).any()
                    )
                    cluster_rows.append(
                        {
                            "callable": not callable_members.empty,
                            "broad_event": broad_positive,
                            "breakpoint": bool(callable_members["breakpoint"].any()),
                            "mixed_status": (
                                callable_members["broad_event"].nunique(dropna=False) > 1
                            ),
                            "event_positive_retained": retained_positive,
                        }
                    )
                clusters = pd.DataFrame(cluster_rows)
                cluster_count = len(clusters)
                callable_clusters = (
                    clusters["callable"] if "callable" in clusters else pd.Series(dtype=bool)
                )
                callable_cluster_count = int(callable_clusters.sum())
                event_clusters = (
                    clusters["broad_event"] & callable_clusters
                    if not clusters.empty
                    else pd.Series(dtype=bool)
                )
                event_cluster_count = int(event_clusters.sum())
                records.extend(
                    (
                        _count_record(
                            base=base,
                            level="cluster",
                            measure="cluster_count",
                            count=cluster_count,
                        ),
                        _rate_record(
                            base=base,
                            level="cluster",
                            measure="broad_other_reference_cluster_rate",
                            numerator=event_cluster_count,
                            denominator=callable_cluster_count,
                        ),
                        _rate_record(
                            base=base,
                            level="cluster",
                            measure="breakpoint_cluster_rate",
                            numerator=(
                                int((clusters["breakpoint"] & callable_clusters).sum())
                                if not clusters.empty
                                else 0
                            ),
                            denominator=callable_cluster_count,
                        ),
                        _rate_record(
                            base=base,
                            level="cluster",
                            measure="mixed_status_cluster_rate",
                            numerator=(
                                int((clusters["mixed_status"] & callable_clusters).sum())
                                if not clusters.empty
                                else 0
                            ),
                            denominator=callable_cluster_count,
                        ),
                        _rate_record(
                            base=base,
                            level="cluster",
                            measure="event_positive_cluster_retention_rate",
                            numerator=(
                                int((clusters["event_positive_retained"] & event_clusters).sum())
                                if not clusters.empty
                                else 0
                            ),
                            denominator=event_cluster_count,
                        ),
                    )
                )
    return pd.DataFrame.from_records(records, columns=_METRIC_COLUMNS)


def write_variant_qc_metric_artifacts(
    obs_path: str | Path,
    output_dir: str | Path,
    *,
    source_generation_id: str,
) -> dict[str, Path]:
    """Write versioned long-form metrics plus compact JSON and TSV summaries."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = calculate_variant_qc_metrics(
        pd.read_parquet(obs_path),
        source_generation_id=source_generation_id,
    )
    metrics_path = output_dir / VARIANT_QC_METRICS
    json_path = output_dir / VARIANT_QC_SUMMARY_JSON
    tsv_path = output_dir / VARIANT_QC_SUMMARY_TSV
    metrics.to_parquet(metrics_path, index=False)
    compact = metrics.loc[metrics["grouping"] == "overall"].copy()
    compact.to_csv(tsv_path, sep="\t", index=False)
    atomic_write_json(
        json_path,
        {
            "schema_version": VARIANT_QC_METRICS_SCHEMA_VERSION,
            "analysis_version": VARIANT_QC_METRICS_ANALYSIS_VERSION,
            "source_generation_id": str(source_generation_id),
            "metrics": json.loads(compact.to_json(orient="records")),
        },
    )
    return {
        "metrics": metrics_path,
        "summary_json": json_path,
        "summary_tsv": tsv_path,
    }


def generate_variant_qc_plots(
    metrics_path: str | Path,
    plot_layout,
) -> list[Path]:
    """Generate bounded cohort plots solely from durable metric artifacts."""
    import matplotlib.pyplot as plt

    from ..cli.stage_artifacts import register_plot_artifact

    metrics_path = Path(metrics_path)
    metrics = pd.read_parquet(metrics_path)
    overall = metrics.loc[metrics["grouping"] == "overall"].copy()
    outputs: list[Path] = []
    plot_specs = (
        (
            "read",
            (
                "broad_other_reference_read_rate",
                "breakpoint_read_rate",
                "noncallable_read_rate",
            ),
            "variant_read_rates_by_cohort",
        ),
        (
            "cluster",
            (
                "broad_other_reference_cluster_rate",
                "breakpoint_cluster_rate",
                "mixed_status_cluster_rate",
                "event_positive_cluster_retention_rate",
            ),
            "variant_cluster_rates_by_cohort",
        ),
    )
    for reference_set_id, set_metrics in overall.groupby(
        "variant_reference_set_id",
        sort=True,
        observed=True,
    ):
        safe_set_id = "".join(
            character if character.isalnum() or character in "-._" else "_"
            for character in str(reference_set_id)
        )[:24]
        for level, measures, plot_type in plot_specs:
            frame = set_metrics.loc[
                (set_metrics["level"] == level) & set_metrics["measure"].isin(measures)
            ]
            if frame.empty:
                continue
            pivot = frame.pivot_table(
                index="cohort",
                columns="measure",
                values="value",
                observed=True,
            ).reindex(COHORTS)
            figure, axis = plt.subplots(figsize=(10, 5))
            pivot.plot.bar(ax=axis, width=0.8)
            axis.set(
                xlabel="Cohort",
                ylabel="Rate",
                title=(f"{plot_type.replace('_', ' ').title()} / {reference_set_id}"),
                ylim=(0, 1.02),
            )
            axis.tick_params(axis="x", labelrotation=25)
            axis.legend(frameon=False, fontsize=8)
            figure.tight_layout()
            path = plot_layout.categories["variant_qc"] / f"{safe_set_id}__{plot_type}.png"
            figure.savefig(path, dpi=160)
            plt.close(figure)
            register_plot_artifact(
                plot_layout,
                path,
                stage="preprocess",
                category="variant_qc",
                plot_type=plot_type,
                model_id=str(reference_set_id),
                source_manifest=metrics_path,
            )
            outputs.append(path)
    return outputs
