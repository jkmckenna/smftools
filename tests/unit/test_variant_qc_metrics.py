import json

import numpy as np
import pandas as pd
import pytest

from smftools.cli.stage_artifacts import prepare_analysis_plot_layout
from smftools.preprocessing.variant_metrics import (
    COHORTS,
    calculate_variant_qc_metrics,
    generate_variant_qc_plots,
    write_variant_qc_metric_artifacts,
)

pytestmark = pytest.mark.unit


def _obs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "read_id": [f"read-{index}" for index in range(1, 7)],
            "molecule_uid": [f"molecule-{index}" for index in range(1, 7)],
            "Reference_strand": ["ref_top"] * 6,
            "Sample": ["sample-a"] * 6,
            "variant_reference_set_id": ["set-1"] * 6,
            "variant_evidence_status": [
                "complete",
                "complete",
                "complete",
                "complete",
                "complete",
                "blocked_missing_input",
            ],
            "variant_callable_site_count": [2, 2, 1, 1, 0, np.nan],
            "chimeric_variant_sites": [True, False, True, False, False, False],
            "variant_has_breakpoint": [True, False, False, False, False, False],
            "passes_nonvariant_qc": [True, True, True, True, False, True],
            "passes_qc": [True, True, False, True, False, True],
            "is_duplicate": [False, True, True, False, False, False],
            "passes_dedup": [True, False, False, True, False, True],
            "duplicate_cluster_id": [0, 0, 1, 1, -1, -1],
        }
    )


def _metric(
    metrics: pd.DataFrame,
    cohort: str,
    measure: str,
) -> pd.Series:
    selected = metrics.loc[
        (metrics["cohort"] == cohort)
        & (metrics["grouping"] == "overall")
        & (metrics["measure"] == measure)
    ]
    assert len(selected) == 1
    return selected.iloc[0]


def test_named_cohort_read_and_cluster_metrics_match_hand_calculation():
    metrics = calculate_variant_qc_metrics(_obs(), source_generation_id="generation-1")
    assert set(metrics["cohort"]) == set(COHORTS)
    assert set(metrics["source_generation_id"]) == {"generation-1"}

    all_reads = _metric(metrics, "all_aligned", "read_count")
    assert (all_reads["numerator"], all_reads["denominator"], all_reads["value"]) == (
        6,
        1,
        6.0,
    )
    callable_rate = _metric(metrics, "all_aligned", "callable_read_rate")
    assert (callable_rate["numerator"], callable_rate["denominator"]) == (4, 6)
    assert callable_rate["value"] == pytest.approx(4 / 6)
    broad_rate = _metric(metrics, "all_aligned", "broad_other_reference_read_rate")
    assert (broad_rate["numerator"], broad_rate["denominator"], broad_rate["value"]) == (
        2,
        4,
        0.5,
    )
    breakpoint_rate = _metric(metrics, "all_aligned", "breakpoint_read_rate")
    assert (breakpoint_rate["numerator"], breakpoint_rate["denominator"]) == (1, 4)

    cluster_rate = _metric(
        metrics,
        "all_aligned",
        "broad_other_reference_cluster_rate",
    )
    assert (cluster_rate["numerator"], cluster_rate["denominator"], cluster_rate["value"]) == (
        2,
        2,
        1.0,
    )
    mixed_rate = _metric(metrics, "all_aligned", "mixed_status_cluster_rate")
    assert (mixed_rate["numerator"], mixed_rate["denominator"], mixed_rate["value"]) == (
        2,
        2,
        1.0,
    )
    retention = _metric(
        metrics,
        "all_aligned",
        "event_positive_cluster_retention_rate",
    )
    assert (retention["numerator"], retention["denominator"], retention["value"]) == (
        1,
        2,
        0.5,
    )


def test_pre_filter_cohort_preserves_event_and_post_dedup_counts_keeper_once():
    metrics = calculate_variant_qc_metrics(_obs(), source_generation_id="generation-1")
    pre_nonvariant = _metric(
        metrics,
        "pre_dedup_nonvariant_qc",
        "broad_other_reference_read_rate",
    )
    pre_final = _metric(metrics, "pre_dedup_final_qc", "broad_other_reference_read_rate")
    assert pre_nonvariant["numerator"] == 2
    assert pre_final["numerator"] == 1

    post_cluster = _metric(
        metrics,
        "post_dedup_nonvariant_qc",
        "broad_other_reference_cluster_rate",
    )
    assert (post_cluster["numerator"], post_cluster["denominator"]) == (1, 2)
    post_mixed = _metric(
        metrics,
        "post_dedup_nonvariant_qc",
        "mixed_status_cluster_rate",
    )
    assert post_mixed["numerator"] == 0


def test_empty_cohorts_and_no_calls_have_defined_denominators():
    obs = _obs()
    obs["passes_nonvariant_qc"] = False
    obs["passes_qc"] = False
    obs["passes_dedup"] = False
    metrics = calculate_variant_qc_metrics(obs, source_generation_id="generation-1")

    empty = _metric(metrics, "pre_dedup_nonvariant_qc", "callable_read_rate")
    assert (empty["numerator"], empty["denominator"]) == (0, 0)
    assert np.isnan(empty["value"])
    noncallable = _metric(metrics, "all_aligned", "noncallable_read_rate")
    assert (noncallable["numerator"], noncallable["denominator"]) == (2, 6)


def test_metric_artifacts_and_plots_are_consumers_of_durable_table(tmp_path):
    obs_path = tmp_path / "obs.parquet"
    _obs().to_parquet(obs_path, index=False)
    outputs = write_variant_qc_metric_artifacts(
        obs_path,
        tmp_path / "variant",
        source_generation_id="generation-1",
    )
    assert outputs["metrics"].is_file()
    assert outputs["summary_tsv"].is_file()
    summary = json.loads(outputs["summary_json"].read_text(encoding="utf-8"))
    assert summary["source_generation_id"] == "generation-1"
    assert summary["metrics"]

    layout = prepare_analysis_plot_layout(tmp_path, stage="preprocess")
    plots = generate_variant_qc_plots(outputs["metrics"], layout)
    assert len(plots) == 2
    catalog = pd.read_parquet(layout.catalog)
    assert set(catalog["plot_type"]) == {
        "variant_read_rates_by_cohort",
        "variant_cluster_rates_by_cohort",
    }
