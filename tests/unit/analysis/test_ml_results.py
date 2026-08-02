from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from smftools.analysis.compute.ml_results import (
    attribution_summary_table,
    class_balance_table,
    confusion_table,
    evaluation_curve_table,
    evaluation_metric_table,
    fold_metric_table,
    training_history_table,
)
from smftools.analysis.plot.ml_results import (
    plot_attribution_summary,
    plot_calibration_curves,
    plot_confusion_matrices,
    plot_evaluation_curves,
    plot_feature_importance,
    plot_metric_comparison,
    plot_training_history,
)
from smftools.machine_learning.artifacts import ExplanationMaskPolicy, ExplanationTarget
from smftools.machine_learning.contracts import InputSchema
from smftools.machine_learning.evaluation import (
    PredictionResult,
    TrainingEvent,
    TrainingHistory,
    aggregate_fold_metrics,
    evaluate_predictions,
)
from smftools.machine_learning.interpretability import AttributionResult, InterpretabilityRequest
from smftools.machine_learning.plan import parse_ml_plan

pytestmark = pytest.mark.unit


def test_compute_module_import_does_not_load_ml_execution_frameworks() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import smftools.analysis.compute.ml_results; "
                "assert 'torch' not in sys.modules; "
                "assert 'shap' not in sys.modules; "
                "assert 'captum' not in sys.modules"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def _evaluation():
    truth = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    positive = np.asarray([0.1, 0.8, 0.2, 0.7, 0.4, 0.9, 0.3, 0.6])
    probabilities = np.column_stack((1.0 - positive, positive))
    predictions = PredictionResult(
        molecule_uids=tuple(f"molecule-{index}" for index in range(len(truth))),
        class_ids=np.argmax(probabilities, axis=1),
        scores=np.log(probabilities),
        probabilities=probabilities,
        class_order=("inactive", "active"),
        split="test",
        experiment_uids=("experiment-a",) * len(truth),
        modalities=("deaminase",) * 4 + ("conversion",) * 4,
        groups=("sample-a",) * 4 + ("sample-b",) * 4,
        truth_class_ids=truth,
        positive_class="active",
        cohort="held-out-natural",
        model_id="model-a",
    )
    return evaluate_predictions(predictions)


def _attribution() -> AttributionResult:
    plan = parse_ml_plan(
        {
            "schema_version": 1,
            "scope": {"kind": "experiment"},
            "datasets": {
                "reads": {
                    "modalities": ["deaminase"],
                    "labels": {
                        "column": "activity",
                        "classes": {"inactive": 0, "active": 1},
                    },
                }
            },
            "splits": {
                "samples": {
                    "strategy": "explicit_groups",
                    "group_by": ["sample"],
                    "train_groups": ["sample-a"],
                    "validation_groups": ["sample-b"],
                    "test_groups": ["sample-c"],
                }
            },
            "models": {"cnn": {"backend": "torch", "recipe": "residual_dilated_cnn_v1"}},
            "jobs": {
                "train": {
                    "action": "train",
                    "dataset": "reads",
                    "split": "samples",
                    "models": ["cnn"],
                }
            },
        }
    )
    schema = InputSchema.from_dataset(
        plan.datasets["reads"],
        reference="locus",
        n_positions=3,
    )
    request = InterpretabilityRequest.create(
        method="Saliency",
        model_id="1" * 64,
        dataset_snapshot_id="2" * 64,
        input_schema_hash=schema.schema_hash,
        split_role="test",
        cohort="held-out-natural",
        observation_uids=("molecule-0", "molecule-1"),
        target=ExplanationTarget(
            output_name="activity_probability",
            class_id=1,
            class_name="active",
        ),
        mask_policy=ExplanationMaskPolicy.create(
            mask_kinds=("observed", "availability", "design"),
            handling="zero attribution where any declared validity mask is false",
        ),
    )
    values = np.asarray(
        [
            [[1.0], [-2.0], [3.0]],
            [[3.0], [2.0], [-1.0]],
        ]
    )
    return AttributionResult.create(
        request=request,
        axes=("observation", "position", "channel"),
        values=values,
        observation_uids=request.observation_uids,
        coordinates=np.asarray([100, 101, 102]),
        channels=schema.channels,
    )


def test_canonical_results_convert_to_tidy_semantic_tables() -> None:
    result = _evaluation()
    results = {"result-a": result}

    metrics = evaluation_metric_table(results)
    curves = evaluation_curve_table(results)
    confusion = confusion_table(results)
    balance = class_balance_table(results)
    folds = fold_metric_table(aggregate_fold_metrics({"fold-a": result}))

    assert set(metrics["split"]) == {"test"}
    assert set(metrics["cohort"]) == {"held-out-natural"}
    assert set(metrics["model_id"]) == {"model-a"}
    assert {"pooled", "modality"} == set(metrics["scope"])
    assert {"roc", "precision_recall", "calibration"} == set(curves["kind"])
    assert (
        curves.groupby(
            ["kind", "scope", "modality", "class_name"],
            dropna=False,
        )["point_index"]
        .min()
        .eq(0)
        .all()
    )
    assert confusion.loc[confusion["scope"] == "pooled", "count"].sum() == 8
    assert set(confusion["actual_class"]) == {"inactive", "active"}
    assert balance.loc[balance["scope"] == "pooled", "fraction"].sum() == pytest.approx(1.0)
    assert set(folds["fold_name"]) == {"fold-a"}
    assert not any("molecule" in column for column in metrics.columns)


def test_training_history_retains_only_real_events_without_fabricating_epochs() -> None:
    table = training_history_table(
        {
            "empty-model": TrainingHistory(backend="sklearn"),
            "sklearn-model": TrainingHistory(
                backend="sklearn",
                events=(
                    TrainingEvent(
                        event_index=0,
                        event_type="fit_completed",
                        metrics={},
                    ),
                ),
            ),
            "torch-model": TrainingHistory(
                backend="torch",
                events=(
                    TrainingEvent(
                        event_index=0,
                        event_type="epoch_completed",
                        epoch=1,
                        metrics={"train_loss": 0.8, "validation_loss": 0.9},
                    ),
                    TrainingEvent(
                        event_index=1,
                        event_type="epoch_completed",
                        epoch=2,
                        metrics={"train_loss": 0.6, "validation_loss": 0.7},
                    ),
                ),
            ),
        }
    )

    empty = table.loc[table["model_id"] == "empty-model"]
    sklearn = table.loc[table["model_id"] == "sklearn-model"].iloc[0]
    torch = table.loc[table["model_id"] == "torch-model"]
    assert empty.empty
    assert pd.isna(sklearn["epoch"])
    assert pd.isna(sklearn["metric_name"])
    assert len(torch) == 4
    assert set(torch["metric_name"]) == {"train_loss", "validation_loss"}


def test_attribution_summary_preserves_biological_axes_and_removes_observations() -> None:
    table = attribution_summary_table({"explanation": _attribution()})

    assert table["coordinate"].tolist() == [100, 101, 102]
    assert set(table["channel"]) == {"accessibility"}
    assert set(table["biological_role"]) == {"accessibility"}
    np.testing.assert_allclose(table["mean_attribution"], [2.0, 0.0, 1.0])
    np.testing.assert_allclose(table["mean_absolute_attribution"], [2.0, 2.0, 2.0])
    assert set(table["n_observations"]) == {2}
    assert set(table["split"]) == {"test"}
    assert set(table["target_class"]) == {"active"}
    assert "observation_uid" not in table.columns


def test_all_ml_result_figures_rebuild_from_tidy_tables(tmp_path: Path) -> None:
    result = _evaluation()
    results = {"result-a": result}
    history = training_history_table(
        {
            "model-a": TrainingHistory(
                backend="torch",
                events=(
                    TrainingEvent(
                        event_index=0,
                        event_type="epoch_completed",
                        epoch=1,
                        metrics={"train_loss": 0.8, "validation_loss": 0.9},
                    ),
                    TrainingEvent(
                        event_index=1,
                        event_type="epoch_completed",
                        epoch=2,
                        metrics={"train_loss": 0.6, "validation_loss": 0.7},
                    ),
                ),
            )
        }
    )
    metrics = evaluation_metric_table(results)
    curves = evaluation_curve_table(results)
    matrices = confusion_table(results)
    attributions = attribution_summary_table({"explanation": _attribution()})

    outputs = {
        "history.svg": lambda path: plot_training_history(history, path),
        "roc-pr.svg": lambda path: plot_evaluation_curves(curves, path),
        "calibration.svg": lambda path: plot_calibration_curves(curves, path),
        "metrics.svg": lambda path: plot_metric_comparison(
            metrics,
            path,
            names=("accuracy", "balanced_accuracy"),
        ),
        "confusion.svg": lambda path: plot_confusion_matrices(matrices, path),
        "importance.svg": lambda path: plot_feature_importance(attributions, path),
        "attribution.svg": lambda path: plot_attribution_summary(attributions, path),
    }
    for name, render in outputs.items():
        output = tmp_path / name
        render(output)
        assert output.is_file()
        assert output.stat().st_size > 0


def test_plotting_rejects_incomplete_tables_and_implicit_directories(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        plot_training_history(pd.DataFrame({"model_id": ["model-a"]}), tmp_path / "plot.svg")

    curves = evaluation_curve_table({"result-a": _evaluation()})
    with pytest.raises(ValueError, match="parent does not exist"):
        plot_evaluation_curves(curves, tmp_path / "missing" / "curves.svg")
