from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from smftools.informatics.molecule_identity import molecule_uid
from smftools.machine_learning.orchestration import (
    MLWorkflowPlanningError,
    plan_ml_workflow,
)
from smftools.machine_learning.plan import parse_ml_plan
from smftools.machine_learning.selection import (
    MLDataSelectionPlan,
    ResolvedChannelSource,
    SelectedExperimentSource,
)

pytestmark = pytest.mark.unit

EXPERIMENT_UID = "12345678-1234-5678-1234-567812345678"


def _plan(*, alpha: object = 1.0, tracking: str = "none"):
    return parse_ml_plan(
        {
            "schema_version": 1,
            "scope": {"kind": "project"},
            "datasets": {
                "reads": {
                    "modalities": ["deaminase"],
                    "references": ["locus"],
                    "samples": {"include": ["exp-a/s1", "exp-a/s2", "exp-a/s3"]},
                    "labels": {
                        "column": "activity",
                        "classes": {"inactive": 0, "active": 1},
                    },
                }
            },
            "splits": {
                "samples": {
                    "strategy": "explicit_groups",
                    "group_by": ["experiment_uid", "Sample"],
                    "train_groups": ["exp-a/s1"],
                    "validation_groups": ["exp-a/s2"],
                    "test_groups": ["exp-a/s3"],
                }
            },
            "models": {
                "nb": {
                    "backend": "sklearn",
                    "family": "bernoulli_nb",
                    "parameters": {"alpha": alpha},
                }
            },
            "jobs": {
                "train": {
                    "action": "train",
                    "dataset": "reads",
                    "split": "samples",
                    "models": ["nb"],
                    "evaluate": ["validation", "test"],
                },
                "explain_external": {
                    "action": "explain",
                    "dataset": "reads",
                    "model": "model:published-model",
                    "explain": ["GradientSHAP"],
                },
                "plots": {
                    "action": "plot",
                    "runs": ["train"],
                    "plots": ["training_history"],
                },
            },
            "tracking": {"provider": tracking},
        }
    )


def _selection(plan, project: Path) -> MLDataSelectionPlan:
    rows = []
    for sample in ("s1", "s2", "s3"):
        for index, class_id in enumerate((0, 1, 0, 1)):
            read_id = f"{sample}-{index}"
            rows.append(
                {
                    "molecule_uid": molecule_uid(EXPERIMENT_UID, read_id),
                    "experiment_uid": EXPERIMENT_UID,
                    "read_id": read_id,
                    "experiment_id": "exp-a",
                    "sample_id": sample,
                    "Sample": sample,
                    "reference": "locus",
                    "physical_reference": "chr1+",
                    "modality": "deaminase",
                    "class_id": class_id,
                }
            )
    identity = pd.DataFrame(rows)
    source = SelectedExperimentSource(
        experiment_id="exp-a",
        experiment_uid=EXPERIMENT_UID,
        modality="deaminase",
        physical_references=("chr1+",),
        canonical_references=("locus",),
        channels=(
            ResolvedChannelSource(
                channel_name="accessibility",
                biological_role="accessibility",
                modality="deaminase",
                stage="preprocess",
                layer="C_site_binary",
                site_context="C",
                catalog_sha256="a" * 64,
            ),
        ),
        membership_artifact=project / "membership.parquet",
        membership_artifact_sha256="b" * 64,
        membership_fingerprint="c" * 64,
        feature_fingerprint="d" * 64,
    )
    return MLDataSelectionPlan(
        schema_version=1,
        selection_id="selection-1",
        dataset_name="reads",
        plan_hash=plan.plan_hash,
        scope_kind="project",
        scope_id=project.name,
        set_name=None,
        channel_policy="single_modality",
        channel_names=("accessibility",),
        group_by=("Sample", "experiment_uid"),
        sources=(source,),
        identity_table=identity,
        membership_fingerprint="c" * 64,
        feature_fingerprint="d" * 64,
        n_observations=len(identity),
        n_features=100,
        estimated_materialization_bytes=9_600,
        class_counts={"0": 6, "1": 6},
        modality_counts={"deaminase": 12},
        sample_counts={"s1": 4, "s2": 4, "s3": 4},
    )


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "project-a"
    root.mkdir()
    (root / "registry.json").write_text(
        json.dumps({"schema_version": 4, "experiments": {}, "sets": {}}),
        encoding="utf-8",
    )
    return root


def test_workflow_dry_run_composes_selection_split_models_and_outputs(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(tracking="wandb")
    selection = _selection(plan, project)
    calls = []

    def fake_selection(resolved_plan, dataset_name, **kwargs):
        calls.append((resolved_plan, dataset_name, kwargs))
        return selection

    monkeypatch.setattr(
        "smftools.machine_learning.orchestration.planning.plan_ml_dataset",
        fake_selection,
    )
    before = sorted(path.relative_to(project) for path in project.rglob("*"))

    report = plan_ml_workflow(
        plan,
        project_dir=project,
        dependency_checker=lambda package: package == "captum",
    )
    result = report.to_dict()

    assert calls[0][1:] == (
        "reads",
        {
            "experiment_dir": None,
            "project_dir": project,
            "experiment_id": None,
        },
    )
    assert result["workspace"]["root"].endswith("project_outputs/ml")
    assert result["datasets"][0]["sample_counts"] == {"s1": 4, "s2": 4, "s3": 4}
    assert result["datasets"][0]["class_counts"] == {"0": 6, "1": 6}
    assert result["datasets"][0]["selectors"]["samples"]["include"] == [
        "exp-a/s1",
        "exp-a/s2",
        "exp-a/s3",
    ]
    assert result["datasets"][0]["input_schema"]["masks"][0]["kind"] == "observed"
    assert result["splits"][0]["overlap_checks"]["observation_roles_disjoint"] is True
    assert result["splits"][0]["requested_groups"]["test"] == ["exp-a/s3"]
    assert result["splits"][0]["estimated_materialization_bytes_by_role"] == {
        "test": 3200,
        "train": 3200,
        "validation": 3200,
    }
    assert result["models"][0]["family"] == "bernoulli_nb"
    assert [job["action"] for job in result["jobs"]] == ["train", "explain", "plot"]
    assert result["jobs"][0]["output_layout"] == "runs/<fresh-run-uuid>"
    assert result["optional_dependencies"] == [
        {
            "package": "captum",
            "extra": "ml-extended",
            "available": True,
            "required_for": ["jobs.explain_external.explain:GradientSHAP"],
        },
        {
            "package": "wandb",
            "extra": "ml-extended",
            "available": False,
            "required_for": ["experiment tracking"],
        },
    ]
    assert result["execution"] == {
        "writes_files": False,
        "trains_models": False,
        "backend_services": ["sklearn", "pytorch"],
        "lightning_required": False,
        "hydra_required": False,
        "tracking_provider": "wandb",
    }
    assert sorted(path.relative_to(project) for path in project.rglob("*")) == before


def test_workflow_requires_explicit_scope_and_never_uses_working_directory() -> None:
    with pytest.raises(
        MLWorkflowPlanningError,
        match="scope: project plans require project_dir",
    ):
        plan_ml_workflow(_plan())


def test_workflow_model_errors_include_the_plan_field(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(alpha="invalid")
    monkeypatch.setattr(
        "smftools.machine_learning.orchestration.planning.plan_ml_dataset",
        lambda *args, **kwargs: _selection(plan, project),
    )

    with pytest.raises(MLWorkflowPlanningError, match=r"models\.nb:.*alpha"):
        plan_ml_workflow(plan, project_dir=project)


def test_workflow_report_is_immutable_but_to_dict_is_detached(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    monkeypatch.setattr(
        "smftools.machine_learning.orchestration.planning.plan_ml_dataset",
        lambda *args, **kwargs: _selection(plan, project),
    )
    report = plan_ml_workflow(plan, project_dir=project)

    with pytest.raises(TypeError):
        report.workspace["root"] = "elsewhere"  # type: ignore[index]
    detached = report.to_dict()
    detached["workspace"]["root"] = "elsewhere"
    assert report.workspace["root"] != "elsewhere"
