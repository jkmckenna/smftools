from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from smftools.constants import (
    ML_DATASETS_DIR,
    ML_EXPERIMENT_OUTPUTS_DIR,
    ML_INDEX_DIR,
    ML_MODELS_DIR,
    ML_PROJECT_OUTPUTS_DIR,
    ML_RUNS_DIR,
)
from smftools.machine_learning.workspace import (
    MLRunPaths,
    MLWorkspace,
    MLWorkspaceError,
    resolve_ml_workspace,
)

pytestmark = pytest.mark.unit


def _experiment_config(output_directory, *, name: str = "experiment-a"):
    return SimpleNamespace(
        output_directory=output_directory,
        experiment_name=name,
    )


def _project(tmp_path, name: str = "project-a"):
    project_dir = tmp_path / name
    project_dir.mkdir()
    (project_dir / "registry.json").write_text(
        json.dumps({"schema_version": 4, "experiments": {}, "sets": {}})
    )
    return project_dir


def test_experiment_workspace_uses_configured_output_root_without_creating_it(
    tmp_path,
) -> None:
    output_directory = tmp_path / "experiment-output"
    workspace = resolve_ml_workspace(
        experiment_config=_experiment_config(output_directory),
    )

    assert workspace.scope_kind == "experiment"
    assert workspace.scope_id == "experiment-a"
    assert workspace.owner_root == output_directory.resolve()
    assert workspace.root == output_directory.resolve() / ML_EXPERIMENT_OUTPUTS_DIR
    assert not output_directory.exists()


def test_project_workspace_uses_project_outputs_ml_without_creating_it(tmp_path) -> None:
    project_dir = _project(tmp_path)
    workspace = resolve_ml_workspace(project_dir=project_dir)

    assert workspace.scope_kind == "project"
    assert workspace.scope_id == "project-a"
    assert workspace.root == (project_dir.resolve() / "project_outputs" / ML_PROJECT_OUTPUTS_DIR)
    assert not workspace.root.exists()


def test_workspace_requires_exactly_one_scope(tmp_path) -> None:
    config = _experiment_config(tmp_path / "output")
    project_dir = _project(tmp_path)

    with pytest.raises(MLWorkspaceError, match="exactly one"):
        resolve_ml_workspace()
    with pytest.raises(MLWorkspaceError, match="exactly one"):
        resolve_ml_workspace(experiment_config=config, project_dir=project_dir)


def test_experiment_workspace_requires_output_directory_and_identity(tmp_path) -> None:
    with pytest.raises(MLWorkspaceError, match="output_directory"):
        resolve_ml_workspace(
            experiment_config=SimpleNamespace(
                output_directory=None,
                experiment_name="experiment-a",
            )
        )

    with pytest.raises(MLWorkspaceError, match="scope_id"):
        resolve_ml_workspace(
            experiment_config=SimpleNamespace(
                output_directory=tmp_path / "output",
                experiment_name=None,
            )
        )


def test_experiment_scope_rejects_multi_experiment_selection(tmp_path) -> None:
    with pytest.raises(MLWorkspaceError, match="more than one experiment"):
        resolve_ml_workspace(
            experiment_config=_experiment_config(tmp_path / "output"),
            selected_experiment_ids=("experiment-a", "experiment-b"),
        )


def test_project_scope_accepts_multi_experiment_selection(tmp_path) -> None:
    workspace = resolve_ml_workspace(
        project_dir=_project(tmp_path),
        selected_experiment_ids=("experiment-a", "experiment-b"),
    )

    assert workspace.scope_kind == "project"


def test_project_workspace_cannot_reference_registered_experiment_outputs(tmp_path) -> None:
    experiment_output = tmp_path / "experiment-output"
    project_dir = _project(tmp_path)
    (project_dir / "registry.json").write_text(
        json.dumps(
            {
                "schema_version": 4,
                "experiments": {
                    "experiment-a": {
                        "status": "active",
                        "run_root": experiment_output.as_posix(),
                    }
                },
                "sets": {},
            }
        )
    )
    workspace = resolve_ml_workspace(project_dir=project_dir)

    assert not workspace.root.is_relative_to(experiment_output)
    with pytest.raises(MLWorkspaceError, match="escapes"):
        workspace.portable_reference(experiment_output / "ml_outputs/run.json")


def test_project_workspace_requires_initialized_project(tmp_path) -> None:
    project_dir = tmp_path / "not-initialized"
    project_dir.mkdir()

    with pytest.raises(MLWorkspaceError, match="initialize the project"):
        resolve_ml_workspace(project_dir=project_dir)


def test_workspace_identity_is_path_neutral_across_relocation(tmp_path) -> None:
    first = resolve_ml_workspace(
        experiment_config=_experiment_config(tmp_path / "machine-a" / "output"),
        scope_id="stable-experiment-uid",
    )
    moved = resolve_ml_workspace(
        experiment_config=_experiment_config(tmp_path / "machine-b" / "output"),
        scope_id="stable-experiment-uid",
    )

    assert first.root != moved.root
    assert first.workspace_id == moved.workspace_id
    assert first.portable_reference(first.run_paths("run-1").manifest) == (
        moved.portable_reference(moved.run_paths("run-1").manifest)
    )


def test_workspace_identity_distinguishes_scope_kind_and_owner(tmp_path) -> None:
    experiment = resolve_ml_workspace(
        experiment_config=_experiment_config(tmp_path / "output"),
        scope_id="owner-a",
    )
    project = resolve_ml_workspace(
        project_dir=_project(tmp_path),
        scope_id="owner-a",
    )
    other_experiment = resolve_ml_workspace(
        experiment_config=_experiment_config(tmp_path / "other"),
        scope_id="owner-b",
    )

    assert len({experiment.workspace_id, project.workspace_id, other_experiment.workspace_id}) == 3


def test_standard_workspace_roots_are_deterministic(tmp_path) -> None:
    workspace = resolve_ml_workspace(
        experiment_config=_experiment_config(tmp_path / "output"),
    )

    assert workspace.datasets_root == workspace.root / ML_DATASETS_DIR
    assert workspace.runs_root == workspace.root / ML_RUNS_DIR
    assert workspace.models_root == workspace.root / ML_MODELS_DIR
    assert workspace.index_root == workspace.root / ML_INDEX_DIR
    assert workspace.dataset_dir("dataset-1") == workspace.datasets_root / "dataset-1"
    assert workspace.model_dir("model-1") == workspace.models_root / "model-1"


def test_run_path_bundle_covers_all_job_output_categories(tmp_path) -> None:
    workspace = resolve_ml_workspace(
        experiment_config=_experiment_config(tmp_path / "output"),
    )
    paths = workspace.run_paths("run-1")

    assert isinstance(paths, MLRunPaths)
    assert paths.root == workspace.runs_root / "run-1"
    assert paths.manifest == paths.root / "run_manifest.json"
    assert paths.resolved_plan == paths.root / "resolved_plan.json"
    assert paths.resolved_config == paths.root / "resolved_config.json"
    assert paths.environment == paths.root / "environment.json"
    assert paths.history == paths.root / "history.parquet"
    assert paths.metrics == paths.root / "metrics.parquet"
    assert paths.prediction_path("validation") == paths.root / "predictions/validation.parquet"
    assert paths.plots == paths.root / "plots"
    assert paths.explanation_dir("explanation-1") == (paths.root / "explanations/explanation-1")
    assert paths.checkpoints == paths.root / "checkpoints"
    assert paths.logs == paths.root / "logs"


@pytest.mark.parametrize(
    "method,value",
    [
        ("run_paths", "../escape"),
        ("dataset_dir", "nested/value"),
        ("model_dir", "/absolute"),
    ],
)
def test_artifact_identifiers_cannot_escape_workspace(
    tmp_path,
    method: str,
    value: str,
) -> None:
    workspace = resolve_ml_workspace(
        experiment_config=_experiment_config(tmp_path / "output"),
    )

    with pytest.raises(MLWorkspaceError, match="path component"):
        getattr(workspace, method)(value)


def test_portable_reference_round_trip_and_containment(tmp_path) -> None:
    workspace = resolve_ml_workspace(
        experiment_config=_experiment_config(tmp_path / "output"),
    )
    manifest = workspace.run_paths("run-1").manifest
    reference = workspace.portable_reference(manifest)

    assert reference == "runs/run-1/run_manifest.json"
    assert workspace.resolve_reference(reference) == manifest

    with pytest.raises(MLWorkspaceError, match="escapes"):
        workspace.portable_reference(tmp_path / "outside.json")
    with pytest.raises(MLWorkspaceError, match="portable path"):
        workspace.resolve_reference("../outside.json")
    with pytest.raises(MLWorkspaceError, match="portable path"):
        workspace.resolve_reference("/absolute/outside.json")


def test_workspace_constructor_rejects_wrong_scope_root(tmp_path) -> None:
    with pytest.raises(MLWorkspaceError, match="workspace must resolve"):
        MLWorkspace(
            scope_kind="experiment",
            scope_id="experiment-a",
            owner_root=tmp_path / "output",
            root=tmp_path / "somewhere-else",
        )


def test_dry_run_reports_paths_without_filesystem_writes(tmp_path) -> None:
    output_directory = tmp_path / "output"
    workspace = resolve_ml_workspace(
        experiment_config=_experiment_config(output_directory),
    )
    report = workspace.to_dry_run_dict(run_id="run-1")

    assert report["workspace_id"] == workspace.workspace_id
    assert report["run"]["root"] == workspace.run_paths("run-1").root.as_posix()
    assert report["run"]["predictions"].endswith("/runs/run-1/predictions")
    assert not output_directory.exists()


@pytest.mark.parametrize("action", ["train", "apply", "evaluate", "explain", "plot"])
def test_every_plan_job_action_reports_intended_run_paths(tmp_path, action: str) -> None:
    workspace = resolve_ml_workspace(
        experiment_config=_experiment_config(tmp_path / "output"),
    )

    report = workspace.to_job_dry_run_dict(action=action, run_id=f"{action}-run")

    assert report["action"] == action
    assert report["run"]["run_id"] == f"{action}-run"
    assert report["run"]["manifest"].endswith(f"/runs/{action}-run/run_manifest.json")
