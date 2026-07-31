from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

from smftools.informatics.experiment_manifest import (
    record_stage_completion,
    update_experiment_manifest,
)
from smftools.informatics.molecule_identity import molecule_uid
from smftools.machine_learning.plan import parse_ml_plan
from smftools.machine_learning.selection import MLSelectionError, plan_ml_dataset
from smftools.project.reference_registry import ReferenceRegistry
from smftools.project.registry import init_project, load_registry, save_registry

pytestmark = pytest.mark.unit


def _plan(
    *,
    scope: str = "project",
    set_name: str | None = None,
    modalities: list[str] | None = None,
    channels: list[dict] | None = None,
    channel_policy: str | None = None,
    references: list[str] | None = None,
    samples: list[str] | None = None,
) -> object:
    dataset: dict = {
        "modalities": modalities or ["deaminase"],
        "references": references or ["locus"],
        "filters": {"mapping_quality_min": 20},
        "labels": {
            "column": "activity",
            "classes": {"inactive": 0, "active": 1},
        },
    }
    if channels is not None:
        dataset["channels"] = channels
    if channel_policy is not None:
        dataset["channel_policy"] = channel_policy
    if samples is not None:
        dataset["samples"] = {"include": samples}
    scope_value: dict[str, str] = {"kind": scope}
    if set_name is not None:
        scope_value["set"] = set_name
    return parse_ml_plan(
        {
            "schema_version": 1,
            "scope": scope_value,
            "datasets": {"reads": dataset},
            "splits": {
                "by_sample": {
                    "strategy": "stratified_group",
                    "group_by": ["experiment_uid", "Sample"],
                }
            },
            "models": {"baseline": {"backend": "sklearn", "family": "bernoulli_nb"}},
            "jobs": {
                "train": {
                    "action": "train",
                    "dataset": "reads",
                    "split": "by_sample",
                    "models": ["baseline"],
                }
            },
        }
    )


def _write_experiment(
    root: Path,
    *,
    experiment_id: str,
    modality: str,
    layers: list[str],
    samples: tuple[str, ...] = ("sample_a", "sample_b"),
) -> dict:
    run_root = root / experiment_id
    raw_dir = run_root / "raw_outputs"
    preprocess_dir = run_root / "preprocess_adata_outputs"
    molecule_index = run_root / "molecule_index"
    read_index = preprocess_dir / "read_index"
    for directory in (raw_dir, preprocess_dir, molecule_index, read_index):
        directory.mkdir(parents=True, exist_ok=True)
    (raw_dir / "spine.h5ad").touch()
    (preprocess_dir / "spine.h5ad").touch()

    experiment_uid = str(uuid4())
    read_ids = [f"{experiment_id}_read_{index}" for index in range(len(samples))]
    identities = [molecule_uid(experiment_uid, read_id) for read_id in read_ids]
    pd.DataFrame(
        {
            "molecule_uid": identities,
            "experiment_uid": experiment_uid,
            "read_id": read_ids,
            "Reference_strand": ["chr1+"] * len(read_ids),
            "Sample": list(samples),
            "Barcode": list(samples),
            "mapping_quality": [30, 10][: len(read_ids)],
            "activity": ["active", "inactive"][: len(read_ids)],
        }
    ).to_parquet(molecule_index / "part.parquet", index=False)
    pd.DataFrame({"molecule_uid": identities}).to_parquet(read_index / "part.parquet", index=False)
    task_catalog = preprocess_dir / "task_catalog.parquet"
    pd.DataFrame(
        {
            "task_id": ["task-0"],
            "reference": ["chr1+"],
            "layers": [layers],
        }
    ).to_parquet(task_catalog, index=False)
    interval_catalog = raw_dir / "interval_catalog.parquet"
    pd.DataFrame({"reference": ["chr1+"], "max_end": [100]}).to_parquet(
        interval_catalog, index=False
    )
    return {
        "path": str(run_root),
        "name": experiment_id,
        "experiment_uid": experiment_uid,
        "modality": modality,
        "schema_version": 1,
        "spines": {
            "raw": str(raw_dir / "spine.h5ad"),
            "preprocess": str(preprocess_dir / "spine.h5ad"),
        },
        "references": {"chr1+": "reference-uid"},
        "n_reads": len(read_ids),
        "status": "active",
        "catalogs": {
            "interval_catalog.parquet": str(interval_catalog),
            "molecule_index": str(molecule_index),
            "preprocess_read_index": str(read_index),
            "preprocess_task_catalog": str(task_catalog),
        },
    }


def _project(tmp_path: Path, entries: dict[str, dict], *, set_ids: list[str] | None = None) -> Path:
    project = tmp_path / "project"
    init_project(project)
    registry = load_registry(project)
    registry["experiments"] = entries
    if set_ids is not None:
        registry["sets"]["training"] = {"kind": "list", "experiments": set_ids}
    save_registry(project, registry)
    ReferenceRegistry(canonical_names={"reference-uid": "locus"}).save(
        project / "reference_registry.yaml"
    )
    return project


def _mixed_channels() -> list[dict]:
    return [
        {
            "name": "accessibility",
            "biological_role": "accessibility",
            "sources": [
                {
                    "modality": "deaminase",
                    "stage": "preprocess",
                    "layer": "C_site_binary",
                    "site_context": "C",
                },
                {
                    "modality": "conversion",
                    "stage": "preprocess",
                    "layer": "GpC_site_binary",
                    "site_context": "GpC",
                },
            ],
        },
        {
            "name": "endogenous_methylation",
            "biological_role": "endogenous_methylation",
            "sources": [
                {
                    "modality": "conversion",
                    "stage": "preprocess",
                    "layer": "CpG_site_binary",
                    "site_context": "CpG",
                }
            ],
        },
    ]


def test_project_selection_resolves_mixed_modalities_without_opening_spines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entries = {
        "deam": _write_experiment(
            tmp_path,
            experiment_id="deam",
            modality="deaminase",
            layers=["C_site_binary"],
        ),
        "conversion": _write_experiment(
            tmp_path,
            experiment_id="conversion",
            modality="conversion",
            layers=["GpC_site_binary", "CpG_site_binary"],
        ),
    }
    project = _project(tmp_path, entries)
    plan = _plan(
        modalities=["deaminase", "conversion"],
        channels=_mixed_channels(),
        channel_policy="union",
    )

    def fail_matrix_read(*args, **kwargs):
        raise AssertionError("selection planning must not open a feature matrix")

    monkeypatch.setattr("smftools.readwrite.safe_read_h5ad", fail_matrix_read)
    result = plan_ml_dataset(plan, "reads", project_dir=project)

    assert result.n_observations == 2
    assert result.n_features == 100
    assert result.modality_counts == {"conversion": 1, "deaminase": 1}
    assert result.class_counts == {"1": 2}
    assert result.group_by == ("Sample", "experiment_uid")
    assert set(result.identity_table["reference"]) == {"locus"}
    assert [len(source.channels) for source in result.sources] == [2, 1]
    assert result.estimated_materialization_bytes == 2 * 100 * 2 * 6
    assert result.to_dry_run_dict()["selection_id"] == result.selection_id


def test_named_project_set_limits_experiments_without_copying_data(tmp_path: Path) -> None:
    entries = {
        name: _write_experiment(
            tmp_path,
            experiment_id=name,
            modality="deaminase",
            layers=["C_site_binary"],
        )
        for name in ("included", "excluded")
    }
    project = _project(tmp_path, entries, set_ids=["included"])
    index_file = tmp_path / "included" / "molecule_index" / "part.parquet"
    index = pd.read_parquet(index_file)
    index.loc[index["Sample"] == "sample_b", "mapping_quality"] = 30
    index.to_parquet(index_file, index=False)

    result = plan_ml_dataset(
        _plan(set_name="training", samples=["included/sample_b"]),
        "reads",
        project_dir=project,
    )

    assert [source.experiment_id for source in result.sources] == ["included"]
    assert result.sample_counts == {"sample_b": 1}
    assert (
        result.sources[0].membership_artifact
        == (tmp_path / "included" / "molecule_index").resolve()
    )


def test_selection_identity_changes_when_eligible_membership_changes(tmp_path: Path) -> None:
    entry = _write_experiment(
        tmp_path,
        experiment_id="deam",
        modality="deaminase",
        layers=["C_site_binary"],
    )
    project = _project(tmp_path, {"deam": entry})
    plan = _plan()
    first = plan_ml_dataset(plan, "reads", project_dir=project)

    index_file = tmp_path / "deam" / "molecule_index" / "part.parquet"
    frame = pd.read_parquet(index_file)
    frame.loc[1, "mapping_quality"] = 30
    frame.to_parquet(index_file, index=False)
    second = plan_ml_dataset(plan, "reads", project_dir=project)

    assert first.n_observations == 1
    assert second.n_observations == 2
    assert first.membership_fingerprint != second.membership_fingerprint
    assert first.selection_id != second.selection_id

    interval_file = tmp_path / "deam" / "raw_outputs" / "interval_catalog.parquet"
    intervals = pd.read_parquet(interval_file)
    intervals.loc[0, "max_end"] = 120
    intervals.to_parquet(interval_file, index=False)
    third = plan_ml_dataset(plan, "reads", project_dir=project)

    assert third.n_features == 120
    assert second.feature_fingerprint != third.feature_fingerprint
    assert second.selection_id != third.selection_id


def test_selection_rejects_missing_layer_and_ambiguous_cpg_role(tmp_path: Path) -> None:
    entry = _write_experiment(
        tmp_path,
        experiment_id="conversion",
        modality="conversion",
        layers=["GpC_site_binary"],
    )
    project = _project(tmp_path, {"conversion": entry})
    with pytest.raises(MLSelectionError, match="CpG_site_binary.*unavailable"):
        plan_ml_dataset(_plan(modalities=["conversion"]), "reads", project_dir=project)

    ambiguous = [
        {
            "name": "cpg",
            "biological_role": "unknown",
            "sources": [
                {
                    "modality": "conversion",
                    "stage": "preprocess",
                    "layer": "GpC_site_binary",
                    "site_context": "CpG",
                }
            ],
        }
    ]
    with pytest.raises(MLSelectionError, match="ambiguous biological meaning"):
        plan_ml_dataset(
            _plan(modalities=["conversion"], channels=ambiguous),
            "reads",
            project_dir=project,
        )


def test_selection_rejects_unknown_project_modality(tmp_path: Path) -> None:
    entry = _write_experiment(
        tmp_path,
        experiment_id="unknown",
        modality="unknown",
        layers=["C_site_binary"],
    )
    project = _project(tmp_path, {"unknown": entry})

    with pytest.raises(MLSelectionError, match="no known modality"):
        plan_ml_dataset(_plan(), "reads", project_dir=project)


def test_experiment_scope_resolves_current_preprocess_generation(tmp_path: Path) -> None:
    entry = _write_experiment(
        tmp_path,
        experiment_id="deam",
        modality="deaminase",
        layers=["unused"],
    )
    run_root = Path(entry["path"])
    preprocess_dir = run_root / "preprocess_adata_outputs"
    generation = preprocess_dir / "generations" / "generation-1"
    (generation / "read_index").mkdir(parents=True)
    identities = pd.read_parquet(run_root / "molecule_index" / "part.parquet")["molecule_uid"]
    pd.DataFrame({"molecule_uid": identities}).to_parquet(
        generation / "read_index" / "part.parquet", index=False
    )
    pd.DataFrame(
        {
            "task_id": ["task-0"],
            "reference": ["chr1+"],
            "layers": [["C_site_binary"]],
        }
    ).to_parquet(generation / "task_catalog.parquet", index=False)
    (preprocess_dir / "current.json").write_text(
        json.dumps({"generation_path": "generations/generation-1"}),
        encoding="utf-8",
    )
    update_experiment_manifest(
        run_root,
        experiment="deam",
        experiment_uid=entry["experiment_uid"],
        modality="deaminase",
        reference_uids={"chr1+": "reference-uid"},
    )
    record_stage_completion(run_root, "raw")
    record_stage_completion(run_root, "preprocess")

    result = plan_ml_dataset(
        _plan(scope="experiment", references=["chr1+"]),
        "reads",
        experiment_dir=run_root,
    )

    assert result.scope_kind == "experiment"
    assert result.scope_id == "deam"
    assert result.sources[0].channels[0].layer == "C_site_binary"
