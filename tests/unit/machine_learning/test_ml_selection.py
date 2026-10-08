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


# --- MLX-01: labels from a project table -----------------------------------


def _table_plan(
    *,
    keys: list[str],
    scope: str = "project",
    filters: dict | None = None,
    group_by: list[str] | None = None,
    missing: str = "drop",
    table: str = "ml/labels.parquet",
):
    return parse_ml_plan(
        {
            "schema_version": 1,
            "scope": {"kind": scope},
            "datasets": {
                "reads": {
                    "modalities": ["deaminase"],
                    "references": ["locus"],
                    "filters": filters or {},
                    "labels": {
                        "source": "table",
                        "table": table,
                        "keys": keys,
                        "column": "label",
                        "classes": {"inactive": 0, "active": 1},
                        "positive_class": "active",
                        "missing": missing,
                    },
                }
            },
            "splits": {
                "by_experiment": {
                    "strategy": "leave_one_group_out",
                    "group_by": group_by or ["experiment_uid"],
                }
            },
            "models": {"nb": {"backend": "sklearn", "family": "bernoulli_nb"}},
            "jobs": {
                "train": {
                    "action": "train",
                    "dataset": "reads",
                    "split": "by_experiment",
                    "models": ["nb"],
                }
            },
        }
    )


def _barcoded_project(tmp_path: Path) -> Path:
    entries = {
        "deam": _write_experiment(
            tmp_path,
            experiment_id="deam",
            modality="deaminase",
            layers=["C_site_binary"],
            samples=("barcode01", "barcode02"),
        )
    }
    return _project(tmp_path, entries)


def _write_labels(project: Path, rows: list[dict], name: str = "ml/labels.parquet") -> None:
    path = project / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if path.suffix == ".csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)


def test_table_labels_join_on_experiment_and_barcode_across_spellings(tmp_path: Path) -> None:
    project = _barcoded_project(tmp_path)
    # The sheet says 1 and 2; the store says barcode01 and barcode02.
    _write_labels(
        project,
        [
            {"experiment_id": "deam", "barcode": 1, "label": "active"},
            {"experiment_id": "deam", "barcode": 2, "label": "inactive"},
        ],
    )

    result = plan_ml_dataset(
        _table_plan(keys=["experiment_id", "barcode"]), "reads", project_dir=project
    )

    by_sample = dict(zip(result.identity_table["sample_id"], result.identity_table["class_id"]))
    assert by_sample == {"barcode01": 1, "barcode02": 0}
    assert result.to_dry_run_dict()["label_table_sha256"] == result.label_table_sha256


def test_table_rows_without_a_label_follow_missing(tmp_path: Path) -> None:
    project = _barcoded_project(tmp_path)
    _write_labels(project, [{"experiment_id": "deam", "barcode": "NB01", "label": "active"}])

    dropped = plan_ml_dataset(
        _table_plan(keys=["experiment_id", "barcode"]), "reads", project_dir=project
    )
    assert dropped.n_observations == 1
    with pytest.raises(MLSelectionError, match="missing values"):
        plan_ml_dataset(
            _table_plan(keys=["experiment_id", "barcode"], missing="error"),
            "reads",
            project_dir=project,
        )


def test_table_labels_select_single_molecules_by_uid(tmp_path: Path) -> None:
    project = _barcoded_project(tmp_path)
    _write_labels(
        project,
        [
            {"experiment_id": "deam", "barcode": 1, "label": "active"},
            {"experiment_id": "deam", "barcode": 2, "label": "inactive"},
        ],
    )
    every = plan_ml_dataset(
        _table_plan(keys=["experiment_id", "barcode"]), "reads", project_dir=project
    )
    uids = every.identity_table["molecule_uid"].astype(str).tolist()
    assert len(uids) == 2
    # A molecule-keyed table lists one molecule; with missing: drop it alone remains.
    _write_labels(project, [{"molecule_uid": uids[1], "label": "active"}])
    chosen = plan_ml_dataset(_table_plan(keys=["molecule_uid"]), "reads", project_dir=project)
    assert chosen.n_observations == 1
    assert chosen.identity_table["molecule_uid"].astype(str).tolist() == [uids[1]]
    assert chosen.identity_table["class_id"].tolist() == [1]


def test_table_keys_must_be_unique_after_normalisation(tmp_path: Path) -> None:
    project = _barcoded_project(tmp_path)
    _write_labels(
        project,
        [
            {"experiment_id": "deam", "barcode": 1, "label": "active"},
            {"experiment_id": "deam", "barcode": "barcode01", "label": "inactive"},
        ],
        name="ml/labels.csv",
    )
    with pytest.raises(MLSelectionError, match="sharing a key"):
        plan_ml_dataset(
            _table_plan(keys=["experiment_id", "barcode"], table="ml/labels.csv"),
            "reads",
            project_dir=project,
        )


def test_table_keys_on_reference_and_sample(tmp_path: Path) -> None:
    project = _barcoded_project(tmp_path)
    _write_labels(
        project,
        [
            {
                "reference": "locus",
                "physical_reference": "chr1+",
                "sample": "barcode01",
                "label": "active",
            },
            {
                "reference": "locus",
                "physical_reference": "chr1+",
                "sample": "barcode02",
                "label": "inactive",
            },
        ],
        name="ml/labels.csv",
    )
    result = plan_ml_dataset(
        _table_plan(keys=["reference", "physical_reference", "sample"], table="ml/labels.csv"),
        "reads",
        project_dir=project,
    )
    assert result.class_counts == {"0": 1, "1": 1}


def test_filters_and_groups_may_name_table_columns(tmp_path: Path) -> None:
    project = _barcoded_project(tmp_path)
    _write_labels(
        project,
        [
            {"experiment_id": "deam", "barcode": 1, "label": "active", "harvest": "fresh"},
            {"experiment_id": "deam", "barcode": 2, "label": "inactive", "harvest": "cycling"},
        ],
    )
    result = plan_ml_dataset(
        _table_plan(
            keys=["experiment_id", "barcode"],
            filters={"harvest": "fresh"},
            group_by=["experiment_uid", "harvest"],
        ),
        "reads",
        project_dir=project,
    )
    assert result.n_observations == 1
    assert list(result.identity_table["harvest"]) == ["fresh"]


def test_table_columns_may_not_shadow_stored_metadata(tmp_path: Path) -> None:
    project = _barcoded_project(tmp_path)
    _write_labels(
        project,
        [{"experiment_id": "deam", "barcode": 1, "label": "active", "activity": "inactive"}],
    )
    with pytest.raises(MLSelectionError, match="collide with molecule metadata"):
        plan_ml_dataset(
            _table_plan(keys=["experiment_id", "barcode"]), "reads", project_dir=project
        )


def test_selection_identity_changes_with_table_content(tmp_path: Path) -> None:
    project = _barcoded_project(tmp_path)
    plan = _table_plan(keys=["experiment_id", "barcode"])
    rows = [
        {"experiment_id": "deam", "barcode": 1, "label": "active"},
        {"experiment_id": "deam", "barcode": 2, "label": "inactive"},
    ]
    _write_labels(project, rows)
    first = plan_ml_dataset(plan, "reads", project_dir=project)
    _write_labels(project, [{**row, "label": "active"} for row in rows])
    second = plan_ml_dataset(plan, "reads", project_dir=project)

    assert first.membership_fingerprint == second.membership_fingerprint
    assert first.selection_id != second.selection_id


def test_missing_label_table_is_reported(tmp_path: Path) -> None:
    project = _barcoded_project(tmp_path)
    with pytest.raises(MLSelectionError, match="label table not found"):
        plan_ml_dataset(
            _table_plan(keys=["experiment_id", "barcode"]), "reads", project_dir=project
        )


# --- MLX-05: stores as the pipeline writes them (F66) ----------------------


def _x_channel(site_context: str = "C") -> list[dict]:
    return [
        {
            "name": "accessibility",
            "biological_role": "accessibility",
            "sources": [
                {
                    "modality": "deaminase",
                    "stage": "preprocess",
                    "layer": "X",
                    "site_context": site_context,
                }
            ],
        }
    ]


def _pipeline_shaped_project(tmp_path: Path, *, stage_obs: dict | None = None) -> Path:
    """Planner catalog without layers, written-store catalog beside the read index."""
    entry = _write_experiment(
        tmp_path, experiment_id="deam", modality="deaminase", layers=["nan0_0minus1"]
    )
    entry["catalogs"].pop("preprocess_task_catalog")
    preprocess = Path(entry["path"]) / "preprocess_adata_outputs"
    planner = pd.read_parquet(preprocess / "task_catalog.parquet").drop(columns=["layers"])
    planner.to_parquet(preprocess / "task_catalog.parquet", index=False)
    written = planner.assign(layers=[["nan0_0minus1", "nan_half"]], has_x=[True])
    written.to_parquet(preprocess / "catalog.parquet", index=False)
    if stage_obs is not None:
        pd.DataFrame(stage_obs).to_parquet(preprocess / "stage_obs.parquet", index=False)
    return _project(tmp_path, {"deam": entry})


def test_selection_reads_the_written_store_catalog_and_x(tmp_path: Path) -> None:
    project = _pipeline_shaped_project(tmp_path)
    result = plan_ml_dataset(_plan(channels=_x_channel()), "reads", project_dir=project)
    assert result.sources[0].channels[0].layer == "X"


def test_missing_layer_error_names_what_the_stage_wrote(tmp_path: Path) -> None:
    project = _pipeline_shaped_project(tmp_path)
    with pytest.raises(MLSelectionError, match=r"wrote \['X', 'nan0_0minus1', 'nan_half'\]"):
        plan_ml_dataset(_plan(), "reads", project_dir=project)


def test_deaminase_gpc_subset_is_accessibility(tmp_path: Path) -> None:
    project = _pipeline_shaped_project(tmp_path)
    result = plan_ml_dataset(_plan(channels=_x_channel("GpC")), "reads", project_dir=project)
    assert result.sources[0].channels[0].site_context == "GpC"
    with pytest.raises(MLSelectionError, match="C or GpC sites"):
        plan_ml_dataset(_plan(channels=_x_channel("CpG")), "reads", project_dir=project)


def test_filters_reach_stage_obs_qc_flags(tmp_path: Path) -> None:
    project = _pipeline_shaped_project(
        tmp_path,
        stage_obs={"read_id": ["deam_read_0", "deam_read_1"], "passes_dedup": [False, True]},
    )
    plan = parse_ml_plan(
        {
            **_plan(channels=_x_channel()).to_dict(),
            "datasets": {
                "reads": {
                    **_plan(channels=_x_channel()).to_dict()["datasets"]["reads"],
                    "filters": {"passes_dedup": True},
                }
            },
        }
    )
    result = plan_ml_dataset(plan, "reads", project_dir=project)
    assert list(result.identity_table["read_id"]) == ["deam_read_1"]
