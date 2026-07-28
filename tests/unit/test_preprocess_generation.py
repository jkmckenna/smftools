from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import pandas as pd
import pytest

from smftools.informatics.experiment_spine import experiment_spine_path
from smftools.informatics.partition_read import relative_uns_path
from smftools.informatics.raw_store import write_raw_store
from smftools.informatics.sidecar_manifest import register_sidecar, sidecar_manifest_path
from smftools.pipeline import (
    AnalysisScope,
    ChannelDependency,
    ChannelSpec,
    NodeInputs,
    PlanState,
    SemanticNodeSpec,
)
from smftools.preprocessing import preprocess_generation
from smftools.preprocessing.preprocess_generation import (
    PREPROCESS_CURRENT_FILENAME,
    PREPROCESS_GENERATION_MANIFEST,
    PreprocessGenerationError,
    publish_preprocess_generation,
    resolve_current_preprocess_generation,
)
from smftools.preprocessing.semantic_upgrade import (
    PREPROCESS_PLOTS_NODE,
    PREPROCESS_REDUCERS_NODE,
    PREPROCESS_TASKS_NODE,
    PREPROCESS_VARIANT_EVIDENCE_NODE,
    PREPROCESS_VARIANT_REFERENCE_NODE,
    load_preprocess_node_results,
    plan_preprocess_upgrade,
)
from smftools.readwrite import safe_write_h5ad

pytestmark = pytest.mark.unit


def _cfg(tmp_path):
    return SimpleNamespace(
        output_directory=tmp_path,
        experiment_name="experiment",
        smf_modality="conversion",
    )


def _source(tmp_path):
    path = tmp_path / "raw_outputs" / "spine.h5ad"
    path.parent.mkdir(parents=True)
    safe_write_h5ad(
        ad.AnnData(obs=pd.DataFrame(index=["read-1"])),
        path,
        backup=False,
        verbose=False,
    )
    return path


def _fake_executor(
    spine_path,
    cfg,
    output_dir,
    *,
    publication_dir,
    run_root,
    refresh_experiment_spine,
):
    del spine_path, cfg
    assert refresh_experiment_spine is False
    output_dir = Path(output_dir)
    publication_dir = Path(publication_dir)
    run_root = Path(run_root)
    store = output_dir / "store"
    group = store / "task-1"
    group.mkdir(parents=True)
    (group / "zarr.json").write_text("{}\n", encoding="utf-8")
    read_index = output_dir / "read_index"
    read_index.mkdir()
    (read_index / "part.parquet").write_bytes(b"index")
    task_catalog = output_dir / "task_catalog.parquet"
    catalog = output_dir / "catalog.parquet"
    pd.DataFrame({"task_id": ["task-1"]}).to_parquet(task_catalog, index=False)
    pd.DataFrame({"task_id": ["task-1"], "group_path": ["store/task-1"]}).to_parquet(
        catalog,
        index=False,
    )
    for name in ("var.parquet", "obs.parquet", "stage_obs.parquet"):
        pd.DataFrame({"value": [1]}).to_parquet(output_dir / name, index=False)
    plot_catalog = output_dir / "plots" / "catalog.parquet"
    plot_catalog.parent.mkdir()
    pd.DataFrame(columns=["path"]).to_parquet(plot_catalog, index=False)

    spine = ad.AnnData(obs=pd.DataFrame(index=["read-1"]))
    pointers = {
        "preprocess_store": publication_dir / "store",
        "preprocess_catalog": publication_dir / "catalog.parquet",
        "preprocess_task_catalog": publication_dir / "task_catalog.parquet",
        "preprocess_read_index": publication_dir / "read_index",
        "preprocess_var": publication_dir / "var.parquet",
        "preprocess_obs": publication_dir / "obs.parquet",
        "preprocess_stage_obs": publication_dir / "stage_obs.parquet",
        "preprocess_plot_catalog": publication_dir / "plots" / "catalog.parquet",
    }
    for key, path in pointers.items():
        spine.uns[key] = relative_uns_path(path, run_root)
    spine_path = output_dir / "spine.h5ad"
    safe_write_h5ad(spine, spine_path, backup=False, verbose=False)

    sidecars = sidecar_manifest_path(output_dir)
    for key, path in {
        "preprocess_store": store,
        "preprocess_catalog": catalog,
        "preprocess_task_catalog": task_catalog,
        "preprocess_read_index": read_index,
        "preprocess_var": output_dir / "var.parquet",
        "preprocess_obs": output_dir / "obs.parquet",
        "preprocess_stage_obs": output_dir / "stage_obs.parquet",
        "preprocess_spine": spine_path,
        "preprocess_plot_catalog": plot_catalog,
    }.items():
        register_sidecar(sidecars, key, path)
    return {"spine": spine_path}


def test_preprocess_generation_publishes_valid_current_and_canonical_spine(tmp_path):
    source = _source(tmp_path)
    output_dir = tmp_path / "preprocess_adata_outputs"

    outputs = publish_preprocess_generation(
        source,
        _cfg(tmp_path),
        output_dir,
        executor=_fake_executor,
    )

    current = resolve_current_preprocess_generation(output_dir)
    assert current is not None
    generation, manifest = current
    assert generation == outputs["generation"]
    assert manifest["generation_id"] == outputs["generation_id"]
    assert outputs["generation_spine"] == generation / "spine.h5ad"
    assert outputs["spine"] == output_dir / "spine.h5ad"
    assert Path(outputs["spine"]).is_file()
    assert not any((output_dir / ".staging").iterdir())


def test_real_partitioned_executor_publishes_generation_scoped_outputs(tmp_path):
    from tests.unit.test_partitioned_preprocess_executor import _cfg as executor_cfg
    from tests.unit.test_partitioned_preprocess_executor import _frame

    raw = write_raw_store(
        _frame(),
        tmp_path / "raw_outputs",
        reference_lengths={"ref_top": 12},
        extra_uns={"References": {"ref_FASTA_sequence": "ACGCGTACGTAC"}},
    )
    cfg = executor_cfg()
    cfg.output_directory = tmp_path
    cfg.experiment_name = "experiment"
    output_dir = tmp_path / "preprocess_adata_outputs"

    outputs = publish_preprocess_generation(raw["spine"], cfg, output_dir)

    generation = Path(outputs["generation"])
    spine = ad.read_h5ad(outputs["spine"])
    assert Path(outputs["store"]).is_relative_to(generation)
    assert Path(outputs["read_index"]).is_relative_to(generation)
    assert Path(outputs["plot_catalog"]).is_relative_to(generation)
    assert spine.uns["preprocess_generation_id"] == outputs["generation_id"]
    assert (
        spine.uns["preprocess_task_catalog"]
        == f"preprocess_adata_outputs/generations/{outputs['generation_id']}/task_catalog.parquet"
    )
    experiment_spine = ad.read_h5ad(experiment_spine_path(tmp_path))
    assert "passes_qc" in experiment_spine.obs


def test_failure_before_staged_artifacts_publishes_no_generation(tmp_path):
    source = _source(tmp_path)
    output_dir = tmp_path / "preprocess_adata_outputs"

    def fail(*_args, **_kwargs):
        raise RuntimeError("injected task failure")

    with pytest.raises(RuntimeError, match="injected task failure"):
        publish_preprocess_generation(source, _cfg(tmp_path), output_dir, executor=fail)

    assert not (output_dir / PREPROCESS_CURRENT_FILENAME).exists()
    assert not list((output_dir / "generations").iterdir())
    assert not any((output_dir / ".staging").iterdir())


def test_validation_failure_preserves_prior_current_generation(tmp_path, monkeypatch):
    source = _source(tmp_path)
    output_dir = tmp_path / "preprocess_adata_outputs"
    first = publish_preprocess_generation(
        source,
        _cfg(tmp_path),
        output_dir,
        executor=_fake_executor,
    )
    original_validate = preprocess_generation.validate_preprocess_generation

    def fail_staging(path, **kwargs):
        if ".staging" in Path(path).parts:
            raise RuntimeError("injected validation failure")
        return original_validate(path, **kwargs)

    monkeypatch.setattr(preprocess_generation, "validate_preprocess_generation", fail_staging)
    with pytest.raises(RuntimeError, match="injected validation failure"):
        publish_preprocess_generation(
            source,
            _cfg(tmp_path),
            output_dir,
            executor=_fake_executor,
        )

    current = resolve_current_preprocess_generation(output_dir)
    assert current is not None
    assert current[1]["generation_id"] == first["generation_id"]
    assert {path.name for path in (output_dir / "generations").iterdir()} == {
        first["generation_id"]
    }


def test_current_pointer_failure_preserves_prior_generation(tmp_path, monkeypatch):
    source = _source(tmp_path)
    output_dir = tmp_path / "preprocess_adata_outputs"
    first = publish_preprocess_generation(
        source,
        _cfg(tmp_path),
        output_dir,
        executor=_fake_executor,
    )
    original_write = preprocess_generation.atomic_write_json

    def fail_current(path, *args, **kwargs):
        if Path(path).name == PREPROCESS_CURRENT_FILENAME:
            raise RuntimeError("injected pointer failure")
        return original_write(path, *args, **kwargs)

    monkeypatch.setattr(preprocess_generation, "atomic_write_json", fail_current)
    with pytest.raises(RuntimeError, match="injected pointer failure"):
        publish_preprocess_generation(
            source,
            _cfg(tmp_path),
            output_dir,
            executor=_fake_executor,
        )

    current = resolve_current_preprocess_generation(output_dir)
    assert current is not None
    assert current[1]["generation_id"] == first["generation_id"]
    assert {path.name for path in (output_dir / "generations").iterdir()} == {
        first["generation_id"]
    }
    canonical = ad.read_h5ad(output_dir / "spine.h5ad")
    assert canonical.uns["preprocess_generation_id"] == first["generation_id"]


@pytest.mark.parametrize("damage", ["malformed", "manifest_checksum", "artifact_checksum"])
def test_current_pointer_rejects_malformed_or_mismatched_manifest(tmp_path, damage):
    source = _source(tmp_path)
    output_dir = tmp_path / "preprocess_adata_outputs"
    outputs = publish_preprocess_generation(
        source,
        _cfg(tmp_path),
        output_dir,
        executor=_fake_executor,
    )
    current_path = output_dir / PREPROCESS_CURRENT_FILENAME
    if damage == "malformed":
        current_path.write_text("{", encoding="utf-8")
    elif damage == "manifest_checksum":
        manifest_path = Path(outputs["generation"]) / PREPROCESS_GENERATION_MANIFEST
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["task_count"] = 99
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        (Path(outputs["store"]) / "task-1" / "unexpected.bin").write_bytes(b"corrupt")

    with pytest.raises(PreprocessGenerationError):
        resolve_current_preprocess_generation(output_dir)


def test_relocated_experiment_resolves_current_generation(tmp_path):
    source = _source(tmp_path / "original")
    original = tmp_path / "original" / "preprocess_adata_outputs"
    first = publish_preprocess_generation(
        source,
        _cfg(tmp_path / "original"),
        original,
        executor=_fake_executor,
    )
    relocated_root = tmp_path / "relocated"
    shutil.copytree(tmp_path / "original", relocated_root)

    current = resolve_current_preprocess_generation(relocated_root / "preprocess_adata_outputs")

    assert current is not None
    assert current[1]["generation_id"] == first["generation_id"]
    assert current[0].is_relative_to(relocated_root)


def test_legacy_canonical_spine_without_pointer_requires_recompute(tmp_path):
    output_dir = tmp_path / "preprocess_adata_outputs"
    output_dir.mkdir()
    safe_write_h5ad(
        ad.AnnData(),
        output_dir / "spine.h5ad",
        backup=False,
        verbose=False,
    )

    assert resolve_current_preprocess_generation(output_dir) is None


def test_compatible_generation_is_reused_without_invoking_executor(tmp_path):
    source = _source(tmp_path)
    output_dir = tmp_path / "preprocess_adata_outputs"
    first = publish_preprocess_generation(
        source,
        _cfg(tmp_path),
        output_dir,
        executor=_fake_executor,
    )

    def fail(*_args, **_kwargs):
        raise AssertionError("compatible preprocess nodes must not execute")

    second = publish_preprocess_generation(
        source,
        _cfg(tmp_path),
        output_dir,
        executor=fail,
    )

    assert second["generation_id"] != first["generation_id"]
    assert Path(first["generation"]).is_dir()
    manifest = json.loads(Path(second["generation_manifest"]).read_text(encoding="utf-8"))
    results = load_preprocess_node_results(manifest)
    assert set(results) == {
        PREPROCESS_TASKS_NODE,
        PREPROCESS_REDUCERS_NODE,
        PREPROCESS_PLOTS_NODE,
    }
    assert {result.reused_from_generation_id for result in results.values()} == {
        first["generation_id"]
    }


@pytest.mark.parametrize(
    ("changed_key", "expected"),
    [
        (
            "bypass_clean_nan",
            {
                PREPROCESS_TASKS_NODE: PlanState.STALE_CONFIG,
                PREPROCESS_REDUCERS_NODE: PlanState.DEPENDENT_RECOMPUTE,
                PREPROCESS_PLOTS_NODE: PlanState.DEPENDENT_RECOMPUTE,
            },
        ),
        (
            "position_max_nan_threshold",
            {
                PREPROCESS_TASKS_NODE: PlanState.COMPATIBLE,
                PREPROCESS_REDUCERS_NODE: PlanState.STALE_CONFIG,
                PREPROCESS_PLOTS_NODE: PlanState.DEPENDENT_RECOMPUTE,
            },
        ),
        (
            "duplicate_detection_site_types",
            {
                PREPROCESS_TASKS_NODE: PlanState.COMPATIBLE,
                PREPROCESS_REDUCERS_NODE: PlanState.STALE_CONFIG,
                PREPROCESS_PLOTS_NODE: PlanState.DEPENDENT_RECOMPUTE,
            },
        ),
        (
            "preprocess_plot_max_heatmap_reads",
            {
                PREPROCESS_TASKS_NODE: PlanState.COMPATIBLE,
                PREPROCESS_REDUCERS_NODE: PlanState.COMPATIBLE,
                PREPROCESS_PLOTS_NODE: PlanState.STALE_CONFIG,
            },
        ),
        (
            "spatial_clustermap_sortby",
            {
                PREPROCESS_TASKS_NODE: PlanState.COMPATIBLE,
                PREPROCESS_REDUCERS_NODE: PlanState.COMPATIBLE,
                PREPROCESS_PLOTS_NODE: PlanState.COMPATIBLE,
            },
        ),
    ],
)
def test_upgrade_plan_invalidates_only_affected_nodes(tmp_path, changed_key, expected):
    source = _source(tmp_path)
    output_dir = tmp_path / "preprocess_adata_outputs"
    cfg = _cfg(tmp_path)
    setattr(cfg, changed_key, 1)
    publish_preprocess_generation(source, cfg, output_dir, executor=_fake_executor)
    current = resolve_current_preprocess_generation(output_dir)
    assert current is not None
    setattr(cfg, changed_key, 2)

    plan = plan_preprocess_upgrade(
        source,
        cfg,
        output_dir,
        current_generation=current,
    )

    assert {decision.analysis_id: decision.state for decision in plan.decisions} == expected


def test_algorithm_change_and_corrupt_artifact_have_distinct_plan_outcomes(tmp_path):
    source = _source(tmp_path)
    output_dir = tmp_path / "preprocess_adata_outputs"
    cfg = _cfg(tmp_path)
    publish_preprocess_generation(source, cfg, output_dir, executor=_fake_executor)
    current = resolve_current_preprocess_generation(output_dir)
    assert current is not None

    algorithm_plan = plan_preprocess_upgrade(
        source,
        cfg,
        output_dir,
        current_generation=current,
        algorithm_versions={PREPROCESS_REDUCERS_NODE: "2"},
    )
    algorithm_states = {
        decision.analysis_id: decision.state for decision in algorithm_plan.decisions
    }
    assert algorithm_states[PREPROCESS_TASKS_NODE] is PlanState.COMPATIBLE
    assert algorithm_states[PREPROCESS_REDUCERS_NODE] is PlanState.STALE_ALGORITHM
    assert algorithm_states[PREPROCESS_PLOTS_NODE] is PlanState.DEPENDENT_RECOMPUTE

    (current[0] / "store" / "task-1" / "unexpected.bin").write_bytes(b"corrupt")
    corrupt_plan = plan_preprocess_upgrade(
        source,
        cfg,
        output_dir,
        current_generation=current,
    )
    corrupt_states = {decision.analysis_id: decision.state for decision in corrupt_plan.decisions}
    assert corrupt_states[PREPROCESS_TASKS_NODE] is PlanState.INVALID_ARTIFACT
    assert corrupt_states[PREPROCESS_REDUCERS_NODE] is PlanState.DEPENDENT_RECOMPUTE
    assert corrupt_states[PREPROCESS_PLOTS_NODE] is PlanState.DEPENDENT_RECOMPUTE


def test_legacy_generation_without_node_provenance_reports_missing_results(tmp_path):
    source = _source(tmp_path)
    output_dir = tmp_path / "preprocess_adata_outputs"
    outputs = publish_preprocess_generation(
        source,
        _cfg(tmp_path),
        output_dir,
        executor=_fake_executor,
    )
    manifest = json.loads(Path(outputs["generation_manifest"]).read_text(encoding="utf-8"))
    manifest.pop("node_results")

    plan = plan_preprocess_upgrade(
        source,
        _cfg(tmp_path),
        output_dir,
        current_generation=(Path(outputs["generation"]), manifest),
    )

    assert all(decision.state is PlanState.MISSING for decision in plan.decisions)


def test_plot_only_upgrade_reuses_compute_without_invoking_executor(tmp_path):
    source = _source(tmp_path)
    output_dir = tmp_path / "preprocess_adata_outputs"
    cfg = _cfg(tmp_path)
    cfg.emit_automated_plots = False
    cfg.preprocess_plot_max_heatmap_reads = 10
    first = publish_preprocess_generation(source, cfg, output_dir, executor=_fake_executor)
    cfg.preprocess_plot_max_heatmap_reads = 20

    def fail(*_args, **_kwargs):
        raise AssertionError("plot-only upgrades must not execute preprocess compute")

    second = publish_preprocess_generation(source, cfg, output_dir, executor=fail)

    manifest = json.loads(Path(second["generation_manifest"]).read_text(encoding="utf-8"))
    results = load_preprocess_node_results(manifest)
    assert results[PREPROCESS_TASKS_NODE].reused_from_generation_id == first["generation_id"]
    assert results[PREPROCESS_REDUCERS_NODE].reused_from_generation_id == first["generation_id"]
    assert results[PREPROCESS_PLOTS_NODE].reused_from_generation_id is None


def test_reducer_upgrade_reuses_real_task_partitions(tmp_path, monkeypatch):
    from tests.unit.test_partitioned_preprocess_executor import _cfg as executor_cfg
    from tests.unit.test_partitioned_preprocess_executor import _frame

    raw = write_raw_store(
        _frame(),
        tmp_path / "raw_outputs",
        reference_lengths={"ref_top": 12},
        extra_uns={"References": {"ref_FASTA_sequence": "ACGCGTACGTAC"}},
    )
    cfg = executor_cfg()
    cfg.output_directory = tmp_path
    cfg.experiment_name = "experiment"
    cfg.emit_automated_plots = False
    output_dir = tmp_path / "preprocess_adata_outputs"
    first = publish_preprocess_generation(raw["spine"], cfg, output_dir)

    def fail_task(*_args, **_kwargs):
        raise AssertionError("compatible preprocess task partitions must be reused")

    monkeypatch.setattr(
        "smftools.preprocessing.partitioned_executor.execute_preprocess_task",
        fail_task,
    )
    cfg.position_max_nan_threshold = 0.7
    second = publish_preprocess_generation(raw["spine"], cfg, output_dir)

    manifest = json.loads(Path(second["generation_manifest"]).read_text(encoding="utf-8"))
    results = load_preprocess_node_results(manifest)
    assert results[PREPROCESS_TASKS_NODE].reused_from_generation_id == first["generation_id"]
    assert results[PREPROCESS_REDUCERS_NODE].reused_from_generation_id is None
    assert Path(first["generation"]).is_dir()


def test_reporting_only_upgrade_reuses_tasks_and_publishes_variant_nodes(
    tmp_path,
    monkeypatch,
):
    from tests.unit.test_partitioned_preprocess_executor import _cfg as executor_cfg
    from tests.unit.test_partitioned_preprocess_executor import _frame

    raw = write_raw_store(
        _frame(),
        tmp_path / "raw_outputs",
        reference_lengths={"ref_top": 12},
        extra_uns={
            "References": {
                "ref_FASTA_sequence": "ACGCGTACGTAC",
                "alt_FASTA_sequence": "ATGCGTACGTAC",
            }
        },
    )
    cfg = executor_cfg()
    cfg.output_directory = tmp_path
    cfg.experiment_name = "experiment"
    cfg.emit_automated_plots = False
    cfg.variant_analysis_mode = "off"
    output_dir = tmp_path / "preprocess_adata_outputs"
    first = publish_preprocess_generation(raw["spine"], cfg, output_dir)

    def fail_task(*_args, **_kwargs):
        raise AssertionError("reporting-only upgrades must reuse preprocess tasks")

    monkeypatch.setattr(
        "smftools.preprocessing.partitioned_executor.execute_preprocess_task",
        fail_task,
    )
    cfg.variant_analysis_mode = "report"
    cfg.references_to_align_for_variant_annotation = [
        "ref_top_strand_FASTA_base",
        "alt_top_strand_FASTA_base",
    ]
    second = publish_preprocess_generation(raw["spine"], cfg, output_dir)

    manifest = json.loads(Path(second["generation_manifest"]).read_text(encoding="utf-8"))
    results = load_preprocess_node_results(manifest)
    assert results[PREPROCESS_TASKS_NODE].reused_from_generation_id == first["generation_id"]
    assert results[PREPROCESS_VARIANT_REFERENCE_NODE].reused_from_generation_id is None
    assert results[PREPROCESS_VARIANT_EVIDENCE_NODE].reused_from_generation_id is None
    assert results[PREPROCESS_REDUCERS_NODE].reused_from_generation_id is None
    assert Path(second["variant_read_index"]).is_dir()


def test_corrupt_reused_copy_prevents_publication_and_preserves_current(
    tmp_path,
    monkeypatch,
):
    from tests.unit.test_partitioned_preprocess_executor import _cfg as executor_cfg
    from tests.unit.test_partitioned_preprocess_executor import _frame

    raw = write_raw_store(
        _frame(),
        tmp_path / "raw_outputs",
        reference_lengths={"ref_top": 12},
        extra_uns={"References": {"ref_FASTA_sequence": "ACGCGTACGTAC"}},
    )
    cfg = executor_cfg()
    cfg.output_directory = tmp_path
    cfg.experiment_name = "experiment"
    cfg.emit_automated_plots = False
    output_dir = tmp_path / "preprocess_adata_outputs"
    first = publish_preprocess_generation(raw["spine"], cfg, output_dir)

    original_copy = shutil.copy2

    def corrupt_catalog_copy(source, destination, *args, **kwargs):
        copied = original_copy(source, destination, *args, **kwargs)
        if Path(destination).name == "catalog.parquet":
            Path(destination).write_bytes(Path(destination).read_bytes() + b"corrupt")
        return copied

    monkeypatch.setattr(
        "smftools.preprocessing.partitioned_executor.shutil.copy2",
        corrupt_catalog_copy,
    )
    cfg.position_max_nan_threshold = 0.7

    with pytest.raises(RuntimeError, match="artifact copy is corrupt"):
        publish_preprocess_generation(raw["spine"], cfg, output_dir)

    current = resolve_current_preprocess_generation(output_dir)
    assert current is not None
    assert current[1]["generation_id"] == first["generation_id"]
    assert {path.name for path in (output_dir / "generations").iterdir()} == {
        first["generation_id"]
    }


def test_source_change_and_missing_source_have_distinct_plan_outcomes(tmp_path):
    source = _source(tmp_path)
    output_dir = tmp_path / "preprocess_adata_outputs"
    cfg = _cfg(tmp_path)
    publish_preprocess_generation(source, cfg, output_dir, executor=_fake_executor)
    current = resolve_current_preprocess_generation(output_dir)
    assert current is not None

    safe_write_h5ad(
        ad.AnnData(obs=pd.DataFrame(index=["read-1", "read-2"])),
        source,
        backup=False,
        verbose=False,
    )
    changed = plan_preprocess_upgrade(
        source,
        cfg,
        output_dir,
        current_generation=current,
    )
    assert changed.decisions[0].state is PlanState.STALE_INPUT

    source.unlink()
    missing = plan_preprocess_upgrade(
        source,
        cfg,
        output_dir,
        current_generation=current,
    )
    assert missing.decisions[0].state is PlanState.BLOCKED_MISSING_INPUT


def test_force_target_is_expressed_in_semantic_plan(tmp_path):
    source = _source(tmp_path)
    output_dir = tmp_path / "preprocess_adata_outputs"
    cfg = _cfg(tmp_path)
    publish_preprocess_generation(source, cfg, output_dir, executor=_fake_executor)
    current = resolve_current_preprocess_generation(output_dir)
    assert current is not None

    plan = plan_preprocess_upgrade(
        source,
        cfg,
        output_dir,
        current_generation=current,
        force_targets={PREPROCESS_REDUCERS_NODE},
    )
    states = {decision.analysis_id: decision.state for decision in plan.decisions}

    assert states[PREPROCESS_TASKS_NODE] is PlanState.COMPATIBLE
    assert states[PREPROCESS_REDUCERS_NODE] is PlanState.STALE_INPUT
    assert states[PREPROCESS_PLOTS_NODE] is PlanState.DEPENDENT_RECOMPUTE


def test_new_independent_node_does_not_invalidate_compatible_tasks(tmp_path):
    source = _source(tmp_path)
    output_dir = tmp_path / "preprocess_adata_outputs"
    cfg = _cfg(tmp_path)
    publish_preprocess_generation(source, cfg, output_dir, executor=_fake_executor)
    current = resolve_current_preprocess_generation(output_dir)
    assert current is not None
    new_node = SemanticNodeSpec(
        analysis_id="preprocess.new_summary",
        scope=AnalysisScope.EXPERIMENT_ANALYSIS,
        dependencies=(PREPROCESS_TASKS_NODE,),
        consumed_channels=(ChannelDependency(PREPROCESS_TASKS_NODE, "derived_partitions", 1),),
        produced_channels=(ChannelSpec("new_summary", 1),),
        semantic_config_keys=("summary_mode",),
        validator_id="preprocess.new_summary",
    )
    new_inputs = NodeInputs(
        semantic_config={"summary_mode": "default"},
        logical_scope_identity="preprocess:experiment",
        logical_task_plan_digest="experiment",
    )

    plan = plan_preprocess_upgrade(
        source,
        cfg,
        output_dir,
        current_generation=current,
        target=new_node.analysis_id,
        additional_specs=(new_node,),
        additional_inputs={new_node.analysis_id: new_inputs},
        additional_validators={"preprocess.new_summary": lambda _result: True},
    )
    states = {decision.analysis_id: decision.state for decision in plan.decisions}

    assert states[PREPROCESS_TASKS_NODE] is PlanState.COMPATIBLE
    assert states[new_node.analysis_id] is PlanState.MISSING
