import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import pandas as pd
import pytest

from smftools.informatics.raw_generation import (
    RAW_CURRENT_FILENAME,
    RAW_GENERATION_MANIFEST,
    RawGenerationError,
    publish_raw_generation,
    resolve_current_raw_generation,
    validate_raw_generation,
)
from smftools.readwrite import safe_read_h5ad, safe_write_h5ad


def _publication_sources(run_root: Path):
    raw_root = run_root / "raw_outputs"
    raw_store = raw_root / "raw"
    raw_store.mkdir(parents=True, exist_ok=True)
    (raw_store / "part-00000.parquet").write_bytes(b"raw-shard")
    pd.DataFrame({"reference": ["ref"]}).to_parquet(
        raw_root / "interval_catalog.parquet", index=False
    )
    pd.DataFrame({"read_id": ["read-1"]}).to_parquet(raw_root / "obs.parquet", index=False)
    pd.DataFrame({"read_id": ["read-1"]}).to_parquet(run_root / "molecules.parquet", index=False)
    pd.DataFrame({"read_id": ["read-1"], "segment_read_id": ["read-1"]}).to_parquet(
        run_root / "segments.parquet", index=False
    )
    molecule_index = run_root / "molecule_index"
    molecule_index.mkdir(exist_ok=True)
    pd.DataFrame({"read_id": ["read-1"]}).to_parquet(
        molecule_index / "part-00000.parquet", index=False
    )
    segment_index = run_root / "segment_index"
    segment_index.mkdir(exist_ok=True)
    pd.DataFrame({"segment_read_id": ["read-1"]}).to_parquet(
        segment_index / "part-00000.parquet", index=False
    )
    pd.DataFrame({"reference": ["ref"]}).to_parquet(
        run_root / "reference_interval_map.parquet", index=False
    )
    input_manifest = raw_root / "input_manifest"
    input_manifest.mkdir(exist_ok=True)
    (input_manifest / "resolved_input_manifest.csv").write_text(
        "schema_version,path\n1,input.bam\n", encoding="utf-8"
    )
    (input_manifest / "resolved_input_manifest.json").write_text("{}\n", encoding="utf-8")
    (input_manifest / "input_resolution_report.json").write_text("{}\n", encoding="utf-8")
    spine = ad.AnnData(obs=pd.DataFrame(index=["read-1"]))
    spine.uns["is_spine"] = True
    safe_write_h5ad(spine, raw_root / "spine.h5ad", backup=False, verbose=False)

    region_root = run_root / "region_catalogs"
    region_root.mkdir(exist_ok=True)
    alignment_regions = region_root / "alignment_regions.parquet"
    pd.DataFrame({"reference": ["ref"]}).to_parquet(alignment_regions, index=False)
    dependency = raw_root / "intermediates" / "alignment.bam"
    dependency.parent.mkdir(exist_ok=True)
    dependency.write_bytes(b"bam")
    sources = {
        "spine": raw_root / "spine.h5ad",
        "ragged_store": raw_store,
        "interval_catalog": raw_root / "interval_catalog.parquet",
        "obs": raw_root / "obs.parquet",
        "molecules": run_root / "molecules.parquet",
        "molecule_index": molecule_index,
        "segments": run_root / "segments.parquet",
        "segment_index": segment_index,
        "reference_interval_map": run_root / "reference_interval_map.parquet",
        "input_manifest_csv": input_manifest / "resolved_input_manifest.csv",
        "input_manifest_json": input_manifest / "resolved_input_manifest.json",
        "input_resolution_report": input_manifest / "input_resolution_report.json",
    }
    return sources, {"alignment-bam": dependency}, {"alignment": alignment_regions}


def _publish(run_root: Path, *, generation_id: str):
    sources, dependencies, regions = _publication_sources(run_root)
    return publish_raw_generation(
        run_root,
        sources,
        config_hash="config-a",
        input_artifact_ids=["input-manifest:abc"],
        dependencies=dependencies,
        region_artifacts=regions,
        generation_id=generation_id,
    )


def test_publish_selects_valid_relocatable_generation(tmp_path):
    outputs = _publish(tmp_path, generation_id="generation-a")

    generation, manifest = resolve_current_raw_generation(tmp_path / "raw_outputs")
    assert generation == outputs["generation"]
    assert manifest["generation_id"] == "generation-a"
    assert manifest["config_hash"] == "config-a"
    assert manifest["input_artifact_ids"] == ["input-manifest:abc"]
    assert outputs["spine"] == generation / "spine.h5ad"

    relocated = tmp_path.parent / f"{tmp_path.name}-relocated"
    shutil.copytree(tmp_path, relocated)
    moved_generation, moved_manifest = resolve_current_raw_generation(relocated / "raw_outputs")
    assert moved_generation == relocated / "raw_outputs" / "generations" / "generation-a"
    assert moved_manifest["generation_id"] == "generation-a"


def test_schema_one_generation_remains_valid_without_segment_artifacts(tmp_path):
    outputs = _publish(tmp_path, generation_id="generation-a")
    generation = Path(outputs["generation"])
    spine_path = generation / "spine.h5ad"
    spine, _ = safe_read_h5ad(spine_path, verbose=False)
    spine.uns.pop("segments_catalog", None)
    spine.uns.pop("segment_index", None)
    safe_write_h5ad(spine, spine_path, backup=False, verbose=False)
    shutil.rmtree(generation / "segment_index")
    (generation / "segments.parquet").unlink()

    manifest_path = generation / RAW_GENERATION_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 1
    manifest["artifacts"].pop("segments")
    manifest["artifacts"].pop("segment_index")
    manifest["artifacts"]["spine"]["sha256"] = hashlib.sha256(spine_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    validated = validate_raw_generation(
        generation,
        expected_generation_id="generation-a",
        final_dir=generation,
        run_root=tmp_path,
    )

    assert validated["schema_version"] == 1


def test_corrupt_artifact_behind_current_pointer_is_rejected(tmp_path):
    outputs = _publish(tmp_path, generation_id="generation-a")
    Path(outputs["interval_catalog"]).write_bytes(b"corrupt")

    with pytest.raises(RawGenerationError, match="missing or corrupt"):
        resolve_current_raw_generation(tmp_path / "raw_outputs")


def test_invalid_or_corrupt_current_pointer_is_rejected(tmp_path):
    _publish(tmp_path, generation_id="generation-a")
    current = tmp_path / "raw_outputs" / RAW_CURRENT_FILENAME
    current.write_text("not-json", encoding="utf-8")
    with pytest.raises(RawGenerationError, match="unreadable"):
        resolve_current_raw_generation(tmp_path / "raw_outputs")

    current.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generation_id": "generation-a",
                "generation_path": "../../outside",
                "manifest_sha256": "invalid",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RawGenerationError, match="not portable"):
        resolve_current_raw_generation(tmp_path / "raw_outputs")


def test_downstream_paths_select_generation_and_reject_corrupt_pointer(tmp_path):
    from smftools.cli.helpers import get_adata_paths
    from smftools.informatics.experiment_spine import write_experiment_spine
    from smftools.project.registry import discover_stage_spines
    from smftools.readwrite import safe_read_h5ad

    outputs = _publish(tmp_path, generation_id="generation-a")
    cfg = SimpleNamespace(
        output_directory=tmp_path,
        experiment_name="experiment",
        smf_modality="conversion",
    )
    assert get_adata_paths(cfg).raw_spine == outputs["spine"]
    run_root, spines = discover_stage_spines(tmp_path)
    assert run_root == tmp_path
    assert spines["raw"] == outputs["spine"]
    consolidated_path = write_experiment_spine(tmp_path)
    consolidated, _ = safe_read_h5ad(consolidated_path, verbose=False)
    assert consolidated.uns["source_base_dir"] == ("raw_outputs/generations/generation-a")

    current = tmp_path / "raw_outputs" / RAW_CURRENT_FILENAME
    current.write_text("not-json", encoding="utf-8")
    with pytest.raises(RawGenerationError, match="unreadable"):
        get_adata_paths(cfg)
    assert get_adata_paths(cfg, allow_invalid_raw=True).raw_spine == (
        tmp_path / "raw_outputs" / "spine.h5ad"
    )


def test_publish_replaces_corrupt_current_pointer(tmp_path):
    (tmp_path / "raw_outputs").mkdir()
    current = tmp_path / "raw_outputs" / RAW_CURRENT_FILENAME
    current.write_text("not-json", encoding="utf-8")

    outputs = _publish(tmp_path, generation_id="generation-a")

    selected, manifest = resolve_current_raw_generation(tmp_path / "raw_outputs")
    assert selected == outputs["generation"]
    assert manifest["generation_id"] == "generation-a"


def test_failed_replacement_preserves_prior_current_generation(tmp_path, monkeypatch):
    first = _publish(tmp_path, generation_id="generation-a")
    sources, dependencies, regions = _publication_sources(tmp_path)
    import smftools.informatics.raw_generation as raw_generation

    real_atomic_write = raw_generation.atomic_write_json

    def fail_current(path, payload):
        if Path(path).name == RAW_CURRENT_FILENAME:
            raise RuntimeError("injected pointer failure")
        return real_atomic_write(path, payload)

    monkeypatch.setattr(raw_generation, "atomic_write_json", fail_current)
    with pytest.raises(RuntimeError, match="injected pointer failure"):
        publish_raw_generation(
            tmp_path,
            sources,
            config_hash="config-b",
            input_artifact_ids=["input-manifest:def"],
            dependencies=dependencies,
            region_artifacts=regions,
            generation_id="generation-b",
        )

    selected, manifest = resolve_current_raw_generation(tmp_path / "raw_outputs")
    assert selected == first["generation"]
    assert manifest["generation_id"] == "generation-a"
    assert not (tmp_path / "raw_outputs" / "generations" / "generation-b").exists()


def test_staging_copy_failure_preserves_prior_current_generation(tmp_path, monkeypatch):
    first = _publish(tmp_path, generation_id="generation-a")
    sources, dependencies, regions = _publication_sources(tmp_path)
    import smftools.informatics.raw_generation as raw_generation

    real_copy = raw_generation._copy_artifact

    def fail_molecule_index(source, destination):
        if Path(destination).name == "molecule_index":
            raise RuntimeError("injected molecule-index failure")
        return real_copy(source, destination)

    monkeypatch.setattr(raw_generation, "_copy_artifact", fail_molecule_index)
    with pytest.raises(RuntimeError, match="injected molecule-index failure"):
        publish_raw_generation(
            tmp_path,
            sources,
            config_hash="config-b",
            input_artifact_ids=["input-manifest:def"],
            dependencies=dependencies,
            region_artifacts=regions,
            generation_id="generation-b",
        )

    selected, manifest = resolve_current_raw_generation(tmp_path / "raw_outputs")
    assert selected == first["generation"]
    assert manifest["generation_id"] == "generation-a"
    assert not (tmp_path / "raw_outputs" / ".staging" / "generation-b").exists()


def test_failure_after_pointer_swap_rolls_back_selection(tmp_path, monkeypatch):
    first = _publish(tmp_path, generation_id="generation-a")
    sources, dependencies, regions = _publication_sources(tmp_path)
    import smftools.informatics.raw_generation as raw_generation

    real_validate = raw_generation.validate_raw_generation
    calls = 0

    def fail_final_validation(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected final validation failure")
        return real_validate(*args, **kwargs)

    monkeypatch.setattr(raw_generation, "validate_raw_generation", fail_final_validation)
    with pytest.raises(RuntimeError, match="injected final validation failure"):
        publish_raw_generation(
            tmp_path,
            sources,
            config_hash="config-b",
            input_artifact_ids=["input-manifest:def"],
            dependencies=dependencies,
            region_artifacts=regions,
            generation_id="generation-b",
        )

    selected, manifest = resolve_current_raw_generation(tmp_path / "raw_outputs")
    assert selected == first["generation"]
    assert manifest["generation_id"] == "generation-a"


def test_missing_or_incomplete_manifest_is_not_valid(tmp_path):
    generation = tmp_path / "raw_outputs" / "generations" / "generation-a"
    generation.mkdir(parents=True)
    with pytest.raises(RawGenerationError, match="missing or unreadable"):
        validate_raw_generation(generation)

    (generation / RAW_GENERATION_MANIFEST).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "generation_id": "generation-a",
                "artifacts": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RawGenerationError, match="incomplete"):
        validate_raw_generation(generation)
