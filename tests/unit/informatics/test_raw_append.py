import json
import os
import shutil
from pathlib import Path

import pandas as pd
import pytest

from smftools.informatics.input_manifest import resolve_input_manifest
from smftools.informatics.partition_read import materialize
from smftools.informatics.raw_append import (
    RawAppendError,
    assemble_raw_append,
    discard_raw_append_assembly,
    plan_raw_append,
)
from smftools.informatics.raw_generation import (
    publish_raw_generation,
    resolve_current_raw_generation,
)
from smftools.informatics.raw_store import write_raw_store
from smftools.readwrite import safe_read_h5ad

EXPERIMENT_UID = "12345678-1234-5678-1234-567812345678"


def _frame(read_id: str, *, start: int = 0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "read_id": read_id,
                "reference": "ref",
                "Reference_strand": "ref_top",
                "sample": "sample",
                "barcode": "bc01",
                "strand": "top",
                "mapping_direction": "fwd",
                "reference_start": start,
                "cigar": "4M",
                "aligned_length": 4,
                "sequence": [0, 1, 2, 3],
                "quality": [30, 30, 30, 30],
                "mismatch": [4, 4, 4, 4],
                "modification_signal": [0.0, 0.0, 0.0, 0.0],
            }
        ]
    )


def _write_store(run_root: Path, read_id: str, *, start: int = 0) -> dict[str, object]:
    return write_raw_store(
        _frame(read_id, start=start),
        run_root / "raw_outputs",
        reference_lengths={"ref_top": 20},
        extra_uns={"experiment_uid": EXPERIMENT_UID},
    )


def _sources(run_root: Path, store: dict[str, object]) -> dict[str, Path]:
    input_root = run_root / "raw_outputs" / "input_manifest"
    reference_map = run_root / "reference_interval_map.parquet"
    if not reference_map.exists():
        pd.DataFrame({"reference": ["ref_top"]}).to_parquet(reference_map, index=False)
    return {
        "spine": Path(store["spine"]),
        "ragged_store": run_root / "raw_outputs" / "raw",
        "interval_catalog": Path(store["interval_catalog"]),
        "obs": Path(store["obs"]),
        "molecules": Path(store["molecules"]),
        "molecule_index": Path(store["molecule_index"]),
        "segments": Path(store["segments"]),
        "segment_index": Path(store["segment_index"]),
        "reference_interval_map": reference_map,
        "sidecar_manifest": Path(store["manifest"]),
        "input_manifest_csv": input_root / "resolved_input_manifest.csv",
        "input_manifest_json": input_root / "resolved_input_manifest.json",
        "input_resolution_report": input_root / "input_resolution_report.json",
    }


def _source_ids(manifest, reference: str = "reference-a") -> list[str]:
    return [
        f"input-manifest:{manifest.digest}",
        *(f"source:{row.source_id}:{row.sha256}" for row in manifest.rows),
        f"alignment-reference-bundle:{reference}",
    ]


def _initial_generation(tmp_path: Path):
    first = tmp_path / "first.fastq"
    first.write_bytes(b"first")
    previous_manifest = resolve_input_manifest(
        output_directory=tmp_path,
        input_paths=[first],
    )
    store = _write_store(tmp_path, "read-one")
    outputs = publish_raw_generation(
        tmp_path,
        _sources(tmp_path, store),
        config_hash="config-a",
        input_artifact_ids=_source_ids(previous_manifest),
        generation_id="generation-a",
    )
    return first, previous_manifest, Path(outputs["generation"])


def test_append_plan_requires_pure_addition_and_stable_config_reference(tmp_path):
    first, _previous, generation = _initial_generation(tmp_path)
    second = tmp_path / "second.fastq"
    second.write_bytes(b"second")
    current = resolve_input_manifest(
        output_directory=tmp_path,
        input_paths=[first, second],
    )

    eligible = plan_raw_append(
        generation,
        current,
        run_root=tmp_path,
        config_hash="config-a",
        input_artifact_ids=_source_ids(current),
    )
    changed_config = plan_raw_append(
        generation,
        current,
        run_root=tmp_path,
        config_hash="config-b",
        input_artifact_ids=_source_ids(current),
    )
    changed_reference = plan_raw_append(
        generation,
        current,
        run_root=tmp_path,
        config_hash="config-a",
        input_artifact_ids=_source_ids(current, "reference-b"),
    )

    assert eligible.eligible
    assert len(eligible.transition.added_source_ids) == 1
    assert not changed_config.eligible and changed_config.reason == "raw configuration changed"
    assert not changed_reference.eligible
    assert "reference" in changed_reference.reason

    removed = resolve_input_manifest(
        output_directory=tmp_path / "removed",
        input_paths=[first],
    )
    removed_plan = plan_raw_append(
        generation,
        removed,
        run_root=tmp_path,
        config_hash="config-a",
        input_artifact_ids=_source_ids(removed),
    )
    assert not removed_plan.eligible
    assert "identical" in removed_plan.reason

    first.write_bytes(b"mutated")
    mutated = resolve_input_manifest(
        output_directory=tmp_path / "mutated",
        input_paths=[first],
    )
    mutated_plan = plan_raw_append(
        generation,
        mutated,
        run_root=tmp_path,
        config_hash="config-a",
        input_artifact_ids=_source_ids(mutated),
    )
    assert not mutated_plan.eligible
    assert "content_mutated" in mutated_plan.reason


def test_append_assembly_reuses_shards_and_publishes_complete_relocatable_generation(tmp_path):
    first, _previous, generation = _initial_generation(tmp_path)
    second = tmp_path / "second.fastq"
    second.write_bytes(b"second")
    current = resolve_input_manifest(
        output_directory=tmp_path,
        input_paths=[first, second],
    )
    plan = plan_raw_append(
        generation,
        current,
        run_root=tmp_path,
        config_hash="config-a",
        input_artifact_ids=_source_ids(current),
    )
    added_store = _write_store(tmp_path, "read-two", start=5)
    assembly = assemble_raw_append(
        tmp_path,
        generation,
        transition=plan.transition.to_dict(),
    )
    sources = {
        **_sources(tmp_path, added_store),
        **assembly.sources,
    }
    outputs = publish_raw_generation(
        tmp_path,
        sources,
        config_hash="config-a",
        input_artifact_ids=_source_ids(current),
        reuse_generation=generation,
        source_transition=plan.transition.to_dict(),
        generation_id="generation-b",
    )
    discard_raw_append_assembly(assembly)

    selected, manifest = resolve_current_raw_generation(tmp_path / "raw_outputs")
    spine, _ = safe_read_h5ad(selected / "spine.h5ad", verbose=False)
    segments = pd.read_parquet(selected / "segments.parquet")
    assert selected == outputs["generation"]
    assert spine.n_obs == 2
    assert set(spine.obs_names) == {"read-one", "read-two"}
    assert set(segments["read_id"].astype(str)) == {"read-one", "read-two"}
    assert spine.obs["canonical_row"].astype(int).tolist() == [0, 1]
    assert set(spine.obs["ragged_shard"].astype(str)) == set(segments["group_path"].astype(str))
    added = materialize(selected / "spine.h5ad", read_ids=["read-two"])
    assert added.obs_names.tolist() == ["read-two"]
    assert added.X.shape == (1, 20)
    assert manifest["source_transition"]["kind"] == "append_only"
    assert manifest["reuse"]["generation_id"] == "generation-a"
    assert manifest["reuse"]["reused_files"] > 0
    old_shard = next((generation / "raw").rglob("*.parquet"))
    reused_shard = selected / old_shard.relative_to(generation)
    assert os.stat(old_shard).st_ino == os.stat(reused_shard).st_ino

    relocated = tmp_path.parent / f"{tmp_path.name}-relocated-append"
    shutil.copytree(tmp_path, relocated)
    moved, moved_manifest = resolve_current_raw_generation(relocated / "raw_outputs")
    assert moved_manifest["generation_id"] == "generation-b"
    assert safe_read_h5ad(moved / "spine.h5ad", verbose=False)[0].n_obs == 2


def test_append_collision_fails_without_advancing_current(tmp_path):
    first, _previous, generation = _initial_generation(tmp_path)
    second = tmp_path / "second.fastq"
    second.write_bytes(b"second")
    current = resolve_input_manifest(
        output_directory=tmp_path,
        input_paths=[first, second],
    )
    plan = plan_raw_append(
        generation,
        current,
        run_root=tmp_path,
        config_hash="config-a",
        input_artifact_ids=_source_ids(current),
    )
    _write_store(tmp_path, "read-one", start=5)

    with pytest.raises(RawAppendError, match="colliding segment identities"):
        assemble_raw_append(
            tmp_path,
            generation,
            transition=plan.transition.to_dict(),
        )

    selected, manifest = resolve_current_raw_generation(tmp_path / "raw_outputs")
    pointer = json.loads((tmp_path / "raw_outputs" / "current.json").read_text())
    assert selected == generation
    assert manifest["generation_id"] == "generation-a"
    assert pointer["generation_id"] == "generation-a"
