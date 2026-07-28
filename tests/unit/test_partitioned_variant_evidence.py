import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from smftools.constants import MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT
from smftools.informatics.raw_store import write_raw_store
from smftools.informatics.sidecar_manifest import resolve_sidecar
from smftools.preprocessing.partitioned_variant import (
    BLOCKED_MISSING_INPUT,
    execute_partitioned_variant_evidence,
    plan_variant_evidence_tasks,
    query_partitioned_variant_evidence,
    validate_variant_evidence_generation,
)
from smftools.preprocessing.variant_reference import (
    VariantReferenceMember,
    VariantReferenceSet,
)

pytest.importorskip("pyarrow")

EXPERIMENT_UID = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"


def _encode(sequence: str) -> list[int]:
    return [MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT[base] for base in sequence]


def _reference_set(*, second_sequence: str = "AACAAAATAA") -> VariantReferenceSet:
    return VariantReferenceSet(
        members=(
            VariantReferenceMember(
                member_id="refA",
                sequence="AAAAAAAAAA",
                aliases=("refA_top",),
            ),
            VariantReferenceMember(
                member_id="refB",
                sequence=second_sequence,
                aliases=("refB_top",),
            ),
        )
    )


def _frame(*, include_sequence: bool = True) -> pd.DataFrame:
    rows = [
        ("shared_read", "refA", "AAAAAAAAAA"),
        ("cross_core", "refA", "AAAAAAATAA"),
        ("discordant", "refA", "AACAAAATAA"),
        ("ref_b_self", "refB", "AACAAAATAA"),
    ]
    records = []
    for read_id, reference, sequence in rows:
        record = {
            "read_id": read_id,
            "reference": reference,
            "Reference_strand": f"{reference}_top",
            "sample": "bc1",
            "barcode": "bc1",
            "reference_start": 0,
            "cigar": "10M",
            "aligned_length": 10,
            "quality": [30] * 10,
            "mismatch": [4] * 10,
            "modification_signal": [0.0] * 10,
        }
        if include_sequence:
            record["sequence"] = _encode(sequence)
        records.append(record)
    return pd.DataFrame(records)


def _raw(
    root: Path,
    *,
    shard_size: int,
    include_sequence: bool = True,
    experiment_uid: str = EXPERIMENT_UID,
):
    return write_raw_store(
        _frame(include_sequence=include_sequence),
        root / "run" / "raw_outputs",
        reference_lengths={"refA_top": 10, "refB_top": 10},
        shard_size=shard_size,
        genome_tile_size=5,
        genome_tile_halo=0,
        analysis_mode="genome",
        extra_uns={"experiment_uid": experiment_uid},
    )


def _semantic_frames(outputs):
    queried = query_partitioned_variant_evidence(outputs["manifest"].parent)
    return {
        kind: frame.drop(columns=["task_id"], errors="ignore")
        .sort_values(list(frame.columns.difference(["task_id"])), kind="stable")
        .reset_index(drop=True)
        for kind, frame in queried.items()
    }


def test_all_molecules_are_finalized_once_and_cross_core_breakpoint_is_detected(
    tmp_path,
) -> None:
    raw = _raw(tmp_path / "source", shard_size=1)
    outputs = execute_partitioned_variant_evidence(
        raw["spine"],
        [_reference_set()],
        tmp_path / "variant",
        max_workers=2,
    )

    obs = pd.read_parquet(outputs["obs"])
    assert len(obs) == len(_frame())
    assert not obs.duplicated(["experiment_uid", "molecule_uid", "variant_reference_set_id"]).any()
    cross_core = obs.set_index("read_id").loc["cross_core"]
    assert cross_core["breakpoint_count"] == 1
    assert bool(cross_core["has_breakpoint"]) is True

    events = query_partitioned_variant_evidence(
        tmp_path / "variant",
        molecule_uids=[cross_core["molecule_uid"]],
    )["events"]
    breakpoints = events.loc[events["event_type"] == "breakpoint", "breakpoint"]
    assert breakpoints.tolist() == [4.5]


def test_results_are_invariant_to_workers_task_order_memory_and_sharding(tmp_path) -> None:
    raw_one = _raw(tmp_path / "one", shard_size=1)
    reference_set = _reference_set()
    tasks = plan_variant_evidence_tasks(raw_one["spine"], [reference_set])
    outputs_one = execute_partitioned_variant_evidence(
        raw_one["spine"],
        [reference_set],
        tmp_path / "out-one",
        tasks=list(reversed(tasks)),
        max_workers=1,
        memory_budget_mb=1,
    )
    outputs_many = execute_partitioned_variant_evidence(
        raw_one["spine"],
        [reference_set],
        tmp_path / "out-many",
        max_workers=8,
        memory_budget_mb=1024,
    )

    raw_chunked = _raw(tmp_path / "chunked", shard_size=4)
    outputs_chunked = execute_partitioned_variant_evidence(
        raw_chunked["spine"],
        [reference_set],
        tmp_path / "out-chunked",
        max_workers=3,
    )
    baseline = _semantic_frames(outputs_one)
    for candidate in (_semantic_frames(outputs_many), _semantic_frames(outputs_chunked)):
        for kind in ("obs", "calls", "events"):
            pd.testing.assert_frame_equal(
                baseline[kind],
                candidate[kind],
                check_like=True,
            )


def test_two_reference_sets_have_distinct_artifacts_and_query_pruning(
    tmp_path,
    monkeypatch,
) -> None:
    raw = _raw(tmp_path / "source", shard_size=1)
    first = _reference_set()
    second = _reference_set(second_sequence="AACAAAGTAA")
    outputs = execute_partitioned_variant_evidence(
        raw["spine"],
        [first, second],
        tmp_path / "variant",
        max_workers=2,
    )
    obs = pd.read_parquet(outputs["obs"])
    assert set(obs["variant_reference_set_id"]) == {
        first.reference_set_id,
        second.reference_set_id,
    }
    assert obs.groupby("variant_reference_set_id").size().tolist() == [4, 4]

    opened: list[str] = []
    real_read_parquet = pd.read_parquet

    def _tracked(path, *args, **kwargs):
        opened.append(str(path))
        return real_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", _tracked)
    selected_row = obs.loc[obs["variant_reference_set_id"] == first.reference_set_id].iloc[0]
    molecule_uid = str(selected_row["molecule_uid"])
    selected_task_id = str(selected_row["task_id"])
    task_catalog = real_read_parquet(outputs["task_catalog"]).set_index("task_id")
    expected_task_store_paths = {
        str((tmp_path / "variant" / task_catalog.loc[selected_task_id, column]).resolve())
        for column in ("calls_path", "events_path")
    }
    expected_obs_path = str(
        (tmp_path / "variant" / task_catalog.loc[selected_task_id, "obs_path"]).resolve()
    )
    opened.clear()
    selected = query_partitioned_variant_evidence(
        tmp_path / "variant",
        variant_reference_set_ids=[first.reference_set_id],
        molecule_uids=[molecule_uid],
    )
    assert set(selected["obs"]["variant_reference_set_id"]) == {first.reference_set_id}
    assert all(second.reference_set_id not in path for path in opened)
    assert {str(Path(path).resolve()) for path in opened if "/task_store/" in path} == (
        expected_task_store_paths
    )
    assert expected_obs_path in {str(Path(path).resolve()) for path in opened}


def test_duplicate_read_ids_across_experiments_have_distinct_molecule_keys(tmp_path) -> None:
    first = _raw(
        tmp_path / "first",
        shard_size=2,
        experiment_uid="11111111-1111-4111-8111-111111111111",
    )
    second = _raw(
        tmp_path / "second",
        shard_size=2,
        experiment_uid="22222222-2222-4222-8222-222222222222",
    )
    first_outputs = execute_partitioned_variant_evidence(
        first["spine"],
        [_reference_set()],
        tmp_path / "first-variant",
    )
    second_outputs = execute_partitioned_variant_evidence(
        second["spine"],
        [_reference_set()],
        tmp_path / "second-variant",
    )
    first_obs = pd.read_parquet(first_outputs["obs"]).set_index("read_id")
    second_obs = pd.read_parquet(second_outputs["obs"]).set_index("read_id")
    assert (
        first_obs.loc["shared_read", "molecule_uid"]
        != second_obs.loc["shared_read", "molecule_uid"]
    )


def test_relocated_outputs_remain_queryable_and_manifests_are_registered(tmp_path) -> None:
    raw = _raw(tmp_path / "source", shard_size=2)
    outputs = execute_partitioned_variant_evidence(
        raw["spine"],
        [_reference_set()],
        tmp_path / "variant",
    )
    relocated = tmp_path / "relocated" / "variant"
    shutil.copytree(tmp_path / "variant", relocated)
    validate_variant_evidence_generation(relocated)
    selected = query_partitioned_variant_evidence(relocated)
    assert len(selected["obs"]) == 4

    with (relocated / "generation_manifest.json").open(encoding="utf-8") as handle:
        generation = json.load(handle)
    assert generation["task_count"] > 0
    assert all(artifact["path_kind"] == "relative" for artifact in generation["artifacts"].values())
    manifest = relocated / "sidecar_manifest.json"
    assert resolve_sidecar(manifest, "variant_read_index") == relocated / "read_index"


def test_missing_sequence_channel_produces_blocked_records(tmp_path) -> None:
    raw = _raw(tmp_path / "source", shard_size=2, include_sequence=False)
    outputs = execute_partitioned_variant_evidence(
        raw["spine"],
        [_reference_set()],
        tmp_path / "variant",
    )
    obs = pd.read_parquet(outputs["obs"])
    assert set(obs["evidence_status"]) == {BLOCKED_MISSING_INPUT}
    assert set(obs["missing_inputs"]) == {"sequence"}
    task_catalog = pd.read_parquet(outputs["task_catalog"])
    assert set(task_catalog["outcome"]) == {BLOCKED_MISSING_INPUT}
