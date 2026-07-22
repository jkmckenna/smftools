from __future__ import annotations

from types import SimpleNamespace

import pytest

from smftools.hmm.fit_plan import HMMModelSpec, build_hmm_fit_plan
from smftools.preprocessing.dispatch_plan import PreprocessTask


def _task(
    barcode: str, chunk: int, read_ids: tuple[str, ...], *, width: int = 100
) -> PreprocessTask:
    return PreprocessTask(
        task_id=f"ref|{barcode}|0-{width}|{chunk:05d}",
        reference="ref",
        barcode=barcode,
        analysis_mode="genome",
        chunk_index=chunk,
        core_start=0,
        core_end=width,
        load_start=0,
        load_end=width,
        n_reads=len(read_ids),
        estimated_memory_bytes=len(read_ids) * width * 8,
        read_ids=read_ids,
    )


def _spec() -> HMMModelSpec:
    return HMMModelSpec(
        name="accessibility",
        label="GpC",
        signals=("GpC",),
        feature_groups=("footprint", "accessible"),
        architecture="single",
    )


def _cfg(**overrides):
    values = {
        "hmm_fit_strategy": "per_group",
        "hmm_groupby": ["sample", "reference", "methbase"],
        "hmm_shared_scope": ["reference", "methbase"],
        "hmm_adapt_emissions": True,
        "hmm_adapt_startprobs": True,
        "hmm_max_fit_reads": 1000,
        "hmm_fit_selection_seed": 0,
        "target_task_memory_mb": 512,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _tasks() -> list[PreprocessTask]:
    return [
        _task("bc1", 0, ("r1", "r2")),
        _task("bc1", 1, ("r3", "r4")),
        _task("bc2", 0, ("r5", "r6")),
    ]


def test_per_group_chunks_share_one_fit_assignment():
    planning = build_hmm_fit_plan(_tasks(), [_spec()], _cfg())

    assert len(planning.base_plans) == 2
    assert not planning.adaptation_plans
    bc1 = next(plan for plan in planning.base_plans if plan.barcode == "bc1")
    assert bc1.selected_read_ids == ("r1", "r2", "r3", "r4")
    assert bc1.candidate_n_reads == 4
    assert {planning.apply_assignments[task_id]["GpC"] for task_id in bc1.apply_task_ids} == {
        bc1.fit_id
    }


def test_fit_selection_is_capped_and_invariant_to_task_order():
    cfg = _cfg(hmm_max_fit_reads=3, hmm_fit_selection_seed=17)
    forward = build_hmm_fit_plan(_tasks()[:2], [_spec()], cfg).base_plans[0]
    reverse = build_hmm_fit_plan(list(reversed(_tasks()[:2])), [_spec()], cfg).base_plans[0]

    assert len(forward.selected_read_ids) == 3
    assert forward.selected_read_ids == reverse.selected_read_ids
    assert forward.selection_sha256 == reverse.selection_sha256


def test_memory_ceiling_can_reduce_configured_fit_limit():
    tasks = [_task("bc1", 0, ("r1", "r2", "r3"), width=100_000)]
    plan = build_hmm_fit_plan(
        tasks,
        [_spec()],
        _cfg(hmm_max_fit_reads=1000, target_task_memory_mb=1),
    ).base_plans[0]

    assert plan.memory_max_fit_reads == 1
    assert len(plan.selected_read_ids) == 1


def test_shared_transitions_fit_base_once_then_adapt_per_barcode():
    planning = build_hmm_fit_plan(
        _tasks(),
        [_spec()],
        _cfg(hmm_fit_strategy="shared_transitions"),
    )

    assert len(planning.base_plans) == 1
    assert planning.base_plans[0].fit_kind == "GLOBAL"
    assert planning.base_plans[0].barcode == "ALL"
    assert len(planning.adaptation_plans) == 2
    assert {plan.barcode for plan in planning.adaptation_plans} == {"bc1", "bc2"}
    assert {plan.parent_fit_id for plan in planning.adaptation_plans} == {
        planning.base_plans[0].fit_id
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("hmm_groupby", ["sample", "reference"]),
        ("hmm_shared_scope", ["reference", "unknown", "methbase"]),
    ],
)
def test_invalid_partitioned_grouping_fails_early(field, value):
    cfg = _cfg(**{field: value})
    with pytest.raises(ValueError):
        build_hmm_fit_plan(_tasks(), [_spec()], cfg)
