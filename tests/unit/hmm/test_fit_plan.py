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


# --- HXR: fits shared across references ------------------------------------------


def _ref_task(
    reference: str, barcode: str, read_ids: tuple[str, ...], *, width: int, core: int = 0
) -> PreprocessTask:
    return PreprocessTask(
        task_id=f"{reference}|{barcode}|{core}-{core + width}|00000",
        reference=reference,
        barcode=barcode,
        analysis_mode="locus",
        chunk_index=0,
        core_start=core,
        core_end=core + width,
        load_start=core,
        load_end=core + width,
        n_reads=len(read_ids),
        estimated_memory_bytes=len(read_ids) * width * 8,
        read_ids=read_ids,
    )


def _allele_tasks() -> list[PreprocessTask]:
    wt = tuple(f"wt{i}" for i in range(10))
    dl = tuple(f"dl{i}" for i in range(4))
    return [
        _ref_task("wt_top", "bc1", wt, width=100),
        _ref_task("del_top", "bc1", dl, width=80),
        _ref_task("spike_top", "bc1", ("s1", "s2"), width=50),
        _ref_task("wt_top", "bc2", ("wt_b",), width=100),
    ]


def test_reference_group_pools_alleles_into_one_fit_per_sample():
    cfg = _cfg(hmm_reference_groups={"alleles": ["wt", "del"]})
    planning = build_hmm_fit_plan(_allele_tasks(), [_spec()], cfg)

    by_reference = {(plan.reference, plan.barcode): plan for plan in planning.base_plans}
    assert set(by_reference) == {
        ("del_top+wt_top", "bc1"),
        ("spike_top", "bc1"),  # not in the group: fits alone, as before
        ("wt_top", "bc2"),  # only one allele present: an ordinary fit
    }
    pooled = by_reference[("del_top+wt_top", "bc1")]
    assert pooled.references == ("del_top", "wt_top")
    assert [m.load_end for m in pooled.members] == [80, 100]
    assert {planning.apply_assignments[t]["GpC"] for t in pooled.apply_task_ids} == {pooled.fit_id}
    assert len(pooled.apply_task_ids) == 2
    assert not by_reference[("wt_top", "bc2")].members
    record = pooled.to_dict()
    assert record["references"] == ["del_top", "wt_top"]
    assert "selected_read_ids" not in record["members"]
    assert pooled.training_metadata()["reference_n_reads"] == {"del_top": 4, "wt_top": 10}


def test_pooled_selection_draws_equal_shares_per_reference():
    cfg = _cfg(hmm_reference_groups={"alleles": ["wt_top", "del_top"]}, hmm_max_fit_reads=6)
    pooled = next(
        plan
        for plan in build_hmm_fit_plan(_allele_tasks(), [_spec()], cfg).base_plans
        if plan.members
    )
    counts = {m.reference: len(m.selected_read_ids) for m in pooled.members}
    assert counts == {"del_top": 3, "wt_top": 3}
    assert pooled.candidate_n_reads == 14

    few = _cfg(hmm_reference_groups={"alleles": ["wt", "del"]}, hmm_max_fit_reads=10)
    pooled = next(
        plan
        for plan in build_hmm_fit_plan(_allele_tasks(), [_spec()], few).base_plans
        if plan.members
    )
    # The smaller allele gives all it has; the larger one keeps to its share.
    assert {m.reference: len(m.selected_read_ids) for m in pooled.members} == {
        "del_top": 4,
        "wt_top": 5,
    }


def test_reference_groups_leave_other_fits_and_ids_unchanged():
    tasks = _allele_tasks()
    plain = build_hmm_fit_plan(tasks, [_spec()], _cfg())
    grouped = build_hmm_fit_plan(tasks, [_spec()], _cfg(hmm_reference_groups={"a": ["wt", "del"]}))
    reverse = build_hmm_fit_plan(
        list(reversed(tasks)), [_spec()], _cfg(hmm_reference_groups={"a": ["del", "wt"]})
    )
    plain_ids = {(p.reference, p.barcode): p.fit_id for p in plain.base_plans}
    grouped_ids = {(p.reference, p.barcode): p.fit_id for p in grouped.base_plans}
    for key in (("spike_top", "bc1"), ("wt_top", "bc2")):
        assert grouped_ids[key] == plain_ids[key]
    assert grouped_ids == {(p.reference, p.barcode): p.fit_id for p in reverse.base_plans}


def test_reference_group_strands_fit_apart():
    tasks = [
        _ref_task("wt_top", "bc1", ("a",), width=100),
        _ref_task("del_top", "bc1", ("b",), width=80),
        _ref_task("wt_bottom", "bc1", ("c",), width=100),
        _ref_task("del_bottom", "bc1", ("d",), width=80),
    ]
    cfg = _cfg(hmm_reference_groups={"alleles": ["wt", "del_top", "del_bottom"]})
    references = {p.reference for p in build_hmm_fit_plan(tasks, [_spec()], cfg).base_plans}
    assert references == {"del_top+wt_top", "del_bottom+wt_bottom"}


def test_reference_group_shared_transitions_pool_base_and_adaptations():
    cfg = _cfg(
        hmm_fit_strategy="shared_transitions",
        hmm_reference_groups={"alleles": ["wt", "del"]},
    )
    tasks = _allele_tasks()[:2] + [_ref_task("del_top", "bc2", ("dl_b",), width=80)]
    tasks.append(_ref_task("wt_top", "bc2", ("wt_b",), width=100))
    planning = build_hmm_fit_plan(tasks, [_spec()], cfg)
    (base,) = planning.base_plans
    assert base.fit_kind == "GLOBAL" and base.references == ("del_top", "wt_top")
    assert {(a.barcode, a.references) for a in planning.adaptation_plans} == {
        ("bc1", ("del_top", "wt_top")),
        ("bc2", ("del_top", "wt_top")),
    }


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"hmm_reference_groups": {"a": ["wt"], "b": ["wt"]}}, "both"),
        (
            {"hmm_reference_groups": {"a": ["wt", "del"]}, "hmm_distance_aware": True},
            "hmm_distance_aware",
        ),
    ],
)
def test_invalid_reference_groups_fail_early(overrides, message):
    with pytest.raises(ValueError, match=message):
        build_hmm_fit_plan(_allele_tasks(), [_spec()], _cfg(**overrides))


def test_reference_group_refuses_references_with_several_cores():
    tasks = [
        _ref_task("wt_top", "bc1", ("a",), width=100, core=0),
        _ref_task("wt_top", "bc1", ("b",), width=100, core=100),
        _ref_task("del_top", "bc1", ("c",), width=80),
    ]
    with pytest.raises(ValueError, match="several analysis cores"):
        build_hmm_fit_plan(tasks, [_spec()], _cfg(hmm_reference_groups={"a": ["wt", "del"]}))


# --- HPS-12: genome mode -- one fit per sample from a genome-wide read subset ----


def _genome_task(reference, barcode, core, read_ids, *, width=100, halo=10):
    return PreprocessTask(
        task_id=f"{reference}|{barcode}|{core}-{core + width}|00000",
        reference=reference,
        barcode=barcode,
        analysis_mode="genome",
        chunk_index=0,
        core_start=core,
        core_end=core + width,
        load_start=max(0, core - halo),
        load_end=core + width + halo,
        n_reads=len(read_ids),
        estimated_memory_bytes=len(read_ids) * width * 8,
        read_ids=tuple(read_ids),
    )


def _genome_tasks():
    tasks = []
    for barcode in ("bc1", "bc2"):
        for reference in ("chr1_top", "chr2_top"):
            for core in (0, 100):
                ids = [f"{barcode}-{reference}-{core}-{i}" for i in range(6)]
                if core == 100:  # a read spanning both tiles is listed in both
                    ids.append(f"{barcode}-{reference}-0-0")
                tasks.append(_genome_task(reference, barcode, core, ids))
    return tasks


def test_genome_mode_fits_one_model_per_sample_from_a_read_subset():
    tasks = _genome_tasks()
    planning = build_hmm_fit_plan(tasks, [_spec()], _cfg(hmm_max_fit_reads=10))
    assert len(planning.base_plans) == 2  # one per sample, not per tile
    for plan in planning.base_plans:
        assert plan.reference == "genome"
        assert len(plan.selected_read_ids) == 10  # bounded subset of 24 reads
        in_members = [r for m in plan.members for r in m.selected_read_ids]
        assert sorted(in_members) == sorted(plan.selected_read_ids)
        assert len(set(in_members)) == len(in_members)  # each read in one tile slice
        sample_tasks = {t.task_id for t in tasks if t.barcode == plan.barcode}
        assert set(plan.apply_task_ids) == sample_tasks
        assert {planning.apply_assignments[t]["GpC"] for t in sample_tasks} == {plan.fit_id}
    again = build_hmm_fit_plan(list(reversed(tasks)), [_spec()], _cfg(hmm_max_fit_reads=10))
    assert {p.fit_id for p in again.base_plans} == {p.fit_id for p in planning.base_plans}
