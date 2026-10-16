from __future__ import annotations

import json
from pathlib import Path

import pytest

from smftools.constants import RAW_DIR
from smftools.informatics.generation_listing import (
    CURRENT_FILENAME,
    GENERATION_MANIFEST,
    GENERATIONS_SUBDIR,
)
from smftools.informatics.generation_pruning import (
    KEEP_CURRENT,
    KEEP_LAST,
    KEEP_PINNED,
    KEEP_RECENT,
    KEEP_UNREADABLE,
    PRUNE_CANDIDATE,
    GenerationPruneError,
    parse_older_than,
    plan_experiment_generation_prune,
)
from smftools.informatics.generation_retention import pin_generation

pytestmark = pytest.mark.unit


def _publish(
    container: Path,
    generation_id: str,
    created_at: str,
    *,
    current: bool = False,
) -> Path:
    generation = container / GENERATIONS_SUBDIR / generation_id
    generation.mkdir(parents=True, exist_ok=True)
    (generation / "payload.bin").write_bytes(generation_id.encode())
    (generation / GENERATION_MANIFEST).write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "complete",
                "generation_id": generation_id,
                "created_at": created_at,
                "artifacts": {"payload": "payload.bin"},
            }
        ),
        encoding="utf-8",
    )
    if current:
        (container / CURRENT_FILENAME).write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "generation_id": generation_id,
                    "generation_path": f"{GENERATIONS_SUBDIR}/{generation_id}",
                }
            ),
            encoding="utf-8",
        )
    return generation


def test_plan_protects_current_pinned_and_keep_last_then_offers_candidates(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "outputs"
    container = run_root / RAW_DIR
    _publish(container, "raw-current", "2026-04-01T00:00:00Z", current=True)
    _publish(container, "raw-new", "2026-03-01T00:00:00Z")
    _publish(container, "raw-pinned", "2026-02-01T00:00:00Z")
    _publish(container, "raw-old", "2026-01-01T00:00:00Z")
    pin_generation(container, "raw-pinned", reason="publication source")
    paths_before = sorted(path.relative_to(run_root) for path in run_root.rglob("*"))

    plan = plan_experiment_generation_prune(
        run_root,
        keep_last=2,
        older_than="2026-03-15T00:00:00Z",
        stages=("raw",),
    )

    dispositions = {decision.generation_id: decision.disposition for decision in plan.decisions}
    assert dispositions == {
        "raw-current": KEEP_CURRENT,
        "raw-new": KEEP_LAST,
        "raw-pinned": KEEP_PINNED,
        "raw-old": PRUNE_CANDIDATE,
    }
    old = next(d for d in plan.decisions if d.generation_id == "raw-old")
    assert old.policy_candidate is True
    assert old.deletion_allowed is True
    assert plan.candidate_bytes > 0
    assert plan.reclaimable_bytes == plan.candidate_bytes
    assert plan.to_dict()["deletion_supported"] is False
    assert sorted(path.relative_to(run_root) for path in run_root.rglob("*")) == paths_before


def test_plan_protects_generations_at_or_after_cutoff(tmp_path: Path) -> None:
    run_root = tmp_path / "outputs"
    container = run_root / RAW_DIR
    _publish(container, "raw-current", "2026-04-01T00:00:00Z", current=True)
    _publish(container, "raw-recent", "2026-03-15T00:00:00Z")
    _publish(container, "raw-old", "2026-03-14T23:59:59Z")

    plan = plan_experiment_generation_prune(
        run_root,
        older_than="2026-03-15T00:00:00Z",
        stages=("raw",),
    )

    dispositions = {decision.generation_id: decision.disposition for decision in plan.decisions}
    assert dispositions["raw-recent"] == KEEP_RECENT
    assert dispositions["raw-old"] == PRUNE_CANDIDATE


def test_plan_protects_every_noncurrent_generation_when_retention_is_corrupt(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "outputs"
    container = run_root / RAW_DIR
    _publish(container, "raw-current", "2026-04-01T00:00:00Z", current=True)
    _publish(container, "raw-old", "2026-01-01T00:00:00Z")
    (container / "retention.json").write_text("{", encoding="utf-8")

    plan = plan_experiment_generation_prune(
        run_root,
        older_than="2100-01-01T00:00:00Z",
        stages=("raw",),
    )

    old = next(decision for decision in plan.decisions if decision.generation_id == "raw-old")
    assert old.disposition == KEEP_UNREADABLE
    assert old.policy_candidate is False
    assert old.deletion_allowed is False
    assert plan.candidate_bytes == 0


@pytest.mark.parametrize("value", ["", "not-a-date"])
def test_parse_older_than_rejects_invalid_values(value: str) -> None:
    with pytest.raises(GenerationPruneError, match="older-than"):
        parse_older_than(value)


def test_plan_requires_policy_and_known_stages(tmp_path: Path) -> None:
    with pytest.raises(GenerationPruneError, match="requires"):
        plan_experiment_generation_prune(tmp_path)
    with pytest.raises(GenerationPruneError, match="unknown generation stage"):
        plan_experiment_generation_prune(
            tmp_path,
            keep_last=1,
            stages=("unknown",),
        )


# --- SLC-06 / SLC-07: generations that live generations read ----------------------


def _spine_with(generation: Path, **uns: str) -> None:
    import anndata as ad

    ad.AnnData(uns=dict(uns)).write_h5ad(generation / "spine.h5ad")


def _lineage(generation: Path, **lineage: object) -> None:
    manifest_path = generation / GENERATION_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(lineage)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _refreshed_experiment(tmp_path: Path) -> Path:
    """Current generations that read non-current ones, as after refreshes:
    the current preprocess is an annotation refresh of an older one, and the
    current spatial spine points at an older raw and that older preprocess."""
    from smftools.constants import PREPROCESS_DIR, SPATIAL_DIR

    root = tmp_path / "outputs"
    raw, pre, spatial = root / RAW_DIR, root / PREPROCESS_DIR, root / SPATIAL_DIR
    _publish(raw, "raw-read", "2026-01-01T00:00:00Z")
    _publish(raw, "raw-stale", "2026-01-02T00:00:00Z")
    _publish(raw, "raw-current", "2026-02-01T00:00:00Z", current=True)
    parent = _publish(pre, "pre-parent", "2026-01-03T00:00:00Z")
    _publish(pre, "pre-stale", "2026-01-04T00:00:00Z")
    refresh = _publish(pre, "pre-current", "2026-02-02T00:00:00Z", current=True)
    _lineage(
        refresh,
        annotation_refresh={"parent_generation_id": "pre-parent"},
        upgrade_plan={"current_generation_id": "pre-stale"},  # bookkeeping, not a read
    )
    _lineage(parent, source={"raw": f"../../{RAW_DIR}/generations/raw-read/spine.h5ad"})
    _publish(spatial, "spatial-stale", "2026-01-05T00:00:00Z")
    current = _publish(spatial, "spatial-current", "2026-02-03T00:00:00Z", current=True)
    _spine_with(
        current,
        molecule_index=f"../../{RAW_DIR}/generations/raw-read/molecule_index",
        preprocess_catalog=f"../../{PREPROCESS_DIR}/generations/pre-parent/catalog.parquet",
    )
    return root


def test_generations_live_ones_read_are_kept(tmp_path: Path) -> None:
    from smftools.informatics.generation_pruning import KEEP_REFERENCED

    root = _refreshed_experiment(tmp_path)
    plan = plan_experiment_generation_prune(root, keep_last=0)
    dispositions = {d.generation_id: d.disposition for d in plan.decisions}
    assert dispositions == {
        "raw-current": KEEP_CURRENT,
        "raw-read": KEEP_REFERENCED,  # read by the current spatial spine and by pre-parent
        "raw-stale": PRUNE_CANDIDATE,
        "pre-current": KEEP_CURRENT,
        "pre-parent": KEEP_REFERENCED,  # annotation-refresh parent
        "pre-stale": PRUNE_CANDIDATE,  # named only by upgrade bookkeeping
        "spatial-current": KEEP_CURRENT,
        "spatial-stale": PRUNE_CANDIDATE,
    }
    parent = next(d for d in plan.decisions if d.generation_id == "pre-parent")
    assert parent.deletion_allowed is False
    assert any("parent_generation_id" in reason for reason in parent.reasons)


def test_a_pinned_generation_protects_what_it_reads(tmp_path: Path) -> None:
    from smftools.constants import SPATIAL_DIR
    from smftools.informatics.generation_pruning import KEEP_REFERENCED

    root = _refreshed_experiment(tmp_path)
    stale = root / SPATIAL_DIR / GENERATIONS_SUBDIR / "spatial-stale"
    _spine_with(stale, molecule_index=f"../../{RAW_DIR}/generations/raw-stale/molecule_index")
    pin_generation(root / SPATIAL_DIR, "spatial-stale", reason="figure 2")
    plan = plan_experiment_generation_prune(root, keep_last=0)
    dispositions = {d.generation_id: d.disposition for d in plan.decisions}
    assert dispositions["spatial-stale"] == KEEP_PINNED
    assert dispositions["raw-stale"] == KEEP_REFERENCED


def test_protect_from_keeps_what_a_registry_names(tmp_path: Path) -> None:
    from smftools.informatics.generation_pruning import KEEP_REFERENCED

    root = _refreshed_experiment(tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    (project / "registry.json").write_text(
        json.dumps({"spines": {"raw": f"runs/x/{RAW_DIR}/generations/raw-stale/spine.h5ad"}}),
        encoding="utf-8",
    )
    plan = plan_experiment_generation_prune(root, keep_last=0, protect_from=[project])
    stale = next(d for d in plan.decisions if d.generation_id == "raw-stale")
    assert stale.disposition == KEEP_REFERENCED
    assert "registry.json" in stale.reasons[0]
    with pytest.raises(GenerationPruneError, match="no file"):
        plan_experiment_generation_prune(root, keep_last=0, protect_from=[tmp_path / "missing"])


def test_unreadable_references_keep_every_noncurrent_generation(tmp_path: Path) -> None:
    from smftools.constants import SPATIAL_DIR
    from smftools.informatics.generation_pruning import KEEP_REFERENCES_UNKNOWN

    root = _refreshed_experiment(tmp_path)
    spine = root / SPATIAL_DIR / GENERATIONS_SUBDIR / "spatial-current" / "spine.h5ad"
    spine.write_bytes(b"not an h5 file")
    plan = plan_experiment_generation_prune(root, keep_last=0)
    assert plan.candidate_bytes == 0
    assert {d.disposition for d in plan.decisions if not d.is_current} <= {
        KEEP_REFERENCES_UNKNOWN,
        "keep_referenced",
    }


def test_reference_graph_lists_edges_with_their_source(tmp_path: Path) -> None:
    from smftools.informatics.generation_references import build_reference_graph

    graph = build_reference_graph(_refreshed_experiment(tmp_path))
    edges = {(r.source_id, r.target_id): r.via for r in graph.references}
    assert edges[("spatial-current", "raw-read")].startswith("spine.h5ad:uns/")
    assert edges[("pre-current", "pre-parent")].endswith("parent_generation_id")
    assert ("pre-current", "pre-stale") not in edges
    assert not graph.unreadable
