from pathlib import Path

import anndata as ad
import pandas as pd
import pytest

from smftools.project import registry as reg
from smftools.project.reference_registry import (
    ReferenceRegistry,
    build_reference_alias_table,
    detect_reference_conflicts,
)
from smftools.readwrite import safe_write_h5ad


def _make_experiment(exp_dir, *, name, modality, reference_uids, n_reads=3):
    exp_dir.mkdir(parents=True, exist_ok=True)
    spine = ad.AnnData(obs=pd.DataFrame(index=[f"{name}_r{i}" for i in range(n_reads)]))
    spine.uns["modality"] = modality
    spine.uns["experiment"] = name
    spine.uns["raw_schema_version"] = 2
    spine.uns["reference_uids"] = reference_uids
    safe_write_h5ad(spine, exp_dir / "spine.h5ad", backup=False, verbose=False)
    return exp_dir


def test_init_add_list_remove(tmp_path):
    proj = tmp_path / "project"
    reg.init_project(proj)
    exp = _make_experiment(
        tmp_path / "expA",
        name="expA",
        modality="direct",
        reference_uids={"chr1_top": "uid1", "chr1_bottom": "uid1"},
    )
    exp_id, entry = reg.add_experiment(proj, exp)
    assert exp_id == "expA"
    assert entry["modality"] == "direct"
    assert entry["references"] == {"chr1_top": "uid1", "chr1_bottom": "uid1"}
    assert entry["n_reads"] == 3
    assert [e["id"] for e in reg.list_experiments(proj)] == ["expA"]

    reg.remove_experiment(proj, "expA")
    assert reg.list_experiments(proj) == []
    assert len(reg.list_experiments(proj, active_only=False)) == 1


def test_add_is_idempotent_same_path(tmp_path):
    proj = tmp_path / "p"
    reg.init_project(proj)
    exp = _make_experiment(tmp_path / "e", name="e", modality="conversion", reference_uids={"r_top": "u"})
    reg.add_experiment(proj, exp)
    reg.add_experiment(proj, exp)  # re-add same path -> refresh, not duplicate
    assert len(reg.list_experiments(proj)) == 1


def test_registry_json_stores_relative_paths_not_absolute(tmp_path):
    import json

    proj = tmp_path / "project"
    reg.init_project(proj)
    exp = _make_experiment(
        tmp_path / "expA", name="expA", modality="direct", reference_uids={"chr1_top": "uid1"}
    )
    reg.add_experiment(proj, exp)

    raw = json.loads((proj / reg.REGISTRY_FILENAME).read_text())
    stored_path = raw["experiments"]["expA"]["path"]
    assert not Path(stored_path).is_absolute()
    assert stored_path == "../expA"


def test_project_survives_being_copied_to_a_different_absolute_path(tmp_path_factory):
    """Simulates transferring the whole tree to another machine/mount point."""
    import shutil

    original_root = tmp_path_factory.mktemp("original")
    proj = original_root / "project"
    reg.init_project(proj)
    exp = _make_experiment(
        original_root / "expA", name="expA", modality="direct", reference_uids={"chr1_top": "uid1"}
    )
    reg.add_experiment(proj, exp)

    # Copy the *whole* tree (project + experiment, preserving their relative
    # layout) to a completely different absolute root, as if moved to another
    # machine. The stale absolute path from `original_root` no longer exists.
    moved_root = tmp_path_factory.mktemp("moved") / "relocated"
    shutil.copytree(original_root, moved_root)
    shutil.rmtree(original_root)

    experiments = reg.list_experiments(moved_root / "project")
    assert len(experiments) == 1
    assert experiments[0]["path"] == str((moved_root / "expA").resolve())
    assert Path(experiments[0]["path"]).exists()


def test_sets_list_and_query(tmp_path):
    proj = tmp_path / "p"
    reg.init_project(proj)
    reg.add_set(proj, "cohort", experiments=["a", "b"])
    reg.add_set(proj, "direct", query="modality='direct'")
    assert reg.resolve_set(proj, "cohort") == {"kind": "list", "experiments": ["a", "b"]}
    assert reg.resolve_set(proj, "direct")["kind"] == "query"
    with pytest.raises(ValueError):
        reg.add_set(proj, "bad", experiments=["a"], query="x")


def test_reference_harmonization_same_sequence_different_names():
    experiments = [
        {"id": "A", "references": {"myGene_top": "uidX", "myGene_bottom": "uidX"}},
        {"id": "B", "references": {"gene1_top": "uidX"}},  # different name, same sequence
    ]
    table = build_reference_alias_table(experiments, ReferenceRegistry())
    assert set(table["canonical_reference"]) == {"uidX"}  # auto-harmonized

    table2 = build_reference_alias_table(experiments, ReferenceRegistry(canonical_names={"uidX": "NKG2A"}))
    assert set(table2["canonical_reference"]) == {"NKG2A"}  # friendly name via YAML


def test_reference_alias_group_merges_near_identical():
    experiments = [
        {"id": "A", "references": {"gene_top": "uid1"}},
        {"id": "B", "references": {"gene_top": "uid2"}},  # near-identical, different hash
    ]
    table = build_reference_alias_table(experiments, ReferenceRegistry(aliases={"gene": ["uid1", "uid2"]}))
    assert set(table["canonical_reference"]) == {"gene"}


def test_detect_conflicts_flags_ambiguous_name():
    experiments = [
        {"id": "A", "references": {"gene_top": "uid1"}},
        {"id": "B", "references": {"gene_top": "uid2"}},  # same name, different sequence
    ]
    table = build_reference_alias_table(experiments, ReferenceRegistry())
    assert any("gene_top" in w for w in detect_reference_conflicts(table))


def test_reference_registry_yaml_round_trip(tmp_path):
    rr = ReferenceRegistry(canonical_names={"uid1": "A"}, aliases={"B": ["uid2", "uid3"]})
    loaded = ReferenceRegistry.load(rr.save(tmp_path / "reference_registry.yaml"))
    assert loaded.canonical_reference("uid1") == "A"
    assert loaded.canonical_reference("uid2") == "B"
    assert loaded.canonical_reference("unknown") == "unknown"
