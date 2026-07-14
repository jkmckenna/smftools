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
    entry = raw["experiments"]["expA"]
    # "path" is now the run root (expA's parent, since expA held a spine
    # directly -- see discover_stage_spines), not expA itself.
    stored_path = entry["path"]
    assert not Path(stored_path).is_absolute()
    assert stored_path == ".."
    stored_spine = entry["spines"]["raw"]
    assert not Path(stored_spine).is_absolute()
    assert stored_spine == "../expA/spine.h5ad"


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
    assert experiments[0]["path"] == str(moved_root.resolve())
    assert experiments[0]["spines"]["raw"] == str((moved_root / "expA" / "spine.h5ad").resolve())
    assert Path(experiments[0]["spines"]["raw"]).exists()


def _make_run(run_dir, *, stages, name, modality, reference_uids, n_reads=3):
    """Build a run directory with a spine under each of `stages` (subset of
    STAGE_DIRS keys), simulating an experiment partway through -- or all the way
    through -- the raw -> preprocess -> spatial -> hmm pipeline."""
    for stage in stages:
        stage_dir = run_dir / reg.STAGE_DIRS[stage]
        stage_dir.mkdir(parents=True, exist_ok=True)
        spine = ad.AnnData(obs=pd.DataFrame(index=[f"{name}_r{i}" for i in range(n_reads)]))
        spine.uns["modality"] = modality
        spine.uns["experiment"] = name
        spine.uns["raw_schema_version"] = 2
        spine.uns["reference_uids"] = reference_uids
        safe_write_h5ad(spine, stage_dir / "spine.h5ad", backup=False, verbose=False)
    return run_dir


def test_discover_stage_spines_from_run_root(tmp_path):
    run_dir = _make_run(
        tmp_path / "expA",
        stages=["raw", "preprocess", "hmm"],
        name="expA",
        modality="direct",
        reference_uids={"chr1_top": "uid1"},
    )
    # Pointing at the run's top-level output directory ...
    run_root, spines = reg.discover_stage_spines(run_dir)
    assert run_root == run_dir
    assert set(spines) == {"raw", "preprocess", "hmm"}
    # ... and pointing directly at one stage dir (the original convention) find
    # the same sibling spines.
    run_root2, spines2 = reg.discover_stage_spines(run_dir / reg.STAGE_DIRS["raw"])
    assert run_root2 == run_dir
    assert set(spines2) == {"raw", "preprocess", "hmm"}


def test_add_experiment_records_every_available_stage(tmp_path):
    proj = tmp_path / "project"
    reg.init_project(proj)
    run_dir = _make_run(
        tmp_path / "expA",
        stages=["raw", "preprocess", "hmm"],
        name="expA",
        modality="direct",
        reference_uids={"chr1_top": "uid1"},
    )
    exp_id, entry = reg.add_experiment(proj, run_dir)
    assert set(entry["spines"]) == {"raw", "preprocess", "hmm"}

    [listed] = reg.list_experiments(proj)
    assert set(listed["spines"]) == {"raw", "preprocess", "hmm"}
    assert listed["spines"]["hmm"] == str((run_dir / reg.STAGE_DIRS["hmm"] / "spine.h5ad").resolve())


def test_resolve_experiment_spine_prefers_most_derived_stage(tmp_path):
    proj = tmp_path / "project"
    reg.init_project(proj)
    _make_run(
        tmp_path / "expA",
        stages=["raw", "preprocess", "hmm"],
        name="expA",
        modality="direct",
        reference_uids={"chr1_top": "uid1"},
    )
    _make_run(
        tmp_path / "expB",
        stages=["raw"],
        name="expB",
        modality="direct",
        reference_uids={"chr1_top": "uid1"},
    )
    reg.add_experiment(proj, tmp_path / "expA")
    reg.add_experiment(proj, tmp_path / "expB")
    entries = {entry["id"]: entry for entry in reg.list_experiments(proj)}

    # Auto (no explicit stage): most-derived stage available per experiment.
    assert reg.resolve_experiment_spine(entries["expA"])[0] == "hmm"
    assert reg.resolve_experiment_spine(entries["expB"])[0] == "raw"

    # Explicit stage: honored if present, None if the experiment hasn't reached it.
    assert reg.resolve_experiment_spine(entries["expA"], stage="raw")[0] == "raw"
    assert reg.resolve_experiment_spine(entries["expB"], stage="hmm") is None


def test_resolve_experiment_spine_prefers_consolidated_experiment_spine(tmp_path):
    proj = tmp_path / "project"
    reg.init_project(proj)
    _make_run(
        tmp_path / "expA",
        stages=["raw", "preprocess", "hmm", "experiment"],
        name="expA",
        modality="direct",
        reference_uids={"chr1_top": "uid1"},
    )
    reg.add_experiment(proj, tmp_path / "expA")
    entry = {e["id"]: e for e in reg.list_experiments(proj)}["expA"]

    # Auto (no explicit stage): the consolidated spine wins over STAGE_PRIORITY.
    assert reg.resolve_experiment_spine(entry)[0] == "experiment"
    # Explicit stage requests are unaffected -- still resolve to that stage's own
    # spine, not the consolidated one.
    assert reg.resolve_experiment_spine(entry, stage="hmm")[0] == "hmm"


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("myexp.h5ad", "raw"),
        ("myexp.h5ad.gz", "raw"),
        ("myexp_preprocessed.h5ad.gz", "preprocess"),
        ("myexp_preprocessed_duplicates_removed.h5ad.gz", "preprocess"),
        ("myexp_spatial.h5ad.gz", "spatial"),
        ("myexp_hmm.h5ad", "hmm"),
        ("myexp_latent.h5ad.gz", "latent"),
        ("myexp_variant.h5ad.gz", "variant"),
        ("myexp_chimeric.h5ad.gz", "chimeric"),
    ],
)
def test_infer_legacy_stage_from_filename(tmp_path, filename, expected):
    assert reg.infer_legacy_stage(tmp_path / filename) == expected


def _legacy_monolithic_spine(*, reference_uids=None, references_fasta=None, ref_strands=None):
    n = 3
    obs = pd.DataFrame(index=[f"r{i}" for i in range(n)])
    if ref_strands is not None:
        obs["Reference_strand"] = ref_strands
    spine = ad.AnnData(obs=obs)
    spine.uns["modality"] = "direct"
    spine.uns["experiment"] = "legacyExp"
    if reference_uids is not None:
        spine.uns["reference_uids"] = reference_uids
    if references_fasta is not None:
        spine.uns["References"] = references_fasta
    return spine


def test_legacy_reference_uids_prefers_existing():
    spine = _legacy_monolithic_spine(reference_uids={"chr1_top": "uidX"})
    assert reg._legacy_reference_uids(spine) == {"chr1_top": "uidX"}


def test_legacy_reference_uids_computed_on_the_fly_from_fasta():
    spine = _legacy_monolithic_spine(
        ref_strands=["chr1_top", "chr1_top", "chr1_bottom"],
        references_fasta={"chr1_FASTA_sequence": "ACGTACGTAC"},
    )
    uids = reg._legacy_reference_uids(spine)
    assert set(uids) == {"chr1_top", "chr1_bottom"}
    # Same underlying sequence -> same computed identity for both strands.
    assert uids["chr1_top"] == uids["chr1_bottom"]
    # Source object itself was never mutated with the computed uids.
    assert "reference_uids" not in spine.uns


def test_legacy_reference_uids_empty_when_no_sequence_info():
    spine = _legacy_monolithic_spine()
    assert reg._legacy_reference_uids(spine) == {}


def _write_legacy_h5ad(path, *, stage_hint="", reference_uids=None, ref_strands=None, fasta=None):
    spine = _legacy_monolithic_spine(
        reference_uids=reference_uids, references_fasta=fasta, ref_strands=ref_strands
    )
    safe_write_h5ad(spine, path, backup=False, verbose=False)
    assert "is_spine" not in spine.uns
    return path


def test_add_experiment_registers_legacy_file_with_inferred_stage(tmp_path):
    proj = tmp_path / "project"
    reg.init_project(proj)
    legacy_file = _write_legacy_h5ad(
        tmp_path / "legacyExp_preprocessed.h5ad",
        reference_uids={"chr1_top": "uid1"},
    )
    exp_id, entry = reg.add_experiment(proj, legacy_file, experiment_id="legacyExp")
    assert exp_id == "legacyExp"
    assert set(entry["spines"]) == {"preprocess"}
    resolved = reg._resolve_registry_path(entry["spines"]["preprocess"], proj)
    assert resolved == legacy_file.resolve()


def test_add_experiment_legacy_file_explicit_stage_overrides_inference(tmp_path):
    proj = tmp_path / "project"
    reg.init_project(proj)
    # Filename looks like "raw" (no suffix match) but caller says it's hmm.
    legacy_file = _write_legacy_h5ad(tmp_path / "weird_name.h5ad", reference_uids={"chr1_top": "uid1"})
    exp_id, entry = reg.add_experiment(proj, legacy_file, experiment_id="legacyExp", stage="hmm")
    assert set(entry["spines"]) == {"hmm"}


def test_add_experiment_legacy_files_accumulate_across_calls(tmp_path):
    proj = tmp_path / "project"
    reg.init_project(proj)
    raw_file = _write_legacy_h5ad(
        tmp_path / "legacyExp.h5ad", reference_uids={"chr1_top": "uid1"}
    )
    preprocess_file = _write_legacy_h5ad(
        tmp_path / "legacyExp_preprocessed.h5ad", reference_uids={"chr1_top": "uid1"}
    )
    hmm_file = _write_legacy_h5ad(
        tmp_path / "legacyExp_hmm.h5ad", reference_uids={"chr1_top": "uid1"}
    )

    reg.add_experiment(proj, raw_file, experiment_id="legacyExp")
    reg.add_experiment(proj, preprocess_file, experiment_id="legacyExp")
    exp_id, entry = reg.add_experiment(proj, hmm_file, experiment_id="legacyExp")

    # Nothing registered by an earlier call was lost -- all three stages present.
    assert set(entry["spines"]) == {"raw", "preprocess", "hmm"}
    [listed] = reg.list_experiments(proj)
    assert set(listed["spines"]) == {"raw", "preprocess", "hmm"}


def test_add_experiment_legacy_file_never_mutates_source(tmp_path):
    proj = tmp_path / "project"
    reg.init_project(proj)
    legacy_file = _write_legacy_h5ad(
        tmp_path / "legacyExp.h5ad",
        ref_strands=["chr1_top", "chr1_top", "chr1_bottom"],
        fasta={"chr1_FASTA_sequence": "ACGTACGTAC"},
    )
    before_bytes = legacy_file.read_bytes()
    before_mtime = legacy_file.stat().st_mtime_ns

    exp_id, entry = reg.add_experiment(proj, legacy_file, experiment_id="legacyExp")

    # Reference identity was computed on the fly (no uns["reference_uids"] on
    # the source file), yet the source file's bytes/mtime are untouched.
    assert entry["references"]
    assert legacy_file.read_bytes() == before_bytes
    assert legacy_file.stat().st_mtime_ns == before_mtime


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
