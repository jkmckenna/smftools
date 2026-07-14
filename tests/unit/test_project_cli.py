import anndata as ad
import pandas as pd
from click.testing import CliRunner

from smftools import cli_entry
from smftools.informatics.raw_store import write_raw_store
from smftools.informatics.reference_identity import reference_uid
from smftools.readwrite import safe_read_h5ad, safe_write_h5ad

SEQUENCE = "ACGTACGTACGT"


def _make_raw_experiment(out_dir, *, reference_strand, uid, npos=12, n=4):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "read_id": f"{reference_strand}_r{i}",
            "reference": reference_strand.rsplit("_", 1)[0],
            "Reference_strand": reference_strand,
            "sample": "bc01",
            "barcode": "bc01",
            "strand": "top",
            "mapping_direction": "fwd",
            "reference_start": 0,
            "cigar": f"{npos}M",
            "aligned_length": npos,
            "sequence": [i % 4 for _ in range(npos)],
            "quality": [30] * npos,
            "mismatch": [4] * npos,
            "modification_signal": [float(i % 2)] * npos,
        }
        for i in range(n)
    ]
    write_raw_store(
        pd.DataFrame(rows),
        out_dir,
        reference_lengths={reference_strand: npos},
        extra_uns={
            "reference_uids": {reference_strand: uid},
            "modality": "direct",
            "experiment": out_dir.name,
        },
    )
    return out_dir


def test_project_init_cli_scaffolds_docs_and_working_dirs(tmp_path):
    proj = tmp_path / "project"
    runner = CliRunner()

    r = runner.invoke(cli_entry.cli, ["project", "init", str(proj)])
    assert r.exit_code == 0, r.output
    assert "Initialized project registry" in r.output
    assert "created" in r.output

    for filename in ("README.md", "AGENTS.md", "CLAUDE.md", "PLAN.md", "project.yaml"):
        assert (proj / filename).is_file()
    assert (proj / "project_scripts").is_dir()
    assert (proj / "project_outputs").is_dir()

    # Re-running is a no-op for the scaffold (idempotent registry init too).
    readme = proj / "README.md"
    readme.write_text("# hand-edited\n")
    r2 = runner.invoke(cli_entry.cli, ["project", "init", str(proj)])
    assert r2.exit_code == 0, r2.output
    assert readme.read_text() == "# hand-edited\n"


def test_project_cli_end_to_end(tmp_path):
    uid = reference_uid(SEQUENCE, 12)
    _make_raw_experiment(tmp_path / "expA", reference_strand="geneA_top", uid=uid, n=4)
    _make_raw_experiment(tmp_path / "expB", reference_strand="geneB_top", uid=uid, n=3)
    proj = tmp_path / "project"
    runner = CliRunner()

    assert runner.invoke(cli_entry.cli, ["project", "init", str(proj)]).exit_code == 0

    r = runner.invoke(cli_entry.cli, ["project", "add", str(proj), str(tmp_path / "expA")])
    assert r.exit_code == 0, r.output
    assert "Registered 'expA'" in r.output

    assert (
        runner.invoke(cli_entry.cli, ["project", "add", str(proj), str(tmp_path / "expB")]).exit_code
        == 0
    )

    r = runner.invoke(cli_entry.cli, ["project", "list", str(proj)])
    assert r.exit_code == 0, r.output
    assert "expA" in r.output and "expB" in r.output
    assert "canonical reference" in r.output

    out = tmp_path / "combined.h5ad"
    r = runner.invoke(cli_entry.cli, ["project", "materialize", str(proj), uid, "-o", str(out)])
    assert r.exit_code == 0, r.output
    assert out.exists()
    combined, _ = safe_read_h5ad(out)
    assert combined.n_obs == 7
    assert set(combined.obs["experiment"]) == {"expA", "expB"}


def test_project_materialize_cli_caches_and_force_recompute_still_works(tmp_path):
    from smftools.project.set_store import set_cache_dir

    uid = reference_uid(SEQUENCE, 12)
    _make_raw_experiment(tmp_path / "expA", reference_strand="geneA_top", uid=uid, n=4)
    proj = tmp_path / "project"
    runner = CliRunner()
    assert runner.invoke(cli_entry.cli, ["project", "init", str(proj)]).exit_code == 0
    assert runner.invoke(cli_entry.cli, ["project", "add", str(proj), str(tmp_path / "expA")]).exit_code == 0

    out = tmp_path / "combined.h5ad"
    r = runner.invoke(cli_entry.cli, ["project", "materialize", str(proj), uid, "-o", str(out)])
    assert r.exit_code == 0, r.output

    cache_dir = set_cache_dir(proj, uid)
    assert (cache_dir / "base.h5ad").exists()

    r2 = runner.invoke(
        cli_entry.cli, ["project", "materialize", str(proj), uid, "-o", str(out), "--force-recompute"]
    )
    assert r2.exit_code == 0, r2.output
    combined, _ = safe_read_h5ad(out)
    assert combined.n_obs == 4


def test_project_sample_store_list_cli(tmp_path):
    uid = reference_uid(SEQUENCE, 12)
    _make_raw_experiment(tmp_path / "expA", reference_strand="geneA_top", uid=uid, n=4)
    _make_raw_experiment(tmp_path / "expB", reference_strand="geneB_top", uid=uid, n=3)
    proj = tmp_path / "project"
    runner = CliRunner()
    assert runner.invoke(cli_entry.cli, ["project", "init", str(proj)]).exit_code == 0
    assert runner.invoke(cli_entry.cli, ["project", "add", str(proj), str(tmp_path / "expA")]).exit_code == 0
    assert runner.invoke(cli_entry.cli, ["project", "add", str(proj), str(tmp_path / "expB")]).exit_code == 0

    r = runner.invoke(cli_entry.cli, ["project", "sample-store-list", str(proj)])
    assert r.exit_code == 0, r.output
    assert "2 partition(s)" in r.output
    assert "expA" in r.output and "geneA_top" in r.output and "bc01" in r.output
    assert "expB" in r.output

    r_filtered = runner.invoke(cli_entry.cli, ["project", "sample-store-list", str(proj), "--experiment-id", "expA"])
    assert r_filtered.exit_code == 0, r_filtered.output
    assert "1 partition(s)" in r_filtered.output
    assert "expB" not in r_filtered.output


def test_project_sample_store_list_cli_empty_project(tmp_path):
    proj = tmp_path / "project"
    runner = CliRunner()
    assert runner.invoke(cli_entry.cli, ["project", "init", str(proj)]).exit_code == 0

    r = runner.invoke(cli_entry.cli, ["project", "sample-store-list", str(proj)])
    assert r.exit_code == 0, r.output
    assert "No per-sample-store partitions" in r.output


def test_project_add_cli_backfills_per_sample_store_for_modern_experiment(tmp_path):
    from smftools.project.sample_store import list_per_sample_partitions

    uid = reference_uid(SEQUENCE, 12)
    _make_raw_experiment(tmp_path / "expA", reference_strand="geneA_top", uid=uid, n=4)
    proj = tmp_path / "project"
    runner = CliRunner()

    assert runner.invoke(cli_entry.cli, ["project", "init", str(proj)]).exit_code == 0
    r = runner.invoke(cli_entry.cli, ["project", "add", str(proj), str(tmp_path / "expA")])
    assert r.exit_code == 0, r.output

    partitions = list_per_sample_partitions(proj, "expA")
    assert len(partitions) == 1
    assert partitions[0] == {
        "kind": "pointer",
        "experiment_id": "expA",
        "reference_strand": "geneA_top",
        "sample": "bc01",
        "n_reads": 4,
    }


def test_project_add_cli_caches_per_sample_store_for_legacy_file(tmp_path):
    from smftools.project.sample_store import list_per_sample_partitions, load_per_sample_partition

    sequence = "ACGTACGTACGT"
    legacy_file = _make_legacy_monolithic_file(
        tmp_path / "legacyExp2_preprocessed.h5ad",
        reference_strand="geneL_top",
        sequence=sequence,
        n=3,
        sample="bc00",
    )
    before_bytes = legacy_file.read_bytes()
    proj = tmp_path / "project"
    runner = CliRunner()

    assert runner.invoke(cli_entry.cli, ["project", "init", str(proj)]).exit_code == 0
    r = runner.invoke(
        cli_entry.cli,
        ["project", "add", str(proj), str(legacy_file), "--id", "legacyExp2", "--stage", "preprocess"],
    )
    assert r.exit_code == 0, r.output

    # Legacy registration caches the partition (has no lazy read path to point at
    # instead), unlike a modern experiment which only gets a pointer.
    partitions = list_per_sample_partitions(proj, "legacyExp2")
    assert len(partitions) == 1
    assert partitions[0]["kind"] == "cache"
    assert partitions[0]["n_reads"] == 3
    loaded = load_per_sample_partition(proj, "legacyExp2", "geneL_top", "bc00")
    assert loaded.n_obs == 3

    # Source legacy file is only ever read, never mutated.
    assert legacy_file.read_bytes() == before_bytes


def _make_legacy_monolithic_file(path, *, reference_strand, sequence, n=3, npos=6, sample="bc00"):
    import numpy as np

    chromosome = reference_strand.rsplit("_", 1)[0]
    obs = pd.DataFrame(
        {"Reference_strand": [reference_strand] * n, "Sample": [sample] * n},
        index=[f"{reference_strand}_leg{i}" for i in range(n)],
    )
    spine = ad.AnnData(X=np.zeros((n, npos), dtype=np.float32), obs=obs)
    spine.var_names = [str(p) for p in range(npos)]
    spine.uns["modality"] = "direct"
    spine.uns["experiment"] = "legacyExp"
    spine.uns["References"] = {f"{chromosome}_FASTA_sequence": sequence}
    safe_write_h5ad(spine, path, backup=False, verbose=False)
    assert "is_spine" not in spine.uns
    return path


def test_project_cli_registers_and_materializes_legacy_monolithic_file(tmp_path):
    """A pre-partitioned-store run (a single monolithic .h5ad, no uns['is_spine'])
    registers via --stage and materializes through the same project/materialize
    chain as a modern partitioned run -- without ever rewriting the source file."""
    sequence = "ACGTACGTACGT"
    legacy_file = _make_legacy_monolithic_file(
        tmp_path / "legacyExp_preprocessed.h5ad",
        reference_strand="geneL_top",
        sequence=sequence,
        n=3,
    )
    before_bytes = legacy_file.read_bytes()

    proj = tmp_path / "project"
    runner = CliRunner()
    assert runner.invoke(cli_entry.cli, ["project", "init", str(proj)]).exit_code == 0

    r = runner.invoke(
        cli_entry.cli,
        ["project", "add", str(proj), str(legacy_file), "--id", "legacyExp", "--stage", "preprocess"],
    )
    assert r.exit_code == 0, r.output
    assert "Registered 'legacyExp'" in r.output

    # Source file untouched by registration (no cached-back reference_uids etc).
    assert legacy_file.read_bytes() == before_bytes

    r = runner.invoke(cli_entry.cli, ["project", "list", str(proj)])
    assert r.exit_code == 0, r.output
    assert "legacyExp" in r.output

    uid = reference_uid(sequence)
    out = tmp_path / "legacy_combined.h5ad"
    r = runner.invoke(
        cli_entry.cli,
        ["project", "materialize", str(proj), uid, "-o", str(out), "--stage", "preprocess"],
    )
    assert r.exit_code == 0, r.output
    combined, _ = safe_read_h5ad(out)
    assert combined.n_obs == 3
    assert set(combined.obs["experiment"]) == {"legacyExp"}
