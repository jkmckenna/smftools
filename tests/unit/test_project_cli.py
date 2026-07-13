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


def _make_legacy_monolithic_file(path, *, reference_strand, sequence, n=3, npos=6):
    import numpy as np

    chromosome = reference_strand.rsplit("_", 1)[0]
    obs = pd.DataFrame(
        {"Reference_strand": [reference_strand] * n},
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
