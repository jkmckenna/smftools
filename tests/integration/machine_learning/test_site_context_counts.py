"""SCB-01: counting a plan dataset's calls per site, on stores the writers make."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from smftools.analysis.compute.site_context_bias import offset_enrichment, site_contexts
from smftools.informatics.molecule_identity import molecule_uid
from smftools.informatics.partition_store import write_experiment_store
from smftools.machine_learning.plan import parse_ml_plan
from smftools.project.reference_registry import ReferenceRegistry
from smftools.project.registry import init_project, load_registry, save_registry
from smftools.tools.site_context_bias import count_site_calls

pytestmark = pytest.mark.integration

#           0         1         2
#           012345678901234567890123456789
SEQUENCE = "ACTGACAAGCTTACGGCTACCAACTTGACA"
C_SITES = [i for i, base in enumerate(SEQUENCE) if base == "C" and 0 < i < len(SEQUENCE) - 1]
READS = 10


def _calls(rng) -> np.ndarray:
    """A C site is modified exactly when the next base is T; some calls missing."""
    calls = np.full((READS, len(SEQUENCE)), np.nan, dtype=np.float32)
    for position in C_SITES:
        calls[:, position] = 1.0 if SEQUENCE[position + 1] == "T" else 0.0
    calls[rng.random(calls.shape) < 0.2] = np.nan
    return calls


@pytest.fixture
def project(tmp_path: Path):
    rng = np.random.default_rng(0)
    run_root = tmp_path / "runs" / "exp"
    preprocess = run_root / "preprocess_adata_outputs"
    barcodes = ["barcode01"] * READS + ["barcode02"] * READS
    read_ids = [f"r{index}" for index in range(2 * READS)]
    calls = np.vstack([_calls(rng), _calls(rng)])
    obs = pd.DataFrame(
        {
            "Reference_strand": pd.Categorical(["locus_top"] * len(read_ids)),
            "Sample": pd.Categorical(barcodes),
        },
        index=read_ids,
    )
    source = ad.AnnData(X=calls, obs=obs)
    source.var_names = [str(position) for position in range(len(SEQUENCE))]
    source.var["locus_top_C_site"] = [position in C_SITES for position in range(len(SEQUENCE))]
    source.uns["References"] = {"locus_FASTA_sequence": SEQUENCE}
    paths = write_experiment_store(source, preprocess, experiment="exp", modality="deaminase")

    experiment_uid = str(uuid4())
    uids = [molecule_uid(experiment_uid, read_id) for read_id in read_ids]
    (run_root / "molecule_index").mkdir(parents=True)
    pd.DataFrame(
        {
            "molecule_uid": uids,
            "experiment_uid": experiment_uid,
            "read_id": read_ids,
            "Reference_strand": "locus_top",
            "Sample": barcodes,
            "Barcode": barcodes,
            "reference_start": 0,
            "reference_end": len(SEQUENCE),
        }
    ).to_parquet(run_root / "molecule_index" / "part.parquet", index=False)
    (preprocess / "read_index").mkdir()
    pd.DataFrame(
        {
            "molecule_uid": uids,
            "group_path": [f"store/{barcode}" for barcode in barcodes],
            "group_row": list(range(READS)) * 2,
        }
    ).to_parquet(preprocess / "read_index" / "part.parquet", index=False)
    pd.DataFrame(
        {"task_id": ["t"], "reference": ["locus_top"], "layers": [[]], "has_x": [True]}
    ).to_parquet(preprocess / "catalog.parquet", index=False)
    raw = run_root / "raw_outputs"
    raw.mkdir()
    (raw / "spine.h5ad").touch()
    pd.DataFrame({"reference": ["locus_top"], "max_end": [len(SEQUENCE)]}).to_parquet(
        raw / "interval_catalog.parquet", index=False
    )
    root = tmp_path / "project"
    init_project(root)
    registry = load_registry(root)
    registry["experiments"] = {
        "exp": {
            "path": str(run_root),
            "name": "exp",
            "experiment_uid": experiment_uid,
            "modality": "deaminase",
            "schema_version": 1,
            "spines": {"raw": str(raw / "spine.h5ad"), "preprocess": str(paths["spine"])},
            "references": {"locus_top": "uid"},
            "n_reads": len(read_ids),
            "status": "active",
            "catalogs": {
                "interval_catalog.parquet": str(raw / "interval_catalog.parquet"),
                "molecule_index": str(run_root / "molecule_index"),
                "preprocess_read_index": str(preprocess / "read_index"),
            },
        }
    }
    save_registry(root, registry)
    ReferenceRegistry(canonical_names={"uid": "locus"}).save(root / "reference_registry.yaml")
    return root, calls, barcodes


def _plan():
    return parse_ml_plan(
        {
            "schema_version": 1,
            "scope": {"kind": "project"},
            "datasets": {
                "sites": {
                    "modalities": ["deaminase"],
                    "references": ["locus"],
                    "channels": [
                        {
                            "name": "C",
                            "biological_role": "accessibility",
                            "sources": [
                                {
                                    "modality": "deaminase",
                                    "stage": "preprocess",
                                    "layer": "X",
                                    "site_context": "C",
                                }
                            ],
                        }
                    ],
                }
            },
            "splits": {},
            "models": {},
            "jobs": {},
        }
    )


def test_counts_equal_a_direct_count_per_group(project) -> None:
    root, calls, barcodes = project
    counts = count_site_calls(_plan(), "sites", project_dir=root, group_by="Barcode")

    assert counts.frame_reference == "locus"
    assert counts.sequence_for == {"locus_top": "locus"}
    table = counts.sites.set_index(["group", "position"])
    for barcode in ("barcode01", "barcode02"):
        rows = np.array([b == barcode for b in barcodes])
        for position in C_SITES:
            column = calls[rows, position]
            observed = int(np.isfinite(column).sum())
            modified = int((column == 1).sum())
            assert table.loc[(barcode, position), "observed"] == observed
            assert table.loc[(barcode, position), "modified"] == modified
    # Only design (C) sites carry calls.
    assert set(counts.sites["position"]) <= set(C_SITES)


def test_worker_split_gives_identical_counts(project) -> None:
    root, _, _ = project
    one = count_site_calls(_plan(), "sites", project_dir=root, group_by="Barcode")
    two = count_site_calls(_plan(), "sites", project_dir=root, group_by="Barcode", workers=2)
    pd.testing.assert_frame_equal(one.sites, two.sites)


def test_enrichment_recovers_the_planted_preference(project) -> None:
    root, _, _ = project
    counts = count_site_calls(_plan(), "sites", project_dir=root)
    sites = site_contexts(
        counts.sites, {"locus": SEQUENCE}, flank=1, sequence_for=counts.sequence_for
    )
    assert set(sites["context"].str[1]) == {"C"}
    table = offset_enrichment(sites, flank=1)
    plus_one = table[table["offset"] == 1].set_index("base")["log2_enrichment"]
    assert plus_one["T"] > 1
    assert (plus_one.drop(["T"]).dropna() < 0).all()


def test_reference_sequences_drop_padding(tmp_path: Path) -> None:
    from smftools.tools.site_context_bias import reference_sequences

    spine = ad.AnnData(obs=pd.DataFrame(index=["r"]))
    spine.uns["References"] = {
        "long_FASTA_sequence": "ACGTACGT",
        "short_FASTA_sequence": "ACGTNNNN",  # padded to the longest
        "bare_FASTA_sequence": "ACGNNNNN",  # no recorded length
    }
    spine.uns["reference_lengths"] = {"long_top": 8, "short_bottom": 4}
    path = tmp_path / "spine.h5ad"
    spine.write_h5ad(path)
    assert reference_sequences([path]) == {"long": "ACGTACGT", "short": "ACGT", "bare": "ACG"}


def _cli(root, tmp_path, *extra):
    import json

    from click.testing import CliRunner

    from smftools.cli_entry import cli

    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_plan().to_dict()))
    out = tmp_path / "out"
    result = CliRunner().invoke(
        cli,
        [
            "project",
            "context-bias",
            str(root),
            "--plan",
            str(plan_path),
            "--dataset",
            "sites",
            "--output",
            str(out),
            *extra,
        ],
    )
    assert result.exit_code == 0, result.output
    return out, json.loads((out / "run.json").read_text()), result.output


def test_cli_writes_every_output_and_reuses_counts(project, tmp_path) -> None:
    root, _, _ = project
    out, record, output = _cli(
        root,
        tmp_path,
        "--group-by",
        "Barcode",
        "--flank",
        "2",
        "--kmer",
        "3",
        "--kmer",
        "5",
        "--reference-group",
        "barcode01",
    )
    assert "counted" in output and not record["counts_reused"]
    for name in (
        "site_counts.parquet",
        "site_counts.json",
        "sites.parquet",
        "offset_enrichment.csv",
        "kmer_rates.csv",
        "group_differences.csv",
        "offset_enrichment.png",
        "enrichment_logo.png",
        "kmer_rates_k3.png",
        "kmer_rates_k5.png",
        "group_differences.png",
        "run.json",
    ):
        assert (out / name).exists(), name
    assert record["groups"] == ["barcode01", "barcode02"]
    assert set(pd.read_csv(out / "kmer_rates.csv")["k"]) == {3, 5}
    assert set(pd.read_parquet(out / "sites.parquet")["context"].str.len()) == {5}

    # Another flank re-uses the counts; the windows follow the new flank.
    _, again, output = _cli(root, tmp_path, "--group-by", "Barcode", "--flank", "1", "--no-figures")
    assert again["counts_reused"] and "reused cached counts" in output
    assert set(pd.read_parquet(out / "sites.parquet")["context"].str.len()) == {3}
    # A different grouping is a different count.
    _, regrouped, _ = _cli(root, tmp_path, "--no-figures")
    assert not regrouped["counts_reused"] and regrouped["groups"] == ["all"]


def test_cli_rejects_bad_arguments(project, tmp_path) -> None:
    import json

    from click.testing import CliRunner

    from smftools.cli_entry import cli

    root, _, _ = project
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_plan().to_dict()))
    base = [
        "project",
        "context-bias",
        str(root),
        "--plan",
        str(plan_path),
        "--dataset",
        "sites",
        "--output",
        str(tmp_path / "o"),
    ]
    for extra, message in (
        (["--flank", "1", "--kmer", "5"], "k-mer size 5"),
        (["--group-by", "Barcode", "--reference-group", "missing"], "reference group"),
        (["--channel", "nope"], "channel"),
    ):
        result = CliRunner().invoke(cli, [*base, *extra])
        assert result.exit_code != 0 and message in result.output, result.output


def test_cli_exports_weight_tables(project, tmp_path) -> None:
    """`HCE-01`: context-bias writes relative k-mer weights for the HMM."""
    from smftools.analysis.compute.site_context_bias import read_weight_table

    root, _, _ = project
    out, record, _ = _cli(
        root,
        tmp_path,
        "--group-by",
        "Barcode",
        "--kmer",
        "1",
        "--kmer",
        "3",
        "--export-weights",
        "--weights-source",
        "naked_dna",
        "--no-figures",
    )
    assert record["weight_tables"] == ["context_weights_k3.parquet"]
    table = read_weight_table(out / "context_weights_k3.parquet")
    assert set(table["group"]) == {"barcode01", "barcode02"}
    assert (table["source"] == "naked_dna").all() and (table["k"] == 3).all()
    # In the fixture a C followed by T is always modified: its weight is above 1.
    assert (table.loc[table["kmer"].str[2] == "T", "weight"] > 1).all()
