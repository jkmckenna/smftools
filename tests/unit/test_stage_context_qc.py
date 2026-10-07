"""Sequence-context QC of a preprocess generation (`SCQ-01`)."""

import json
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from smftools.preprocessing.stage_context_qc import (
    context_statistics,
    count_stage_sites,
    site_types_for,
    write_stage_context_qc,
)


def _store(path: Path, reference: str, reads: list[str], positions, sites, values) -> Path:
    var = pd.DataFrame(
        {f"{reference}_C_site": sites, f"{reference}_GpC_site": np.zeros(len(sites), bool)},
        index=[str(p) for p in positions],
    )
    ad.AnnData(
        X=np.asarray(values, dtype=float), obs=pd.DataFrame(index=reads), var=var
    ).write_zarr(path)
    return path


def _catalog_and_paths(tmp_path):
    nan = np.nan
    stores = {
        ("ref_top", "bc1", 0): _store(
            tmp_path / "a",
            "ref_top",
            ["r1", "r2", "dup"],
            [0, 1, 2, 3],
            [False, True, False, True],
            [[1, 1, 0, 0], [0, nan, 1, 1], [1, 1, 1, 1]],
        ),
        ("ref_top", "bc1", 1): _store(
            tmp_path / "b",
            "ref_top",
            ["r3", "fail"],
            [0, 1, 2, 3],
            [False, True, False, True],
            [[nan, 0, 1, 1], [1, 1, 1, 1]],
        ),
    }
    catalog = pd.DataFrame(
        [{"reference": ref, "barcode": bc, "chunk_index": chunk} for ref, bc, chunk in stores]
    )
    obs = pd.DataFrame(
        {
            "Barcode": "bc1",
            "passes_qc": [True, True, True, True, False],
            "passes_dedup": [True, True, False, True, False],
        },
        index=["r1", "r2", "dup", "r3", "fail"],
    )
    return catalog, (lambda row: stores[(row.reference, row.barcode, row.chunk_index)]), obs


def test_tallies_count_passing_reads_across_chunks(tmp_path):
    catalog, path, obs = _catalog_and_paths(tmp_path)
    counts = count_stage_sites(catalog, path, obs, ["C_site"])
    by_position = counts.set_index("position")[["observed", "modified"]]
    # C sites 1 and 3; passing reads r1, r2 (chunk 0) and r3 (chunk 1).
    assert by_position.loc[1].tolist() == [2, 1]  # r1=1, r2 unobserved, r3=0
    assert by_position.loc[3].tolist() == [3, 2]  # r1=0, r2=1, r3=1
    assert set(counts["site_type"]) == {"C_site"} and set(counts["barcode"]) == {"bc1"}


def test_without_dedup_the_qc_flag_decides(tmp_path):
    catalog, path, obs = _catalog_and_paths(tmp_path)
    counts = count_stage_sites(catalog, path, obs.drop(columns="passes_dedup"), ["C_site"])
    assert counts.set_index("position").loc[3, "observed"] == 4  # dup now counted


def test_bottom_strand_contexts_are_reverse_complemented():
    counts = pd.DataFrame(
        {
            "barcode": "bc1",
            "physical_reference": ["ref_top", "ref_bottom"],
            "site_type": "C_site",
            "position": [3, 4],
            "observed": [10, 10],
            "modified": [5, 5],
        }
    )
    #          0123456
    sequence = "AATCGAT"
    sites, enrichment, rates = context_statistics(counts, {"ref": sequence}, flank=1, kmers=[1, 3])
    contexts = sites.set_index("physical_reference")["context"]
    assert contexts["ref_top"] == "TCG"
    assert contexts["ref_bottom"] == "TCG"  # forward CGA, reverse complement TCG
    assert sites["cpg"].all()
    assert {"barcode", "physical_reference", "site_type", "cpg"} <= set(enrichment.columns)
    assert set(rates["k"]) == {1, 3}


def test_site_types_follow_the_modality():
    assert site_types_for(SimpleNamespace(smf_modality="deaminase")) == ("C_site",)
    assert site_types_for(SimpleNamespace(smf_modality="conversion")) == ("GpC_site", "CpG_site")
    assert site_types_for(SimpleNamespace(smf_modality="direct")) == ()


def test_settings_off_write_nothing(tmp_path):
    cfg = SimpleNamespace(smf_modality="deaminase", stage_context_qc=False)
    assert write_stage_context_qc(tmp_path, tmp_path / "missing.parquet", None, {}, cfg) is None
    assert not (tmp_path / "context_qc").exists()


@pytest.mark.parametrize("stage", ["preprocess", "hmm"])
def test_settings_never_make_a_stage_stale(stage):
    from smftools.cli.helpers import stage_config_hash
    from smftools.config.experiment_config import ExperimentConfig

    base = ExperimentConfig(smf_modality="deaminase")
    changed = ExperimentConfig(
        smf_modality="deaminase",
        stage_context_qc=False,
        stage_context_qc_flank=5,
        stage_context_qc_kmers=[1, 5],
    )
    assert stage_config_hash(base, stage) == stage_config_hash(changed, stage)


def test_preprocess_stage_writes_context_qc_from_passing_reads(tmp_path):
    from smftools.informatics.partition_read import materialize
    from smftools.informatics.raw_store import write_raw_store
    from smftools.preprocessing.partitioned_executor import execute_partitioned_preprocessing

    from .test_partitioned_preprocess_executor import _cfg, _frame

    raw = write_raw_store(
        _frame(),
        tmp_path / "raw_outputs",
        reference_lengths={"ref_top": 12},
        analysis_mode="locus",
        extra_uns={"References": {"ref_FASTA_sequence": "ACGCGTACGTAC"}},
    )
    cfg = _cfg()
    cfg.smf_modality = "deaminase"
    cfg.bypass_label_deaminase_pcr_chimeras = True
    output_dir = tmp_path / "preprocess_outputs"
    outputs = execute_partitioned_preprocessing(raw["spine"], cfg, output_dir)
    counts = pd.read_parquet(output_dir / "context_qc" / "site_counts.parquet")
    assert set(counts["site_type"]) == {"C_site"} and len(counts)
    for name in ("sites.parquet", "offset_enrichment.csv", "kmer_rates.csv", "run.json"):
        assert (output_dir / "context_qc" / name).exists()

    # Equal to a direct count of the passing reads (read2 fails the length filter).
    adata = materialize(outputs["spine"], references=["ref_top"], start=0, end=12)
    passing = adata[adata.obs["passes_qc"].astype(bool).to_numpy()]
    assert passing.n_obs == 1
    values = np.asarray(passing.X, dtype=float)
    expected = []
    for site_type in ("C_site",):
        mask = passing.var[f"ref_top_{site_type}"].to_numpy(dtype=bool)
        observed = (~np.isnan(values[:, mask])).sum(0)
        modified = (np.nan_to_num(values[:, mask]) >= 0.5).sum(0)
        positions = passing.var.index.astype(int).to_numpy()[mask]
        for position, o, m in zip(positions, observed, modified, strict=True):
            if o:
                expected.append((site_type, int(position), int(o), int(m)))
    got = sorted(
        counts[["site_type", "position", "observed", "modified"]].itertuples(index=False, name=None)
    )
    assert got == sorted(expected)


def test_no_passing_calls_give_empty_tables():
    empty = pd.DataFrame(
        columns=["barcode", "physical_reference", "site_type", "position", "observed", "modified"]
    )
    sites, enrichment, rates = context_statistics(empty, {}, flank=3, kmers=[1, 3])
    assert sites.empty and enrichment.empty and rates.empty


# --- SCQ-03: backfill for finished stages ---------------------------------------


def _tables(directory: Path) -> dict:
    return {
        "counts": pd.read_parquet(directory / "site_counts.parquet"),
        "sites": pd.read_parquet(directory / "sites.parquet"),
        "rates": pd.read_csv(directory / "kmer_rates.csv"),
    }


def _assert_backfill_matches(monkeypatch, generation: Path, stage: str, cfg):
    from smftools.tools import context_qc_backfill

    written = _tables(generation / "context_qc")
    plots_before = sorted(p.name for p in (generation / "plots").rglob("*"))
    monkeypatch.setattr(context_qc_backfill, "current_generation", lambda *_: generation)

    result = context_qc_backfill.backfill_context_qc(generation.parent, stage, cfg)
    assert result["status"] == "exists"  # present: left alone without refresh
    result = context_qc_backfill.backfill_context_qc(generation.parent, stage, cfg, refresh=True)
    assert result["status"] == "written"

    backfilled = _tables(generation / "context_qc")
    for name, frame in written.items():
        pd.testing.assert_frame_equal(backfilled[name], frame, check_dtype=False)
    run = json.loads((generation / "context_qc" / "run.json").read_text())
    assert run["backfilled"] is True
    # The generation's own (checksummed) plot tree is untouched.
    assert sorted(p.name for p in (generation / "plots").rglob("*")) == plots_before
    assert (generation / "context_qc" / "plots" / "catalog.parquet").exists()


def test_preprocess_backfill_equals_the_stage_outputs(tmp_path, monkeypatch):
    from smftools.informatics.raw_store import write_raw_store
    from smftools.preprocessing.partitioned_executor import execute_partitioned_preprocessing

    from .test_partitioned_preprocess_executor import _cfg, _frame

    raw = write_raw_store(
        _frame(),
        tmp_path / "raw_outputs",
        reference_lengths={"ref_top": 12},
        analysis_mode="locus",
        extra_uns={"References": {"ref_FASTA_sequence": "ACGCGTACGTAC"}},
    )
    cfg = _cfg()
    cfg.smf_modality = "deaminase"
    cfg.bypass_label_deaminase_pcr_chimeras = True
    generation = tmp_path / "preprocess_outputs"
    execute_partitioned_preprocessing(raw["spine"], cfg, generation)
    _assert_backfill_matches(monkeypatch, generation, "preprocess", cfg)


def test_hmm_backfill_equals_the_stage_outputs(tmp_path, monkeypatch):
    from .test_hmm_partitioned_cli import _context_run

    cfg, outputs, _ = _context_run(
        tmp_path, hmm_variants={"learned": {"hmm_context_model": "learned"}}
    )
    _assert_backfill_matches(monkeypatch, outputs["task_catalog"].parent, "hmm", cfg)


def test_backfill_config_comes_from_the_experiment_manifest(tmp_path):
    from smftools.tools.context_qc_backfill import backfill_context_qc, experiment_config

    (tmp_path / "experiment_manifest.json").write_text(
        json.dumps({"config": {"smf_modality": "deaminase", "hmm_methbases": ["C"]}})
    )
    cfg = experiment_config(tmp_path)
    assert cfg.smf_modality == "deaminase"
    # No published generation: reported, nothing written.
    assert backfill_context_qc(tmp_path, "hmm", cfg)["status"] == "no_generation"
    with pytest.raises(FileNotFoundError):
        experiment_config(tmp_path / "missing")


def test_context_qc_commands_report_per_stage(tmp_path):
    from click.testing import CliRunner

    from smftools.cli_entry import cli

    (tmp_path / "experiment_manifest.json").write_text(
        json.dumps({"config": {"smf_modality": "deaminase"}})
    )
    result = CliRunner().invoke(cli, ["experiment", "context-qc", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert f"{tmp_path.name} preprocess: no_generation" in result.output
    assert f"{tmp_path.name} hmm: no_generation" in result.output


# --- HCE-10: backfill per-read HMM and raw fractions ---------------------------


def test_fraction_backfill_equals_what_the_hmm_stage_stores(tmp_path, monkeypatch):
    from smftools.readwrite import safe_read_zarr
    from smftools.tools import context_qc_backfill

    from .test_hmm_partitioned_cli import _context_run

    cfg, outputs, _ = _context_run(
        tmp_path, hmm_variants={"learned": {"hmm_context_model": "learned"}}
    )
    generation = outputs["task_catalog"].parent
    monkeypatch.setattr(context_qc_backfill, "current_generation", lambda *_: generation)
    plots_before = sorted(p.name for p in (generation / "plots").rglob("*"))

    result = context_qc_backfill.backfill_context_qc(generation.parent, "hmm-fractions", cfg)
    assert result["status"] == "written"
    table = pd.read_parquet(generation / "molecule_fractions" / "molecule_fractions.parquet")
    columns = [c for c in table.columns if c.endswith("fraction")]
    assert {
        "C_all_accessible_features_fraction",
        "C_learned_all_accessible_features_site_fraction",
        "C_site_modified_fraction",
    } <= set(columns)

    for record in pd.read_parquet(outputs["task_catalog"]).itertuples():
        task, _ = safe_read_zarr(generation / record.group_path)
        rows = table[table["task_id"] == record.task_id].reindex(task.obs_names)
        for column in columns:
            np.testing.assert_allclose(
                rows[column].to_numpy(dtype=float),
                task.obs[column].to_numpy(dtype=float),
                err_msg=column,
            )
    figures = sorted(
        p.name for p in (generation / "molecule_fractions" / "plots" / "features").glob("*.png")
    )
    assert any(name.endswith("molecule_fractions.png") for name in figures)
    assert any(name.endswith("hmm_vs_raw_scatter.png") for name in figures)
    assert sorted(p.name for p in (generation / "plots").rglob("*")) == plots_before
    assert (
        context_qc_backfill.backfill_context_qc(generation.parent, "hmm-fractions", cfg)["status"]
        == "exists"
    )
