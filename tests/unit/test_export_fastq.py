from __future__ import annotations

import gzip
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from smftools.cli import helpers
from smftools.cli.export_fastq import export_fastq_for_experiment, export_fastq_for_project
from smftools.informatics.raw_store import write_raw_store
from smftools.preprocessing.partitioned_executor import execute_partitioned_preprocessing
from smftools.project.registry import add_experiment, init_project
from smftools.readwrite import safe_write_h5ad


def _read(read_id, barcode, seq_ints, qual_ints, read_length=None):
    cigar = f"{len(seq_ints)}M"
    n = read_length if read_length is not None else len(seq_ints)
    return dict(
        read_id=read_id,
        reference="ref",
        Reference_strand="ref_top",
        barcode=barcode,
        sample=barcode,
        reference_start=0,
        cigar=cigar,
        aligned_length=len(seq_ints),
        sequence=seq_ints,
        quality=qual_ints,
        mismatch=[4] * len(seq_ints),
        read_length=n,
        mapped_length=n,
        reference_length=12,
        read_quality=30,
        mapping_quality=60,
        read_length_to_reference_length_ratio=n / 12,
        mapped_length_to_reference_length_ratio=n / 12,
        mapped_length_to_read_length_ratio=1.0,
    )


def _preprocess_cfg(**overrides):
    base = dict(
        smf_modality="conversion",
        output_binary_layer_name="binarized_methylation",
        bypass_clean_nan=False,
        clean_nan_layers=["nan0_0minus1", "nan_half"],
        reference_column="Reference_strand",
        mod_target_bases=["GpC", "CpG"],
        bypass_append_base_context=False,
        target_task_memory_mb=1,
        position_max_nan_threshold=0.6,
        read_len_filter_thresholds=[None, None],
        mapped_len_filter_thresholds=[None, None],
        read_len_to_ref_ratio_filter_thresholds=[None, None],
        mapped_len_to_ref_ratio_filter_thresholds=[None, None],
        mapped_len_to_read_len_ratio_filter_thresholds=[None, None],
        read_quality_filter_thresholds=[None, None],
        read_mapping_quality_filter_thresholds=[None, None],
        bypass_filter_reads_on_length_quality_mapping=False,
        read_mod_filtering_gpc_thresholds=None,
        read_mod_filtering_cpg_thresholds=None,
        read_mod_filtering_c_thresholds=None,
        read_mod_filtering_a_thresholds=None,
        read_mod_filtering_use_other_c_as_background=False,
        min_valid_fraction_positions_in_read_vs_ref=None,
        bypass_filter_reads_on_modification_thresholds=False,
        bypass_flag_duplicate_reads=True,
        sample_name_col_for_plotting="Sample",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture
def patch_config(monkeypatch):
    """Patch helpers.load_experiment_config / get_adata_paths for a given cfg/paths pair."""

    def _apply(cfg, paths):
        monkeypatch.setattr(helpers, "load_experiment_config", lambda _path: cfg)
        monkeypatch.setattr(helpers, "get_adata_paths", lambda _cfg: paths)

    return _apply


def test_export_fastq_for_experiment_uses_partitioned_qc(tmp_path, patch_config):
    rows = [
        _read("read1", "bc01", [0, 1, 2, 3], [30] * 4),  # passes length filter
        _read("read2", "bc01", [3, 2, 1, 0], [20] * 4, read_length=1),  # fails
        _read("read3", "bc02", [1, 1, 1, 1], [25] * 4),  # passes
    ]
    raw_out = write_raw_store(
        pd.DataFrame(rows),
        tmp_path / "raw_outputs",
        reference_lengths={"ref_top": 12},
        analysis_mode="locus",
        extra_uns={"References": {"ref_FASTA_sequence": "ACGCGTACGTAC"}},
    )
    pp_cfg = _preprocess_cfg(read_len_filter_thresholds=[2, None])
    outputs = execute_partitioned_preprocessing(
        raw_out["spine"], pp_cfg, tmp_path / "preprocess_adata_outputs"
    )

    paths = SimpleNamespace(
        raw_spine=raw_out["spine"], preprocess_spine=outputs["spine"], pp_dedup=None, pp=None
    )
    patch_config(SimpleNamespace(sample_name_col_for_plotting="Sample"), paths)

    outdir = tmp_path / "fastq_out"
    result = export_fastq_for_experiment("fake.csv", outdir)

    assert result == outdir
    with gzip.open(outdir / "bc01.fastq.gz", "rt") as handle:
        assert handle.read() == "@read1\nACGT\n+\n????\n"
    with gzip.open(outdir / "bc02.fastq.gz", "rt") as handle:
        assert handle.read() == "@read3\nCCCC\n+\n::::\n"
    manifest = pd.read_csv(outdir / "fastq_manifest.csv")
    assert dict(zip(manifest["barcode"], manifest["n_reads"])) == {"bc01": 1, "bc02": 1}


def test_export_fastq_for_experiment_falls_back_to_legacy_pp_dedup(tmp_path, patch_config):
    rows = [
        _read("read1", "bc01", [0, 1, 2, 3], [30] * 4),
        _read("read2", "bc01", [3, 2, 1, 0], [20] * 4),
    ]
    raw_out = write_raw_store(
        pd.DataFrame(rows),
        tmp_path / "raw_outputs",
        reference_lengths={"ref_top": 12},
        analysis_mode="locus",
    )
    # Legacy dedup adata: only read1 survived filtering/dedup.
    pp_dedup = ad.AnnData(
        X=np.zeros((1, 1)), obs=pd.DataFrame({"Sample": ["bc01"]}, index=["read1"])
    )
    pp_dedup_path = tmp_path / "pp_dedup.h5ad.gz"
    safe_write_h5ad(pp_dedup, pp_dedup_path, backup=False, verbose=False)

    paths = SimpleNamespace(
        raw_spine=raw_out["spine"], preprocess_spine=None, pp_dedup=pp_dedup_path, pp=None
    )
    patch_config(SimpleNamespace(sample_name_col_for_plotting="Sample"), paths)

    outdir = tmp_path / "fastq_out"
    export_fastq_for_experiment("fake.csv", outdir)

    files = sorted(outdir.glob("*.fastq.gz"))
    assert [f.name for f in files] == ["bc01.fastq.gz"]
    with gzip.open(files[0], "rt") as handle:
        assert handle.read() == "@read1\nACGT\n+\n????\n"


def test_export_fastq_for_experiment_raises_without_qc_source(tmp_path, patch_config):
    rows = [_read("read1", "bc01", [0, 1, 2, 3], [30] * 4)]
    raw_out = write_raw_store(
        pd.DataFrame(rows),
        tmp_path / "raw_outputs",
        reference_lengths={"ref_top": 12},
        analysis_mode="locus",
    )
    paths = SimpleNamespace(
        raw_spine=raw_out["spine"], preprocess_spine=None, pp_dedup=None, pp=None
    )
    patch_config(SimpleNamespace(sample_name_col_for_plotting="Sample"), paths)

    with pytest.raises(ValueError, match="no QC-passed read set found"):
        export_fastq_for_experiment("fake.csv", tmp_path / "fastq_out")


def test_export_fastq_for_experiment_allow_unfiltered_writes_all_reads(tmp_path, patch_config):
    rows = [
        _read("read1", "bc01", [0, 1, 2, 3], [30] * 4),
        _read("read2", "bc01", [3, 2, 1, 0], [20] * 4),
    ]
    raw_out = write_raw_store(
        pd.DataFrame(rows),
        tmp_path / "raw_outputs",
        reference_lengths={"ref_top": 12},
        analysis_mode="locus",
    )
    paths = SimpleNamespace(
        raw_spine=raw_out["spine"], preprocess_spine=None, pp_dedup=None, pp=None
    )
    patch_config(SimpleNamespace(sample_name_col_for_plotting="Sample"), paths)

    outdir = tmp_path / "fastq_out"
    export_fastq_for_experiment("fake.csv", outdir, allow_unfiltered=True)

    with gzip.open(outdir / "bc01.fastq.gz", "rt") as handle:
        content = handle.read()
    assert content.count("@read") == 2


def test_export_fastq_for_experiment_missing_raw_spine_raises(tmp_path, patch_config):
    paths = SimpleNamespace(
        raw_spine=tmp_path / "raw_outputs" / "spine.h5ad",
        preprocess_spine=None,
        pp_dedup=None,
        pp=None,
    )
    patch_config(SimpleNamespace(sample_name_col_for_plotting="Sample"), paths)

    with pytest.raises(FileNotFoundError, match="smftools experiment raw"):
        export_fastq_for_experiment("fake.csv", tmp_path / "fastq_out")


def test_export_fastq_for_project_namespaces_by_experiment(tmp_path):
    project_dir = tmp_path / "project"
    init_project(project_dir)
    pp_cfg = _preprocess_cfg()

    for exp_id, barcode, seq in [("expA", "bc01", [0, 1, 2, 3]), ("expB", "bc01", [1, 1, 1, 1])]:
        exp_root = tmp_path / exp_id
        raw_out = write_raw_store(
            pd.DataFrame([_read("r1", barcode, seq, [30] * 4)]),
            exp_root / "raw_outputs",
            reference_lengths={"ref_top": 12},
            analysis_mode="locus",
            extra_uns={"References": {"ref_FASTA_sequence": "ACGCGTACGTAC"}},
        )
        execute_partitioned_preprocessing(
            raw_out["spine"], pp_cfg, exp_root / "preprocess_adata_outputs"
        )
        add_experiment(project_dir, raw_out["spine"].parent, experiment_id=exp_id)

    outdir = tmp_path / "fastq_out"
    export_fastq_for_project(project_dir, outdir)

    files = sorted(f.name for f in outdir.glob("*.fastq.gz"))
    assert files == ["expA__bc01.fastq.gz", "expB__bc01.fastq.gz"]
    with gzip.open(outdir / "expA__bc01.fastq.gz", "rt") as handle:
        assert "ACGT" in handle.read()
    with gzip.open(outdir / "expB__bc01.fastq.gz", "rt") as handle:
        assert "CCCC" in handle.read()


def test_export_fastq_for_project_skips_experiment_without_preprocess_spine(tmp_path, caplog):
    project_dir = tmp_path / "project"
    init_project(project_dir)
    exp_root = tmp_path / "expA"
    raw_out = write_raw_store(
        pd.DataFrame([_read("r1", "bc01", [0, 1, 2, 3], [30] * 4)]),
        exp_root / "raw_outputs",
        reference_lengths={"ref_top": 12},
        analysis_mode="locus",
    )
    add_experiment(project_dir, raw_out["spine"].parent, experiment_id="expA")

    outdir = tmp_path / "fastq_out"
    export_fastq_for_project(project_dir, outdir)

    assert list(outdir.glob("*.fastq.gz")) == []
    manifest = pd.read_csv(outdir / "fastq_manifest.csv")
    assert manifest.empty


def test_export_fastq_for_project_filters_by_experiments_list(tmp_path):
    project_dir = tmp_path / "project"
    init_project(project_dir)
    pp_cfg = _preprocess_cfg()

    for exp_id in ("expA", "expB"):
        exp_root = tmp_path / exp_id
        raw_out = write_raw_store(
            pd.DataFrame([_read("r1", "bc01", [0, 1, 2, 3], [30] * 4)]),
            exp_root / "raw_outputs",
            reference_lengths={"ref_top": 12},
            analysis_mode="locus",
            extra_uns={"References": {"ref_FASTA_sequence": "ACGCGTACGTAC"}},
        )
        execute_partitioned_preprocessing(
            raw_out["spine"], pp_cfg, exp_root / "preprocess_adata_outputs"
        )
        add_experiment(project_dir, raw_out["spine"].parent, experiment_id=exp_id)

    outdir = tmp_path / "fastq_out"
    export_fastq_for_project(project_dir, outdir, experiments=["expA"])

    files = sorted(f.name for f in outdir.glob("*.fastq.gz"))
    assert files == ["expA__bc01.fastq.gz"]
