from __future__ import annotations

import gzip
import json
from array import array
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from smftools.cli import helpers
from smftools.cli.export_bundle import export_bundle_for_experiment
from smftools.cli.export_fastq import export_fastq_for_experiment, export_fastq_for_project
from smftools.informatics.bam_functions import concatenate_fastqs_to_bam
from smftools.informatics.export_bundle import ExportBundleError, read_bundle_manifest
from smftools.informatics.input_manifest import InputManifestError, resolve_input_manifest_readonly
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
    identity = pd.read_csv(outdir / "identity_map.csv")
    with gzip.open(outdir / "bc01.fastq.gz", "rt") as handle:
        assert (
            handle.read()
            == f"@{identity.loc[identity['source_read_id'] == 'read1', 'bundle_read_id'].item()}\nACGT\n+\n????\n"
        )
    with gzip.open(outdir / "bc02.fastq.gz", "rt") as handle:
        assert "\nCCCC\n+\n::::\n" in handle.read()
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
        assert "\nACGT\n+\n????\n" in handle.read()


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
    assert content.count("\n+") == 2
    bundle = json.loads((outdir / "bundle_manifest.json").read_text())
    assert bundle["selection"]["unfiltered"] is True


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
    identity = pd.read_csv(outdir / "identity_map.csv")
    assert identity["bundle_read_id"].is_unique
    assert set(identity["experiment_id"]) == {"expA", "expB"}
    resolved = resolve_input_manifest_readonly(
        input_manifest_path=outdir / "bundle_manifest.json", modality="conversion"
    )
    normalized_bam = tmp_path / "project-reingested.bam"
    concatenate_fastqs_to_bam(
        resolved.fastq_inputs(), normalized_bam, progress=False, samtools_backend="python"
    )
    import pysam

    with pysam.AlignmentFile(normalized_bam, "rb", check_sq=False) as handle:
        assert len({read.query_name for read in handle.fetch(until_eof=True)}) == 2


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


def test_paired_fastq_bundle_preserves_layout_and_molecule_identity(tmp_path, patch_config):
    rows = [
        _read("template/1", "bc01", [0, 1, 2, 3], [30] * 4),
        _read("template/2", "bc01", [3, 2, 1, 0], [30] * 4),
    ]
    for row, mate in zip(rows, ("R1", "R2"), strict=True):
        row.update(
            template_id="template",
            mate=mate,
            paired=True,
            proper_pair=True,
            mate_unmapped=False,
        )
    raw = write_raw_store(
        pd.DataFrame(rows), tmp_path / "raw_outputs", reference_lengths={"ref_top": 12}
    )
    paths = SimpleNamespace(raw_spine=raw["spine"], preprocess_spine=None, pp_dedup=None, pp=None)
    patch_config(
        SimpleNamespace(
            experiment_name="paired",
            sample_name_col_for_plotting="Sample",
            smf_modality="conversion",
        ),
        paths,
    )

    outdir = tmp_path / "bundle"
    export_fastq_for_experiment("fake.csv", outdir, allow_unfiltered=True)

    inputs = pd.read_csv(outdir / "inputs.csv")
    assert set(inputs["mate"]) == {"R1", "R2"}
    assert inputs["pair_id"].nunique() == 1
    identity = pd.read_csv(outdir / "identity_map.csv")
    assert identity["molecule_uid"].nunique() == 1
    assert set(identity["mate"]) == {"R1", "R2"}
    headers = []
    for path in sorted(outdir.glob("*__R?.fastq.gz")):
        with gzip.open(path, "rt") as handle:
            headers.append(handle.readline().strip().removeprefix("@"))
    assert {header.rsplit("/", 1)[1] for header in headers} == {"1", "2"}
    assert len({header.rsplit("/", 1)[0] for header in headers}) == 1
    resolved = resolve_input_manifest_readonly(
        input_manifest_path=outdir / "bundle_manifest.json", modality="conversion"
    )
    normalized_bam = tmp_path / "paired-reingested.bam"
    concatenate_fastqs_to_bam(
        resolved.fastq_inputs(), normalized_bam, progress=False, samtools_backend="python"
    )
    import pysam

    with pysam.AlignmentFile(normalized_bam, "rb", check_sq=False) as handle:
        reads = list(handle.fetch(until_eof=True))
    assert len(reads) == 2
    assert {read.query_name for read in reads} == {identity["molecule_uid"].iloc[0]}
    assert {read.is_read1 for read in reads} == {False, True}


def test_sequence_bundle_relocates_and_rejects_direct_reingestion(tmp_path, patch_config):
    raw = write_raw_store(
        pd.DataFrame([_read("read1", "bc01", [0, 1, 2, 3], [30] * 4)]),
        tmp_path / "raw_outputs",
        reference_lengths={"ref_top": 12},
    )
    paths = SimpleNamespace(raw_spine=raw["spine"], preprocess_spine=None, pp_dedup=None, pp=None)
    patch_config(
        SimpleNamespace(
            experiment_name="exp", sample_name_col_for_plotting="Sample", smf_modality="conversion"
        ),
        paths,
    )
    original = tmp_path / "bundle"
    export_fastq_for_experiment("fake.csv", original, allow_unfiltered=True)
    moved = tmp_path / "relocated" / "bundle"
    moved.parent.mkdir()
    original.rename(moved)

    payload = read_bundle_manifest(moved)
    assert payload["bundle_kind"] == "sequence_only"
    resolved = resolve_input_manifest_readonly(
        input_manifest_path=moved / "bundle_manifest.json", modality="conversion"
    )
    assert resolved.rows[0].sample == "bc01"
    with pytest.raises(InputManifestError, match="Direct-modification analysis requires"):
        resolve_input_manifest_readonly(
            input_manifest_path=moved / "bundle_manifest.json", modality="direct"
        )
    fastq = next(moved.glob("*.fastq.gz"))
    fastq.write_bytes(fastq.read_bytes() + b"tamper")
    with pytest.raises(ExportBundleError, match="checksum mismatch|size mismatch"):
        read_bundle_manifest(moved)


def test_bam_bundle_preserves_auxiliary_tags_and_checksums(tmp_path, patch_config):
    import pysam

    bam = tmp_path / "bam_outputs" / "aligned.bam"
    bam.parent.mkdir()
    header = {
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": "ref", "LN": 12}],
        "RG": [{"ID": "rg1", "SM": "sample1"}],
    }
    with pysam.AlignmentFile(bam, "wb", header=header) as handle:
        read = pysam.AlignedSegment()
        read.query_name = "read1"
        read.query_sequence = "ACGT"
        read.query_qualities = pysam.qualitystring_to_array("????")
        read.reference_id = 0
        read.reference_start = 0
        read.cigarstring = "4M"
        read.mapping_quality = 60
        read.set_tag("BC", "bc01")
        read.set_tag("RG", "rg1")
        read.set_tag("MM", "C+m,0;")
        read.set_tag("ML", array("B", [200]))
        handle.write(read)
    pysam.index(str(bam))
    row = _read("read1", "bc01", [0, 1, 2, 3], [30] * 4)
    row["source_read_id"] = "read1"
    raw = write_raw_store(
        pd.DataFrame([row]),
        tmp_path / "raw_outputs",
        reference_lengths={"ref_top": 12},
        bam_path=bam,
        extra_uns={"modality": "direct"},
    )
    paths = SimpleNamespace(raw_spine=raw["spine"], preprocess_spine=None, pp_dedup=None, pp=None)
    patch_config(
        SimpleNamespace(
            experiment_name="direct", sample_name_col_for_plotting="Sample", smf_modality="direct"
        ),
        paths,
    )

    outdir = tmp_path / "bam_bundle"
    export_bundle_for_experiment("fake.csv", outdir, bundle_format="bam", allow_unfiltered=True)

    payload = read_bundle_manifest(outdir)
    assert payload["bundle_kind"] == "lossless_bam"
    exported = next((outdir / "alignments").glob("*.bam"))
    with pysam.AlignmentFile(exported, "rb") as handle:
        read = next(handle.fetch(until_eof=True))
        assert read.get_tag("BC") == "bc01"
        assert read.get_tag("RG") == "rg1"
        assert read.get_tag("MM") == "C+m,0;"
        assert list(read.get_tag("ML")) == [200]
    resolved = resolve_input_manifest_readonly(
        input_manifest_path=outdir / "bundle_manifest.json",
        alignment_mode="existing",
        modality="direct",
    )
    assert resolved.rows[0].modification_capability == "mm_ml"
