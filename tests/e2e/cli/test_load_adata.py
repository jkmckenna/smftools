"""Pytest-based end-to-end check for load_adata."""

from __future__ import annotations

import csv
import importlib
import importlib.resources as resources
import random
import shutil
import sys
import types
from importlib.machinery import ModuleSpec
from pathlib import Path

import pandas as pd
import pytest

from smftools.cli.load_adata import load_adata
from smftools.cli.raw_adata import raw_adata

CONFIGS = [
    Path("tests/_test_inputs/test_experiment_config_direct_I.csv"),
    Path("tests/_test_inputs/test_experiment_config_deaminase_I.csv"),
    Path("tests/_test_inputs/test_experiment_config_conversion_I.csv"),
]


@pytest.mark.e2e
@pytest.mark.parametrize("config_path", CONFIGS, ids=lambda p: p.name)
def test_load_adata_e2e(config_path: Path):
    adata, adata_path, _cfg = load_adata(str(config_path))

    # `load_adata` returns `adata=None` only when a later pipeline stage's output
    # already existed and load was skipped -- not expected for a fresh test run,
    # but tolerate it by reading back from disk so these invariants still cover
    # the actual on-disk artifact either way.
    import anndata as ad

    if adata is None:
        adata = ad.read_h5ad(adata_path)

    # Basic non-empty-result invariants. These exist to catch the class of bug
    # where a refactor (e.g. of the AnnData-concatenation or dict-skip logic in
    # modkit_extract_to_adata.py) silently produces an empty or malformed result
    # instead of raising -- "does not raise" alone would not catch that.
    assert adata.n_obs > 0, "expected at least one read in the final AnnData"
    assert adata.n_vars > 0, "expected at least one position/var in the final AnnData"
    assert adata.obs_names.is_unique, "expected unique read names (obs_names)"


@pytest.mark.e2e
def test_partitioned_existing_bams_produce_one_raw_generation(tmp_path: Path):
    pysam = pytest.importorskip("pysam")
    source = Path("tests/_test_inputs/parallel_dispatch/sample.bam").resolve()
    fasta = tmp_path / "sample.fasta"
    shutil.copy2(Path("tests/_test_inputs/parallel_dispatch/sample.fasta").resolve(), fasta)
    partitions = [tmp_path / "lane-1.bam", tmp_path / "lane-2.bam"]
    with pysam.AlignmentFile(str(source), "rb") as input_bam:
        outputs = [
            pysam.AlignmentFile(str(path), "wb", header=input_bam.header) for path in partitions
        ]
        try:
            for index, record in enumerate(input_bam.fetch(until_eof=True)):
                outputs[index % len(outputs)].write(record)
        finally:
            for output in outputs:
                output.close()

    manifest = tmp_path / "inputs.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["path", "source_kind", "namespace", "sample", "barcode"]
        )
        writer.writeheader()
        for index, partition in enumerate(partitions, start=1):
            writer.writerow(
                {
                    "path": partition.name,
                    "source_kind": "aligned_bam",
                    "namespace": f"lane-{index}",
                    "sample": "sample",
                    "barcode": "sample",
                }
            )

    config = tmp_path / "config.csv"
    values = {
        "smf_modality": "direct",
        "alignment_mode": "existing",
        "input_manifest_path": str(manifest),
        "fasta": str(fasta),
        "output_directory": str(tmp_path / "output"),
        "experiment_name": "partition-e2e",
        "direct_signal_backend": "pysam",
        "samtools_backend": "python",
        "skip_bam_split": "True",
        "skip_bam_qc": "True",
        "input_already_demuxed": "True",
        "make_beds": "False",
        "make_bigwigs": "False",
        "threads": "1",
        "max_memory_gb": "4",
    }
    with config.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variable", "value", "type"])
        writer.writeheader()
        for variable, value in values.items():
            value_type = (
                "bool"
                if value in {"True", "False"}
                else "int"
                if variable == "threads"
                else "float"
                if variable == "max_memory_gb"
                else "str"
            )
            writer.writerow({"variable": variable, "value": value, "type": value_type})

    spine, spine_path, _cfg = raw_adata(str(config))

    assert spine_path.is_file()
    assert spine.n_obs == 4
    assert spine.obs_names.is_unique
    assert set(spine.obs["namespace"].astype(str)) == {"lane-1", "lane-2"}


@pytest.mark.e2e
def test_sequence_export_bundle_reingests_as_fresh_raw_generation(tmp_path: Path, monkeypatch):
    if shutil.which("minimap2") is None:
        pytest.skip("minimap2 is required for the export-bundle round trip")
    from types import SimpleNamespace

    from smftools.cli import helpers
    from smftools.cli.export_fastq import export_fastq_for_experiment
    from smftools.informatics.raw_store import write_raw_store

    rng = random.Random(13)
    reference_sequence = "".join(rng.choices("ACGT", k=2400))
    read_sequence = reference_sequence[500:1100]
    encoding = {base: index for index, base in enumerate("ACGT")}
    source = write_raw_store(
        pd.DataFrame(
            [
                {
                    "read_id": "duplicate-prone-name",
                    "reference": "ref",
                    "Reference_strand": "ref_top",
                    "barcode": "bc01",
                    "sample": "sample-one",
                    "reference_start": 500,
                    "cigar": "600M",
                    "aligned_length": 600,
                    "sequence": [encoding[base] for base in read_sequence],
                    "quality": [30] * 600,
                    "mismatch": [4] * 600,
                    "read_length": 600,
                    "mapped_length": 600,
                    "reference_length": 2400,
                    "read_quality": 30,
                    "mapping_quality": 60,
                    "read_length_to_reference_length_ratio": 0.25,
                    "mapped_length_to_reference_length_ratio": 0.25,
                    "mapped_length_to_read_length_ratio": 1.0,
                }
            ]
        ),
        tmp_path / "source" / "raw_outputs",
        reference_lengths={"ref_top": 2400},
        extra_uns={"modality": "conversion"},
    )
    source_paths = SimpleNamespace(
        raw_spine=source["spine"], preprocess_spine=None, pp_dedup=None, pp=None
    )
    source_cfg = SimpleNamespace(
        experiment_name="source",
        sample_name_col_for_plotting="Sample",
        smf_modality="conversion",
        trim=False,
    )
    monkeypatch.setattr(helpers, "load_experiment_config", lambda _path: source_cfg)
    monkeypatch.setattr(helpers, "get_adata_paths", lambda _cfg: source_paths)
    bundle = tmp_path / "bundle"
    export_fastq_for_experiment("source.csv", bundle, allow_unfiltered=True)

    fasta = tmp_path / "reference.fasta"
    fasta.write_text(f">ref\n{reference_sequence}\n", encoding="utf-8")
    config = tmp_path / "reingest.csv"
    values = {
        "smf_modality": "conversion",
        "alignment_mode": "align",
        "aligner": "minimap2",
        "align_from_bam": "False",
        "input_manifest_path": str(bundle / "bundle_manifest.json"),
        "fasta": str(fasta),
        "output_directory": str(tmp_path / "reingested"),
        "experiment_name": "reingested",
        "samtools_backend": "python",
        "skip_bam_split": "True",
        "skip_bam_qc": "True",
        "input_already_demuxed": "True",
        "make_beds": "False",
        "make_bigwigs": "False",
        "threads": "1",
        "max_memory_gb": "4",
    }
    with config.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variable", "value", "type"])
        writer.writeheader()
        for variable, value in values.items():
            value_type = (
                "bool"
                if value in {"True", "False"}
                else "int"
                if variable == "threads"
                else "float"
                if variable == "max_memory_gb"
                else "str"
            )
            writer.writerow({"variable": variable, "value": value, "type": value_type})

    monkeypatch.undo()
    spine, spine_path, _cfg = raw_adata(str(config))

    identity = pd.read_csv(bundle / "identity_map.csv").iloc[0]
    assert spine_path.is_file()
    assert spine.n_obs == 1
    assert spine.obs.iloc[0]["template_id"] == identity["bundle_template_id"]
    assert spine.obs.iloc[0]["Sample"] == "sample-one"
