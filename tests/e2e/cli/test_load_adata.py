"""Pytest-based end-to-end check for load_adata."""

from __future__ import annotations

import csv
import importlib
import importlib.resources as resources
import json
import random
import shutil
import stat
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


def _config_values(config_path: Path) -> dict[str, str]:
    """Read a test experiment config without loading or mutating anything."""
    with config_path.open(newline="", encoding="utf-8-sig") as handle:
        return {
            str(row["variable"]).strip(): str(row.get("value", "")).strip()
            for row in csv.DictReader(handle)
            if row.get("variable")
        }


def _skip_without_basecalling_prerequisites(config_path: Path) -> None:
    """Skip rather than fail when this config's external prerequisites are absent.

    A basecalling config needs the dorado executable, a populated model directory,
    and (for the direct modality) modkit. None of these are in the repository, and
    the model directory is gitignored, so a fresh checkout cannot satisfy them.
    Reporting that as a skip keeps an absent external tool an explicit deferment
    instead of a failure indistinguishable from a real regression.
    """
    values = _config_values(config_path)
    missing = []
    if shutil.which("dorado") is None:
        missing.append("the dorado executable")
    model_dir = values.get("model_dir", "")
    # Checked separately from the executables: a missing model directory is not
    # visible to a PATH lookup, so nothing else in the stack catches it early.
    if model_dir and not Path(model_dir).is_dir():
        missing.append(f"a dorado model directory at {model_dir}")
    if (
        values.get("smf_modality") == "direct"
        and values.get("direct_signal_backend", "modkit") == "modkit"
        and shutil.which("modkit") is None
    ):
        missing.append("the modkit executable")
    if missing:
        pytest.skip(f"{config_path.name} requires {', '.join(missing)}")


@pytest.mark.e2e
@pytest.mark.parametrize("config_path", CONFIGS, ids=lambda p: p.name)
def test_load_adata_e2e(config_path: Path):
    _skip_without_basecalling_prerequisites(config_path)
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
def test_existing_bam_source_append_publishes_new_complete_generation(tmp_path: Path):
    pysam = pytest.importorskip("pysam")
    from smftools.informatics.raw_generation import resolve_current_raw_generation

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
    fieldnames = ["path", "source_kind", "namespace", "sample", "barcode"]
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "path": partitions[0].name,
                "source_kind": "aligned_bam",
                "namespace": "lane-1",
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
        "experiment_name": "append-e2e",
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

    first_spine, _first_path, _cfg = raw_adata(str(config))
    first_generation, _first_manifest = resolve_current_raw_generation(
        tmp_path / "output" / "raw_outputs"
    )
    with manifest.open("a", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fieldnames).writerow(
            {
                "path": partitions[1].name,
                "source_kind": "aligned_bam",
                "namespace": "lane-2",
                "sample": "sample",
                "barcode": "sample",
            }
        )

    appended_spine, _appended_path, _cfg = raw_adata(str(config))
    second_generation, second_manifest = resolve_current_raw_generation(
        tmp_path / "output" / "raw_outputs"
    )

    assert first_spine.n_obs == 2
    assert appended_spine.n_obs == 4
    assert second_generation != first_generation
    assert first_generation.is_dir()
    assert second_manifest["source_transition"]["kind"] == "append_only"
    assert len(second_manifest["source_transition"]["added_source_ids"]) == 1
    assert second_manifest["reuse"]["reused_files"] > 0


def _write_experiment_config(path: Path, values: dict[str, str]) -> Path:
    """Write a minimal experiment config, inferring each value's declared type."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variable", "value", "type"])
        writer.writeheader()
        for variable, value in values.items():
            if value in {"True", "False"}:
                value_type = "bool"
            elif variable == "threads":
                value_type = "int"
            elif variable == "max_memory_gb":
                value_type = "float"
            else:
                value_type = "str"
            writer.writerow({"variable": variable, "value": value, "type": value_type})
    return path


@pytest.mark.e2e
@pytest.mark.parametrize(
    ("aligner", "required_tools"),
    [
        ("minimap2", ("minimap2",)),
        ("bwa-mem2", ("bwa-mem2",)),
        # bowtie2 builds its native index with a separate binary.
        ("bowtie2", ("bowtie2", "bowtie2-build")),
    ],
)
def test_every_supported_aligner_produces_a_complete_raw_generation(
    tmp_path: Path, aligner: str, required_tools: tuple[str, ...]
):
    """Each adapter must drive a full raw ingestion, not just an isolated call.

    The adapter-level integration tests exercise ``execute`` directly; this covers
    the configured end-to-end path so an aligner cannot be selectable in config
    yet unusable through the pipeline.
    """
    missing = [tool for tool in required_tools if shutil.which(tool) is None]
    if missing:
        pytest.skip(f"{aligner} requires {', '.join(missing)}")
    from smftools.informatics.raw_generation import resolve_current_raw_generation

    rng = random.Random(21)
    reference_sequence = "".join(rng.choices("ACGT", k=2600))
    fasta = tmp_path / "reference.fasta"
    fasta.write_text(f">ref\n{reference_sequence}\n", encoding="utf-8")
    reads = tmp_path / "reads.fastq"
    records = []
    for index in range(2):
        sequence = reference_sequence[400 + index * 700 : 1000 + index * 700]
        records.append(f"@read-{index + 1}\n{sequence}\n+\n{'I' * len(sequence)}\n")
    reads.write_text("".join(records), encoding="utf-8")

    manifest = tmp_path / "inputs.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "sample", "barcode"])
        writer.writeheader()
        writer.writerow({"path": reads.name, "sample": "sample", "barcode": "sample"})

    config = _write_experiment_config(
        tmp_path / "config.csv",
        {
            "smf_modality": "conversion",
            "alignment_mode": "align",
            "aligner": aligner,
            "align_from_bam": "False",
            "input_manifest_path": str(manifest),
            "fasta": str(fasta),
            "output_directory": str(tmp_path / "output"),
            "experiment_name": f"aligner-{aligner}",
            "samtools_backend": "python",
            "skip_bam_split": "True",
            "skip_bam_qc": "True",
            "input_already_demuxed": "True",
            "make_beds": "False",
            "make_bigwigs": "False",
            "threads": "1",
            "max_memory_gb": "4",
        },
    )

    spine, spine_path, _cfg = raw_adata(str(config))
    generation, generation_manifest = resolve_current_raw_generation(
        tmp_path / "output" / "raw_outputs"
    )

    assert spine_path.is_file()
    assert spine.n_obs == 2
    assert spine.obs_names.is_unique
    assert generation.is_dir()
    assert generation_manifest["generation_id"]

    # The generation pins its alignment provenance as a checksummed dependency
    # rather than copying it in, so follow that record to the adapter that really
    # ran. A configured-but-unused aligner would not appear here.
    dependency = generation_manifest["dependencies"]["sidecar:alignment_manifest"]
    assert dependency["anchor"] == "run_root"
    alignment_manifest_path = tmp_path / "output" / dependency["path"]
    assert alignment_manifest_path.is_file()
    alignment_manifest = json.loads(alignment_manifest_path.read_text(encoding="utf-8"))
    assert alignment_manifest["adapter"]["name"] == aligner
    assert alignment_manifest["adapter"]["normalized_argv"][0] == aligner
    if aligner != "minimap2":
        # Short-read adapters build a native index; record which one produced it.
        assert alignment_manifest["adapter"]["reference_index"]["adapter"] == aligner


_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def _reverse_complement(sequence: str) -> str:
    return sequence.translate(_COMPLEMENT)[::-1]


@pytest.mark.e2e
def test_fastq_directory_input_discovers_and_ingests_every_source(tmp_path: Path):
    """A directory of FASTQs is one homogeneous source set, not one file.

    Covers the discovery path (``input_data_path`` pointing at a directory)
    rather than an explicit manifest, and asserts the mixed-type guard still
    rejects a directory holding more than one recognized input kind.
    """
    if shutil.which("minimap2") is None:
        pytest.skip("minimap2 is required for the FASTQ directory round trip")
    from smftools.informatics.raw_generation import resolve_current_raw_generation

    rng = random.Random(31)
    reference_sequence = "".join(rng.choices("ACGT", k=2600))
    fasta = tmp_path / "reference.fasta"
    fasta.write_text(f">ref\n{reference_sequence}\n", encoding="utf-8")
    reads_dir = tmp_path / "reads"
    reads_dir.mkdir()
    # Names deliberately without a trailing mate token so auto-pairing leaves
    # them as two independent unpaired sources.
    for index, stem in enumerate(("alpha", "beta")):
        sequence = reference_sequence[400 + index * 700 : 1000 + index * 700]
        (reads_dir / f"{stem}.fastq").write_text(
            f"@{stem}-read\n{sequence}\n+\n{'I' * len(sequence)}\n", encoding="utf-8"
        )

    base_values = {
        "smf_modality": "conversion",
        "alignment_mode": "align",
        "aligner": "minimap2",
        "align_from_bam": "False",
        "input_data_path": str(reads_dir),
        "fasta": str(fasta),
        "output_directory": str(tmp_path / "output"),
        "experiment_name": "fastq-directory",
        "samtools_backend": "python",
        "skip_bam_split": "True",
        "skip_bam_qc": "True",
        "input_already_demuxed": "True",
        "make_beds": "False",
        "make_bigwigs": "False",
        "threads": "1",
        "max_memory_gb": "4",
    }
    config = _write_experiment_config(tmp_path / "config.csv", base_values)

    spine, _path, _cfg = raw_adata(str(config))
    generation, _manifest = resolve_current_raw_generation(tmp_path / "output" / "raw_outputs")

    assert spine.n_obs == 2
    assert spine.obs_names.is_unique
    assert generation.is_dir()

    # A directory mixing recognized input kinds must fail before any execution.
    (reads_dir / "stray.bam").write_bytes(b"not-a-real-bam")
    mixed_config = _write_experiment_config(
        tmp_path / "mixed.csv",
        {**base_values, "output_directory": str(tmp_path / "mixed-output")},
    )
    with pytest.raises(ValueError, match="mixed recognized input types"):
        raw_adata(str(mixed_config))


@pytest.mark.e2e
def test_paired_illumina_overlap_becomes_one_consensus_molecule(tmp_path: Path):
    """One paired template must become one molecule with two segments.

    This is the IAR-08/IAR-10 contract end to end: mate layout survives
    alignment, both mates keep distinct segment identity, and the overlap is
    reconciled into a single molecule rather than two independent reads.
    """
    missing = [tool for tool in ("bwa-mem2",) if shutil.which(tool) is None]
    if missing:
        pytest.skip(f"paired alignment requires {', '.join(missing)}")
    from smftools.informatics.raw_generation import resolve_current_raw_generation

    rng = random.Random(37)
    reference_sequence = "".join(rng.choices("ACGT", k=2600))
    fasta = tmp_path / "reference.fasta"
    fasta.write_text(f">ref\n{reference_sequence}\n", encoding="utf-8")
    # Mates overlap across reference[700:900]; the insert spans [400, 1200).
    forward = reference_sequence[400:900]
    reverse = _reverse_complement(reference_sequence[700:1200])
    reads_dir = tmp_path / "reads"
    reads_dir.mkdir()
    for name, sequence in (("sample_R1.fastq", forward), ("sample_R2.fastq", reverse)):
        (reads_dir / name).write_text(
            f"@pair-one\n{sequence}\n+\n{'I' * len(sequence)}\n", encoding="utf-8"
        )

    config = _write_experiment_config(
        tmp_path / "config.csv",
        {
            "smf_modality": "conversion",
            "alignment_mode": "align",
            "aligner": "bwa-mem2",
            "align_from_bam": "False",
            "input_data_path": str(reads_dir),
            "fastq_auto_pairing": "True",
            "fasta": str(fasta),
            "output_directory": str(tmp_path / "output"),
            "experiment_name": "paired-illumina",
            "samtools_backend": "python",
            "skip_bam_split": "True",
            "skip_bam_qc": "True",
            "input_already_demuxed": "True",
            "make_beds": "False",
            "make_bigwigs": "False",
            "threads": "1",
            "max_memory_gb": "4",
        },
    )

    spine, _path, _cfg = raw_adata(str(config))
    generation, _manifest = resolve_current_raw_generation(tmp_path / "output" / "raw_outputs")
    segments = pd.read_parquet(generation / "segments.parquet")

    # One template collapses to one molecule, still backed by two segments.
    assert spine.n_obs == 1
    assert len(segments) == 2
    assert segments["segment_uid"].is_unique
    assert segments["molecule_uid"].nunique() == 1
    assert int(spine.obs.iloc[0]["segment_count"]) == 2


@pytest.mark.e2e
def test_fastq_source_append_aligns_only_new_source(tmp_path: Path):
    if shutil.which("minimap2") is None:
        pytest.skip("minimap2 is required for the FASTQ append round trip")
    from smftools.informatics.raw_generation import resolve_current_raw_generation

    rng = random.Random(14)
    reference_sequence = "".join(rng.choices("ACGT", k=2600))
    fasta = tmp_path / "reference.fasta"
    fasta.write_text(f">ref\n{reference_sequence}\n", encoding="utf-8")
    fastqs = [tmp_path / "first.fastq", tmp_path / "second.fastq"]
    for index, path in enumerate(fastqs):
        sequence = reference_sequence[400 + index * 700 : 1000 + index * 700]
        path.write_text(
            f"@read-{index + 1}\n{sequence}\n+\n{'I' * len(sequence)}\n",
            encoding="utf-8",
        )

    manifest = tmp_path / "inputs.csv"
    fieldnames = ["path", "namespace", "sample", "barcode"]
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "path": fastqs[0].name,
                "namespace": "lane-1",
                "sample": "sample",
                "barcode": "sample",
            }
        )

    config = tmp_path / "config.csv"
    values = {
        "smf_modality": "conversion",
        "alignment_mode": "align",
        "aligner": "minimap2",
        "align_from_bam": "False",
        "input_manifest_path": str(manifest),
        "fasta": str(fasta),
        "output_directory": str(tmp_path / "output"),
        "experiment_name": "fastq-append-e2e",
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

    first_spine, _path, _cfg = raw_adata(str(config))
    first_generation, _manifest = resolve_current_raw_generation(
        tmp_path / "output" / "raw_outputs"
    )
    with manifest.open("a", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fieldnames).writerow(
            {
                "path": fastqs[1].name,
                "namespace": "lane-2",
                "sample": "sample",
                "barcode": "sample",
            }
        )

    appended_spine, _path, _cfg = raw_adata(str(config))
    second_generation, second_manifest = resolve_current_raw_generation(
        tmp_path / "output" / "raw_outputs"
    )

    assert first_spine.n_obs == 1
    assert appended_spine.n_obs == 2
    assert first_generation.is_dir() and second_generation != first_generation
    assert second_manifest["source_transition"]["kind"] == "append_only"
    assert second_manifest["reuse"]["reused_files"] > 0


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


@pytest.mark.e2e
def test_lossless_bam_bundle_reingests_with_modification_capability_intact(tmp_path: Path):
    """The two bundle kinds must differ in what a re-ingestion can still do.

    ``sequence_only`` and ``lossless_bam`` are both re-ingestible, so a round trip
    that only checks "it loads" cannot tell them apart. What separates them is
    capability: the BAM bundle carries the owned alignment forward, so re-ingesting
    it declares alignment-grade sources and reproduces the source coordinates
    without an aligner ever running; the FASTQ bundle declares those capabilities
    lost and must realign from sequence alone.
    """
    if shutil.which("minimap2") is None:
        pytest.skip("minimap2 is required to produce the owned alignment being bundled")
    pytest.importorskip("pysam")
    from smftools.cli.export_bundle import export_bundle_for_experiment
    from smftools.informatics.export_bundle import read_bundle_manifest
    from smftools.informatics.input_manifest import input_manifest_artifact_paths
    from smftools.informatics.raw_generation import resolve_current_raw_generation

    rng = random.Random(37)
    reference_sequence = "".join(rng.choices("ACGT", k=2600))
    fasta = tmp_path / "reference.fasta"
    fasta.write_text(f">ref\n{reference_sequence}\n", encoding="utf-8")
    reads = tmp_path / "reads.fastq"
    reads.write_text(
        "".join(
            f"@read-{index + 1}\n{reference_sequence[400 + index * 600 : 1000 + index * 600]}\n"
            f"+\n{'I' * 600}\n"
            for index in range(3)
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "inputs.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "sample", "barcode"])
        writer.writeheader()
        writer.writerow({"path": reads.name, "sample": "sample-one", "barcode": "bc01"})

    source_config = _write_experiment_config(
        tmp_path / "source.csv",
        {
            "smf_modality": "conversion",
            "alignment_mode": "align",
            "aligner": "minimap2",
            "align_from_bam": "False",
            "input_manifest_path": str(manifest),
            "fasta": str(fasta),
            "output_directory": str(tmp_path / "source"),
            "experiment_name": "bundle-source",
            "samtools_backend": "python",
            "skip_bam_split": "True",
            "skip_bam_qc": "True",
            "input_already_demuxed": "True",
            "make_beds": "False",
            "make_bigwigs": "False",
            "threads": "1",
            "max_memory_gb": "4",
        },
    )
    source_spine, _source_path, _cfg = raw_adata(str(source_config))
    assert source_spine.n_obs == 3

    lossless_dir = tmp_path / "lossless-bundle"
    sequence_dir = tmp_path / "sequence-bundle"
    export_bundle_for_experiment(
        str(source_config), lossless_dir, bundle_format="bam", allow_unfiltered=True
    )
    export_bundle_for_experiment(
        str(source_config),
        sequence_dir,
        bundle_format="fastq",
        allow_unfiltered=True,
        gzip_output=False,
    )

    lossless = read_bundle_manifest(lossless_dir)
    sequence_only = read_bundle_manifest(sequence_dir)
    assert lossless["bundle_kind"] == "lossless_bam"
    assert lossless["lost_capabilities"] == []
    assert sequence_only["bundle_kind"] == "sequence_only"
    assert "alignment" in sequence_only["lost_capabilities"]
    assert "mm_ml" in sequence_only["lost_capabilities"]
    # Both kinds must name the generation they came from; a bundle whose origin is
    # unrecorded cannot be audited back to the experiment that produced it.
    assert {str(item["raw_generation_id"]) for item in lossless["source_generations"]} == {
        str(source_spine.uns["raw_generation_id"])
    }

    declarations = pd.read_csv(lossless_dir / "inputs.csv")
    assert set(declarations["source_kind"]) == {"aligned_bam"}
    assert set(declarations["source_role"]) == {"alignment"}
    assert set(declarations["modification_capability"]) == {"conversion_sequence"}
    assert set(pd.read_csv(sequence_dir / "inputs.csv")["modification_capability"]) == {
        "sequence_only"
    }

    reingest_output = tmp_path / "reingested"
    reingest_config = _write_experiment_config(
        tmp_path / "reingest.csv",
        {
            "smf_modality": "conversion",
            "alignment_mode": "existing",
            "input_manifest_path": str(lossless_dir / "bundle_manifest.json"),
            "fasta": str(fasta),
            "output_directory": str(reingest_output),
            "experiment_name": "bundle-reingested",
            "samtools_backend": "python",
            "skip_bam_split": "True",
            "skip_bam_qc": "True",
            "input_already_demuxed": "True",
            "make_beds": "False",
            "make_bigwigs": "False",
            "threads": "1",
            "max_memory_gb": "4",
        },
    )
    spine, spine_path, _cfg = raw_adata(str(reingest_config))
    generation, generation_manifest = resolve_current_raw_generation(
        reingest_output / "raw_outputs"
    )

    assert spine_path.is_file()
    assert generation.is_dir()
    assert generation_manifest["generation_id"]
    assert spine.n_obs == source_spine.n_obs

    # The re-ingested run resolves its own source identity from the bundle. That
    # record -- not the bundle's own declaration -- is what proves the capability
    # survived the round trip into a fresh experiment.
    resolved = json.loads(
        input_manifest_artifact_paths(reingest_output)["input_manifest_json"].read_text(
            encoding="utf-8"
        )
    )
    assert {row["modification_capability"] for row in resolved["sources"]} == {
        "conversion_sequence"
    }
    assert {row["source_role"] for row in resolved["sources"]} == {"alignment"}

    # No aligner ran here, so the coordinates must be the ones the source
    # generation already established rather than a fresh alignment that happens to
    # agree. The re-ingested alignment manifest proves that: it declares no adapter
    # of its own and still carries the source run's minimap2 @PG record.
    dependency = generation_manifest["dependencies"]["sidecar:alignment_manifest"]
    alignment_manifest = json.loads(
        (reingest_output / dependency["path"]).read_text(encoding="utf-8")
    )
    assert alignment_manifest["alignment_mode"] == "existing"
    assert "adapter" not in alignment_manifest
    normalized = alignment_manifest["validation"]["normalized"]
    assert normalized["external_aligner"] == "minimap2"
    assert normalized["mapped_primary_records"] == source_spine.n_obs

    identity = pd.read_csv(lossless_dir / "identity_map.csv")
    source_by_read = source_spine.obs.set_index(source_spine.obs["template_id"].astype(str))
    reingested_by_read = spine.obs.set_index(spine.obs["template_id"].astype(str))
    assert set(reingested_by_read.index) == set(identity["source_read_id"].astype(str))
    for read_id in reingested_by_read.index:
        original = source_by_read.loc[read_id]
        replayed = reingested_by_read.loc[read_id]
        assert int(replayed["reference_start"]) == int(original["reference_start"])
        assert int(replayed["mapped_length"]) == int(original["mapped_length"])
        assert str(replayed["Reference_strand"]) == str(original["Reference_strand"])


def _restore_writable(root: Path) -> None:
    for path in [root, *root.rglob("*")]:
        path.chmod(path.stat().st_mode | stat.S_IWUSR | (stat.S_IXUSR if path.is_dir() else 0))


@pytest.mark.e2e
def test_owned_output_validates_after_relocation_under_a_foreign_container_uid(tmp_path: Path):
    """A completed run must validate from a new path, read-only, and unowned.

    Container tasks stage outputs somewhere, then hand the directory to another
    step that mounts it elsewhere, often read-only and under an arbitrary UID that
    owns none of the files. Validation therefore may not depend on the original
    absolute location, on write access, or on ownership -- and immutability is
    detect-not-prevent, so the published tree stays writable rather than being
    chmod'd read-only, which is what makes an arbitrary UID workable at all.
    """
    if shutil.which("minimap2") is None:
        pytest.skip("minimap2 is required to produce an owned alignment to relocate")
    from smftools.cli.workflow_contract import run_experiment_workflow, validate_workflow_output
    from smftools.informatics.raw_generation import (
        RawGenerationError,
        resolve_current_raw_generation,
    )

    rng = random.Random(53)
    reference_sequence = "".join(rng.choices("ACGT", k=2600))
    fasta = tmp_path / "reference.fasta"
    fasta.write_text(f">ref\n{reference_sequence}\n", encoding="utf-8")
    reads = tmp_path / "reads.fastq"
    reads.write_text(
        "".join(
            f"@read-{index + 1}\n{reference_sequence[400 + index * 600 : 1000 + index * 600]}\n"
            f"+\n{'I' * 600}\n"
            for index in range(2)
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "inputs.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "sample", "barcode"])
        writer.writeheader()
        writer.writerow({"path": reads.name, "sample": "sample-one", "barcode": "bc01"})

    origin = tmp_path / "origin"
    config = _write_experiment_config(
        tmp_path / "config.csv",
        {
            "smf_modality": "conversion",
            "alignment_mode": "align",
            "aligner": "minimap2",
            "align_from_bam": "False",
            "input_manifest_path": str(manifest),
            "fasta": str(fasta),
            "output_directory": str(origin),
            "experiment_name": "relocation-e2e",
            "samtools_backend": "python",
            "skip_bam_split": "True",
            "skip_bam_qc": "True",
            "input_already_demuxed": "True",
            "make_beds": "False",
            "make_bigwigs": "False",
            "threads": "1",
            "max_memory_gb": "4",
        },
    )
    result_path = run_experiment_workflow(str(config), target="raw", output_root=origin)
    assert validate_workflow_output(origin)["valid"]

    generation, generation_manifest = resolve_current_raw_generation(origin / "raw_outputs")
    # Immutability is enforced by checksum, not by permissions: an owned artifact
    # that lost its write bit would be unmanageable for a UID that does not own it.
    published = [path for path in generation.rglob("*") if path.is_file()]
    assert published
    assert all(path.stat().st_mode & stat.S_IWUSR for path in published)
    # Nothing may pin the original absolute location, or the move below breaks.
    assert str(origin) not in (generation / "generation_manifest.json").read_text(encoding="utf-8")
    assert str(origin) not in result_path.read_text(encoding="utf-8")

    relocated = tmp_path / "consumer" / "mounted" / "run"
    relocated.parent.mkdir(parents=True)
    shutil.move(str(origin), str(relocated))

    moved_generation, moved_manifest = resolve_current_raw_generation(relocated / "raw_outputs")
    assert moved_manifest["generation_id"] == generation_manifest["generation_id"]
    assert moved_generation == relocated / generation.relative_to(origin)
    assert validate_workflow_output(relocated)["valid"]

    # A read-only mount is the container case that write access would silently
    # pass: validation must complete without needing to write anywhere.
    try:
        for path in sorted(relocated.rglob("*"), reverse=True):
            path.chmod(0o555 if path.is_dir() else 0o444)
        relocated.chmod(0o555)
        assert validate_workflow_output(relocated)["valid"]
        read_only_generation, read_only_manifest = resolve_current_raw_generation(
            relocated / "raw_outputs"
        )
        assert read_only_generation == moved_generation
        assert read_only_manifest["generation_id"] == moved_manifest["generation_id"]
    finally:
        _restore_writable(relocated)

    # Detect-not-prevent: nothing stopped the edit, so validation must catch it.
    molecules = moved_generation / "molecules.parquet"
    molecules.write_bytes(molecules.read_bytes() + b"\0")
    with pytest.raises(RawGenerationError, match="missing or corrupt"):
        resolve_current_raw_generation(relocated / "raw_outputs")
