"""A run directory with several read representations is a valid input (`BCS-01`).

It used to be refused outright -- "input_data_path contains mixed recognized
input types" -- so the practice was to point at one subdirectory by hand and
record the reason in a config comment.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from smftools.config import ExperimentConfig, LoadExperimentConfig

pytestmark = pytest.mark.unit

HAC_5 = "dna_r10.4.1_e8.2_400bps_hac@v5.0.0"


def _run_directory(tmp_path: Path, *, model: str = HAC_5) -> Path:
    """A run root shaped like MinKNOW's: signal, passing reads, failing reads."""
    root = tmp_path / "run"
    (root / "pod5").mkdir(parents=True)
    (root / "pod5" / "signal.pod5").write_bytes(b"")
    for tree, count in (("fastq_pass", 2), ("fastq_fail", 3)):
        for index in range(count):
            path = root / tree / f"reads_{index}.fastq.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(path, "wt") as handle:
                handle.write(f"@r{index} basecall_model_version_id={model}\nACGT\n+\nIIII\n")
    return root


def _config(tmp_path: Path, input_path: Path, *, model: str = "hac", modality="deaminase") -> Path:
    fasta = tmp_path / "ref.fasta"
    fasta.write_text(">ref\nACGT\n", encoding="utf-8")
    config = tmp_path / "experiment_config.csv"
    config.write_text(
        "variable,value\n"
        f"smf_modality,{modality}\n"
        f"input_data_path,{input_path}\n"
        f"model,{model}\n"
        f"fasta,{fasta}\n"
        f"output_directory,{tmp_path / 'store'}\n"
        "experiment_id,probe\n",
        encoding="utf-8",
    )
    return config


def _load(config_path: Path) -> ExperimentConfig:
    cfg, _ = ExperimentConfig.from_var_dict(
        LoadExperimentConfig(config_path).var_dict, date_str="260101"
    )
    return cfg


def test_run_root_resolves_to_the_passing_reads(tmp_path):
    cfg = _load(_config(tmp_path, _run_directory(tmp_path)))
    assert cfg.input_type == "fastq"
    assert len(cfg.input_files) == 2
    assert all("fastq_pass" in str(path) for path in cfg.input_files)


def test_unmatched_model_falls_through_to_the_signal(tmp_path):
    """No derivative for the requested model, but POD5 is there to basecall."""
    cfg = _load(_config(tmp_path, _run_directory(tmp_path), model="sup"))
    assert cfg.input_type == "pod5"
    assert len(cfg.input_files) == 1


def test_direct_modality_ignores_a_canonical_fastq_and_takes_the_signal(tmp_path):
    cfg = _load(_config(tmp_path, _run_directory(tmp_path), modality="direct"))
    assert cfg.input_type == "pod5"


def test_homogeneous_directory_is_unchanged(tmp_path):
    """Selection must not disturb the single-type case that always worked."""
    root = tmp_path / "reads"
    root.mkdir()
    for index in range(2):
        with gzip.open(root / f"r{index}.fastq.gz", "wt") as handle:
            handle.write(f"@r{index}\nACGT\n+\nIIII\n")
    cfg = _load(_config(tmp_path, root))
    assert cfg.input_type == "fastq"
    assert len(cfg.input_files) == 2


def test_no_qualifying_source_and_no_signal_explains_itself(tmp_path):
    """The refusal names what was found and which rule each candidate failed."""
    root = tmp_path / "reads"
    root.mkdir()
    with gzip.open(root / "a.fastq.gz", "wt") as handle:
        handle.write(f"@r0 basecall_model_version_id={HAC_5}\nACGT\n+\nIIII\n")
    (root / "b.bam").write_bytes(b"")
    with pytest.raises(ValueError) as excinfo:
        _load(_config(tmp_path, root, model="sup"))
    message = str(excinfo.value)
    assert "sup" in message
    assert "a.fastq.gz" in message
