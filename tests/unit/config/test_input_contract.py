from pathlib import Path

import pytest

from smftools.config import ExperimentConfig
from smftools.config.discover_input_files import discover_input_files


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


@pytest.mark.parametrize(
    ("filename", "kind"),
    [
        ("reads.pod5", "pod5"),
        ("reads.fast5", "fast5"),
        ("reads.fastq.gz", "fastq"),
        ("reads.bam", "bam"),
        ("reads.sam", "sam"),
        ("reads.cram", "cram"),
        ("reads.h5ad", "h5ad"),
    ],
)
def test_discovery_categorizes_recognized_single_files(tmp_path, filename, kind):
    input_path = _touch(tmp_path / filename)

    found = discover_input_files(input_path)

    assert found[f"{kind}_paths"] == [input_path.resolve()]
    assert found[f"input_is_{kind}"] is True


@pytest.mark.parametrize(
    ("filename", "expected_type", "expected_role"),
    [
        ("reads.pod5", "pod5", "raw_signal"),
        ("reads.fast5", "fast5", "raw_signal"),
        ("reads.fastq.gz", "fastq", "reads"),
        ("reads.bam", "bam", "reads"),
        ("reads.h5ad", "h5ad", None),
    ],
)
def test_supported_single_files_retain_input_type(tmp_path, filename, expected_type, expected_role):
    input_path = _touch(tmp_path / filename)

    config, _ = ExperimentConfig.from_var_dict(
        {"input_data_path": str(input_path)}, defaults_map={}
    )

    assert config.input_type == expected_type
    assert config.input_files == [input_path.resolve()]
    assert config.input_source_role == expected_role


@pytest.mark.parametrize("suffix", [".pod5", ".fast5", ".fastq", ".h5ad"])
def test_supported_homogeneous_directories_retain_input_type(tmp_path, suffix):
    input_dir = tmp_path / "inputs"
    _touch(input_dir / f"one{suffix}")
    _touch(input_dir / f"two{suffix}")

    config, _ = ExperimentConfig.from_var_dict(
        {"input_data_path": str(input_dir), "recursive_input_search": False},
        defaults_map={},
    )

    assert (
        config.input_type
        == {
            ".pod5": "pod5",
            ".fast5": "fast5",
            ".fastq": "fastq",
            ".h5ad": "h5ad",
        }[suffix]
    )
    assert len(config.input_files) == 2


@pytest.mark.parametrize("recursive", [False, True])
def test_mixed_directory_fails_with_deterministic_counts(tmp_path, recursive):
    input_dir = tmp_path / "inputs"
    _touch(input_dir / "reads.pod5")
    _touch(input_dir / "reads.fastq")
    _touch(input_dir / "reads.bam")

    with pytest.raises(
        ValueError,
        match=r"mixed recognized input types \(pod5=1, fastq=1, bam=1\)",
    ):
        ExperimentConfig.from_var_dict(
            {
                "input_data_path": str(input_dir),
                "recursive_input_search": recursive,
            },
            defaults_map={},
        )


def test_bam_directory_fails_before_output_directory_creation(tmp_path):
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "output"
    _touch(input_dir / "reads.bam")

    with pytest.raises(ValueError, match="BAM directory input is not supported"):
        ExperimentConfig.from_var_dict(
            {
                "input_data_path": str(input_dir),
                "output_directory": str(output_dir),
            },
            defaults_map={},
        )

    assert not output_dir.exists()


@pytest.mark.parametrize(("suffix", "label"), [(".sam", "SAM"), (".cram", "CRAM")])
def test_sam_and_cram_fail_with_current_support_guidance(tmp_path, suffix, label):
    input_path = _touch(tmp_path / f"reads{suffix}")

    with pytest.raises(ValueError, match=f"{label} input is not supported yet"):
        ExperimentConfig.from_var_dict({"input_data_path": str(input_path)}, defaults_map={})


def test_unknown_aligner_fails_during_config_loading():
    with pytest.raises(ValueError, match="aligner must be one of: dorado, minimap2"):
        ExperimentConfig.from_var_dict({"aligner": "bowtie2"}, defaults_map={})


def test_unknown_alignment_mode_fails_during_config_loading():
    with pytest.raises(ValueError, match="alignment_mode must be one of: align, existing"):
        ExperimentConfig.from_var_dict({"alignment_mode": "trust_existing"}, defaults_map={})


@pytest.mark.parametrize("alias", ["mm2", "minimap", "minimap-2"])
def test_minimap2_aliases_are_normalized(alias):
    config, _ = ExperimentConfig.from_var_dict({"aligner": alias}, defaults_map={})

    assert config.aligner == "minimap2"


def test_legacy_bam_defaults_to_align_mode(tmp_path):
    input_path = _touch(tmp_path / "reads.bam")

    config, _ = ExperimentConfig.from_var_dict(
        {"input_data_path": str(input_path)}, defaults_map={}
    )

    assert config.alignment_mode == "align"
    assert config.input_source_role == "reads"


def test_existing_alignment_mode_accepts_one_bam_as_alignment_input(tmp_path):
    input_path = _touch(tmp_path / "aligned.bam")

    config, _ = ExperimentConfig.from_var_dict(
        {
            "input_data_path": str(input_path),
            "alignment_mode": "existing",
        },
        defaults_map={},
    )

    assert config.alignment_mode == "existing"
    assert config.input_source_role == "alignment"


def test_existing_alignment_does_not_validate_the_external_aligner_name(tmp_path):
    input_path = _touch(tmp_path / "aligned.bam")

    config, _ = ExperimentConfig.from_var_dict(
        {
            "input_data_path": str(input_path),
            "alignment_mode": "existing",
            "aligner": "external-workflow",
        },
        defaults_map={},
    )

    assert config.aligner == "external-workflow"


def test_direct_modification_fastq_fails_as_signal_incapable(tmp_path):
    input_path = _touch(tmp_path / "reads.fastq")

    with pytest.raises(ValueError, match="FASTQ is sequence-only"):
        ExperimentConfig.from_var_dict(
            {
                "input_data_path": str(input_path),
                "smf_modality": "direct",
            },
            defaults_map={},
        )


def test_conversion_fastq_remains_supported(tmp_path):
    input_path = _touch(tmp_path / "reads.fastq")

    config, _ = ExperimentConfig.from_var_dict(
        {
            "input_data_path": str(input_path),
            "smf_modality": "conversion",
        },
        defaults_map={},
    )

    assert config.input_type == "fastq"
    assert config.input_source_role == "reads"


def test_input_manifest_path_resolves_relative_sources(tmp_path):
    input_path = _touch(tmp_path / "reads" / "sample.fastq")
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text("path\nreads/sample.fastq\n", encoding="utf-8")

    config, _ = ExperimentConfig.from_var_dict(
        {"input_manifest_path": str(manifest_path)}, defaults_map={}
    )

    assert config.input_manifest_path == manifest_path.resolve()
    assert config.input_files == [input_path.resolve()]
    assert config.input_type == "fastq"


def test_input_path_and_manifest_are_mutually_exclusive(tmp_path):
    input_path = _touch(tmp_path / "sample.fastq")
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text("path\nsample.fastq\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one"):
        ExperimentConfig.from_var_dict(
            {
                "input_data_path": str(input_path),
                "input_manifest_path": str(manifest_path),
            },
            defaults_map={},
        )
