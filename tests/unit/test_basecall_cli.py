"""`smftools basecall` (`BCS-05`): publish a basecall generation from signal."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from smftools.cli.basecall import (
    BasecallInputError,
    BasecallMismatchError,
    basecall,
    basecall_core,
    run_from_paths,
)
from smftools.cli.helpers import load_experiment_config
from smftools.cli_entry import cli
from smftools.constants import BASECALL_DIR
from smftools.informatics import basecall_execution as be

pytestmark = pytest.mark.unit


def _config(tmp_path: Path, *, input_path: Path | None = None, model: str = "hac") -> Path:
    pod5_dir = input_path or (tmp_path / "pod5")
    pod5_dir.mkdir(parents=True, exist_ok=True)
    (pod5_dir / "signal.pod5").write_bytes(b"fake-pod5-bytes")
    fasta = tmp_path / "ref.fasta"
    fasta.write_text(">ref\nACGT\n", encoding="utf-8")
    model_dir = tmp_path / "models"
    model_dir.mkdir(exist_ok=True)
    config = tmp_path / "experiment_config.csv"
    config.write_text(
        "variable,value\n"
        "smf_modality,deaminase\n"
        f"input_data_path,{pod5_dir}\n"
        f"model,{model}\n"
        f"model_dir,{model_dir}\n"
        f"fasta,{fasta}\n"
        f"output_directory,{tmp_path / 'store'}\n"
        "experiment_id,probe\n",
        encoding="utf-8",
    )
    return config


@pytest.fixture(autouse=True)
def fake_dorado(monkeypatch):
    """Never invoke real dorado; write a plausible fake BAM instead."""

    def fake_canoncall(model_dir, model, input_path, kit, out_prefix, suffix, *rest):
        Path(out_prefix + suffix).write_bytes(b"fake-bam-bytes")

    def fake_modcall(model_dir, model, input_path, kit, mods, out_prefix, suffix, *rest):
        Path(out_prefix + suffix).write_bytes(b"fake-bam-bytes")

    monkeypatch.setattr(be, "canoncall", fake_canoncall)
    monkeypatch.setattr(be, "modcall", fake_modcall)
    # Skip the real memory-headroom preflight; irrelevant to this behavior and
    # depends on machine resources the test should not care about.
    monkeypatch.setattr(
        "smftools.memory_guard.require_memory_headroom", lambda *args, **kwargs: None
    )


def test_basecall_core_publishes_a_generation(tmp_path: Path) -> None:
    cfg = load_experiment_config(_config(tmp_path))

    result = basecall_core(cfg)

    assert result["reused_generation"] is False
    generation_dir = (
        Path(cfg.output_directory) / BASECALL_DIR / "generations" / result["generation_id"]
    )
    assert generation_dir.is_dir()
    manifest = json.loads((generation_dir / "generation_manifest.json").read_text())
    assert manifest["model"] == "dna_r10.4.1_e8.2_400bps_hac@v5.0.0" or manifest["model"] == "hac"
    input_ids = manifest["input_artifact_ids"]
    assert input_ids[0].startswith("input-manifest:")
    assert all(entry.startswith("source:") for entry in input_ids[1:])
    assert not any(entry.startswith("alignment-reference-bundle:") for entry in input_ids)


def test_basecall_core_is_idempotent_on_a_rerun(tmp_path: Path) -> None:
    config_path = _config(tmp_path)
    cfg = load_experiment_config(config_path)
    first = basecall_core(cfg)

    cfg_again = load_experiment_config(config_path)
    second = basecall_core(cfg_again)

    assert second["reused_generation"] is True
    assert second["generation_id"] == first["generation_id"]


def test_basecall_core_refuses_non_signal_input(tmp_path: Path) -> None:
    bam_dir = tmp_path / "bam_input"
    bam_dir.mkdir()
    (bam_dir / "reads.bam").write_bytes(b"not-a-real-bam-but-present")
    config = tmp_path / "experiment_config.csv"
    fasta = tmp_path / "ref.fasta"
    fasta.write_text(">ref\nACGT\n", encoding="utf-8")
    config.write_text(
        "variable,value\n"
        "smf_modality,deaminase\n"
        f"input_data_path,{bam_dir / 'reads.bam'}\n"
        f"fasta,{fasta}\n"
        f"output_directory,{tmp_path / 'store'}\n"
        "experiment_id,probe\n",
        encoding="utf-8",
    )
    cfg = load_experiment_config(config)

    with pytest.raises(BasecallInputError, match="POD5 or FAST5"):
        basecall_core(cfg)


def test_basecall_function_loads_the_config_itself(tmp_path: Path) -> None:
    result = basecall(str(_config(tmp_path)))

    assert result["reused_generation"] is False


def test_basecall_cli_publishes_and_reports(tmp_path: Path) -> None:
    config_path = _config(tmp_path)
    runner = CliRunner()

    result = runner.invoke(cli, ["basecall", str(config_path)])

    assert result.exit_code == 0, result.output
    assert "Published basecall generation" in result.output


def test_basecall_cli_reports_already_current_on_rerun(tmp_path: Path) -> None:
    config_path = _config(tmp_path)
    runner = CliRunner()
    runner.invoke(cli, ["basecall", str(config_path)])

    result = runner.invoke(cli, ["basecall", str(config_path)])

    assert result.exit_code == 0, result.output
    assert "already current" in result.output


def test_basecall_cli_rejects_non_signal_input_with_a_clean_error(tmp_path: Path) -> None:
    bam_dir = tmp_path / "bam_input"
    bam_dir.mkdir()
    (bam_dir / "reads.bam").write_bytes(b"not-a-real-bam-but-present")
    config = tmp_path / "experiment_config.csv"
    fasta = tmp_path / "ref.fasta"
    fasta.write_text(">ref\nACGT\n", encoding="utf-8")
    config.write_text(
        "variable,value\n"
        "smf_modality,deaminase\n"
        f"input_data_path,{bam_dir / 'reads.bam'}\n"
        f"fasta,{fasta}\n"
        f"output_directory,{tmp_path / 'store'}\n"
        "experiment_id,probe\n",
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(cli, ["basecall", str(config)])

    assert result.exit_code != 0
    assert "POD5 or FAST5" in result.output


def _bare_pod5_dir(tmp_path: Path, name: str = "pod5") -> Path:
    pod5_dir = tmp_path / name
    pod5_dir.mkdir(parents=True, exist_ok=True)
    (pod5_dir / "signal.pod5").write_bytes(b"fake-pod5-bytes")
    return pod5_dir


def test_run_from_paths_publishes_a_generation_with_no_config(tmp_path: Path) -> None:
    pod5_dir = _bare_pod5_dir(tmp_path)
    output_dir = tmp_path / "store"

    result = run_from_paths(
        input_path=pod5_dir,
        output_directory=output_dir,
        model="hac",
        model_dir=tmp_path / "models",
    )

    assert result["reused_generation"] is False
    generation_dir = output_dir / BASECALL_DIR / "generations" / result["generation_id"]
    assert generation_dir.is_dir()
    manifest = json.loads((generation_dir / "generation_manifest.json").read_text())
    assert manifest["input_artifact_ids"][0].startswith("input-manifest:")


def test_run_from_paths_is_idempotent_on_a_rerun(tmp_path: Path) -> None:
    pod5_dir = _bare_pod5_dir(tmp_path)
    output_dir = tmp_path / "store"
    kwargs = dict(
        input_path=pod5_dir, output_directory=output_dir, model="hac", model_dir=tmp_path / "models"
    )

    first = run_from_paths(**kwargs)
    second = run_from_paths(**kwargs)

    assert second["reused_generation"] is True
    assert second["generation_id"] == first["generation_id"]


def test_run_from_paths_modifications_selects_modified_basecalling(
    tmp_path: Path, monkeypatch
) -> None:
    calls = []
    monkeypatch.setattr(be, "canoncall", lambda *a, **k: (_ for _ in ()).throw(AssertionError))

    def fake_modcall(model_dir, model, input_path, kit, mods, out_prefix, suffix, *rest):
        calls.append(mods)
        Path(out_prefix + suffix).write_bytes(b"fake-bam-bytes")

    monkeypatch.setattr(be, "modcall", fake_modcall)
    pod5_dir = _bare_pod5_dir(tmp_path)

    run_from_paths(
        input_path=pod5_dir,
        output_directory=tmp_path / "store",
        model="hac",
        model_dir=tmp_path / "models",
        modifications="5mC_5hmC",
    )

    assert calls == [["5mC_5hmC"]]


def test_run_from_paths_refuses_a_directory_with_no_signal(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(BasecallInputError, match="no POD5 or FAST5"):
        run_from_paths(
            input_path=empty_dir,
            output_directory=tmp_path / "store",
            model="hac",
            model_dir=tmp_path / "models",
        )


def test_basecall_cli_config_free_form_publishes(tmp_path: Path) -> None:
    pod5_dir = _bare_pod5_dir(tmp_path)
    output_dir = tmp_path / "store"
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "basecall",
            "--input",
            str(pod5_dir),
            "--output",
            str(output_dir),
            "--model",
            "hac",
            "--model-dir",
            str(tmp_path / "models"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Published basecall generation" in result.output
    assert (output_dir / BASECALL_DIR / "current.json").is_file()


def test_basecall_cli_rejects_config_path_and_config_free_flags_together(tmp_path: Path) -> None:
    config_path = _config(tmp_path)
    runner = CliRunner()

    result = runner.invoke(cli, ["basecall", str(config_path), "--input", str(tmp_path / "pod5")])

    assert result.exit_code != 0
    assert "not both" in result.output


def test_basecall_cli_rejects_partial_config_free_flags(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(cli, ["basecall", "--input", str(_bare_pod5_dir(tmp_path))])

    assert result.exit_code != 0
    assert "CONFIG_PATH" in result.output


def test_basecall_core_reuses_when_a_pod5_is_pruned_after_basecalling(tmp_path: Path) -> None:
    pod5_dir = tmp_path / "pod5"
    config_path = _config(tmp_path, input_path=pod5_dir)
    (pod5_dir / "extra.pod5").write_bytes(b"fake-pod5-extra")
    first = basecall_core(load_experiment_config(config_path))

    (pod5_dir / "extra.pod5").unlink()

    second = basecall_core(load_experiment_config(config_path))

    assert second["reused_generation"] is True
    assert second["generation_id"] == first["generation_id"]


def test_basecall_core_refuses_when_a_new_pod5_appears_without_a_recorded_subsample(
    tmp_path: Path,
) -> None:
    pod5_dir = tmp_path / "pod5"
    config_path = _config(tmp_path, input_path=pod5_dir)
    basecall_core(load_experiment_config(config_path))

    (pod5_dir / "extra.pod5").write_bytes(b"fake-pod5-extra")

    with pytest.raises(BasecallMismatchError, match="fewer sources"):
        basecall_core(load_experiment_config(config_path))


def test_basecall_core_refuses_a_disjoint_source_set(tmp_path: Path) -> None:
    pod5_dir = tmp_path / "pod5"
    config_path = _config(tmp_path, input_path=pod5_dir)
    basecall_core(load_experiment_config(config_path))

    (pod5_dir / "signal.pod5").unlink()
    (pod5_dir / "different.pod5").write_bytes(b"totally-different-signal")

    with pytest.raises(BasecallMismatchError, match="do not match"):
        basecall_core(load_experiment_config(config_path))


def test_basecall_core_reuses_a_deliberate_subsample_despite_new_sources(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "smftools.informatics.pod5_functions.subsample_pod5_for_basecalling",
        lambda input_path, max_reads, output_dir, seed=42: input_path,
    )
    pod5_dir = tmp_path / "pod5"
    config_path = _config(tmp_path, input_path=pod5_dir)
    cfg = load_experiment_config(config_path)
    cfg.max_basecall_reads = 10
    first = basecall_core(cfg)

    (pod5_dir / "extra.pod5").write_bytes(b"fake-pod5-extra")

    cfg_again = load_experiment_config(config_path)
    cfg_again.max_basecall_reads = 10
    second = basecall_core(cfg_again)

    assert second["reused_generation"] is True
    assert second["generation_id"] == first["generation_id"]
