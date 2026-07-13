from __future__ import annotations

from click.testing import CliRunner

from smftools.cli_entry import cli


def test_export_fastq_requires_exactly_one_of_config_or_project(tmp_path):
    runner = CliRunner()

    result = runner.invoke(cli, ["export-fastq", "--outdir", str(tmp_path / "out")])
    assert result.exit_code != 0
    assert "exactly one of --config or --project" in result.output

    config_path = tmp_path / "a.csv"
    config_path.touch()
    result = runner.invoke(
        cli,
        [
            "export-fastq",
            "--config",
            str(config_path),
            "--project",
            str(tmp_path),
            "--outdir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code != 0
    assert "exactly one of --config or --project" in result.output


def test_export_fastq_dispatches_to_experiment_export(tmp_path, monkeypatch):
    import smftools.cli.export_fastq as export_fastq_module

    captured = {}

    def fake_export(config_path, outdir, **kwargs):
        captured["config_path"] = config_path
        captured["outdir"] = outdir
        captured["kwargs"] = kwargs
        return outdir

    monkeypatch.setattr(export_fastq_module, "export_fastq_for_experiment", fake_export)

    config_path = tmp_path / "a.csv"
    config_path.touch()
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "export-fastq",
            "--config",
            str(config_path),
            "--outdir",
            str(tmp_path / "out"),
            "--group-by",
            "Sample",
            "--no-gzip",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["config_path"] == str(config_path)
    assert captured["kwargs"]["group_by"] == "Sample"
    assert captured["kwargs"]["gzip_output"] is False
