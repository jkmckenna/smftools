from __future__ import annotations

from click.testing import CliRunner

from smftools.cli_entry import cli


def test_experiment_export_fastq_dispatches_to_experiment_export(tmp_path, monkeypatch):
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
            "experiment",
            "export-fastq",
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


def test_project_export_fastq_dispatches_to_project_export(tmp_path, monkeypatch):
    import smftools.cli.export_fastq as export_fastq_module

    captured = {}

    def fake_export(project_dir, outdir, **kwargs):
        captured["project_dir"] = project_dir
        captured["outdir"] = outdir
        captured["kwargs"] = kwargs
        return outdir

    monkeypatch.setattr(export_fastq_module, "export_fastq_for_project", fake_export)

    project_dir = tmp_path / "project"
    project_dir.mkdir()
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "project",
            "export-fastq",
            str(project_dir),
            "--outdir",
            str(tmp_path / "out"),
            "--experiments",
            "exp1,exp2",
            "--no-gzip",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["project_dir"] == project_dir
    assert captured["kwargs"]["experiments"] == ["exp1", "exp2"]
    assert captured["kwargs"]["gzip_output"] is False
