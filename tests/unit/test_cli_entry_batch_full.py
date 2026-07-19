from __future__ import annotations

from click.testing import CliRunner

from smftools.cli_entry import cli


def test_batch_full_dispatches_to_full_flow_per_config(tmp_path, monkeypatch):
    import smftools.cli.recipes as recipes_module

    captured = []

    def fake_full_flow(config_path):
        captured.append(config_path)

    monkeypatch.setattr(recipes_module, "full_flow", fake_full_flow)

    config_a = tmp_path / "a.csv"
    config_a.touch()
    config_b = tmp_path / "b.csv"
    config_b.touch()

    config_table = tmp_path / "configs.txt"
    config_table.write_text(f"{config_a}\n{config_b}\n")

    runner = CliRunner()
    result = runner.invoke(cli, ["experiment", "batch", "full", str(config_table)])

    assert result.exit_code == 0, result.output
    assert captured == [str(config_a), str(config_b)]
