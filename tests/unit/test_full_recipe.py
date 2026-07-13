from click.testing import CliRunner

from smftools import cli_entry
from smftools.cli import recipes


def test_full_flow_runs_raw_preprocess_spatial_hmm_in_order(monkeypatch):
    calls = []
    monkeypatch.setattr(recipes, "raw_adata", lambda path: calls.append(("raw", path)))
    monkeypatch.setattr(
        recipes, "preprocess_adata", lambda path: calls.append(("preprocess", path))
    )
    monkeypatch.setattr(recipes, "spatial_adata", lambda path: calls.append(("spatial", path)))

    def run_hmm(path):
        calls.append(("hmm", path))
        return "adata", "hmm-output"

    monkeypatch.setattr(recipes, "hmm_adata", run_hmm)

    result = recipes.full_flow("experiment.csv")

    assert calls == [
        ("raw", "experiment.csv"),
        ("preprocess", "experiment.csv"),
        ("spatial", "experiment.csv"),
        ("hmm", "experiment.csv"),
    ]
    assert result == ("adata", "hmm-output")


def test_full_cli_invokes_four_stage_recipe(tmp_path, monkeypatch):
    config = tmp_path / "experiment.csv"
    config.write_text("variable,value\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr(cli_entry, "full_flow", lambda path: calls.append(path))

    result = CliRunner().invoke(cli_entry.cli, ["full", str(config)])

    assert result.exit_code == 0
    assert calls == [str(config)]
