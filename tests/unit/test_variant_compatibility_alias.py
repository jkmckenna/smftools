from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import pandas as pd
import pytest
from click.testing import CliRunner

from smftools import cli_entry
from smftools.cli import variant_adata as variant_module
from smftools.readwrite import safe_write_h5ad


def _cfg(tmp_path, *, mode="report"):
    return SimpleNamespace(
        output_directory=tmp_path,
        variant_analysis_mode=mode,
        references_to_align_for_variant_annotation=["ref_a", "ref_b"],
    )


def _paths(tmp_path):
    return SimpleNamespace(
        spine=tmp_path / "load" / "spine.h5ad",
        raw_spine=tmp_path / "raw" / "spine.h5ad",
        raw=tmp_path / "legacy_raw.h5ad.gz",
        pp=tmp_path / "legacy_preprocess.h5ad.gz",
        pp_dedup=tmp_path / "legacy_preprocess_deduplicated.h5ad.gz",
        variant=tmp_path / "legacy_variant.h5ad.gz",
    )


def _patch_request(monkeypatch, tmp_path, *, mode="report", current=False, source=True):
    cfg = _cfg(tmp_path, mode=mode)
    paths = _paths(tmp_path)
    monkeypatch.setattr("smftools.cli.helpers.load_experiment_config", lambda _path: cfg)
    monkeypatch.setattr("smftools.cli.helpers.get_adata_paths", lambda _cfg: paths)
    monkeypatch.setattr(
        variant_module,
        "_current_generation_has_integrated_variant",
        lambda _output: current,
    )
    monkeypatch.setattr(
        variant_module,
        "_partitioned_raw_source",
        lambda _paths: Path("raw/spine.h5ad") if source else None,
    )
    calls = []

    def preprocess(config_path, *, cfg=None):
        calls.append((config_path, cfg))
        return Path("preprocess/spine.h5ad"), None

    monkeypatch.setattr("smftools.cli.preprocess_adata.preprocess_adata", preprocess)
    return cfg, paths, calls


def test_variant_command_requests_integrated_preprocess_and_ignores_legacy_output(
    tmp_path,
    monkeypatch,
):
    cfg, paths, calls = _patch_request(monkeypatch, tmp_path)
    paths.variant.touch()

    with pytest.warns(FutureWarning, match="deprecated"):
        result = variant_module.variant_adata("config.csv")

    assert result == (Path("preprocess/spine.h5ad"), None)
    assert calls == [("config.csv", cfg)]


def test_compatible_integrated_generation_can_be_reused_without_original_raw(
    tmp_path,
    monkeypatch,
):
    cfg, _paths_value, calls = _patch_request(
        monkeypatch,
        tmp_path,
        current=True,
        source=False,
    )

    with pytest.warns(FutureWarning):
        variant_module.variant_adata("config.csv")
    with pytest.warns(FutureWarning):
        variant_module.variant_adata("config.csv")

    assert calls == [("config.csv", cfg), ("config.csv", cfg)]


def test_variant_alias_requires_integrated_mode(tmp_path, monkeypatch):
    _patch_request(monkeypatch, tmp_path, mode="off")

    with (
        pytest.warns(FutureWarning),
        pytest.raises(
            ValueError,
            match="variant_analysis_mode='report'",
        ),
    ):
        variant_module.variant_adata("config.csv")


@pytest.mark.parametrize(
    ("legacy_kind", "message"),
    [
        ("deduplicated", "retained rows only"),
        ("preprocess", "legacy monolithic preprocess"),
        ("raw", "legacy monolithic raw"),
        ("variant", "original raw source is missing"),
        ("missing", "original partitioned raw source is missing"),
    ],
)
def test_legacy_inputs_report_explicit_upgrade_limitations(
    tmp_path,
    monkeypatch,
    legacy_kind,
    message,
):
    _cfg_value, paths, _calls = _patch_request(
        monkeypatch,
        tmp_path,
        current=False,
        source=False,
    )
    selected = {
        "deduplicated": paths.pp_dedup,
        "preprocess": paths.pp,
        "raw": paths.raw,
        "variant": paths.variant,
    }.get(legacy_kind)
    if selected is not None:
        if legacy_kind == "variant":
            legacy = ad.AnnData(obs=pd.DataFrame(index=["retained-read"]))
            legacy.uns["append_variant_call_layer_performed"] = True
            safe_write_h5ad(legacy, selected, backup=False, verbose=False)
        else:
            selected.parent.mkdir(parents=True, exist_ok=True)
            selected.touch()

    with (
        pytest.warns(FutureWarning),
        pytest.raises(
            variant_module.LegacyVariantMigrationError,
            match=message,
        ),
    ):
        variant_module.variant_adata("config.csv")


def test_legacy_variant_reader_remains_available(tmp_path):
    path = tmp_path / "legacy_variant.h5ad"
    safe_write_h5ad(
        ad.AnnData(obs=pd.DataFrame(index=["read"])),
        path,
        backup=False,
        verbose=False,
    )

    with pytest.warns(FutureWarning, match="retained-row snapshots"):
        loaded, _ = variant_module.read_legacy_variant_adata(path)

    assert loaded.obs_names.tolist() == ["read"]


def test_variant_cli_emits_migration_notice(tmp_path, monkeypatch):
    config = tmp_path / "config.csv"
    config.touch()
    calls = []
    monkeypatch.setattr(variant_module, "variant_adata", lambda path: calls.append(path))

    result = CliRunner().invoke(
        cli_entry.cli,
        ["experiment", "variant", str(config)],
    )

    assert result.exit_code == 0, result.output
    assert "DEPRECATED" in result.output
    assert "variant_analysis_mode: report" in result.output
    assert calls == [str(config)]


def test_batch_variant_dispatches_same_compatibility_target(tmp_path, monkeypatch):
    config = tmp_path / "config.csv"
    config.touch()
    config_table = tmp_path / "configs.txt"
    config_table.write_text(f"{config}\n")
    calls = []
    monkeypatch.setattr(variant_module, "variant_adata", lambda path: calls.append(path))

    result = CliRunner().invoke(
        cli_entry.cli,
        ["experiment", "batch", "variant", str(config_table)],
    )

    assert result.exit_code == 0, result.output
    assert calls == [str(config)]
