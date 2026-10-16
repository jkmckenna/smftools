"""HMM defaults (`HPS`): bp axis; learned per-state context emissions for
deaminase data; emissions shared by enzyme with dose-per-cell offsets when
the sample sheet allows, else per sample with a warning."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from smftools.cli.helpers import load_experiment_config

pytestmark = pytest.mark.unit


def _config(tmp_path, *rows):
    path = tmp_path / "experiment_config.csv"
    lines = ["variable,value,help,options,type", "experiment_name,x,,,str", *rows]
    path.write_text("\n".join(lines) + "\n")
    return load_experiment_config(str(path))


def test_deaminase_defaults(tmp_path):
    cfg = _config(tmp_path, "smf_modality,deaminase,,,str")
    assert cfg.hmm_axis == "bp"
    assert cfg.hmm_context_model == "learned"
    assert cfg.hmm_context_states == "all"
    assert cfg.hmm_emission_scope == "auto"
    assert cfg.hmm_emission_groupby == ["enzyme"]
    assert cfg.hmm_efficiency_covariates == ["enzyme_per_cell"]


def test_other_modalities_keep_plain_emissions(tmp_path):
    cfg = _config(tmp_path, "smf_modality,conversion,,,str")
    assert cfg.hmm_axis == "bp"
    assert cfg.hmm_context_model == "none"


def test_explicit_settings_win(tmp_path):
    cfg = _config(
        tmp_path,
        "smf_modality,deaminase,,,str",
        "hmm_context_model,none,,,str",
        "hmm_axis,sites,,,str",
        "hmm_emission_scope,fit_group,,,str",
    )
    assert (cfg.hmm_context_model, cfg.hmm_axis, cfg.hmm_emission_scope) == (
        "none",
        "sites",
        "fit_group",
    )


def test_distance_aware_implies_the_site_axis():
    from smftools.hmm.model_artifacts import resolve_hmm_axis

    assert resolve_hmm_axis(SimpleNamespace(hmm_axis="bp", hmm_distance_aware=True)) == "sites"
    assert resolve_hmm_axis(SimpleNamespace(hmm_axis="bp")) == "bp"
    assert resolve_hmm_axis(SimpleNamespace()) == "sites"  # a missing key: legacy stores


def _spine(barcodes):
    return SimpleNamespace(
        obs=pd.DataFrame(
            {"Barcode": barcodes, "Experiment_name_and_barcode": [f"x_{b}" for b in barcodes]}
        )
    )


def _auto_cfg(tmp_path, sheet: pd.DataFrame | None, **extra):
    path = None
    if sheet is not None:
        path = tmp_path / "samples.csv"
        sheet.to_csv(path, index=False)
    return SimpleNamespace(
        hmm_emission_scope="auto",
        hmm_emission_groupby=["enzyme"],
        hmm_efficiency_groupby=[],
        hmm_efficiency_covariates=["enzyme_per_cell"],
        hmm_fit_strategy="per_group",
        hmm_groupby=["sample", "reference", "methbase"],
        sample_sheet_path=str(path) if path else None,
        sample_sheet_mapping_column="Experiment_name_and_barcode",
        **extra,
    )


def test_auto_scope_resolves_from_the_sample_sheet(tmp_path, caplog):
    from smftools.tools.partitioned_hmm import resolve_emission_scope

    spine = _spine(["bc1", "bc2"])
    sheet = pd.DataFrame(
        {
            "Experiment_name_and_barcode": ["x_bc1", "x_bc2"],
            "enzyme": ["E1", "E1"],
            "enzyme_per_cell": [1e-4, 2e-4],
        }
    )
    assert resolve_emission_scope(spine, _auto_cfg(tmp_path, sheet)).hmm_emission_scope == (
        "shared_efficiency"
    )
    # no sheet / a missing column: per-sample emissions, with a warning
    assert resolve_emission_scope(spine, _auto_cfg(tmp_path, None)).hmm_emission_scope == (
        "fit_group"
    )
    partial = sheet.drop(columns=["enzyme_per_cell"])
    assert resolve_emission_scope(spine, _auto_cfg(tmp_path, partial)).hmm_emission_scope == (
        "fit_group"
    )
    assert "fit_group" in caplog.text
    # an explicit scope is left alone
    explicit = _auto_cfg(tmp_path, None)
    explicit.hmm_emission_scope = "shared_efficiency"
    assert resolve_emission_scope(spine, explicit) is explicit
