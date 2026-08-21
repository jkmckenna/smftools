"""Deamination bypass and the clustermap read cap (`EGL-25`, `EGL-27`)."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from smftools.config.experiment_config import ExperimentConfig
from smftools.preprocessing.chimera_classes import (
    COMPOSITE_COLUMN,
    append_composite_chimera_column,
)
from smftools.preprocessing.partitioned_deamination import deamination_reporting_enabled
from smftools.preprocessing.partitioned_variant_plots import _select_reads

pytestmark = pytest.mark.unit


def _cfg(**overrides):
    base = dict(
        smf_modality="deaminase", conversion_types=["5mC"], bypass_deamination_segmentation=False
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# --- EGL-25: bypass ----------------------------------------------------------


def test_bypass_disables_segmentation():
    assert deamination_reporting_enabled(_cfg()) is True
    assert deamination_reporting_enabled(_cfg(bypass_deamination_segmentation=True)) is False


def test_bypass_is_distinct_from_the_modality_gate():
    """`direct` has no chemistry; bypass is for having it and declining it.

    Both return False, but for different reasons -- conflating them would make
    the bypass look redundant and invite removing it.
    """
    assert deamination_reporting_enabled(_cfg(smf_modality="direct")) is False
    assert deamination_reporting_enabled(_cfg(bypass_deamination_segmentation=True)) is False


def test_composite_degrades_to_the_remaining_methods_under_bypass():
    """The union must not treat an absent segment column as "not chimeric".

    Under bypass the segment column does not exist; `omit_chimeric_reads` has to
    fall back to the methods that did run, which is the `F13`/`F17` failure
    shape avoided.
    """
    obs = pd.DataFrame(
        {
            "chimeric_variant_sites": [True, False, False],
            "deaminase_PCR_chimera": [False, True, False],
        }
    )
    assert list(append_composite_chimera_column(obs)[COMPOSITE_COLUMN]) == [True, True, False]


def test_bypass_flag_is_configurable(tmp_path):
    path = tmp_path / "experiment_config.csv"
    path.write_text(
        "variable,value,help,options,type\nbypass_deamination_segmentation,TRUE,,,bool\n",
        encoding="utf-8",
    )
    cfg, _ = ExperimentConfig.from_csv(path)
    assert cfg.bypass_deamination_segmentation is True


# --- EGL-27: cap and selection ----------------------------------------------


def test_cap_default_is_ten_thousand():
    assert ExperimentConfig().clustermap_max_reads_per_plot == 10000


def _panel(n, reference="ref_top", sample="bc1"):
    return pd.DataFrame(
        {
            "Reference_strand": [reference] * n,
            "Sample": [sample] * n,
            "read_id": [f"r{index:05d}" for index in range(n)],
        }
    )


def test_selection_is_random_not_first_n():
    """First-N by read id is a biased slice; read ids are not randomly ordered.

    This is the defect `EGL-27` records against `EGL-17`'s original selection.
    """
    selected = _select_reads(_panel(500), "Sample", 50, seed=0)
    assert len(selected) == 50
    assert list(selected["read_id"]) != [f"r{index:05d}" for index in range(50)]


def test_selection_is_reproducible_for_a_given_seed():
    """Reproducibility comes from the seed, not from refusing to sample."""
    first = list(_select_reads(_panel(500), "Sample", 50, seed=7)["read_id"])
    second = list(_select_reads(_panel(500), "Sample", 50, seed=7)["read_id"])
    assert first == second


def test_different_seeds_select_differently():
    first = list(_select_reads(_panel(500), "Sample", 50, seed=1)["read_id"])
    second = list(_select_reads(_panel(500), "Sample", 50, seed=2)["read_id"])
    assert first != second


def test_groups_below_the_cap_are_untouched():
    assert len(_select_reads(_panel(10), "Sample", 50, seed=0)) == 10


def test_cap_applies_per_panel_not_globally():
    """Small barcodes must not be squeezed out by large ones."""
    panel = pd.concat([_panel(200, sample="bc1"), _panel(10, sample="bc2")], ignore_index=True)
    selected = _select_reads(panel, "Sample", 50, seed=0)
    counts = selected.groupby("Sample").size().to_dict()
    assert counts == {"bc1": 50, "bc2": 10}
