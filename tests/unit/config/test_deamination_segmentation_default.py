"""PELT deamination segmentation is opt-in, not opt-out."""

from __future__ import annotations

import pytest

from smftools.cli.helpers import load_experiment_config
from smftools.config.experiment_config import ExperimentConfig

pytestmark = pytest.mark.unit


def test_segmentation_is_bypassed_by_default(tmp_path):
    """Segmentation costs far more than it changes on measured data.

    ~2h of saturated 12-way CPU on a 1.3M-read run, against 0 changed chimera
    calls and ~0.03% of variant calls (330,814 -> 330,705) on the one dataset
    where both paths were compared. The scalar `deaminase_PCR_chimera` column is
    produced either way, so the default keeps the cheap chimera call and drops
    the expensive located-segment evidence.

    Both the dataclass default and the parser fallback are pinned: they are
    consulted in different situations, and disagreeing would make the effective
    default depend on which config file was loaded.
    """
    config = tmp_path / "experiment_config.csv"
    config.write_text("variable,value,help,options,type\nexperiment_name,x,,,str\n")

    assert load_experiment_config(str(config)).bypass_deamination_segmentation is True
    field = ExperimentConfig.__dataclass_fields__["bypass_deamination_segmentation"]
    assert field.default is True


def test_segmentation_can_still_be_requested(tmp_path):
    """A run that needs located strand switches must be able to ask for them."""
    config = tmp_path / "experiment_config.csv"
    config.write_text(
        "variable,value,help,options,type\n"
        "experiment_name,x,,,str\n"
        "bypass_deamination_segmentation,FALSE,,,bool\n"
    )

    assert load_experiment_config(str(config)).bypass_deamination_segmentation is False
