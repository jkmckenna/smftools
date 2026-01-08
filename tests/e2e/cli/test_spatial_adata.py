"""Pytest-based end-to-end check for spatial_adata."""

from __future__ import annotations

import importlib
import importlib.resources as resources
import sys
import types
from importlib.machinery import ModuleSpec
from pathlib import Path

import pytest

from smftools.cli.spatial_adata import spatial_adata

CONFIGS = [
    Path("tests/_test_inputs/test_experiment_config_direct_I.csv"),
    Path("tests/_test_inputs/test_experiment_config_deaminase_I.csv"),
    Path("tests/_test_inputs/test_experiment_config_conversion_I.csv"),
]


@pytest.mark.e2e
@pytest.mark.parametrize("config_path", CONFIGS, ids=lambda p: p.name)
def test_spatial_adata_e2e(config_path: Path):
    try:
        spatial_adata(str(config_path))
    except RuntimeError as exc:
        raise
