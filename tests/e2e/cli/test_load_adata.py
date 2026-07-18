"""Pytest-based end-to-end check for load_adata."""

from __future__ import annotations

import importlib
import importlib.resources as resources
import sys
import types
from importlib.machinery import ModuleSpec
from pathlib import Path

import pytest

from smftools.cli.load_adata import load_adata

CONFIGS = [
    Path("tests/_test_inputs/test_experiment_config_direct_I.csv"),
    Path("tests/_test_inputs/test_experiment_config_deaminase_I.csv"),
    Path("tests/_test_inputs/test_experiment_config_conversion_I.csv"),
]


@pytest.mark.e2e
@pytest.mark.parametrize("config_path", CONFIGS, ids=lambda p: p.name)
def test_load_adata_e2e(config_path: Path):
    adata, adata_path, _cfg = load_adata(str(config_path))

    # `load_adata` returns `adata=None` only when a later pipeline stage's output
    # already existed and load was skipped -- not expected for a fresh test run,
    # but tolerate it by reading back from disk so these invariants still cover
    # the actual on-disk artifact either way.
    import anndata as ad

    if adata is None:
        adata = ad.read_h5ad(adata_path)

    # Basic non-empty-result invariants. These exist to catch the class of bug
    # where a refactor (e.g. of the AnnData-concatenation or dict-skip logic in
    # modkit_extract_to_adata.py) silently produces an empty or malformed result
    # instead of raising -- "does not raise" alone would not catch that.
    assert adata.n_obs > 0, "expected at least one read in the final AnnData"
    assert adata.n_vars > 0, "expected at least one position/var in the final AnnData"
    assert adata.obs_names.is_unique, "expected unique read names (obs_names)"
