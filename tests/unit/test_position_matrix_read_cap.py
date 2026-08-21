"""Read cap for the automated position-correlation matrices (`EGL-24`).

Clustermaps have bounded their input since `EGL-27`; the correlation path had
no cap at all. These pin the two properties that make a cap safe to turn on by
default: it must bound each barcode independently, and it must be reproducible,
because a matrix whose composition changes between runs over unchanged data is
not a diagnostic.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from smftools.tools.partitioned_spatial import (
    _cap_reads_per_group,
    _write_position_matrix_sidecars,
)

pytestmark = pytest.mark.unit


def _adata(sizes: dict[tuple[str, str], int]):
    import anndata as ad

    references, samples = [], []
    for (reference, sample), count in sizes.items():
        references.extend([reference] * count)
        samples.extend([sample] * count)
    total = len(references)
    return ad.AnnData(
        X=np.zeros((total, 3), dtype=np.float32),
        obs=pd.DataFrame(
            {"Reference_strand": references, "Sample": samples},
            index=[f"read{index:05d}" for index in range(total)],
        ),
    )


def _sizes(adata):
    return adata.obs.groupby(["Reference_strand", "Sample"], observed=True).size().to_dict()


def test_cap_is_per_group_not_global():
    """A global cap lets one large barcode squeeze the small ones out entirely.

    The matrices are computed and read one barcode at a time, so a barcode
    reduced to a handful of reads is worse than one merely capped.
    """
    adata = _adata({("r1", "A"): 2500, ("r1", "B"): 50, ("r2", "A"): 300})

    capped = _cap_reads_per_group(adata, "Sample", 1000, seed=0)

    assert _sizes(capped) == {("r1", "A"): 1000, ("r1", "B"): 50, ("r2", "A"): 300}


def test_groups_under_the_cap_are_untouched():
    adata = _adata({("r1", "A"): 10, ("r1", "B"): 20})
    assert _cap_reads_per_group(adata, "Sample", 1000, seed=0) is adata


def test_nothing_to_cap_avoids_a_copy():
    """The common small-dataset case must not pay for a materialized copy."""
    adata = _adata({("r1", "A"): 5})
    assert _cap_reads_per_group(adata, "Sample", 1000, seed=0) is adata


def test_selection_is_reproducible_for_a_seed():
    adata = _adata({("r1", "A"): 500})
    first = list(_cap_reads_per_group(adata, "Sample", 100, seed=3).obs_names)
    second = list(_cap_reads_per_group(adata, "Sample", 100, seed=3).obs_names)
    assert first == second


def test_different_seeds_select_differently():
    adata = _adata({("r1", "A"): 500})
    first = list(_cap_reads_per_group(adata, "Sample", 100, seed=1).obs_names)
    second = list(_cap_reads_per_group(adata, "Sample", 100, seed=2).obs_names)
    assert first != second


def test_selection_is_random_not_the_first_n():
    adata = _adata({("r1", "A"): 500})
    selected = list(_cap_reads_per_group(adata, "Sample", 50, seed=0).obs_names)
    assert selected != [f"read{index:05d}" for index in range(50)]


def test_original_row_order_is_preserved():
    """Kept rows stay in store order so downstream ordering is unaffected."""
    adata = _adata({("r1", "A"): 400})
    names = list(_cap_reads_per_group(adata, "Sample", 100, seed=0).obs_names)
    assert names == sorted(names)


@pytest.mark.parametrize("cap", [0, None, -1])
def test_falsy_cap_is_a_no_op(cap):
    adata = _adata({("r1", "A"): 500})
    assert _cap_reads_per_group(adata, "Sample", cap, seed=0) is adata


def test_missing_sample_column_is_a_no_op():
    """Better uncapped than silently capped across the wrong grouping."""
    adata = _adata({("r1", "A"): 500})
    assert _cap_reads_per_group(adata, "Barcode_absent", 100, seed=0) is adata


# --- provenance --------------------------------------------------------------


def test_selection_provenance_is_recorded_beside_the_matrices(tmp_path):
    """A surprising matrix should be traceable to its input without a recompute."""
    import anndata as ad

    adata = ad.AnnData(X=np.zeros((2, 2), dtype=np.float32))
    matrix = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]], index=[1, 2], columns=[1, 2])
    adata.uns["key"] = {"pearson": {"bc1": matrix}}
    region = {"reference": "r1", "start": 0, "end": 10}
    provenance = {
        "reads_used": 1000,
        "reads_available": 2740,
        "max_reads_per_barcode": 1000,
        "selection_seed": 0,
        "min_count_for_pairwise": 10,
    }

    paths = _write_position_matrix_sidecars(adata, tmp_path, region, "key", provenance=provenance)

    manifests = [path for path in paths if path.name == "selection.json"]
    assert manifests, "the selection that produced the matrices must be recorded"
    recorded = json.loads(manifests[0].read_text())
    assert recorded["reads_used"] == 1000
    assert recorded["reads_available"] == 2740
    assert recorded["min_count_for_pairwise"] == 10


def test_no_provenance_written_when_none_requested(tmp_path):
    """Library callers keep the previous output shape exactly."""
    import anndata as ad

    adata = ad.AnnData(X=np.zeros((2, 2), dtype=np.float32))
    adata.uns["key"] = {"pearson": {"bc1": pd.DataFrame([[1.0]], index=[1], columns=[1])}}

    paths = _write_position_matrix_sidecars(
        adata, tmp_path, {"reference": "r1", "start": 0, "end": 10}, "key"
    )

    assert not any(path.name == "selection.json" for path in paths)


# --- config ------------------------------------------------------------------


def test_config_defaults_and_overrides():
    from smftools.config.experiment_config import ExperimentConfig

    cfg = ExperimentConfig()
    assert cfg.spatial_position_matrix_max_reads == 1000
    assert cfg.spatial_position_matrix_min_count_for_pairwise == 10

    overridden, _ = ExperimentConfig.from_var_dict(
        {
            "spatial_position_matrix_max_reads": "250",
            "spatial_position_matrix_min_count_for_pairwise": "4",
        }
    )
    assert overridden.spatial_position_matrix_max_reads == 250
    assert overridden.spatial_position_matrix_min_count_for_pairwise == 4
