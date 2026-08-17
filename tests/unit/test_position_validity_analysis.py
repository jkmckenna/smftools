"""Position validity measured over the analysed reads (`F9` / `EGL-11`).

`reduce_partial_coverage` answers "was this position measured in the reads the
assay produced" and runs before any QC column exists. That is the right question
for position statistics and the wrong one for a shared-position set: reads that
fail QC are largely the ones covering least, so they dilute every position and
are then discarded anyway.

These tests pin the distinction. The fixture is built so that a position's
apparent validity is decided *entirely* by reads that QC throws away.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smftools.preprocessing.partitioned_executor import (
    ANALYSIS_COVERAGE_COLUMNS,
    analysed_read_population,
    reduce_analysis_coverage,
)

pytestmark = pytest.mark.unit


def _write_task(tmp_path, *, read_ids, matrix, positions, reference="ref_top"):
    """Write one task store the reducer can read, plus its catalog row."""
    import anndata as ad

    from smftools.readwrite import safe_write_zarr

    adata = ad.AnnData(
        X=np.asarray(matrix, dtype=float),
        obs=pd.DataFrame(index=[str(r) for r in read_ids]),
        var=pd.DataFrame(index=[str(p) for p in positions]),
    )
    group = tmp_path / "task-1.zarr"
    safe_write_zarr(adata, group, backup=False, verbose=False, zarr_format=3)
    catalog = tmp_path / "catalog.parquet"
    pd.DataFrame(
        [{"group_path": group.name, "reference": reference, "n_positions": len(positions)}]
    ).to_parquet(catalog, index=False)
    return catalog


def _write_var_catalog(tmp_path, positions, reference="ref_top"):
    path = tmp_path / "var.parquet"
    pd.DataFrame(
        {
            "reference": [reference] * len(positions),
            "position": list(positions),
            "valid_count": [0] * len(positions),
            "valid_fraction": [0.0] * len(positions),
            "position_valid": [False] * len(positions),
        }
    ).to_parquet(path, index=False)
    return path


def _write_obs(tmp_path, *, read_ids, passing, reference="ref_top"):
    path = tmp_path / "obs.parquet"
    pd.DataFrame(
        {
            "read_id": [str(r) for r in read_ids],
            "Reference_strand": [reference] * len(read_ids),
            "passes_read_qc": [True] * len(read_ids),
            "passes_qc": list(passing),
            "passes_dedup": list(passing),
        }
    ).to_parquet(path, index=False)
    return path


def test_a_position_carried_by_discarded_reads_is_invalid_for_analysis(tmp_path):
    """The defect, stated as a test.

    Position 0 is measured in every read, but only the failing reads carry it in
    a way that would matter; position 1 is measured *only* by reads that QC
    discards. Over the assay-wide population position 1 looks well covered; over
    the analysed population it is not covered at all.
    """
    read_ids = [f"read{i}" for i in range(10)]
    passing = [True] * 2 + [False] * 8
    # position 0: measured in the 2 analysed reads. position 1: measured only in
    # the 8 discarded ones.
    matrix = [[1.0, np.nan] if index < 2 else [1.0, 1.0] for index in range(10)]
    catalog = _write_task(tmp_path, read_ids=read_ids, matrix=matrix, positions=[0, 1])
    var_catalog = _write_var_catalog(tmp_path, [0, 1])
    obs = _write_obs(tmp_path, read_ids=read_ids, passing=passing)

    reduce_analysis_coverage(catalog, obs, var_catalog, minimum_valid_fraction=0.8)

    result = pd.read_parquet(var_catalog).set_index("position")
    assert set(ANALYSIS_COVERAGE_COLUMNS) <= set(result.columns)
    # Measured in both analysed reads.
    assert result.loc[0, "valid_count_analysis"] == 2
    assert bool(result.loc[0, "position_valid_analysis"]) is True
    # Measured only by reads that are thrown away.
    assert result.loc[1, "valid_count_analysis"] == 0
    assert bool(result.loc[1, "position_valid_analysis"]) is False
    # The assay-wide columns are untouched, so the original meaning survives.
    assert result["position_valid"].tolist() == [False, False]


def test_the_analysis_denominator_excludes_discarded_reads(tmp_path):
    """Diluting reads must leave the fraction alone, not depress it."""
    read_ids = [f"read{i}" for i in range(10)]
    passing = [True] * 2 + [False] * 8
    # Position 0 is measured in the 2 analysed reads and in none of the others.
    matrix = [[1.0] if index < 2 else [np.nan] for index in range(10)]
    catalog = _write_task(tmp_path, read_ids=read_ids, matrix=matrix, positions=[0])
    var_catalog = _write_var_catalog(tmp_path, [0])
    obs = _write_obs(tmp_path, read_ids=read_ids, passing=passing)

    reduce_analysis_coverage(catalog, obs, var_catalog, minimum_valid_fraction=0.8)

    result = pd.read_parquet(var_catalog).iloc[0]
    # 2/2 analysed reads, not 2/10 of everything the assay produced.
    assert result["valid_fraction_analysis"] == pytest.approx(1.0)
    assert bool(result["position_valid_analysis"]) is True


def test_read_qc_alone_is_not_the_analysis_population(tmp_path):
    """`passes_read_qc` is deliberately not a fallback.

    On real data it admits reads that later fail modification QC, and that was
    enough to leave a reference with zero shared positions.
    """
    obs = pd.DataFrame(
        {
            "read_id": ["a", "b"],
            "passes_read_qc": [True, True],
            "passes_qc": [True, False],
            "passes_dedup": [True, False],
        }
    )
    mask, population = analysed_read_population(obs)
    assert population == "passes_dedup"
    assert mask.tolist() == [True, False]

    mask, population = analysed_read_population(obs.drop(columns=["passes_dedup"]))
    assert population == "passes_qc"
    assert mask.tolist() == [True, False]

    # Read QC alone never selects the population, even when it is all that is
    # present -- the reducer falls through to every read and says so.
    mask, population = analysed_read_population(obs.drop(columns=["passes_dedup", "passes_qc"]))
    assert population == "all_reads"
    assert mask.tolist() == [True, True]
