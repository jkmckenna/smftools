"""Membership and coverage are separate questions (`F9` / `F11`).

`position_in_<reference>` used to be assigned `position_valid`, a coverage
density statistic, while every consumer ANDs it with a site-type mask to ask a
structural question: does this column belong to this reference. On a run where
no position cleared the density threshold, that emptied every such mask.

The loud failure was latent ("no eligible positions"). The quiet one was
duplicate detection, which returned `None` for every chunk and reported zero
duplicates out of 19,328 reads -- inflating library complexity with no error
anywhere. These tests pin the separation so neither can recur.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.unit


# --- the overlay that defines the column ------------------------------------


def _overlaid_var(tmp_path, *, positions, member_positions, valid_positions):
    """Run `_overlay_preprocess_var` and hand back the `.var` it produced.

    `member_positions` are the rows the preprocess reduction emitted for this
    reference; `valid_positions` are the subset that cleared the coverage
    threshold. Keeping the two independent is the whole point.
    """
    import anndata as ad

    from smftools.informatics.partition_read import _overlay_preprocess_var

    path = tmp_path / "preprocess_var.parquet"
    pd.DataFrame(
        {
            "reference": ["ref_top"] * len(member_positions),
            "position": list(member_positions),
            "position_valid": [p in valid_positions for p in member_positions],
            "valid_count": [0] * len(member_positions),
            "valid_fraction": [0.0] * len(member_positions),
        }
    ).to_parquet(path)

    spine = ad.AnnData(
        X=np.zeros((1, 1)),
        obs=pd.DataFrame(index=["read0"]),
        var=pd.DataFrame(index=["0"]),
    )
    spine.uns["preprocess_var"] = str(path)

    result = ad.AnnData(
        X=np.zeros((2, len(positions))),
        obs=pd.DataFrame({"Reference_strand": ["ref_top", "ref_top"]}, index=["r0", "r1"]),
        var=pd.DataFrame(index=[str(p) for p in positions]),
    )
    _overlay_preprocess_var(spine, result, None)
    return result.var


def test_membership_does_not_collapse_when_no_position_is_densely_covered(tmp_path):
    """The defect: coverage was written into the membership column.

    Every position belongs to the reference; none clears the density threshold.
    Membership must still be true, or every downstream site mask empties.
    """
    var = _overlaid_var(
        tmp_path,
        positions=[10, 20, 30],
        member_positions=[10, 20, 30],
        valid_positions=[],
    )

    assert var["position_in_ref_top"].tolist() == [True, True, True]
    assert var["N_Reference_strand_with_position"].tolist() == [1, 1, 1]


def test_membership_excludes_positions_the_reference_never_covered(tmp_path):
    """Membership must stay a real distinction, not become all-true."""
    var = _overlaid_var(
        tmp_path,
        positions=[10, 20, 30],
        member_positions=[10, 30],
        valid_positions=[10, 30],
    )

    assert var["position_in_ref_top"].tolist() == [True, False, True]


def test_coverage_survives_under_its_own_reference_qualified_name(tmp_path):
    """Separating the two must not throw the density statistic away."""
    var = _overlaid_var(
        tmp_path,
        positions=[10, 20, 30],
        member_positions=[10, 20, 30],
        valid_positions=[30],
    )

    assert var["ref_top_position_valid"].tolist() == [False, False, True]
    assert var["position_in_ref_top"].tolist() == [True, True, True]


# --- duplicate detection, the quiet casualty --------------------------------


class _Var:
    """Minimal stand-in for a materialized window's `.var`."""

    def __init__(self, frame):
        self._frame = frame

    def __contains__(self, key):
        return key in self._frame.columns

    def __getitem__(self, key):
        return self._frame[key]


class _Window:
    def __init__(self, frame):
        self.var = _Var(frame)
        self.n_vars = len(frame)


def _cfg(site_types=("GpC", "CpG")):
    from types import SimpleNamespace

    return SimpleNamespace(duplicate_detection_site_types=list(site_types))


def test_dedup_compares_sites_when_no_position_is_densely_covered(tmp_path):
    """End to end from the overlay: sparse coverage must not disable dedup.

    This is the failure that produced 0 duplicates in 19,328 reads. It is built
    from the real overlay rather than a hand-written `.var` because the defect
    was in what the overlay wrote, not in what dedup did with it.
    """
    from smftools.preprocessing.duplicate_detection_dispatch import (
        _build_duplicate_detection_context_mask,
    )

    var = _overlaid_var(
        tmp_path,
        positions=[10, 20, 30, 40],
        member_positions=[10, 20, 30, 40],
        valid_positions=[],
    ).copy()
    var["ref_top_GpC_site"] = [True, False, True, False]
    var["ref_top_CpG_site"] = [False, True, False, False]

    mask = _build_duplicate_detection_context_mask(_Window(var), "ref_top", _cfg())

    assert mask.tolist() == [True, True, True, False]
    assert mask.any(), "an empty mask makes duplicate detection silently return None"


def test_dedup_still_restricts_to_the_configured_site_types():
    """The mask must stay a real filter, not become a pass-through."""
    from smftools.preprocessing.duplicate_detection_dispatch import (
        _build_duplicate_detection_context_mask,
    )

    frame = pd.DataFrame(
        {
            "ref_top_GpC_site": [True, False, False],
            "ref_top_CpG_site": [False, False, False],
            "position_in_ref_top": [True, True, True],
        }
    )

    mask = _build_duplicate_detection_context_mask(_Window(frame), "ref_top", _cfg(["GpC"]))

    assert mask.tolist() == [True, False, False]


def test_dedup_excludes_positions_belonging_to_another_reference():
    from smftools.preprocessing.duplicate_detection_dispatch import (
        _build_duplicate_detection_context_mask,
    )

    frame = pd.DataFrame(
        {
            "ref_top_GpC_site": [True, True, True],
            "position_in_ref_top": [True, False, True],
        }
    )

    mask = _build_duplicate_detection_context_mask(_Window(frame), "ref_top", _cfg(["GpC"]))

    assert mask.tolist() == [True, False, True]


# --- latent, the loud casualty ----------------------------------------------


def _adata(matrix, *, reference="ref_top", member=None):
    import anndata as ad

    matrix = np.asarray(matrix, dtype=float)
    var = pd.DataFrame(
        {f"position_in_{reference}": [True] * matrix.shape[1] if member is None else member},
        index=[str(i) for i in range(matrix.shape[1])],
    )
    return ad.AnnData(
        X=matrix,
        obs=pd.DataFrame(index=[f"read{i}" for i in range(matrix.shape[0])]),
        var=var,
    )


def test_latent_density_is_measured_over_the_reads_it_will_factorize():
    """Density depends on which reads you look at, so it is computed locally."""
    from smftools.cli.latent_adata import _build_reference_position_mask

    # Column 0 is measured in every read; column 1 in one read of four.
    matrix = [
        [1.0, 1.0],
        [1.0, np.nan],
        [1.0, np.nan],
        [1.0, np.nan],
    ]
    adata = _adata(matrix)

    dense = _build_reference_position_mask(adata, ["ref_top"], minimum_valid_fraction=0.8)
    permissive = _build_reference_position_mask(adata, ["ref_top"], minimum_valid_fraction=0.0)

    assert dense.tolist() == [True, False]
    # With no density requirement the mask is pure membership.
    assert permissive.tolist() == [True, True]


def test_latent_density_follows_the_subset_not_a_global_flag():
    from smftools.cli.latent_adata import _build_reference_position_mask

    matrix = [
        [1.0, np.nan],
        [1.0, np.nan],
        [1.0, 1.0],
        [1.0, 1.0],
    ]
    full = _adata(matrix)
    # The same column, judged over only the reads that measured it.
    subset = _adata(matrix[2:])

    assert _build_reference_position_mask(
        full, ["ref_top"], minimum_valid_fraction=0.8
    ).tolist() == [True, False]
    assert _build_reference_position_mask(
        subset, ["ref_top"], minimum_valid_fraction=0.8
    ).tolist() == [True, True]


def test_latent_still_honours_reference_membership():
    from smftools.cli.latent_adata import _build_reference_position_mask

    adata = _adata([[1.0, 1.0], [1.0, 1.0]], member=[True, False])

    mask = _build_reference_position_mask(adata, ["ref_top"], minimum_valid_fraction=0.5)

    assert mask.tolist() == [True, False]
