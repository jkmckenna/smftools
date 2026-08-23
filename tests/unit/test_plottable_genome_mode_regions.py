"""Genome-mode references that are short enough still get plot regions (`F27b`).

`analysis_mode` answers a *storage* question -- can every read for this
reference be held in one dense matrix -- and `auto` correctly answers "no" for a
reference with hundreds of thousands of reads. The plotting layer then treated
that storage answer as a *presentation* declaration ("the user will name regions
via BED"), which is right for a genome and nonsense for a 4 kb amplicon.

The result on the 260820 run: the two references with the most reads produced no
spatial clustermaps or position matrices at all, while references holding 1 and
5 reads plotted normally.

Plots never draw every read -- at most `clustermap_max_reads_per_plot` -- so
feasibility is judged against the plotted population.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from smftools.tools.partitioned_spatial import _dense_product_regions, _plottable_as_locus

pytestmark = pytest.mark.unit

GB = 1024**3


def _cfg(cap=10_000, max_gb=8.0):
    return SimpleNamespace(clustermap_max_reads_per_plot=cap, max_full_matrix_gb=max_gb)


def _spine(plans):
    return SimpleNamespace(uns={"reference_plans": plans})


def _no_bed():
    return pd.DataFrame(columns=["reference", "start", "end", "name", "source"])


# --- the feasibility test itself ---------------------------------------------


def test_a_short_reference_with_many_reads_is_plottable():
    """The motivating case: 4,168 bp x 622,281 reads.

    19.3 GB across every read, but 0.33 GB across the 10,000 a plot draws.
    """
    plan = {"reference_length": 4168, "n_reads": 622_281}
    assert _plottable_as_locus(plan, plot_read_cap=10_000, max_bytes=8 * GB)


def test_a_genome_scale_contig_is_not_plottable():
    """Length alone must still exclude the case genome mode exists for."""
    plan = {"reference_length": 61_000_000, "n_reads": 500_000}
    assert not _plottable_as_locus(plan, plot_read_cap=10_000, max_bytes=8 * GB)


def test_feasibility_uses_the_capped_population_not_the_full_one():
    """Judging on total reads is precisely the conflation being fixed."""
    plan = {"reference_length": 4168, "n_reads": 10_000_000}
    assert _plottable_as_locus(plan, plot_read_cap=10_000, max_bytes=8 * GB)


def test_a_group_below_the_cap_uses_its_own_size():
    plan = {"reference_length": 4168, "n_reads": 50}
    assert _plottable_as_locus(plan, plot_read_cap=10_000, max_bytes=8 * GB)


def test_a_lower_ceiling_excludes_more():
    plan = {"reference_length": 4168, "n_reads": 622_281}
    assert not _plottable_as_locus(plan, plot_read_cap=10_000, max_bytes=0.01 * GB)


def test_a_missing_read_count_falls_back_to_the_cap():
    """Plans predating this field must not crash or silently pass everything."""
    assert _plottable_as_locus({"reference_length": 4168}, plot_read_cap=10_000, max_bytes=8 * GB)
    assert not _plottable_as_locus(
        {"reference_length": 61_000_000}, plot_read_cap=10_000, max_bytes=8 * GB
    )


# --- region catalog behaviour -------------------------------------------------


def test_genome_mode_amplicons_now_get_locus_regions():
    plans = {
        "6B6_top": {"analysis_mode": "genome", "reference_length": 4690, "n_reads": 466_183},
        "6B6_bottom": {"analysis_mode": "locus", "reference_length": 4690, "n_reads": 58_560},
    }

    regions = _dense_product_regions(_spine(plans), _no_bed(), _cfg())

    assert set(regions["reference"]) == {"6B6_top", "6B6_bottom"}
    assert int(regions.loc[regions["reference"] == "6B6_top", "end"].iloc[0]) == 4690


def test_genuinely_large_references_still_get_none(caplog):
    plans = {
        "chr19": {"analysis_mode": "genome", "reference_length": 61_000_000, "n_reads": 500_000}
    }

    with caplog.at_level("WARNING", logger="smftools.tools.partitioned_spatial"):
        regions = _dense_product_regions(_spine(plans), _no_bed(), _cfg())

    assert regions.empty
    assert any("chr19" in record.getMessage() for record in caplog.records), (
        "F27a's warning must still fire for references that really cannot be plotted"
    )


def test_no_warning_for_references_that_now_get_regions(caplog):
    """The F27a warning must not fire where F27b has supplied a region."""
    plans = {"6B6_top": {"analysis_mode": "genome", "reference_length": 4690, "n_reads": 466_183}}

    with caplog.at_level("WARNING", logger="smftools.tools.partitioned_spatial"):
        _dense_product_regions(_spine(plans), _no_bed(), _cfg())

    assert not [r for r in caplog.records if r.levelname == "WARNING"]


def test_bed_regions_still_apply_to_genome_mode_references():
    """Explicit BED regions remain the way to subset a large reference."""
    plans = {"chr19": {"analysis_mode": "genome", "reference_length": 61_000_000, "n_reads": 5_000}}
    bed = pd.DataFrame(
        [{"reference": "chr19", "start": 100, "end": 2000, "name": "promoter", "source": "bed"}]
    )

    regions = _dense_product_regions(_spine(plans), bed, _cfg())

    assert list(regions["name"]) == ["promoter"]


def test_absent_cfg_uses_defaults():
    """The function is called from a path that may not thread cfg through."""
    plans = {"6B6_top": {"analysis_mode": "genome", "reference_length": 4690, "n_reads": 466_183}}
    regions = _dense_product_regions(_spine(plans), _no_bed(), None)
    assert set(regions["reference"]) == {"6B6_top"}
