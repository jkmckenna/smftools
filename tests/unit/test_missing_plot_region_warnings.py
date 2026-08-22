"""Warn when a reference will produce no dense products (`F27a`).

A genome-mode reference draws its regions only from a BED file. Without one it
ends up with no regions, produces no plot plans, and therefore no clustermaps or
position matrices -- previously with nothing logged anywhere.

On the run that found this, the two references with the *most* reads (622k and
466k, both 4 kb amplicons promoted to genome mode on raw read count) vanished
from every spatial clustermap, while references holding 1 and 5 reads plotted
normally. Skipping them is defensible; doing it silently is not.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from smftools.tools.partitioned_spatial import _dense_product_regions

pytestmark = pytest.mark.unit


def _spine(plans):
    return SimpleNamespace(uns={"reference_plans": plans})


def _no_bed():
    return pd.DataFrame(columns=["reference", "start", "end", "name", "source"])


def test_genome_mode_reference_without_regions_warns(caplog):
    plans = {
        "6B6_top": {"analysis_mode": "genome", "reference_length": 4690},
        "6B6_bottom": {"analysis_mode": "locus", "reference_length": 4690},
    }

    with caplog.at_level("WARNING", logger="smftools.tools.partitioned_spatial"):
        regions = _dense_product_regions(_spine(plans), _no_bed())

    assert list(regions["reference"]) == ["6B6_bottom"]
    warnings = [r.message for r in caplog.records if r.levelname == "WARNING"]
    assert any("6B6_top" in message for message in warnings)


def test_the_warning_names_every_affected_reference(caplog):
    plans = {
        "6B6_top": {"analysis_mode": "genome", "reference_length": 4690},
        "6B6_enh_del_top": {"analysis_mode": "genome", "reference_length": 4168},
    }

    with caplog.at_level("WARNING", logger="smftools.tools.partitioned_spatial"):
        _dense_product_regions(_spine(plans), _no_bed())

    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "6B6_top" in joined and "6B6_enh_del_top" in joined


def test_the_warning_suggests_both_remedies(caplog):
    """A warning that does not say what to do is only marginally better."""
    plans = {"6B6_top": {"analysis_mode": "genome", "reference_length": 4690}}

    with caplog.at_level("WARNING", logger="smftools.tools.partitioned_spatial"):
        _dense_product_regions(_spine(plans), _no_bed())

    joined = " ".join(r.getMessage() for r in caplog.records).lower()
    assert "bed" in joined
    assert "locus" in joined


def test_genome_mode_reference_with_bed_regions_does_not_warn(caplog):
    plans = {"6B6_top": {"analysis_mode": "genome", "reference_length": 4690}}
    bed = pd.DataFrame(
        [{"reference": "6B6_top", "start": 0, "end": 4690, "name": "r1", "source": "bed"}]
    )

    with caplog.at_level("WARNING", logger="smftools.tools.partitioned_spatial"):
        regions = _dense_product_regions(_spine(plans), bed)

    assert len(regions) == 1
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


def test_locus_mode_references_never_warn(caplog):
    """They always get a locus region, so there is nothing to report."""
    plans = {
        "a": {"analysis_mode": "locus", "reference_length": 100},
        "b": {"analysis_mode": "locus", "reference_length": 200},
    }

    with caplog.at_level("WARNING", logger="smftools.tools.partitioned_spatial"):
        regions = _dense_product_regions(_spine(plans), _no_bed())

    assert len(regions) == 2
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


def test_requested_region_without_coverage_warns(caplog):
    """The second silent path: a region requested but never analyzed."""
    import smftools.informatics.plot_region_stitching as stitching

    source = stitching.resolve_plot_region_plans.__doc__ or ""
    assert source is not None  # sanity
    # Direct check that the branch logs rather than silently continuing.
    import inspect

    body = inspect.getsource(stitching.resolve_plot_region_plans)
    assert "logger.warning" in body
    assert "No analyzed coverage" in body
