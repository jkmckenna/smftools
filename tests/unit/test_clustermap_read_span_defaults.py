"""Clustermaps crop to the reads they actually contain.

Uncropped panels spend their width on positions no molecule covers: on the
`241213` pilot the union of read spans is 944..3795 against a 4,690-position
reference, so ~40% of every HMM and spatial clustermap was empty axis.

Cropping is per *reference*, not per barcode, and that is deliberate -- a shared
x-axis is what makes panels comparable, and on the same data per-barcode spans
differ from the union by under 2% of positions. Paying comparability for 2% is
a bad trade.
"""

from __future__ import annotations

import pytest

from smftools.config.experiment_config import ExperimentConfig

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "field",
    ["hmm_clustermap_restrict_to_read_span", "spatial_clustermap_restrict_to_read_span"],
)
def test_read_span_cropping_is_on_by_default(field):
    assert getattr(ExperimentConfig(), field) is True


@pytest.mark.parametrize(
    "field",
    ["hmm_clustermap_restrict_to_read_span", "spatial_clustermap_restrict_to_read_span"],
)
def test_read_span_cropping_can_be_disabled(field, tmp_path):
    """The uncropped view stays reachable for anyone comparing to older plots."""
    path = tmp_path / "experiment_config.csv"
    path.write_text(f"variable,value,help,options,type\n{field},FALSE,,,bool\n", encoding="utf-8")
    cfg, _report = ExperimentConfig.from_csv(path)
    assert getattr(cfg, field) is False
