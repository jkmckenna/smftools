"""The single-threaded phase before the extraction pool must not run silent (`F48`)."""

from __future__ import annotations

import inspect

import pytest

from smftools.cli.raw_adata import (
    _build_ragged_records_streaming_convertible,
    _build_ragged_records_streaming_direct,
    _log_planning_progress,
)

pytestmark = pytest.mark.unit


def test_planning_progress_names_the_reference_and_its_cost(caplog):
    with caplog.at_level("INFO", logger="smftools.cli.raw_adata"):
        _log_planning_progress(
            "bucketing references",
            started_at=0.0,
            record="6B6_top",
            done=3,
            total=10,
            reads=124_500,
            buckets=32,
        )

    message = caplog.records[-1].message
    assert "3/10" in message
    assert "6B6_top" in message
    assert "124,500 reads" in message
    assert "32 bucket(s)" in message


def test_planning_progress_handles_the_phase_form(caplog):
    """The whole-BAM feature scan has no per-reference identity to report."""
    with caplog.at_level("INFO", logger="smftools.cli.raw_adata"):
        _log_planning_progress("scanning BAM for read features", started_at=0.0)

    assert "scanning BAM for read features" in caplog.records[-1].message


@pytest.mark.parametrize(
    "builder",
    [_build_ragged_records_streaming_convertible, _build_ragged_records_streaming_direct],
)
def test_both_extraction_paths_report_planning(builder):
    """Convertible and direct paths each have their own planning loop.

    Instrumenting one and not the other leaves the silence in place for
    whichever modality the run happens to use -- and the second loop was
    initially left with an undefined counter, which would have raised only on
    a direct-modality run.
    """
    source = inspect.getsource(builder)
    assert "_log_planning_progress" in source
    assert "planning_started" in source
