"""Demux status from `sequencing_summary.txt` (`EGL-29c`).

This route answers a subtly different question than the `BM` tag does -- a
score threshold rather than a classifier assertion -- so what these pin is
mostly that the difference stays visible: the provenance travels with the
value, low-confidence reads near the threshold stay identifiable, and this
route never silently overwrites the stronger one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smftools.informatics.sequencing_summary import (
    DEFAULT_END_SCORE_THRESHOLD,
    SOURCE,
    attach_demux_status,
    classify_end_status,
    find_sequencing_summary,
    read_demux_status,
)

pytestmark = pytest.mark.unit


# --- classification ----------------------------------------------------------


def test_both_ends_above_threshold_is_double():
    status, _ = classify_end_status(pd.Series([95.0]), pd.Series([90.0]), threshold=62.0)
    assert list(status) == ["double"]


def test_one_end_above_threshold_is_single():
    status, _ = classify_end_status(pd.Series([95.0]), pd.Series([45.0]), threshold=62.0)
    assert list(status) == ["single"]


def test_a_high_rear_alone_is_still_single():
    """Requiring *both* states the real criterion.

    Thresholding the rear alone happens to work on the motivating run because
    its front score is almost always high; a run where the front end is
    unreliable would then be silently counted as double.
    """
    status, _ = classify_end_status(pd.Series([20.0]), pd.Series([99.0]), threshold=62.0)
    assert list(status) == ["single"]


def test_neither_end_above_threshold_is_unclassified():
    status, _ = classify_end_status(pd.Series([10.0]), pd.Series([15.0]), threshold=62.0)
    assert list(status) == ["unclassified"]


def test_missing_scores_are_unclassified_with_no_confidence():
    status, confidence = classify_end_status(
        pd.Series([np.nan]), pd.Series([np.nan]), threshold=62.0
    )
    assert list(status) == ["unclassified"]
    assert list(confidence) == [0.0]


def test_an_unreadable_score_counts_as_not_barcoded():
    """Real summaries carry stray non-numeric values in these columns.

    Treating unreadable as "not barcoded at that end" is the conservative
    direction: it can only downgrade a read from double to single, never invent
    a double. The confidence is zeroed so such reads stay findable.
    """
    status, confidence = classify_end_status(pd.Series(["-"]), pd.Series(["95.0"]), threshold=62.0)
    assert list(status) == ["single"]
    assert list(confidence) == [0.0]


# --- confidence --------------------------------------------------------------


def test_scores_at_the_threshold_get_no_confidence():
    """Reads in the valley are exactly the ones the threshold cannot separate."""
    _status, confidence = classify_end_status(pd.Series([62.0]), pd.Series([62.0]), threshold=62.0)
    assert confidence.iloc[0] == pytest.approx(0.0)


def test_decisive_scores_get_high_confidence():
    _status, confidence = classify_end_status(
        pd.Series([100.0]), pd.Series([100.0]), threshold=62.0
    )
    assert confidence.iloc[0] > 0.9


def test_a_double_is_judged_by_its_weaker_end():
    """A read is only as convincingly double as its worst end."""
    _status, strong = classify_end_status(pd.Series([100.0]), pd.Series([100.0]), threshold=62.0)
    _status, marginal = classify_end_status(pd.Series([100.0]), pd.Series([63.0]), threshold=62.0)
    assert marginal.iloc[0] < strong.iloc[0]


# --- reading -----------------------------------------------------------------


def _write_summary(path, rows):
    frame = pd.DataFrame(
        rows,
        columns=["read_id", "barcode_arrangement", "barcode_front_score", "barcode_rear_score"],
    )
    frame.to_csv(path, sep="\t", index=False)
    return path


def test_reading_produces_status_and_provenance(tmp_path):
    path = _write_summary(
        tmp_path / "sequencing_summary_x.txt",
        [("r0", "barcode01", 99.0, 95.0), ("r1", "barcode01", 99.0, 40.0)],
    )
    result = read_demux_status(path, threshold=62.0)

    assert list(result["demux_type"]) == ["double", "single"]
    assert set(result["demux_type_source"]) == {SOURCE}
    assert "demux_type_confidence" in result.columns


def test_a_summary_without_per_end_scores_is_an_error(tmp_path):
    """Failing loudly beats silently classifying everything as unclassified."""
    path = tmp_path / "sequencing_summary_x.txt"
    pd.DataFrame({"read_id": ["r0"], "barcode_arrangement": ["barcode01"]}).to_csv(
        path, sep="\t", index=False
    )
    with pytest.raises(ValueError, match="per-end barcode scores"):
        read_demux_status(path)


def test_chunked_reading_matches_a_single_pass(tmp_path):
    """These files run to millions of rows, so chunking is not optional."""
    rows = [(f"r{index}", "barcode01", 99.0, 90.0 if index % 2 else 30.0) for index in range(50)]
    path = _write_summary(tmp_path / "sequencing_summary_x.txt", rows)
    whole = read_demux_status(path, chunk_size=10_000)
    chunked = read_demux_status(path, chunk_size=7)
    pd.testing.assert_frame_equal(whole, chunked)


# --- discovery ---------------------------------------------------------------


def test_summary_is_found_from_the_fastq_directory(tmp_path):
    """`input_data_path` naturally points at `fastq_pass/`, not the run root."""
    (tmp_path / "fastq_pass").mkdir()
    summary = tmp_path / "sequencing_summary_FBF1.txt"
    summary.write_text("read_id\n")
    assert find_sequencing_summary(tmp_path / "fastq_pass") == summary


def test_missing_summary_returns_none(tmp_path):
    assert find_sequencing_summary(tmp_path) is None


# --- attaching ---------------------------------------------------------------


def _status_frame(read_ids, kinds):
    return pd.DataFrame(
        {
            "read_id": read_ids,
            "demux_type": kinds,
            "demux_type_source": SOURCE,
            "demux_type_confidence": [0.5] * len(read_ids),
        }
    )


def test_attaching_fills_reads_and_records_provenance():
    obs = pd.DataFrame(index=["r0", "r1"])
    filled = attach_demux_status(obs, _status_frame(["r0", "r1"], ["double", "single"]))

    assert filled == 2
    assert list(obs["demux_type"]) == ["double", "single"]
    assert set(obs["demux_type_source"]) == {SOURCE}


def test_an_existing_demux_type_is_not_overwritten():
    """`BM` is a classifier assertion; this is a score threshold.

    Where both exist the assertion is the better evidence, so this route must
    defer rather than clobber it.
    """
    obs = pd.DataFrame({"demux_type": ["double"]}, index=["r0"])
    filled = attach_demux_status(obs, _status_frame(["r0"], ["single"]))

    assert filled == 0
    assert list(obs["demux_type"]) == ["double"]


def test_overwrite_is_available_when_asked():
    obs = pd.DataFrame({"demux_type": ["double"]}, index=["r0"])
    attach_demux_status(obs, _status_frame(["r0"], ["single"]), overwrite=True)
    assert list(obs["demux_type"]) == ["single"]


def test_reads_absent_from_the_summary_are_left_alone():
    obs = pd.DataFrame(index=["r0", "missing"])
    filled = attach_demux_status(obs, _status_frame(["r0"], ["double"]))

    assert filled == 1
    assert pd.isna(obs.loc["missing", "demux_type"]) or obs.loc["missing", "demux_type"] == ""


# --- config ------------------------------------------------------------------


def test_config_defaults():
    from smftools.config.experiment_config import ExperimentConfig

    cfg = ExperimentConfig()
    assert cfg.use_sequencing_summary_demux_status is True
    assert cfg.sequencing_summary_path is None
    assert cfg.barcode_end_score_threshold == DEFAULT_END_SCORE_THRESHOLD
