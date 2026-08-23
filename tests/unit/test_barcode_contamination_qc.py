"""Spike-in barcode contamination QC (`EGL-31`)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from smftools.preprocessing.barcode_contamination_qc import (
    ContaminationQCError,
    barcode_contamination_report,
    contamination_by_barcode,
    end_disagreement,
    mislabeling_by_demux_type,
    poisson_interval,
    spike_in_mask,
    write_barcode_contamination_qc,
)

pytestmark = pytest.mark.unit

SPIKE = ["ctcf_mNanog"]


def _obs(rows):
    return pd.DataFrame(rows)


def _row(reference, assigned, demux_type="double", front="", rear=""):
    return {
        "Reference_strand": reference,
        "barcode_assigned": assigned,
        "demux_type": demux_type,
        "barcode_front": front,
        "barcode_rear": rear,
    }


def _run(*, clean_spike, single_errors, double_errors, library_per_barcode):
    """A synthetic run whose contamination rate is known by construction."""
    rows = []
    for index in range(clean_spike):
        rows.append(
            _row(f"ctcf_mNanog_{'top' if index % 2 else 'bottom'}", "unclassified", "single")
        )
    for index in range(single_errors):
        rows.append(_row("ctcf_mNanog_top", f"NB{index % 2 + 1:02d}", "single"))
    for index in range(double_errors):
        rows.append(_row("ctcf_mNanog_top", "NB01", "double"))
    for barcode, count in library_per_barcode.items():
        for _ in range(count):
            rows.append(_row("6B6_top", barcode, "double"))
    return _obs(rows)


# --- the spike-in is identified by amplicon, not by strand ---------------------


def test_spike_in_matches_both_strands_of_the_named_amplicon():
    """A spike-in is named as an amplicon; references are stored per strand.

    Requiring both strands in config would fail silently on the one forgotten.
    """
    obs = _obs(
        [
            _row("ctcf_mNanog_top", "NB01"),
            _row("ctcf_mNanog_bottom", "NB01"),
            _row("6B6_top", "NB01"),
        ]
    )
    assert list(spike_in_mask(obs, SPIKE)) == [True, True, False]


def test_unconfigured_spike_in_matches_nothing():
    obs = _obs([_row("ctcf_mNanog_top", "NB01")])
    assert not spike_in_mask(obs, []).any()


# --- rates ---------------------------------------------------------------------


def test_contamination_rate_uses_the_whole_spike_in_population():
    """The denominator is every spike-in read, not just the contaminated ones.

    Counting only barcoded spike-in reads would report 100% contamination on any
    input, which is why the unclassified population has to be ingested.
    """
    obs = _run(
        clean_spike=900, single_errors=90, double_errors=10, library_per_barcode={"NB01": 1000}
    )

    summary = mislabeling_by_demux_type(obs, spike_in_references=SPIKE)

    assert summary["spike_in_reads"] == 1000
    assert summary["mislabeled_reads"] == 100
    assert summary["contamination_rate"] == pytest.approx(0.1)


def test_single_ended_discrimination_is_reported():
    """The number that licenses filtering on double-ended assignment."""
    obs = _run(
        clean_spike=900, single_errors=90, double_errors=10, library_per_barcode={"NB01": 1000}
    )

    summary = mislabeling_by_demux_type(obs, spike_in_references=SPIKE)
    by_type = summary["by_demux_type"]

    # 90 of 990 single-ended spike reads are errors; 10 of 10 double-ended.
    assert by_type["single"]["mislabeled_reads"] == 90
    assert by_type["double"]["mislabeled_reads"] == 10
    assert summary["single_over_double_discrimination"] == pytest.approx(
        by_type["single"]["mislabeling_rate"] / by_type["double"]["mislabeling_rate"]
    )


def test_enrichment_is_against_library_share_not_raw_count():
    """Contamination scales with library size, so raw counts rank the wrong thing.

    NB02 contributes a tenth of NB01's library but the same number of spike-in
    reads, so it is the more contaminating barcode despite an equal count.
    """
    rows = []
    rows += [_row("ctcf_mNanog_top", "NB01", "single")] * 10
    rows += [_row("ctcf_mNanog_top", "NB02", "single")] * 10
    rows += [_row("6B6_top", "NB01", "double")] * 10000
    rows += [_row("6B6_top", "NB02", "double")] * 1000

    frame = contamination_by_barcode(_obs(rows), spike_in_references=SPIKE)

    assert frame.loc["NB01", "spike_in_reads"] == frame.loc["NB02", "spike_in_reads"]
    assert frame.loc["NB02", "enrichment"] > frame.loc["NB01", "enrichment"]
    assert frame.loc["NB01", "enrichment"] == pytest.approx(0.55, abs=0.01)


def test_thin_counts_get_intervals_that_span_the_null():
    """Twenty reads cannot distinguish an enrichment of 1.3 from 1.0.

    Presenting a point estimate at this depth invites exactly the over-reading
    this QC exists to prevent, so the interval has to say so.
    """
    rows = []
    rows += [_row("ctcf_mNanog_top", "NB01", "single")] * 13
    rows += [_row("ctcf_mNanog_top", "NB02", "single")] * 10
    rows += [_row("6B6_top", "NB01", "double")] * 1000
    rows += [_row("6B6_top", "NB02", "double")] * 1000

    frame = contamination_by_barcode(_obs(rows), spike_in_references=SPIKE)

    assert frame.loc["NB01", "enrichment"] > 1.0
    assert frame.loc["NB01", "enrichment_low"] < 1.0 < frame.loc["NB01", "enrichment_high"]
    assert not frame.loc["NB01", "significant"]


# --- end disagreement needs no spike-in ----------------------------------------


def test_end_disagreement_counts_reads_whose_ends_name_different_barcodes():
    obs = _obs(
        [
            _row("6B6_top", "NB01", front="NB01", rear="NB01"),
            _row("6B6_top", "NB01", front="NB01", rear="NB47"),
            _row("6B6_top", "NB01", front="NB01", rear=""),
        ]
    )

    result = end_disagreement(obs)

    # The read with only one end called is not comparable, not agreeing.
    assert result["comparable_reads"] == 2
    assert result["disagreeing_reads"] == 1
    assert result["disagreement_rate"] == pytest.approx(0.5)
    assert result["top_pairs"][0] == {"front": "NB01", "rear": "NB47", "reads": 1}


def test_end_disagreement_runs_without_any_spike_in():
    """It is the measure that applies to the whole run rather than 0.1% of it."""
    obs = _obs([_row("6B6_top", "NB01", front="NB01", rear="NB47")])
    _per_barcode, summary = barcode_contamination_report(obs, spike_in_references=[])

    assert summary["spike_in"] is None
    assert summary["end_disagreement"]["disagreeing_reads"] == 1


# --- refusing to measure dishonestly -------------------------------------------


def test_a_pre_f35_store_is_refused_rather_than_measured():
    """Before `F35` the assignment columns were one collapsed value.

    Falling back to it would reproduce the vacuous self-comparison the fix
    removed, so the QC refuses and says what to rebuild.
    """
    obs = _obs([{"Reference_strand": "ctcf_mNanog_top", "barcode": "NB01", "demux_type": "single"}])

    with pytest.raises(ContaminationQCError, match="raw identity schema 2"):
        contamination_by_barcode(obs, spike_in_references=SPIKE)


def test_poisson_interval_brackets_the_count():
    for count in (0, 1, 5, 100, 1268):
        low, high = poisson_interval(count)
        assert low <= count <= high


# --- the written artifacts -----------------------------------------------------


def test_writer_emits_a_table_and_a_summary(tmp_path):
    obs = _run(
        clean_spike=900, single_errors=90, double_errors=10, library_per_barcode={"NB01": 1000}
    )

    paths = write_barcode_contamination_qc(obs, tmp_path, spike_in_references=SPIKE)

    table = pd.read_parquet(paths["per_barcode"])
    assert "barcode" in table.columns and "enrichment_low" in table.columns
    summary = json.loads(paths["summary"].read_text())
    assert summary["spike_in"]["contamination_rate"] == pytest.approx(0.1)
    assert summary["spike_in_references"] == SPIKE
