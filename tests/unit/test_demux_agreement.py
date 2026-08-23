"""Assigned-vs-re-derived barcode agreement (`EGL-29b`).

When FASTQs arrive already demultiplexed, the directory carries the barcode and
the end reason has to be re-derived from sequence -- which yields a second,
independent barcode call. Reporting where the two differ is the point: a silent
preference for either throws away the only signal that the assignment, the kit,
or the extraction parameters are wrong.
"""

from __future__ import annotations

import pandas as pd
import pytest

from smftools.informatics.demux_agreement import (
    AGREE,
    AGREEMENT_COLUMN,
    ASSIGNED_UNCLASSIFIED,
    DISAGREE,
    NOT_COMPARED,
    REDERIVED_UNCLASSIFIED,
    classify_agreement,
    report_barcode_agreement,
    summarize_agreement,
)

pytestmark = pytest.mark.unit


def _series(values):
    return pd.Series(values, index=[f"r{index}" for index in range(len(values))])


def test_matching_barcodes_agree():
    result = classify_agreement(_series(["bc01", "bc02"]), _series(["bc01", "bc02"]))
    assert list(result) == [AGREE, AGREE]


def test_differing_barcodes_disagree():
    result = classify_agreement(_series(["bc01"]), _series(["bc02"]))
    assert list(result) == [DISAGREE]


def test_comparison_ignores_case_and_whitespace():
    """Cosmetic differences are not evidence of a misassignment."""
    result = classify_agreement(_series([" BC01 "]), _series(["bc01"]))
    assert list(result) == [AGREE]


def test_the_two_kinds_of_missing_are_kept_distinct():
    """They mean different things operationally.

    A re-derivation that found nothing is a sensitivity problem; one that found
    a *different* barcode is a correctness problem. Only the second calls the
    assignment into question, so collapsing them to a boolean would hide which
    is happening.
    """
    assigned = _series(["bc01", "unclassified", "bc03"])
    rederived = _series(["unclassified", "bc02", "bc03"])
    assert list(classify_agreement(assigned, rederived)) == [
        REDERIVED_UNCLASSIFIED,
        ASSIGNED_UNCLASSIFIED,
        AGREE,
    ]


@pytest.mark.parametrize("blank", ["", "nan", "none", "unknown", "unassigned"])
def test_blank_forms_count_as_unclassified(blank):
    result = classify_agreement(_series(["bc01"]), _series([blank]))
    assert list(result) == [REDERIVED_UNCLASSIFIED]


def test_both_missing_is_not_compared():
    result = classify_agreement(_series(["unclassified"]), _series([""]))
    assert list(result) == [NOT_COMPARED]


# --- summary -----------------------------------------------------------------


def test_rate_is_over_comparable_reads_not_all_reads():
    """Reads with nothing to compare must not dilute the rate.

    Counting them as agreement would make a real problem look small exactly
    when re-derivation is failing most often.
    """
    assigned = _series(["bc01", "bc01", "bc02", "unclassified"])
    rederived = _series(["bc01", "bc02", "unclassified", "unclassified"])
    agreement = classify_agreement(assigned, rederived)

    summary = summarize_agreement(agreement, assigned, rederived)

    assert summary["comparable"] == 2
    assert summary["disagree"] == 1
    assert summary["disagreement_rate"] == pytest.approx(0.5)
    assert summary["reads"] == 4


def test_confusions_are_reported_so_the_pattern_is_visible():
    """A rate says something is wrong; a concentrated pair says what."""
    assigned = _series(["bc01"] * 5 + ["bc07"])
    rederived = _series(["bc02"] * 5 + ["bc09"])
    summary = summarize_agreement(classify_agreement(assigned, rederived), assigned, rederived)

    assert summary["top_confusions"][0] == {"assigned": "bc01", "rederived": "bc02", "reads": 5}


def test_perfect_agreement_reports_no_confusions():
    assigned = _series(["bc01", "bc02"])
    summary = summarize_agreement(classify_agreement(assigned, assigned), assigned, assigned)
    assert summary["disagreement_rate"] == 0.0
    assert summary["top_confusions"] == []


# --- the obs-level entry point -----------------------------------------------


def test_report_adds_the_column_and_returns_a_summary():
    obs = pd.DataFrame(
        {"barcode_assigned": ["bc01", "bc01"], "barcode_rederived": ["bc01", "bc02"]}
    )
    summary = report_barcode_agreement(obs)

    assert AGREEMENT_COLUMN in obs.columns
    assert summary["disagree"] == 1


def test_report_does_not_change_the_assigned_barcode():
    """Disagreement is reported, never resolved -- the assignment is authoritative."""
    obs = pd.DataFrame({"barcode_assigned": ["bc01"], "barcode_rederived": ["bc02"]})
    report_barcode_agreement(obs)
    assert list(obs["barcode_assigned"]) == ["bc01"]


def test_no_second_assignment_is_not_an_error():
    """The normal case for input that was never re-demultiplexed."""
    obs = pd.DataFrame({"barcode_assigned": ["bc01"]})
    assert report_barcode_agreement(obs) is None
    assert AGREEMENT_COLUMN not in obs.columns


def test_high_disagreement_warns(caplog):
    """A high rate means every downstream per-sample number is suspect."""
    obs = pd.DataFrame({"barcode_assigned": ["bc01"] * 10, "barcode_rederived": ["bc02"] * 10})
    with caplog.at_level("WARNING"):
        report_barcode_agreement(obs, warn_above=0.01)
    assert any("disagree" in record.message for record in caplog.records)


def test_agreement_below_the_threshold_does_not_warn(caplog):
    obs = pd.DataFrame({"barcode_assigned": ["bc01"] * 10, "barcode_rederived": ["bc01"] * 10})
    with caplog.at_level("WARNING"):
        report_barcode_agreement(obs, warn_above=0.01)
    assert not [record for record in caplog.records if record.levelname == "WARNING"]


# --- config ------------------------------------------------------------------


def test_status_rederivation_is_a_separate_flag_from_already_demuxed():
    """`input_already_demuxed` means "do not demux" and stays true here.

    Overloading it would make "keep my barcodes" and "re-scan sequences"
    inexpressible together, which is exactly the combination this needs.
    """
    from smftools.config.experiment_config import ExperimentConfig

    cfg = ExperimentConfig()
    assert cfg.input_already_demuxed is False
    assert cfg.derive_demux_status_from_sequence is False
    assert cfg.barcode_disagreement_warn_fraction == 0.01

    parsed, _ = ExperimentConfig.from_var_dict(
        {"input_already_demuxed": "TRUE", "derive_demux_status_from_sequence": "TRUE"}
    )
    assert parsed.input_already_demuxed is True
    assert parsed.derive_demux_status_from_sequence is True


# --- when the status gets re-derived (`EGL-29a`) ------------------------------


def _cfg(**overrides):
    from types import SimpleNamespace

    base = dict(
        barcode_kit="SQK-NBD114-24",
        input_already_demuxed=False,
        derive_demux_status_from_sequence=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_status_is_derived_on_a_normal_smftools_demux():
    from smftools.informatics.demux_agreement import should_derive_demux_status

    assert should_derive_demux_status(_cfg(), "smftools") is True


def test_already_demuxed_input_skips_derivation_by_default():
    """Unchanged behaviour: nothing re-scans sequences unless asked."""
    from smftools.informatics.demux_agreement import should_derive_demux_status

    assert should_derive_demux_status(_cfg(input_already_demuxed=True), "smftools") is False


def test_the_new_flag_enables_derivation_on_already_demuxed_input():
    """The case the lane exists for: keep the barcodes, recover the end reason."""
    from smftools.informatics.demux_agreement import should_derive_demux_status

    cfg = _cfg(input_already_demuxed=True, derive_demux_status_from_sequence=True)
    assert should_derive_demux_status(cfg, "smftools") is True


def test_derivation_requires_a_barcode_kit():
    """There is nothing to match sequences against without one."""
    from smftools.informatics.demux_agreement import should_derive_demux_status

    cfg = _cfg(barcode_kit=None, derive_demux_status_from_sequence=True)
    assert should_derive_demux_status(cfg, "smftools") is False


def test_dorado_cannot_rederive_on_already_demuxed_input():
    """The new flag is specific to the sequence scanner.

    Dorado has no second pass to attach an end reason to, so honouring the flag
    there would promise something the backend cannot deliver.
    """
    from smftools.informatics.demux_agreement import should_derive_demux_status

    cfg = _cfg(input_already_demuxed=True, derive_demux_status_from_sequence=True)
    assert should_derive_demux_status(cfg, "dorado", dorado_supports=True) is False


def test_old_dorado_does_not_derive():
    from smftools.informatics.demux_agreement import should_derive_demux_status

    assert should_derive_demux_status(_cfg(), "dorado", dorado_supports=False) is False
    assert should_derive_demux_status(_cfg(), "dorado", dorado_supports=True) is True


def test_unknown_backend_does_not_derive():
    from smftools.informatics.demux_agreement import should_derive_demux_status

    assert should_derive_demux_status(_cfg(), "somethingelse") is False
