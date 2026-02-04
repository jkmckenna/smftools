import pytest

from smftools.informatics import bam_functions


def test_validate_umi_config_requires_adapters_when_enabled():
    with pytest.raises(ValueError, match="no UMI adapter sequences were provided"):
        bam_functions.validate_umi_config(True, [None, None], 8)


def test_validate_umi_config_requires_two_slot_adapter_list():
    with pytest.raises(ValueError, match="two-item list"):
        bam_functions.validate_umi_config(True, ["ACGT"], 10)


def test_validate_umi_config_accepts_directional_two_slot_adapters():
    adapters, length = bam_functions.validate_umi_config(True, ["ACGT", None], 10)
    assert adapters == ["ACGT", None]
    assert length == 10

    adapters, length = bam_functions.validate_umi_config(True, [None, "TTAA"], 12)
    assert adapters == [None, "TTAA"]
    assert length == 12


def test_extract_umi_from_read_start_reports_same_orientation():
    read = "ACGTAAACTGCTGATCGTAG"
    umi = bam_functions._extract_umi_adjacent_to_adapter_on_read_end(
        read_sequence=read,
        adapter_sequence="ACGT",
        umi_length=5,
        umi_search_window=10,
        search_from_start=True,
    )
    assert umi == "AAACT"


def test_extract_umi_from_read_end_reports_same_orientation():
    read = "GATTACAACCCCGGGTTTT"
    umi = bam_functions._extract_umi_adjacent_to_adapter_on_read_end(
        read_sequence=read,
        adapter_sequence="ACGT",
        umi_length=4,
        umi_search_window=10,
        search_from_start=False,
    )
    assert umi is None


def test_extract_umi_from_read_end_with_match():
    read = "GATTACAACCCCGGGTTTT"
    umi = bam_functions._extract_umi_adjacent_to_adapter_on_read_end(
        read_sequence=read,
        adapter_sequence="GGG",
        umi_length=4,
        umi_search_window=10,
        search_from_start=False,
    )
    assert umi == "CCCC"


def test_extract_umi_respects_search_window():
    read = "TTTTACGTAAAATTTT"
    umi = bam_functions._extract_umi_adjacent_to_adapter_on_read_end(
        read_sequence=read,
        adapter_sequence="ACGT",
        umi_length=4,
        umi_search_window=1,
        search_from_start=True,
    )
    assert umi is None


def test_extract_umi_uses_adapter_occurrence_nearest_targeted_end():
    read = "NNNNNNNNACGTAAAATTTTACGTGGGG"
    umi = bam_functions._extract_umi_adjacent_to_adapter_on_read_end(
        read_sequence=read,
        adapter_sequence="ACGT",
        umi_length=4,
        umi_search_window=10,
        search_from_start=False,
    )
    assert umi == "GGGG"


def test_extract_umi_rejects_unknown_matcher():
    with pytest.raises(ValueError, match="adapter_matcher must be one of"):
        bam_functions._extract_umi_adjacent_to_adapter_on_read_end(
            read_sequence="ACGTAAAA",
            adapter_sequence="ACGT",
            umi_length=4,
            umi_search_window=10,
            search_from_start=True,
            adapter_matcher="unknown",
        )


def test_extract_umi_can_use_edlib_matcher(monkeypatch):
    class _FakeEdlib:
        @staticmethod
        def align(_query, _target, mode, task, k):
            assert mode == "HW"
            assert task == "locations"
            assert k == 1
            return {"editDistance": 1, "locations": [(0, 3)]}

    monkeypatch.setattr(bam_functions, "require", lambda *args, **kwargs: _FakeEdlib())
    umi = bam_functions._extract_umi_adjacent_to_adapter_on_read_end(
        read_sequence="ACGTAAAA",
        adapter_sequence="ACGA",
        umi_length=4,
        umi_search_window=10,
        search_from_start=True,
        adapter_matcher="edlib",
        adapter_max_edits=1,
    )
    assert umi == "AAAA"


def test_target_read_end_for_ref_side_respects_strand():
    assert bam_functions._target_read_end_for_ref_side(False, "left") == "start"
    assert bam_functions._target_read_end_for_ref_side(False, "right") == "end"
    assert bam_functions._target_read_end_for_ref_side(True, "left") == "end"
    assert bam_functions._target_read_end_for_ref_side(True, "right") == "start"
