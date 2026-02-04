"""Tests for barcode extraction functions."""

import pytest

from smftools.informatics import bam_functions


class TestExtractBarcodeAdjacentToAdapter:
    """Tests for barcode extraction from read sequences."""

    def test_extract_barcode_from_read_start(self):
        """Test barcode extraction when adapter is at read start."""
        read = "ACGTAAAACCCCGGGGTTTT"
        bc, pos = bam_functions._extract_barcode_adjacent_to_adapter_on_read_end(
            read_sequence=read,
            adapter_sequence="ACGT",
            barcode_length=4,
            barcode_search_window=10,
            search_from_start=True,
            adapter_matcher="exact",
        )
        assert bc == "AAAA"
        assert pos == 0

    def test_extract_barcode_from_read_end(self):
        """Test barcode extraction when adapter is at read end."""
        read = "TTTTCCCCGGGGACGT"
        bc, pos = bam_functions._extract_barcode_adjacent_to_adapter_on_read_end(
            read_sequence=read,
            adapter_sequence="ACGT",
            barcode_length=4,
            barcode_search_window=10,
            search_from_start=False,
            adapter_matcher="exact",
        )
        assert bc == "GGGG"
        assert pos == 12

    def test_extract_barcode_no_adapter_found(self):
        """Test that None is returned when adapter not found."""
        read = "TTTTCCCCGGGGAAAA"
        bc, pos = bam_functions._extract_barcode_adjacent_to_adapter_on_read_end(
            read_sequence=read,
            adapter_sequence="ACGT",
            barcode_length=4,
            barcode_search_window=10,
            search_from_start=True,
            adapter_matcher="exact",
        )
        assert bc is None
        assert pos is None

    def test_extract_barcode_outside_search_window(self):
        """Test that adapter outside search window is not found."""
        read = "TTTTTTTTTTACGTAAAA"
        bc, pos = bam_functions._extract_barcode_adjacent_to_adapter_on_read_end(
            read_sequence=read,
            adapter_sequence="ACGT",
            barcode_length=4,
            barcode_search_window=5,
            search_from_start=True,
            adapter_matcher="exact",
        )
        assert bc is None
        assert pos is None

    def test_extract_barcode_insufficient_length(self):
        """Test that None is returned when barcode region is too short."""
        read = "ACGTAA"  # Only 2 bases after adapter
        bc, pos = bam_functions._extract_barcode_adjacent_to_adapter_on_read_end(
            read_sequence=read,
            adapter_sequence="ACGT",
            barcode_length=4,
            barcode_search_window=10,
            search_from_start=True,
            adapter_matcher="exact",
        )
        assert bc is None
        assert pos is None

    def test_extract_barcode_empty_sequence(self):
        """Test handling of empty read sequence."""
        bc, pos = bam_functions._extract_barcode_adjacent_to_adapter_on_read_end(
            read_sequence="",
            adapter_sequence="ACGT",
            barcode_length=4,
            barcode_search_window=10,
            search_from_start=True,
            adapter_matcher="exact",
        )
        assert bc is None
        assert pos is None

    def test_extract_barcode_rejects_unknown_matcher(self):
        """Test that unknown matcher raises ValueError."""
        with pytest.raises(ValueError, match="adapter_matcher must be one of"):
            bam_functions._extract_barcode_adjacent_to_adapter_on_read_end(
                read_sequence="ACGTAAAA",
                adapter_sequence="ACGT",
                barcode_length=4,
                barcode_search_window=10,
                search_from_start=True,
                adapter_matcher="invalid",
            )


class TestMatchBarcodeToReferences:
    """Tests for matching extracted barcodes to reference set."""

    def test_exact_match(self):
        """Test exact barcode matching."""
        references = {
            "barcode01": "AAAA",
            "barcode02": "CCCC",
            "barcode03": "GGGG",
        }
        name, dist = bam_functions._match_barcode_to_references(
            "AAAA", references, max_edit_distance=1
        )
        assert name == "barcode01"
        assert dist == 0

    def test_match_with_one_edit(self):
        """Test barcode matching with one edit distance."""
        references = {
            "barcode01": "AAAA",
            "barcode02": "CCCC",
            "barcode03": "GGGG",
        }
        name, dist = bam_functions._match_barcode_to_references(
            "AAAT", references, max_edit_distance=1
        )
        assert name == "barcode01"
        assert dist == 1

    def test_no_match_exceeds_threshold(self):
        """Test that None is returned when no barcode matches within threshold."""
        references = {
            "barcode01": "AAAA",
            "barcode02": "CCCC",
        }
        name, dist = bam_functions._match_barcode_to_references(
            "TTTT", references, max_edit_distance=1
        )
        assert name is None
        assert dist is None

    def test_best_match_selected(self):
        """Test that the best matching barcode is selected."""
        references = {
            "barcode01": "AAAA",
            "barcode02": "AAAT",  # 1 edit from AAAG
            "barcode03": "AAAG",  # exact match
        }
        name, dist = bam_functions._match_barcode_to_references(
            "AAAG", references, max_edit_distance=2
        )
        assert name == "barcode03"
        assert dist == 0

    def test_empty_barcode(self):
        """Test handling of empty extracted barcode."""
        references = {"barcode01": "AAAA"}
        name, dist = bam_functions._match_barcode_to_references(
            "", references, max_edit_distance=1
        )
        assert name is None
        assert dist is None

    def test_empty_references(self):
        """Test handling of empty reference set."""
        name, dist = bam_functions._match_barcode_to_references(
            "AAAA", {}, max_edit_distance=1
        )
        assert name is None
        assert dist is None

    def test_case_insensitive_matching(self):
        """Test that matching is case-insensitive."""
        references = {"barcode01": "aaaa"}
        name, dist = bam_functions._match_barcode_to_references(
            "AAAA", references, max_edit_distance=0
        )
        assert name == "barcode01"
        assert dist == 0

    def test_multiple_close_matches_returns_best(self):
        """Test that when multiple barcodes are close, the best is returned."""
        references = {
            "barcode01": "AAAA",  # 2 edits from AATT
            "barcode02": "AATT",  # exact match
            "barcode03": "TTTT",  # 2 edits from AATT
        }
        name, dist = bam_functions._match_barcode_to_references(
            "AATT", references, max_edit_distance=2
        )
        assert name == "barcode02"
        assert dist == 0
