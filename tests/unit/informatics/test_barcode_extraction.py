"""Tests for barcode extraction functions."""

import numpy as np
import pandas as pd
import pytest

from smftools.informatics import bam_functions
from smftools.informatics.h5ad_functions import add_demux_type_from_bm_tag


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


class TestLoadBarcodeReferencesFromYaml:
    """Tests for loading barcode references from YAML files."""

    def test_load_flat_structure(self, tmp_path):
        """Test loading from flat YAML structure."""
        yaml_file = tmp_path / "barcodes.yaml"
        yaml_file.write_text(
            "barcode01: ACGTACGT\nbarcode02: TGCATGCA\nbarcode03: AAAACCCC\n"
        )
        refs, length = bam_functions.load_barcode_references_from_yaml(yaml_file)
        assert refs == {
            "barcode01": "ACGTACGT",
            "barcode02": "TGCATGCA",
            "barcode03": "AAAACCCC",
        }
        assert length == 8

    def test_load_nested_structure(self, tmp_path):
        """Test loading from nested YAML structure with 'barcodes' key."""
        yaml_file = tmp_path / "barcodes.yaml"
        yaml_file.write_text(
            "barcodes:\n  barcode01: ACGTACGT\n  barcode02: TGCATGCA\n"
        )
        refs, length = bam_functions.load_barcode_references_from_yaml(yaml_file)
        assert refs == {
            "barcode01": "ACGTACGT",
            "barcode02": "TGCATGCA",
        }
        assert length == 8

    def test_sequences_uppercased(self, tmp_path):
        """Test that sequences are uppercased."""
        yaml_file = tmp_path / "barcodes.yaml"
        yaml_file.write_text("barcode01: acgtacgt\nbarcode02: TgCaTgCa\n")
        refs, length = bam_functions.load_barcode_references_from_yaml(yaml_file)
        assert refs["barcode01"] == "ACGTACGT"
        assert refs["barcode02"] == "TGCATGCA"
        assert length == 8

    def test_barcode_length_derived(self, tmp_path):
        """Test that barcode length is correctly derived."""
        yaml_file = tmp_path / "barcodes.yaml"
        yaml_file.write_text("barcode01: ACGTACGTACGTACGT\n")  # 16 bases
        refs, length = bam_functions.load_barcode_references_from_yaml(yaml_file)
        assert length == 16

    def test_inconsistent_lengths_raises(self, tmp_path):
        """Test that inconsistent barcode lengths raise ValueError."""
        yaml_file = tmp_path / "inconsistent.yaml"
        yaml_file.write_text("barcode01: ACGT\nbarcode02: ACGTACGT\n")  # 4 vs 8
        with pytest.raises(ValueError, match="inconsistent lengths"):
            bam_functions.load_barcode_references_from_yaml(yaml_file)

    def test_file_not_found_raises(self):
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError, match="Barcode YAML file not found"):
            bam_functions.load_barcode_references_from_yaml("/nonexistent/path.yaml")

    def test_empty_file_raises(self, tmp_path):
        """Test that ValueError is raised for empty file."""
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")
        with pytest.raises(ValueError, match="empty"):
            bam_functions.load_barcode_references_from_yaml(yaml_file)

    def test_no_valid_barcodes_raises(self, tmp_path):
        """Test that ValueError is raised when no valid barcodes found."""
        yaml_file = tmp_path / "invalid.yaml"
        # Use only non-string values or nested dicts to avoid matching as sequences
        yaml_file.write_text("config:\n  option: 123\ncount: 456\n")
        with pytest.raises(ValueError, match="No valid barcode sequences found"):
            bam_functions.load_barcode_references_from_yaml(yaml_file)

    def test_invalid_characters_raises(self, tmp_path):
        """Test that ValueError is raised for invalid DNA characters."""
        yaml_file = tmp_path / "invalid_chars.yaml"
        yaml_file.write_text("barcode01: ACGTXYZ\n")
        with pytest.raises(ValueError, match="invalid characters"):
            bam_functions.load_barcode_references_from_yaml(yaml_file)

    def test_n_bases_allowed(self, tmp_path):
        """Test that N bases are allowed in sequences."""
        yaml_file = tmp_path / "with_n.yaml"
        yaml_file.write_text("barcode01: ACNTNACGT\n")
        refs, length = bam_functions.load_barcode_references_from_yaml(yaml_file)
        assert refs["barcode01"] == "ACNTNACGT"
        assert length == 9

    def test_non_string_value_skipped(self, tmp_path):
        """Test that non-string values are skipped (results in no valid barcodes)."""
        yaml_file = tmp_path / "non_string.yaml"
        yaml_file.write_text("barcode01: 12345\nbarcode02: 67890\n")
        # Non-string values are filtered out, resulting in no valid barcodes
        with pytest.raises(ValueError, match="No valid barcode sequences found"):
            bam_functions.load_barcode_references_from_yaml(yaml_file)


class TestAddDemuxTypeFromBmTag:
    """Tests for deriving demux_type from BM tag."""

    def _make_adata(self, bm_values):
        """Create a mock AnnData with BM column."""
        import anndata as ad

        n = len(bm_values)
        adata = ad.AnnData(X=np.zeros((n, 5)))
        adata.obs["BM"] = bm_values
        adata.obs_names = [f"read_{i}" for i in range(n)]
        return adata

    def test_both_becomes_double(self):
        """Test that BM='both' maps to demux_type='double'."""
        adata = self._make_adata(["both", "both", "both"])
        add_demux_type_from_bm_tag(adata)
        assert all(adata.obs["demux_type"] == "double")

    def test_left_only_becomes_single(self):
        """Test that BM='left_only' maps to demux_type='single'."""
        adata = self._make_adata(["left_only", "left_only"])
        add_demux_type_from_bm_tag(adata)
        assert all(adata.obs["demux_type"] == "single")

    def test_right_only_becomes_single(self):
        """Test that BM='right_only' maps to demux_type='single'."""
        adata = self._make_adata(["right_only", "right_only"])
        add_demux_type_from_bm_tag(adata)
        assert all(adata.obs["demux_type"] == "single")

    def test_mismatch_becomes_unclassified(self):
        """Test that BM='mismatch' maps to demux_type='unclassified'."""
        adata = self._make_adata(["mismatch", "mismatch"])
        add_demux_type_from_bm_tag(adata)
        assert all(adata.obs["demux_type"] == "unclassified")

    def test_unclassified_stays_unclassified(self):
        """Test that BM='unclassified' maps to demux_type='unclassified'."""
        adata = self._make_adata(["unclassified", "unclassified"])
        add_demux_type_from_bm_tag(adata)
        assert all(adata.obs["demux_type"] == "unclassified")

    def test_mixed_values(self):
        """Test correct mapping with mixed BM values."""
        adata = self._make_adata(["both", "left_only", "right_only", "mismatch", "unclassified"])
        add_demux_type_from_bm_tag(adata)
        expected = ["double", "single", "single", "unclassified", "unclassified"]
        assert list(adata.obs["demux_type"]) == expected

    def test_case_insensitive(self):
        """Test that BM matching is case-insensitive."""
        adata = self._make_adata(["BOTH", "Left_Only", "RIGHT_ONLY"])
        add_demux_type_from_bm_tag(adata)
        expected = ["double", "single", "single"]
        assert list(adata.obs["demux_type"]) == expected

    def test_missing_bm_column_warns(self):
        """Test that missing BM column sets demux_type to 'unknown'."""
        import anndata as ad

        adata = ad.AnnData(X=np.zeros((3, 5)))
        adata.obs_names = ["read_0", "read_1", "read_2"]
        add_demux_type_from_bm_tag(adata, bm_column="BM")
        assert all(adata.obs["demux_type"] == "unknown")
