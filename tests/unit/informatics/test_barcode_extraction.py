"""Tests for barcode extraction functions."""

import numpy as np
import pandas as pd
import pytest

from smftools.informatics import bam_functions
from smftools.informatics.bam_functions import (
    BarcodeKitConfig,
    FlankingConfig,
    PerEndFlankingConfig,
    UMIKitConfig,
    _build_flanking_from_adapters,
    _extract_sequence_with_flanking,
    _parse_flanking_config_from_dict,
    _parse_per_end_flanking,
    load_umi_config_from_yaml,
    resolve_barcode_config,
    resolve_umi_config,
)
from smftools.informatics.h5ad_functions import add_demux_type_from_bm_tag

# ---------------------------------------------------------------------------
# BAM helpers for integration tests (require pysam)
# ---------------------------------------------------------------------------
try:
    import pysam as _pysam

    HAS_PYSAM = True
except ImportError:
    _pysam = None
    HAS_PYSAM = False

requires_pysam = pytest.mark.skipif(not HAS_PYSAM, reason="pysam not installed")


def _create_test_bam(bam_path, reads, ref_name="chr1", ref_length=10000):
    """Create a coordinate-sorted, indexed BAM file.

    *reads* is a list of dicts with keys ``name``, ``sequence``, and
    optionally ``is_reverse`` (default ``False``).
    """
    header = {
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": ref_name, "LN": ref_length}],
    }
    with _pysam.AlignmentFile(str(bam_path), "wb", header=header) as outf:
        for i, info in enumerate(reads):
            a = _pysam.AlignedSegment()
            a.query_name = info.get("name", f"read_{i}")
            a.query_sequence = info["sequence"]
            a.flag = 16 if info.get("is_reverse", False) else 0
            a.reference_id = 0
            a.reference_start = i * 100
            a.mapping_quality = 60
            n = len(info["sequence"])
            a.cigartuples = [(0, n)]
            a.query_qualities = _pysam.qualitystring_to_array("I" * n)
            outf.write(a)
    _pysam.index(str(bam_path))


def _read_bam_tags(bam_path):
    """Return ``{read_name: {tag: value}}`` for every read in *bam_path*."""
    out = {}
    with _pysam.AlignmentFile(str(bam_path), "rb") as fh:
        for read in fh.fetch(until_eof=True):
            out[read.query_name] = dict(read.get_tags())
    return out


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


class TestExtractSequenceWithFlanking:
    """Tests for flanking-based sequence extraction."""

    def test_adapter_only_mode_from_start(self):
        """Find adapter at start, extract barcode after it."""
        # Structure: ADAPTER + BARCODE + rest
        read = "AAGGTTAACACAAAGACACCGACAACTTTCTTNNNNNNNN"
        flanking = FlankingConfig(adapter_side="AAGGTTAA")
        bc, start, end = _extract_sequence_with_flanking(
            read_sequence=read,
            target_length=24,
            search_window=50,
            search_from_start=True,
            flanking=flanking,
            flank_mode="adapter_only",
            adapter_matcher="exact",
        )
        assert bc == "CACAAAGACACCGACAACTTTCTT"[:24]
        assert start == 8
        assert end == 32

    def test_adapter_only_mode_from_end(self):
        """Find adapter at end, extract barcode before it."""
        # Structure: rest + BARCODE + ADAPTER
        # NNNNNNNN(8) CCCC(12) AAAA(16) TTTT(20) GGGG(24) AAGGTTAA(32)
        read = "NNNNNNNNCCCCAAAATTTTGGGGAAGGTTAA"
        flanking = FlankingConfig(adapter_side="AAGGTTAA")
        bc, start, end = _extract_sequence_with_flanking(
            read_sequence=read,
            target_length=4,
            search_window=50,
            search_from_start=False,
            flanking=flanking,
            flank_mode="adapter_only",
            adapter_matcher="exact",
        )
        assert bc == "GGGG"
        # Adapter starts at 23 (AAGGTTAA), barcode is 4 bases before: 19-23
        # Wait: len("NNNNNNNNCCCCAAAATTTTGGGG") = 24, adapter at 24-32 (exclusive)
        # barcode at 20-24
        assert start == 20
        assert end == 24

    def test_amplicon_only_mode_from_start(self):
        """Find amplicon at start region, extract barcode before it."""
        # Structure: BARCODE + AMPLICON + rest
        # AAAA(0-3) CCCC(4-7) CAGCACCT(8-15) NNNNNNNN(16-23)
        read = "AAAACCCCCAGCACCTNNNNNNNN"
        flanking = FlankingConfig(amplicon_side="CAGCACCT")
        bc, start, end = _extract_sequence_with_flanking(
            read_sequence=read,
            target_length=4,
            search_window=50,
            search_from_start=True,
            flanking=flanking,
            flank_mode="amplicon_only",
            adapter_matcher="exact",
        )
        # Amplicon found at position 8. Extract 4 bases before it: positions 4-8 = "CCCC"
        assert bc == "CCCC"
        assert start == 4
        assert end == 8

    def test_amplicon_only_mode_from_end(self):
        """Find amplicon at end region, extract barcode after it."""
        # Structure: rest + AMPLICON + BARCODE
        read = "NNNNNNNNCAGCACCTGGGG"
        flanking = FlankingConfig(amplicon_side="CAGCACCT")
        bc, start, end = _extract_sequence_with_flanking(
            read_sequence=read,
            target_length=4,
            search_window=50,
            search_from_start=False,
            flanking=flanking,
            flank_mode="amplicon_only",
            adapter_matcher="exact",
        )
        assert bc == "GGGG"
        assert start == 16
        assert end == 20

    def test_both_mode_validates_amplicon(self):
        """Find adapter, extract barcode, validate amplicon present."""
        # Structure: ADAPTER(0-7) + BARCODE(8-15) + AMPLICON(16-23) + rest(24-31)
        read = "AAGGTTAACCCCGGGGCAGCACCTNNNNNNNN"
        flanking = FlankingConfig(adapter_side="AAGGTTAA", amplicon_side="CAGCACCT")
        bc, start, end = _extract_sequence_with_flanking(
            read_sequence=read,
            target_length=8,
            search_window=50,
            search_from_start=True,
            flanking=flanking,
            flank_mode="both",
            adapter_matcher="exact",
        )
        assert bc == "CCCCGGGG"
        assert start == 8
        assert end == 16

    def test_both_mode_fails_without_amplicon(self):
        """Returns None if amplicon validation fails."""
        # Structure: ADAPTER(0-7) + BARCODE(8-15) + WRONG(16+)
        read = "AAGGTTAACCCCGGGGXXXXXXXXNNNN"
        flanking = FlankingConfig(adapter_side="AAGGTTAA", amplicon_side="CAGCACCT")
        bc, start, end = _extract_sequence_with_flanking(
            read_sequence=read,
            target_length=8,
            search_window=50,
            search_from_start=True,
            flanking=flanking,
            flank_mode="both",
            adapter_matcher="exact",
        )
        assert bc is None
        assert start is None
        assert end is None

    def test_no_adapter_side_returns_none_for_adapter_only(self):
        """adapter_only mode with no adapter_side returns None."""
        flanking = FlankingConfig(amplicon_side="CAGCACCT")
        bc, start, end = _extract_sequence_with_flanking(
            read_sequence="AAGGTTAACCCCNNNN",
            target_length=4,
            search_window=50,
            search_from_start=True,
            flanking=flanking,
            flank_mode="adapter_only",
            adapter_matcher="exact",
        )
        assert bc is None

    def test_empty_sequence_returns_none(self):
        flanking = FlankingConfig(adapter_side="ACGT")
        bc, start, end = _extract_sequence_with_flanking(
            read_sequence="",
            target_length=4,
            search_window=50,
            search_from_start=True,
            flanking=flanking,
            flank_mode="adapter_only",
            adapter_matcher="exact",
        )
        assert bc is None

    def test_invalid_flank_mode_raises(self):
        flanking = FlankingConfig(adapter_side="ACGT")
        with pytest.raises(ValueError, match="flank_mode must be one of"):
            _extract_sequence_with_flanking(
                read_sequence="ACGTAAAA",
                target_length=4,
                search_window=50,
                search_from_start=True,
                flanking=flanking,
                flank_mode="invalid",
                adapter_matcher="exact",
            )

    def test_out_of_bounds_returns_none(self):
        """Barcode region extends beyond sequence bounds."""
        read = "ACGT"  # Only adapter, no room for barcode
        flanking = FlankingConfig(adapter_side="ACGT")
        bc, start, end = _extract_sequence_with_flanking(
            read_sequence=read,
            target_length=8,
            search_window=50,
            search_from_start=True,
            flanking=flanking,
            flank_mode="adapter_only",
            adapter_matcher="exact",
        )
        assert bc is None


class TestLoadBarcodeYamlNewFormat:
    """Tests for loading barcode references from new YAML format."""

    def test_global_flanking(self, tmp_path):
        """Parse global adapter_side/amplicon_side."""
        yaml_file = tmp_path / "barcodes.yaml"
        yaml_file.write_text(
            "name: SQK-NBD114-96\n"
            "flanking:\n"
            "  adapter_side: AAGGTTAA\n"
            "  amplicon_side: CAGCACCT\n"
            "barcode_ends: both\n"
            "barcode_flank_mode: both\n"
            "barcodes:\n"
            "  NB01: CACAAAGACACCGACAACTTTCTT\n"
            "  NB02: ACAGACGACTACAAACGGAATCGA\n"
        )
        result = bam_functions.load_barcode_references_from_yaml(yaml_file)
        assert isinstance(result, BarcodeKitConfig)
        assert result.name == "SQK-NBD114-96"
        assert result.barcode_ends == "both"
        assert result.barcode_flank_mode == "both"
        assert len(result.barcodes) == 2
        assert result.barcode_length == 24
        assert result.flanking is not None
        assert result.flanking.left_ref_end.adapter_side == "AAGGTTAA"
        assert result.flanking.left_ref_end.amplicon_side == "CAGCACCT"
        # Global flanking applies to both ends
        assert result.flanking.right_ref_end.adapter_side == "AAGGTTAA"

    def test_per_end_flanking(self, tmp_path):
        """Parse left_ref_end/right_ref_end configs."""
        yaml_file = tmp_path / "barcodes.yaml"
        yaml_file.write_text(
            "flanking:\n"
            "  left_ref_end:\n"
            "    adapter_side: AAGGTTAA\n"
            "    amplicon_side: CAGCACCT\n"
            "  right_ref_end:\n"
            "    adapter_side: DIFFERENT\n"
            "barcode_flank_mode: adapter_only\n"
            "barcodes:\n"
            "  NB01: CACAAAGACACCGACAACTTTCTT\n"
        )
        result = bam_functions.load_barcode_references_from_yaml(yaml_file)
        assert isinstance(result, BarcodeKitConfig)
        assert result.flanking.left_ref_end.adapter_side == "AAGGTTAA"
        assert result.flanking.left_ref_end.amplicon_side == "CAGCACCT"
        assert result.flanking.right_ref_end.adapter_side == "DIFFERENT"
        assert result.flanking.right_ref_end.amplicon_side is None

    def test_backward_compat_old_format(self, tmp_path):
        """Old format without flanking still returns tuple."""
        yaml_file = tmp_path / "barcodes.yaml"
        yaml_file.write_text(
            "barcode01: ACGTACGT\nbarcode02: TGCATGCA\n"
        )
        result = bam_functions.load_barcode_references_from_yaml(yaml_file)
        assert isinstance(result, tuple)
        refs, length = result
        assert len(refs) == 2
        assert length == 8

    def test_config_params_from_yaml(self, tmp_path):
        """Parse barcode_ends, barcode_flank_mode, etc."""
        yaml_file = tmp_path / "barcodes.yaml"
        yaml_file.write_text(
            "barcode_ends: left_only\n"
            "barcode_flank_mode: amplicon_only\n"
            "barcode_max_edit_distance: 5\n"
            "barcode_adapter_max_edits: 4\n"
            "barcodes:\n"
            "  NB01: ACGTACGT\n"
        )
        result = bam_functions.load_barcode_references_from_yaml(yaml_file)
        assert isinstance(result, BarcodeKitConfig)
        assert result.barcode_ends == "left_only"
        assert result.barcode_flank_mode == "amplicon_only"
        assert result.barcode_max_edit_distance == 5
        assert result.barcode_adapter_max_edits == 4

    def test_nested_barcodes_key_new_format(self, tmp_path):
        """New format with nested barcodes key."""
        yaml_file = tmp_path / "barcodes.yaml"
        yaml_file.write_text(
            "flanking:\n"
            "  adapter_side: ACGT\n"
            "barcodes:\n"
            "  BC01: AAAACCCC\n"
            "  BC02: GGGGTTTT\n"
        )
        result = bam_functions.load_barcode_references_from_yaml(yaml_file)
        assert isinstance(result, BarcodeKitConfig)
        assert result.barcodes == {"BC01": "AAAACCCC", "BC02": "GGGGTTTT"}


class TestLoadUmiConfigFromYaml:
    """Tests for loading UMI config from YAML."""

    def test_load_basic_umi_config(self, tmp_path):
        yaml_file = tmp_path / "umi.yaml"
        yaml_file.write_text(
            "flanking:\n"
            "  adapter_side: GTACTGAC\n"
            "  amplicon_side: AATTCCGG\n"
            "length: 12\n"
            "umi_ends: left_only\n"
            "umi_flank_mode: both\n"
            "adapter_max_edits: 1\n"
        )
        result = load_umi_config_from_yaml(yaml_file)
        assert isinstance(result, UMIKitConfig)
        assert result.length == 12
        assert result.umi_ends == "left_only"
        assert result.umi_flank_mode == "both"
        assert result.adapter_max_edits == 1
        assert result.flanking is not None
        assert result.flanking.left_ref_end.adapter_side == "GTACTGAC"

    def test_load_nested_umi_key(self, tmp_path):
        """Support umi: top-level key."""
        yaml_file = tmp_path / "umi.yaml"
        yaml_file.write_text(
            "umi:\n"
            "  flanking:\n"
            "    adapter_side: ACGT\n"
            "  length: 8\n"
        )
        result = load_umi_config_from_yaml(yaml_file)
        assert result.length == 8
        assert result.flanking.left_ref_end.adapter_side == "ACGT"

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_umi_config_from_yaml("/nonexistent/umi.yaml")


class TestConfigResolution:
    """Tests for config resolution priority."""

    def test_config_overrides_yaml(self):
        """Experiment config takes precedence over YAML."""
        kit = BarcodeKitConfig(
            barcodes={"BC01": "ACGT"},
            barcode_length=4,
            barcode_ends="both",
            barcode_flank_mode="adapter_only",
        )

        class FakeCfg:
            barcode_ends = "left_only"
            barcode_flank_mode = "both"
            barcode_max_edit_distance = 5
            barcode_adapter_max_edits = 4
            barcode_amplicon_max_edits = 3

        resolved = resolve_barcode_config(kit, FakeCfg())
        assert resolved["barcode_ends"] == "left_only"
        assert resolved["barcode_flank_mode"] == "both"
        assert resolved["barcode_max_edit_distance"] == 5
        assert resolved["barcode_amplicon_max_edits"] == 3

    def test_yaml_overrides_defaults(self):
        """YAML values used when config attrs are None."""
        kit = BarcodeKitConfig(
            barcodes={"BC01": "ACGT"},
            barcode_length=4,
            barcode_ends="right_only",
            barcode_flank_mode="amplicon_only",
            barcode_max_edit_distance=7,
        )

        class FakeCfg:
            pass

        resolved = resolve_barcode_config(kit, FakeCfg())
        assert resolved["barcode_ends"] == "right_only"
        assert resolved["barcode_flank_mode"] == "amplicon_only"
        assert resolved["barcode_max_edit_distance"] == 7

    def test_umi_config_resolution(self):
        umi_cfg = UMIKitConfig(
            length=12,
            umi_ends="left_only",
            umi_flank_mode="both",
            adapter_max_edits=2,
        )

        class FakeCfg:
            umi_ends = None
            umi_flank_mode = None
            umi_adapter_max_edits = None
            umi_amplicon_max_edits = None

        resolved = resolve_umi_config(umi_cfg, FakeCfg())
        assert resolved["umi_ends"] == "left_only"
        assert resolved["umi_flank_mode"] == "both"

    def test_umi_config_resolution_with_none_config(self):
        """When umi_config is None, defaults are used."""

        class FakeCfg:
            pass

        resolved = resolve_umi_config(None, FakeCfg())
        assert resolved["umi_ends"] == "both"
        assert resolved["umi_flank_mode"] == "adapter_only"


class TestBuildFlankingFromAdapters:
    """Tests for legacy adapter conversion."""

    def test_both_adapters(self):
        result = _build_flanking_from_adapters(["ACGT", "TGCA"])
        assert result.left_ref_end.adapter_side == "ACGT"
        assert result.right_ref_end.adapter_side == "TGCA"

    def test_left_only(self):
        result = _build_flanking_from_adapters(["ACGT", None])
        assert result.left_ref_end.adapter_side == "ACGT"
        assert result.right_ref_end is None

    def test_right_only(self):
        result = _build_flanking_from_adapters([None, "TGCA"])
        assert result.left_ref_end is None
        assert result.right_ref_end.adapter_side == "TGCA"

    def test_no_amplicon_side(self):
        result = _build_flanking_from_adapters(["ACGT", "TGCA"])
        assert result.left_ref_end.amplicon_side is None
        assert result.right_ref_end.amplicon_side is None


# ---------------------------------------------------------------------------
# Tier 1: extract_and_assign_barcodes_in_bam with mock BAM data
# ---------------------------------------------------------------------------
@requires_pysam
class TestExtractAndAssignBarcodesInBam:
    """Integration tests for barcode extraction orchestration."""

    BC_REFS = {"BC01": "AAAA", "BC02": "CCCC", "BC03": "GGGG"}
    LEFT_ADAPTER = "ACGT"
    RIGHT_ADAPTER = "TGCA"

    def _run(self, tmp_path, reads, **kwargs):
        """Create BAM → run extraction → return {name: {tag: val}}."""
        bam = tmp_path / "test.bam"
        _create_test_bam(bam, reads)
        defaults = dict(
            barcode_adapters=[self.LEFT_ADAPTER, self.RIGHT_ADAPTER],
            barcode_references=self.BC_REFS,
            barcode_length=4,
            barcode_search_window=200,
            barcode_max_edit_distance=3,
            barcode_adapter_matcher="exact",
            barcode_adapter_max_edits=0,
            samtools_backend="python",
        )
        defaults.update(kwargs)
        bam_functions.extract_and_assign_barcodes_in_bam(bam, **defaults)
        return _read_bam_tags(bam)

    # -- Legacy adapter path --------------------------------------------------

    def test_legacy_both_ends_match(self, tmp_path):
        """Both ends match BC01 → BC='BC01', BM='both'."""
        # ACGT(0-3) AAAA(4-7) NNNNNNNN(8-15) AAAA(16-19) TGCA(20-23)
        reads = [{"name": "r1", "sequence": "ACGTAAAANNNNNNNNAAAATGCA"}]
        tags = self._run(tmp_path, reads)
        assert tags["r1"]["BC"] == "BC01"
        assert tags["r1"]["BM"] == "both"
        assert tags["r1"]["B1"] == "BC01"
        assert tags["r1"]["B2"] == "BC01"
        assert tags["r1"]["BE"] == 0
        assert tags["r1"]["BF"] == 0

    def test_legacy_mismatch_ends(self, tmp_path):
        """Different barcodes at each end → 'mismatch', 'unclassified'."""
        reads = [{"name": "r1", "sequence": "ACGTAAAANNNNNNNNCCCCTGCA"}]
        tags = self._run(tmp_path, reads)
        assert tags["r1"]["BC"] == "unclassified"
        assert tags["r1"]["BM"] == "mismatch"
        assert tags["r1"]["B1"] == "BC01"
        assert tags["r1"]["B2"] == "BC02"

    def test_legacy_left_only(self, tmp_path):
        """Only left adapter found → assigned from left, BM='left_only'."""
        reads = [{"name": "r1", "sequence": "ACGTAAAANNNNNNNNNNNNNNNN"}]
        tags = self._run(tmp_path, reads)
        assert tags["r1"]["BC"] == "BC01"
        assert tags["r1"]["BM"] == "left_only"
        assert tags["r1"]["B1"] == "BC01"
        assert "B2" not in tags["r1"]

    def test_legacy_right_only(self, tmp_path):
        """Only right adapter found → assigned from right, BM='right_only'."""
        reads = [{"name": "r1", "sequence": "NNNNNNNNNNNNNNNNCCCCTGCA"}]
        tags = self._run(tmp_path, reads)
        assert tags["r1"]["BC"] == "BC02"
        assert tags["r1"]["BM"] == "right_only"
        assert "B1" not in tags["r1"]
        assert tags["r1"]["B2"] == "BC02"

    def test_legacy_unclassified(self, tmp_path):
        """No adapters found → 'unclassified'."""
        reads = [{"name": "r1", "sequence": "TTTTTTTTTTTTTTTTTTTTTTTT"}]
        tags = self._run(tmp_path, reads)
        assert tags["r1"]["BC"] == "unclassified"
        assert tags["r1"]["BM"] == "unclassified"
        assert "B1" not in tags["r1"]
        assert "B2" not in tags["r1"]

    # -- Filtering options ----------------------------------------------------

    def test_require_both_ends_rejects_single(self, tmp_path):
        """require_both_ends=True with only left match → 'unclassified'."""
        reads = [{"name": "r1", "sequence": "ACGTAAAANNNNNNNNNNNNNNNN"}]
        tags = self._run(tmp_path, reads, require_both_ends=True)
        assert tags["r1"]["BC"] == "unclassified"
        assert tags["r1"]["BM"] == "left_only"
        assert tags["r1"]["B1"] == "BC01"

    def test_min_barcode_score_filters_weak_match(self, tmp_path):
        """min_barcode_score=0 rejects matches with edit distance > 0."""
        # AAAT has Hamming distance 1 from AAAA (BC01)
        reads = [{"name": "r1", "sequence": "ACGTAAATNNNNNNNNAAATTGCA"}]
        tags = self._run(tmp_path, reads, min_barcode_score=0)
        assert tags["r1"]["BC"] == "unclassified"
        assert tags["r1"]["BM"] == "unclassified"

    # -- barcode_ends ---------------------------------------------------------

    def test_barcode_ends_left_only(self, tmp_path):
        """barcode_ends='left_only' ignores right end entirely."""
        reads = [{"name": "r1", "sequence": "ACGTAAAANNNNNNNNCCCCTGCA"}]
        tags = self._run(tmp_path, reads, barcode_ends="left_only")
        assert tags["r1"]["BC"] == "BC01"
        assert tags["r1"]["BM"] == "left_only"
        assert tags["r1"]["B1"] == "BC01"
        assert "B2" not in tags["r1"]

    def test_barcode_ends_right_only(self, tmp_path):
        """barcode_ends='right_only' ignores left end entirely."""
        reads = [{"name": "r1", "sequence": "ACGTAAAANNNNNNNNCCCCTGCA"}]
        tags = self._run(tmp_path, reads, barcode_ends="right_only")
        assert tags["r1"]["BC"] == "BC02"
        assert tags["r1"]["BM"] == "right_only"
        assert "B1" not in tags["r1"]
        assert tags["r1"]["B2"] == "BC02"

    # -- Flanking-based extraction -------------------------------------------

    def test_flanking_based_both_ends(self, tmp_path):
        """Flanking-based extraction with BarcodeKitConfig matches both ends."""
        kit = BarcodeKitConfig(
            barcodes=self.BC_REFS,
            barcode_length=4,
            flanking=PerEndFlankingConfig(
                left_ref_end=FlankingConfig(adapter_side="ACGT"),
                right_ref_end=FlankingConfig(adapter_side="TGCA"),
            ),
        )
        reads = [{"name": "r1", "sequence": "ACGTAAAANNNNNNNNAAAATGCA"}]
        tags = self._run(
            tmp_path, reads,
            barcode_kit_config=kit,
            barcode_flank_mode="adapter_only",
        )
        assert tags["r1"]["BC"] == "BC01"
        assert tags["r1"]["BM"] == "both"

    # -- Strand handling ------------------------------------------------------

    def test_reverse_strand_flips_ends(self, tmp_path):
        """Reverse read: left ref → read end, right ref → read start."""
        # For a reverse read, right adapter should be at READ start,
        # left adapter at READ end.
        # TGCA(0-3) GGGG(4-7) NNNNNNNN(8-15) GGGG(16-19) ACGT(20-23)
        reads = [{"name": "r1", "sequence": "TGCAGGGGNNNNNNNNGGGGACGT",
                  "is_reverse": True}]
        tags = self._run(tmp_path, reads)
        assert tags["r1"]["BC"] == "BC03"
        assert tags["r1"]["BM"] == "both"
        assert tags["r1"]["B1"] == "BC03"
        assert tags["r1"]["B2"] == "BC03"

    # -- Error handling -------------------------------------------------------

    def test_empty_references_raises(self, tmp_path):
        """Empty barcode_references raises ValueError."""
        bam = tmp_path / "test.bam"
        _create_test_bam(bam, [{"name": "r1", "sequence": "ACGTAAAANNNNNNNN"}])
        with pytest.raises(ValueError, match="barcode_references"):
            bam_functions.extract_and_assign_barcodes_in_bam(
                bam,
                barcode_adapters=["ACGT", "TGCA"],
                barcode_references={},
                barcode_adapter_matcher="exact",
                samtools_backend="python",
            )

    # -- Multiple reads -------------------------------------------------------

    def test_multiple_reads_mixed_outcomes(self, tmp_path):
        """Multiple reads produce correct per-read tags."""
        reads = [
            {"name": "both_bc01", "sequence": "ACGTAAAANNNNNNNNAAAATGCA"},
            {"name": "mismatch", "sequence": "ACGTAAAANNNNNNNNCCCCTGCA"},
            {"name": "left_bc03", "sequence": "ACGTGGGGNNNNNNNNNNNNNNNN"},
            {"name": "unclassified", "sequence": "TTTTTTTTTTTTTTTTTTTTTTTT"},
        ]
        tags = self._run(tmp_path, reads)
        assert tags["both_bc01"]["BC"] == "BC01"
        assert tags["both_bc01"]["BM"] == "both"
        assert tags["mismatch"]["BC"] == "unclassified"
        assert tags["mismatch"]["BM"] == "mismatch"
        assert tags["left_bc03"]["BC"] == "BC03"
        assert tags["left_bc03"]["BM"] == "left_only"
        assert tags["unclassified"]["BC"] == "unclassified"
        assert tags["unclassified"]["BM"] == "unclassified"
