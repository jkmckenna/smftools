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
    _match_barcode_to_references,
    _parse_flanking_config_from_dict,
    _parse_per_end_flanking,
    _reverse_complement,
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


def _read_parquet_tags(parquet_path):
    """Return ``{read_name: {tag: value}}`` from a barcode sidecar parquet."""
    from pathlib import Path

    p = Path(parquet_path)
    df = pd.read_parquet(p)
    out = {}
    for _, row in df.iterrows():
        tags = {}
        for col in ["BC", "BM", "B1", "B2", "B3", "B4", "B5", "B6"]:
            if col in row.index and pd.notna(row[col]):
                tags[col] = row[col]
        out[row["read_name"]] = tags
    return out


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
        name, dist = bam_functions._match_barcode_to_references("", references, max_edit_distance=1)
        assert name is None
        assert dist is None

    def test_empty_references(self):
        """Test handling of empty reference set."""
        name, dist = bam_functions._match_barcode_to_references("AAAA", {}, max_edit_distance=1)
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
        yaml_file.write_text("barcode01: ACGTACGT\nbarcode02: TGCATGCA\nbarcode03: AAAACCCC\n")
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
        yaml_file.write_text("barcodes:\n  barcode01: ACGTACGT\n  barcode02: TGCATGCA\n")
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
        """Test that BM='read_start_only' maps to demux_type='single'."""
        adata = self._make_adata(["read_start_only", "read_start_only"])
        add_demux_type_from_bm_tag(adata)
        assert all(adata.obs["demux_type"] == "single")

    def test_right_only_becomes_single(self):
        """Test that BM='read_end_only' maps to demux_type='single'."""
        adata = self._make_adata(["read_end_only", "read_end_only"])
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
        adata = self._make_adata(
            ["both", "read_start_only", "read_end_only", "mismatch", "unclassified"]
        )
        add_demux_type_from_bm_tag(adata)
        expected = ["double", "single", "single", "unclassified", "unclassified"]
        assert list(adata.obs["demux_type"]) == expected

    def test_case_insensitive(self):
        """Test that BM matching is case-insensitive."""
        adata = self._make_adata(["BOTH", "Read_Start_Only", "READ_END_ONLY"])
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

    def test_amplicon_only_same_orientation_from_end(self):
        """With same_orientation=True, barcode is BEFORE amplicon even from end."""
        # Layout at end of read: ...NNNNNNNN + BARCODE + AMPLICON
        # Without same_orientation, amplicon_only from end would look for barcode AFTER amplicon.
        # With same_orientation, it should still extract barcode BEFORE amplicon.
        read = "NNNNNNNNCCCCGGGGCAGCACCT"
        flanking = FlankingConfig(amplicon_side="CAGCACCT")
        bc, start, end = _extract_sequence_with_flanking(
            read_sequence=read,
            target_length=8,
            search_window=50,
            search_from_start=False,
            flanking=flanking,
            flank_mode="amplicon_only",
            adapter_matcher="exact",
            same_orientation=True,
        )
        # Amplicon at position 16-24 (CAGCACCT). Barcode 8 bases before: 8-16 (CCCCGGGG)
        assert bc == "CCCCGGGG"
        assert start == 8
        assert end == 16

    def test_amplicon_only_default_orientation_from_end(self):
        """Default (same_orientation=False) preserves old behavior: barcode AFTER amplicon from end."""
        # Layout: ...AMPLICON + BARCODE at end
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
            same_orientation=False,
        )
        assert bc == "GGGG"
        assert start == 16
        assert end == 20

    def test_adapter_only_same_orientation_from_end(self):
        """With same_orientation=True, adapter_only from end still extracts barcode AFTER adapter."""
        # Layout at end of read: ...NNNNNNNN + ADAPTER + BARCODE
        read = "NNNNNNNNAAGGTTAACCCC"
        flanking = FlankingConfig(adapter_side="AAGGTTAA")
        bc, start, end = _extract_sequence_with_flanking(
            read_sequence=read,
            target_length=4,
            search_window=50,
            search_from_start=False,
            flanking=flanking,
            flank_mode="adapter_only",
            adapter_matcher="exact",
            same_orientation=True,
        )
        # Adapter at 8-16 (AAGGTTAA). Barcode 4 bases after: 16-20 (CCCC)
        assert bc == "CCCC"
        assert start == 16
        assert end == 20


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
            "barcode_composite_max_edits: 4\n"
            "barcodes:\n"
            "  NB01: CACAAAGACACCGACAACTTTCTT\n"
            "  NB02: ACAGACGACTACAAACGGAATCGA\n"
        )
        result = bam_functions.load_barcode_references_from_yaml(yaml_file)
        assert isinstance(result, BarcodeKitConfig)
        assert result.name == "SQK-NBD114-96"
        assert result.barcode_ends == "both"
        assert result.barcode_composite_max_edits == 4
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
            "barcode_composite_max_edits: 4\n"
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
        yaml_file.write_text("barcode01: ACGTACGT\nbarcode02: TGCATGCA\n")
        result = bam_functions.load_barcode_references_from_yaml(yaml_file)
        assert isinstance(result, tuple)
        refs, length = result
        assert len(refs) == 2
        assert length == 8

    def test_config_params_from_yaml(self, tmp_path):
        """Parse barcode_ends, barcode_composite_max_edits, etc."""
        yaml_file = tmp_path / "barcodes.yaml"
        yaml_file.write_text(
            "barcode_ends: left_only\n"
            "barcode_max_edit_distance: 5\n"
            "barcode_composite_max_edits: 6\n"
            "barcodes:\n"
            "  NB01: ACGTACGT\n"
        )
        result = bam_functions.load_barcode_references_from_yaml(yaml_file)
        assert isinstance(result, BarcodeKitConfig)
        assert result.barcode_ends == "left_only"
        assert result.barcode_max_edit_distance == 5
        assert result.barcode_composite_max_edits == 6

    def test_nested_barcodes_key_new_format(self, tmp_path):
        """New format with nested barcodes key."""
        yaml_file = tmp_path / "barcodes.yaml"
        yaml_file.write_text(
            "flanking:\n  adapter_side: ACGT\nbarcodes:\n  BC01: AAAACCCC\n  BC02: GGGGTTTT\n"
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
        yaml_file.write_text("umi:\n  flanking:\n    adapter_side: ACGT\n  length: 8\n")
        result = load_umi_config_from_yaml(yaml_file)
        assert result.length == 8
        assert result.flanking.left_ref_end.adapter_side == "ACGT"

    def test_load_top_bottom_flanking(self, tmp_path):
        yaml_file = tmp_path / "umi.yaml"
        yaml_file.write_text(
            "top_flanking:\n"
            "  adapter_side: AAAA\n"
            "bottom_flanking:\n"
            "  adapter_side: TTTT\n"
            "length: 6\n"
        )
        result = load_umi_config_from_yaml(yaml_file)
        assert result.length == 6
        assert result.flanking.left_ref_end.adapter_side == "AAAA"
        assert result.flanking.right_ref_end.adapter_side == "TTTT"

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
            barcode_composite_max_edits=4,
        )

        class FakeCfg:
            barcode_ends = "left_only"
            barcode_max_edit_distance = 5
            barcode_composite_max_edits = 6

        resolved = resolve_barcode_config(kit, FakeCfg())
        assert resolved["barcode_ends"] == "left_only"
        assert resolved["barcode_max_edit_distance"] == 5
        assert resolved["barcode_composite_max_edits"] == 6

    def test_yaml_overrides_defaults(self):
        """YAML values used when config attrs are None."""
        kit = BarcodeKitConfig(
            barcodes={"BC01": "ACGT"},
            barcode_length=4,
            barcode_ends="right_only",
            barcode_composite_max_edits=5,
            barcode_max_edit_distance=7,
        )

        class FakeCfg:
            pass

        resolved = resolve_barcode_config(kit, FakeCfg())
        assert resolved["barcode_ends"] == "right_only"
        assert resolved["barcode_max_edit_distance"] == 7
        assert resolved["barcode_composite_max_edits"] == 5

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


class TestBarcodeMinSeparation:
    """Tests for barcode_min_separation behavior."""

    def test_min_separation_blocks_ambiguous(self):
        refs = {"BC1": "AAAT", "BC2": "AATT"}
        best, dist = _match_barcode_to_references(
            "AAAA",
            refs,
            max_edit_distance=2,
            min_separation=2,
        )
        assert best is None
        assert dist is None

    def test_min_separation_allows_clear_winner(self):
        refs = {"BC1": "AAAT", "BC2": "TTTT"}
        best, dist = _match_barcode_to_references(
            "AAAA",
            refs,
            max_edit_distance=2,
            min_separation=2,
        )
        assert best == "BC1"
        assert dist == 1


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
            barcode_composite_max_edits=0,
            samtools_backend="python",
        )
        # Provide a default flanking kit config unless the caller overrides
        if "barcode_kit_config" not in kwargs:
            defaults["barcode_kit_config"] = BarcodeKitConfig(
                barcodes=self.BC_REFS,
                barcode_length=4,
                flanking=PerEndFlankingConfig(
                    left_ref_end=FlankingConfig(
                        adapter_side=self.LEFT_ADAPTER,
                        amplicon_side=None,
                    ),
                    right_ref_end=FlankingConfig(
                        adapter_side=self.RIGHT_ADAPTER,
                        amplicon_side=None,
                    ),
                ),
            )
        defaults.update(kwargs)
        sidecar = bam_functions.extract_and_assign_barcodes_in_bam(bam, **defaults)
        return _read_parquet_tags(sidecar)

    # -- Flanking-based extraction -------------------------------------------

    def test_flanking_based_both_ends(self, tmp_path):
        """Flanking-based extraction with BarcodeKitConfig matches both ends.

        YAML stores forward-strand sequences. For a forward read:
        - Left end (read start): adapter + barcode + ...
        - Right end (read end):  ... + RC(barcode) + RC(adapter)
        """
        adapter = "ATCG"
        rc_adapter = _reverse_complement(adapter)  # CGAT
        rc_bc = _reverse_complement("AAAA")  # TTTT
        kit = BarcodeKitConfig(
            barcodes=self.BC_REFS,
            barcode_length=4,
            flanking=PerEndFlankingConfig(
                left_ref_end=FlankingConfig(adapter_side=adapter),
                right_ref_end=FlankingConfig(adapter_side=adapter),
            ),
        )
        # Left: ATCG + AAAA + ... Right: ... + TTTT + CGAT
        reads = [{"name": "r1", "sequence": f"ATCGAAAANNNNNNNN{rc_bc}{rc_adapter}"}]
        tags = self._run(
            tmp_path,
            reads,
            barcode_kit_config=kit,
        )
        assert tags["r1"]["BC"] == "BC01"
        assert tags["r1"]["BM"] == "both"

    # -- Strand handling ------------------------------------------------------

    @pytest.mark.skip(reason="Barcode extraction now uses read start/end for unaligned BAMs.")
    def test_reverse_strand_flips_ends(self, tmp_path):
        """Reverse read: left ref → read end, right ref → read start.

        Barcodes appear in RC orientation in the read.  After extraction the
        code RC's them back to forward orientation before matching.
        """
        # For a reverse read, right adapter (TGCA) appears as RC(TGCA)=TGCA
        # at READ start and left adapter (ACGT) as RC(ACGT)=ACGT at READ end.
        # Barcodes appear as RC(GGGG)=CCCC in the read.
        # RC(TGCA)(0-3) RC(GGGG)(4-7) NNNNNNNN(8-15) RC(GGGG)(16-19) RC(ACGT)(20-23)
        reads = [{"name": "r1", "sequence": "TGCACCCCNNNNNNNNCCCCACGT", "is_reverse": True}]
        tags = self._run(tmp_path, reads)
        assert tags["r1"]["BC"] == "BC03"
        assert tags["r1"]["BM"] == "both"
        assert tags["r1"]["B5"] == "BC03"
        assert tags["r1"]["B6"] == "BC03"

    @pytest.mark.skip(reason="Barcode extraction now uses read start/end for unaligned BAMs.")
    def test_reverse_strand_non_palindromic_adapters(self, tmp_path):
        """Reverse read with non-palindromic adapters verifies RC handling."""
        # Adapters: left=ATCG (RC=CGAT), right=GCTA (RC=TAGC)
        # BC01 = AAAA, RC(AAAA) = TTTT
        # Reverse read layout:
        #   read start: RC(right adapter)=TAGC + RC(BC01)=TTTT
        #   read end:   RC(BC01)=TTTT + RC(left adapter)=CGAT
        reads = [{"name": "r1", "sequence": "TAGCTTTTNNNNNNNNTTTTCGAT", "is_reverse": True}]
        tags = self._run(
            tmp_path,
            reads,
            barcode_adapters=["ATCG", "GCTA"],
        )
        assert tags["r1"]["BC"] == "BC01"
        assert tags["r1"]["BM"] == "both"
        assert tags["r1"]["B5"] == "BC01"
        assert tags["r1"]["B6"] == "BC01"

    # -- "either" flank mode -------------------------------------------------

    def test_either_mode_uses_both_when_available(self, tmp_path):
        """'either' mode finds both flanks → same as 'both' mode.

        Forward read with RC construct at right end:
        - Left:  adapter + BC01 + amplicon + ...
        - Right: ... + RC(amplicon) + RC(BC01) + RC(adapter)
        """
        adapter = "ATCG"
        amplicon = "CCCCCCCC"
        rc_adapter = _reverse_complement(adapter)  # CGAT
        rc_amplicon = _reverse_complement(amplicon)  # GGGGGGGG
        rc_bc = _reverse_complement("AAAA")  # TTTT
        kit = BarcodeKitConfig(
            barcodes=self.BC_REFS,
            barcode_length=4,
            flanking=PerEndFlankingConfig(
                left_ref_end=FlankingConfig(adapter_side=adapter, amplicon_side=amplicon),
                right_ref_end=FlankingConfig(adapter_side=adapter, amplicon_side=amplicon),
            ),
        )
        reads = [
            {
                "name": "r1",
                "sequence": f"{adapter}AAAA{amplicon}NNNN{rc_amplicon}{rc_bc}{rc_adapter}",
            }
        ]
        tags = self._run(
            tmp_path,
            reads,
            barcode_kit_config=kit,
        )
        assert tags["r1"]["BC"] == "BC01"
        assert tags["r1"]["BM"] == "both"

    # -- Error handling -------------------------------------------------------

    def test_empty_references_raises(self, tmp_path):
        """Empty barcode_references raises ValueError."""
        bam = tmp_path / "test.bam"
        _create_test_bam(bam, [{"name": "r1", "sequence": "ACGTAAAANNNNNNNN"}])
        kit = BarcodeKitConfig(
            barcodes={},
            barcode_length=4,
            flanking=PerEndFlankingConfig(
                left_ref_end=FlankingConfig(adapter_side="ACGT", amplicon_side=None),
                right_ref_end=FlankingConfig(adapter_side="TGCA", amplicon_side=None),
            ),
        )
        with pytest.raises(ValueError, match="barcode_references"):
            bam_functions.extract_and_assign_barcodes_in_bam(
                bam,
                barcode_adapters=["ACGT", "TGCA"],
                barcode_references={},
                barcode_adapter_matcher="exact",
                barcode_kit_config=kit,
                samtools_backend="python",
            )

    # -- Multiple reads -------------------------------------------------------

    def test_multiple_reads_mixed_outcomes(self, tmp_path):
        """Multiple reads produce correct per-read tags.

        Uses flanking-based extraction with adapter-only flanking.
        Right end: RC(adapter) + RC(barcode), so for BC01=AAAA the right end
        has RC(TGCA)=TGCA followed by RC(AAAA)=TTTT, but the code extracts
        the barcode before RC(adapter) and then reverse-complements it back.
        """
        rc_adapter = _reverse_complement("TGCA")  # TGCA
        rc_bc01 = _reverse_complement("AAAA")  # TTTT
        rc_bc02 = _reverse_complement("CCCC")  # GGGG
        reads = [
            # Both ends match BC01
            {"name": "both_bc01", "sequence": f"ACGTAAAANNNNNNNN{rc_bc01}{rc_adapter}"},
            # Start=BC01, End=BC02 → mismatch
            {"name": "mismatch", "sequence": f"ACGTAAAANNNNNNNN{rc_bc02}{rc_adapter}"},
            # Only left adapter found
            {"name": "left_bc03", "sequence": "ACGTGGGGNNNNNNNNNNNNNNNN"},
            # No adapters found
            {"name": "unclassified", "sequence": "TTTTTTTTTTTTTTTTTTTTTTTT"},
        ]
        tags = self._run(tmp_path, reads)
        assert tags["both_bc01"]["BC"] == "BC01"
        assert tags["both_bc01"]["BM"] == "both"
        assert tags["mismatch"]["BC"] == "unclassified"
        assert tags["mismatch"]["BM"] == "mismatch"
        assert tags["left_bc03"]["BC"] == "BC03"
        assert tags["left_bc03"]["BM"] == "read_start_only"
        assert tags["unclassified"]["BC"] == "unclassified"
        assert tags["unclassified"]["BM"] == "unclassified"

    # -- RC / right-ref-end integration ----------------------------------------

    def test_flanking_forward_right_end_rc(self, tmp_path):
        """Forward read: right ref end has RC'd construct, code RC's flanking to match.

        Kit construct:
        - Left ref end:  [adapter][barcode][amplicon][reference...]
        - Right ref end: [...reference][RC(amplicon)][RC(barcode)][RC(adapter)]

        For a forward read:
        - Left end (read start): adapter + barcode + amplicon + ...
        - Right end (read end):  ... + RC(amplicon) + RC(barcode) + RC(adapter)

        The code should RC the YAML flanking seqs for the right end, find
        them in the read, extract the RC'd barcode, and RC it back.
        """
        adapter = "ACGT"
        amplicon = "TTTTTTTT"
        rc_adapter = _reverse_complement(adapter)  # ACGT → ACGT (palindrome)
        rc_amplicon = _reverse_complement(amplicon)  # AAAAAAAA
        kit = BarcodeKitConfig(
            barcodes=self.BC_REFS,
            barcode_length=4,
            flanking=PerEndFlankingConfig(
                left_ref_end=FlankingConfig(adapter_side=adapter, amplicon_side=amplicon),
                right_ref_end=FlankingConfig(adapter_side=adapter, amplicon_side=amplicon),
            ),
        )
        # Use non-palindromic barcode BC03=GGGG, RC(GGGG)=CCCC
        rc_bc = _reverse_complement("GGGG")  # CCCC
        # Left end:  ACGT + GGGG + TTTTTTTT
        # Right end: AAAAAAAA + CCCC + ACGT
        seq = f"ACGTGGGGTTTTTTTTNNNNNNNNAAAAAAAA{rc_bc}ACGT"
        reads = [{"name": "r1", "sequence": seq}]
        tags = self._run(
            tmp_path,
            reads,
            barcode_kit_config=kit,
            barcode_adapter_matcher="exact",
        )
        assert tags["r1"]["BC"] == "BC03"
        assert tags["r1"]["BM"] == "both"
        assert tags["r1"]["B5"] == "BC03"
        assert tags["r1"]["B6"] == "BC03"

    @pytest.mark.skip(reason="Barcode extraction now uses read start/end for unaligned BAMs.")
    def test_flanking_reverse_strand_both_ends(self, tmp_path):
        """Reverse read: both ends should find barcode with correct RC logic.

        For a reverse-strand read, pysam gives the reverse-complemented sequence.
        - Left ref end now appears at READ end: RC(amplicon) + RC(barcode) + RC(adapter)
        - Right ref end now appears at READ start: adapter + barcode + amplicon

        So: left ref end needs RC (is_reverse=True, ref_side=left → XOR=True)
            right ref end does NOT need RC (is_reverse=True, ref_side=right → XOR=False)
        """
        adapter = "ATCG"
        amplicon = "CCCCCCCC"
        rc_adapter = _reverse_complement(adapter)  # CGAT
        rc_amplicon = _reverse_complement(amplicon)  # GGGGGGGG
        kit = BarcodeKitConfig(
            barcodes=self.BC_REFS,
            barcode_length=4,
            flanking=PerEndFlankingConfig(
                left_ref_end=FlankingConfig(adapter_side=adapter, amplicon_side=amplicon),
                right_ref_end=FlankingConfig(adapter_side=adapter, amplicon_side=amplicon),
            ),
        )
        # BC01=AAAA, RC=TTTT
        rc_bc = _reverse_complement("AAAA")  # TTTT
        # Reverse read layout (what pysam returns):
        # READ start = right ref end (no RC): adapter + barcode + amplicon
        # READ end = left ref end (RC'd):     RC(amplicon) + RC(barcode) + RC(adapter)
        seq = f"{adapter}AAAA{amplicon}NNNNNNNN{rc_amplicon}{rc_bc}{rc_adapter}"
        reads = [{"name": "r1", "sequence": seq, "is_reverse": True}]
        tags = self._run(
            tmp_path,
            reads,
            barcode_kit_config=kit,
            barcode_adapter_matcher="exact",
        )
        assert tags["r1"]["BC"] == "BC01"
        assert tags["r1"]["BM"] == "both"
        assert tags["r1"]["B5"] == "BC01"
        assert tags["r1"]["B6"] == "BC01"


# ---------------------------------------------------------------------------
# Tier 2: Realistic RBK114-style RC logic tests
# ---------------------------------------------------------------------------
@requires_pysam
class TestRBK114RealisticRC:
    """Integration tests using realistic RBK114 kit sequences.

    The RBK114 kit construct:
    - Left ref end:  [adapter][barcode][amplicon][reference...]
    - Right ref end: [...reference][RC(amplicon)][RC(barcode)][RC(adapter)]

    YAML stores forward-strand (left ref end) sequences.  The code must
    RC them when searching the right ref end of a forward read or the
    left ref end of a reverse read (``needs_rc = is_reverse XOR right``).
    """

    ADAPTER = "GCTTGGGTGTTTAACC"  # 16 bp
    AMPLICON = "GTTTTCGCATTTATCGTGAAACGCTTTCGCGTTTTTCGTGCGCCGCTTCA"  # 50 bp
    RC_ADAPTER = _reverse_complement("GCTTGGGTGTTTAACC")  # GGTTAAACACCCAAGC
    RC_AMPLICON = _reverse_complement(
        "GTTTTCGCATTTATCGTGAAACGCTTTCGCGTTTTTCGTGCGCCGCTTCA"
    )  # TGAAGCGGCGCAC...
    FILLER = "N" * 40  # generic reference

    BC_REFS = {
        "RB01": "AAGAAAGTTGTCGGTGTCTTTGTG",
        "RB02": "TCGATTCCGTTTGTAGTCGTCTGT",
        "RB03": "GAGTCTTGTGTCCCAGTTACCAGG",
    }
    RC_RB01 = _reverse_complement("AAGAAAGTTGTCGGTGTCTTTGTG")  # CACAAAGACACCGACAACTTTCTT
    RC_RB02 = _reverse_complement("TCGATTCCGTTTGTAGTCGTCTGT")
    BARCODE_LENGTH = 24

    def _run(self, tmp_path, reads, **kwargs):
        bam = tmp_path / "test.bam"
        _create_test_bam(bam, reads)
        defaults = dict(
            barcode_adapters=[None, None],
            barcode_references=self.BC_REFS,
            barcode_length=self.BARCODE_LENGTH,
            barcode_search_window=200,
            barcode_max_edit_distance=5,
            barcode_adapter_matcher="edlib",
            barcode_composite_max_edits=12,
            samtools_backend="python",
        )
        defaults.update(kwargs)
        sidecar = bam_functions.extract_and_assign_barcodes_in_bam(bam, **defaults)
        return _read_parquet_tags(sidecar)

    def _make_kit(self, **overrides):
        kw = dict(
            barcodes=self.BC_REFS,
            barcode_length=self.BARCODE_LENGTH,
            flanking=PerEndFlankingConfig(
                left_ref_end=FlankingConfig(
                    adapter_side=self.ADAPTER,
                    amplicon_side=self.AMPLICON,
                ),
                right_ref_end=FlankingConfig(
                    adapter_side=self.ADAPTER,
                    amplicon_side=self.AMPLICON,
                ),
            ),
        )
        kw.update(overrides)
        return BarcodeKitConfig(**kw)

    # -- Forward read, both ends -----------------------------------------------

    def test_forward_both_ends(self, tmp_path):
        """Forward read with full construct at both ends → BM='both'."""
        left = self.ADAPTER + self.BC_REFS["RB01"] + self.AMPLICON
        right = self.RC_AMPLICON + self.RC_RB01 + self.RC_ADAPTER
        seq = left + self.FILLER + right
        reads = [{"name": "r1", "sequence": seq}]
        tags = self._run(
            tmp_path,
            reads,
            barcode_kit_config=self._make_kit(),
        )
        assert tags["r1"]["BC"] == "RB01"
        assert tags["r1"]["BM"] == "both"
        assert tags["r1"]["B5"] == "RB01"
        assert tags["r1"]["B6"] == "RB01"

    # -- Forward read, right end only ------------------------------------------

    def test_forward_right_end_only(self, tmp_path):
        """Forward read: only read end has construct → BM='read_end_only'.

        This is the case dorado caught but pre-fix smftools missed.
        """
        right = self.RC_AMPLICON + self.RC_RB01 + self.RC_ADAPTER
        seq = self.FILLER + right
        reads = [{"name": "r1", "sequence": seq}]
        tags = self._run(
            tmp_path,
            reads,
            barcode_kit_config=self._make_kit(),
        )
        assert tags["r1"]["BC"] == "RB01"
        assert tags["r1"]["BM"] == "read_end_only"
        assert "B5" not in tags["r1"]
        assert tags["r1"]["B6"] == "RB01"

    # -- Forward read, left end only -------------------------------------------

    def test_forward_left_end_only(self, tmp_path):
        """Forward read: only read start has construct → BM='read_start_only'."""
        left = self.ADAPTER + self.BC_REFS["RB01"] + self.AMPLICON
        seq = left + self.FILLER
        reads = [{"name": "r1", "sequence": seq}]
        tags = self._run(
            tmp_path,
            reads,
            barcode_kit_config=self._make_kit(),
        )
        assert tags["r1"]["BC"] == "RB01"
        assert tags["r1"]["BM"] == "read_start_only"
        assert tags["r1"]["B5"] == "RB01"
        assert "B6" not in tags["r1"]

    @pytest.mark.skip(reason="Amplicon-gap tolerance removed with composite/single-flank logic.")
    def test_forward_amplicon_gap_tolerance(self, tmp_path):
        """Forward read: barcode is near amplicon with small gap, recovered with tolerance."""
        gap = "NNN"
        left = self.BC_REFS["RB01"] + gap + self.AMPLICON
        seq = left + self.FILLER
        reads = [{"name": "r1", "sequence": seq}]
        tags = self._run(
            tmp_path,
            reads,
            barcode_kit_config=self._make_kit(),
            barcode_adapter_matcher="exact",
            barcode_amplicon_gap_tolerance=5,
        )
        assert tags["r1"]["BC"] == "RB01"
        assert tags["r1"]["BM"] == "read_start_only"
        assert tags["r1"]["B5"] == "RB01"
        assert "B6" not in tags["r1"]

    # -- Reverse read, both ends -----------------------------------------------

    @pytest.mark.skip(reason="Barcode extraction now uses read start/end for unaligned BAMs.")
    def test_reverse_both_ends(self, tmp_path):
        """Reverse read: pysam returns RC of original molecule.

        Read start = right ref end in forward orientation (needs_rc=False)
        Read end = left ref end in RC orientation (needs_rc=True)
        """
        # Right ref end appears at read start in forward orientation
        read_start = self.ADAPTER + self.BC_REFS["RB01"] + self.AMPLICON
        # Left ref end appears at read end in RC orientation
        read_end = self.RC_AMPLICON + self.RC_RB01 + self.RC_ADAPTER
        seq = read_start + self.FILLER + read_end
        reads = [{"name": "r1", "sequence": seq, "is_reverse": True}]
        tags = self._run(
            tmp_path,
            reads,
            barcode_kit_config=self._make_kit(),
        )
        assert tags["r1"]["BC"] == "RB01"
        assert tags["r1"]["BM"] == "both"
        assert tags["r1"]["B5"] == "RB01"
        assert tags["r1"]["B6"] == "RB01"

    # -- Reverse read, left end only (appears at read end) ---------------------

    @pytest.mark.skip(reason="Barcode extraction now uses read start/end for unaligned BAMs.")
    def test_reverse_left_end_only(self, tmp_path):
        """Reverse read: only left ref end present (at read end, RC'd) → 'read_end_only'."""
        read_end = self.RC_AMPLICON + self.RC_RB01 + self.RC_ADAPTER
        seq = self.FILLER + read_end
        reads = [{"name": "r1", "sequence": seq, "is_reverse": True}]
        tags = self._run(
            tmp_path,
            reads,
            barcode_kit_config=self._make_kit(),
        )
        assert tags["r1"]["BC"] == "RB01"
        assert tags["r1"]["BM"] == "read_end_only"
        assert tags["r1"]["B5"] == "RB01"
        assert "B6" not in tags["r1"]

    # -- Reverse read, right end only (appears at read start) ------------------

    @pytest.mark.skip(reason="Barcode extraction now uses read start/end for unaligned BAMs.")
    def test_reverse_right_end_only(self, tmp_path):
        """Reverse read: only right ref end present (at read start, fwd) → 'read_start_only'."""
        read_start = self.ADAPTER + self.BC_REFS["RB01"] + self.AMPLICON
        seq = read_start + self.FILLER
        reads = [{"name": "r1", "sequence": seq, "is_reverse": True}]
        tags = self._run(
            tmp_path,
            reads,
            barcode_kit_config=self._make_kit(),
        )
        assert tags["r1"]["BC"] == "RB01"
        assert tags["r1"]["BM"] == "read_start_only"
        assert "B5" not in tags["r1"]
        assert tags["r1"]["B6"] == "RB01"

    # -- Mismatch between ends -------------------------------------------------

    def test_forward_mismatch_ends(self, tmp_path):
        """Forward read: different barcodes at each end → 'mismatch'."""
        left = self.ADAPTER + self.BC_REFS["RB01"] + self.AMPLICON
        right = self.RC_AMPLICON + self.RC_RB02 + self.RC_ADAPTER
        seq = left + self.FILLER + right
        reads = [{"name": "r1", "sequence": seq}]
        tags = self._run(
            tmp_path,
            reads,
            barcode_kit_config=self._make_kit(),
        )
        assert tags["r1"]["BC"] == "unclassified"
        assert tags["r1"]["BM"] == "mismatch"
        assert tags["r1"]["B5"] == "RB01"
        assert tags["r1"]["B6"] == "RB02"

    # -- Mixed forward + reverse reads ----------------------------------------

    @pytest.mark.skip(reason="Barcode extraction now uses read start/end for unaligned BAMs.")
    def test_mixed_strand_batch(self, tmp_path):
        """Multiple reads on both strands produce correct results."""
        fwd_left = self.ADAPTER + self.BC_REFS["RB01"] + self.AMPLICON
        fwd_right = self.RC_AMPLICON + self.RC_RB01 + self.RC_ADAPTER
        rev_start = self.ADAPTER + self.BC_REFS["RB01"] + self.AMPLICON
        rev_end = self.RC_AMPLICON + self.RC_RB01 + self.RC_ADAPTER

        reads = [
            {"name": "fwd_both", "sequence": fwd_left + self.FILLER + fwd_right},
            {"name": "fwd_left", "sequence": fwd_left + self.FILLER},
            {"name": "fwd_right", "sequence": self.FILLER + fwd_right},
            {"name": "rev_both", "sequence": rev_start + self.FILLER + rev_end, "is_reverse": True},
            {"name": "rev_left", "sequence": self.FILLER + rev_end, "is_reverse": True},
            {"name": "rev_right", "sequence": rev_start + self.FILLER, "is_reverse": True},
            {"name": "nothing", "sequence": "N" * 100},
        ]
        tags = self._run(
            tmp_path,
            reads,
            barcode_kit_config=self._make_kit(),
        )
        assert tags["fwd_both"]["BM"] == "both"
        assert tags["fwd_both"]["BC"] == "RB01"
        assert tags["fwd_left"]["BM"] == "read_start_only"
        assert tags["fwd_left"]["BC"] == "RB01"
        assert tags["fwd_right"]["BM"] == "read_end_only"
        assert tags["fwd_right"]["BC"] == "RB01"
        assert tags["rev_both"]["BM"] == "both"
        assert tags["rev_both"]["BC"] == "RB01"
        assert tags["rev_left"]["BM"] == "read_start_only"
        assert tags["rev_left"]["BC"] == "RB01"
        assert tags["rev_right"]["BM"] == "read_end_only"
        assert tags["rev_right"]["BC"] == "RB01"
        assert tags["nothing"]["BM"] == "unclassified"

    # -- Adapter-only flank mode -----------------------------------------------

    def test_forward_both_ends_adapter_only(self, tmp_path):
        """adapter_only mode still works with RC logic."""
        left = self.ADAPTER + self.BC_REFS["RB01"] + self.AMPLICON
        right = self.RC_AMPLICON + self.RC_RB01 + self.RC_ADAPTER
        seq = left + self.FILLER + right
        reads = [{"name": "r1", "sequence": seq}]
        tags = self._run(
            tmp_path,
            reads,
            barcode_kit_config=self._make_kit(),
        )
        assert tags["r1"]["BC"] == "RB01"
        assert tags["r1"]["BM"] == "both"

    # -- Fuzzy matching (edlib with errors) ------------------------------------

    def test_forward_both_ends_with_errors(self, tmp_path):
        """Barcode with 2 base errors still matches within edit distance 5."""
        # Introduce 2 substitution errors into RB01 at left end
        mutated_bc = "TAGAAAGTTGTCGGTGTCTTTGTG"  # A→T at pos 0
        # Right end: RC of a differently mutated copy
        mutated_rc_bc = _reverse_complement("AAGAAAGTTGTCGGTGTCTTTGCG")  # G→C at pos 22
        left = self.ADAPTER + mutated_bc + self.AMPLICON
        right = self.RC_AMPLICON + mutated_rc_bc + self.RC_ADAPTER
        seq = left + self.FILLER + right
        reads = [{"name": "r1", "sequence": seq}]
        tags = self._run(
            tmp_path,
            reads,
            barcode_kit_config=self._make_kit(),
        )
        assert tags["r1"]["BC"] == "RB01"
        assert tags["r1"]["BM"] == "both"
