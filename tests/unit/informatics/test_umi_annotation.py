import pandas as pd
import pytest

from smftools.informatics import bam_functions
from smftools.informatics.bam_functions import (
    FlankingConfig,
    PerEndFlankingConfig,
    UMIKitConfig,
    _extract_sequence_with_flanking,
)

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
    """Create a coordinate-sorted, indexed BAM file."""
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
    """Return ``{read_name: {tag: value}}`` for every read."""
    out = {}
    with _pysam.AlignmentFile(str(bam_path), "rb") as fh:
        for read in fh.fetch(until_eof=True):
            out[read.query_name] = dict(read.get_tags())
    return out


def _read_parquet_tags(parquet_path):
    """Return ``{read_name: {tag: value}}`` from a Parquet sidecar file.

    Only includes non-null tag values to match the BAM tag semantics
    (missing tag == not present).
    """
    df = pd.read_parquet(parquet_path)
    out = {}
    for _, row in df.iterrows():
        tags = {}
        for col in df.columns:
            if col == "read_name":
                continue
            val = row[col]
            if pd.notna(val):
                tags[col] = val
        out[row["read_name"]] = tags
    return out


class TestUMIFlankingExtraction:
    """Tests for UMI extraction using flanking sequences."""

    def test_umi_adapter_only_from_start(self):
        """Extract UMI after adapter_side at read start."""
        # Structure: ADAPTER + UMI + rest
        read = "GTACTGACAATTCCGGDDDDNNNNNNNN"
        flanking = FlankingConfig(adapter_side="GTACTGAC")
        umi, start, end = _extract_sequence_with_flanking(
            read_sequence=read,
            target_length=8,
            search_window=50,
            search_from_start=True,
            flanking=flanking,
            flank_mode="adapter_only",
            adapter_matcher="exact",
        )
        assert umi == "AATTCCGG"
        assert start == 8
        assert end == 16

    def test_umi_both_mode(self):
        """Extract UMI with both flanking sequences validated."""
        # Structure: ADAPTER + UMI + AMPLICON + rest
        read = "GTACTGACAATTCCGGAACCTTGGNNNNNNNN"
        flanking = FlankingConfig(adapter_side="GTACTGAC", amplicon_side="AACCTTGG")
        umi, start, end = _extract_sequence_with_flanking(
            read_sequence=read,
            target_length=8,
            search_window=50,
            search_from_start=True,
            flanking=flanking,
            flank_mode="both",
            adapter_matcher="exact",
        )
        assert umi == "AATTCCGG"

    def test_umi_both_mode_fails_wrong_amplicon(self):
        """Both mode returns None when amplicon is wrong."""
        read = "GTACTGACAATTCCGGXXXXXXXXNNNN"
        flanking = FlankingConfig(adapter_side="GTACTGAC", amplicon_side="AACCTTGG")
        umi, start, end = _extract_sequence_with_flanking(
            read_sequence=read,
            target_length=8,
            search_window=50,
            search_from_start=True,
            flanking=flanking,
            flank_mode="both",
            adapter_matcher="exact",
        )
        assert umi is None

    def test_umi_amplicon_only_from_start(self):
        """Extract UMI before amplicon_side at read start."""
        read = "AATTCCGGAACCTTGGNNNNNNNN"
        flanking = FlankingConfig(amplicon_side="AACCTTGG")
        umi, start, end = _extract_sequence_with_flanking(
            read_sequence=read,
            target_length=8,
            search_window=50,
            search_from_start=True,
            flanking=flanking,
            flank_mode="amplicon_only",
            adapter_matcher="exact",
        )
        assert umi == "AATTCCGG"

    def test_umi_composite_from_start(self, monkeypatch):
        """Extract UMI with composite alignment."""

        class _FakeEdlib:
            @staticmethod
            def align(query, target, mode, task, k, additionalEqualities=None):
                assert mode == "HW"
                assert task == "path"
                assert k == 0
                # Exact match for entire query against target
                return {
                    "editDistance": 0,
                    "locations": [(0, len(query) - 1)],
                    "cigar": f"{len(query)}=",
                }

        monkeypatch.setattr(bam_functions, "require", lambda *args, **kwargs: _FakeEdlib())
        read = "GTACTGACAATTCCGGAACCTTGGNNNN"
        flanking = FlankingConfig(adapter_side="GTACTGAC", amplicon_side="AACCTTGG")
        umi, start, end = _extract_sequence_with_flanking(
            read_sequence=read,
            target_length=8,
            search_window=50,
            search_from_start=True,
            flanking=flanking,
            flank_mode="composite",
            adapter_matcher="edlib",
            adapter_max_edits=0,
            amplicon_max_edits=0,
        )
        assert umi == "AATTCCGG"

    def test_per_end_different_flanking(self):
        """Different flanking config per reference end."""
        left_flanking = FlankingConfig(adapter_side="AAAA")
        right_flanking = FlankingConfig(adapter_side="TTTT")

        # Test left end extraction (search from start)
        read_left = "AAAACCCCGGGG"
        umi_left, _, _ = _extract_sequence_with_flanking(
            read_sequence=read_left,
            target_length=4,
            search_window=50,
            search_from_start=True,
            flanking=left_flanking,
            flank_mode="adapter_only",
            adapter_matcher="exact",
        )
        assert umi_left == "CCCC"

        # Test right end extraction (search from end)
        read_right = "GGGGCCCCTTTT"
        umi_right, _, _ = _extract_sequence_with_flanking(
            read_sequence=read_right,
            target_length=4,
            search_window=50,
            search_from_start=False,
            flanking=right_flanking,
            flank_mode="adapter_only",
            adapter_matcher="exact",
        )
        assert umi_right == "CCCC"


# ---------------------------------------------------------------------------
# Tier 1: annotate_umi_tags_in_bam with mock BAM data
# ---------------------------------------------------------------------------
@requires_pysam
class TestAnnotateUmiTagsInBam:
    """Integration tests for UMI annotation orchestration."""

    def _run(self, tmp_path, reads, **kwargs):
        """Create BAM -> run UMI annotation -> return {name: {tag: val}} from Parquet sidecar."""
        bam = tmp_path / "test.bam"
        _create_test_bam(bam, reads)
        defaults = dict(
            use_umi=True,
            umi_kit_config=UMIKitConfig(
                flanking=PerEndFlankingConfig(
                    left_ref_end=FlankingConfig(adapter_side="ACGT"),
                    right_ref_end=FlankingConfig(adapter_side="TGCA"),
                ),
                length=4,
                umi_ends="both",
                umi_flank_mode="adapter_only",
            ),
            umi_search_window=200,
            umi_adapter_matcher="exact",
            umi_adapter_max_edits=0,
            samtools_backend="python",
        )
        defaults.update(kwargs)
        sidecar = bam_functions.annotate_umi_tags_in_bam(bam, **defaults)
        return _read_parquet_tags(sidecar)

    # -- use_umi=False --------------------------------------------------------

    def test_use_umi_false_returns_early(self, tmp_path):
        """use_umi=False returns input path without modification."""
        bam = tmp_path / "test.bam"
        _create_test_bam(bam, [{"name": "r1", "sequence": "ACGTAAAANNNNNNNN"}])
        result = bam_functions.annotate_umi_tags_in_bam(
            bam,
            use_umi=False,
            umi_kit_config=UMIKitConfig(
                flanking=PerEndFlankingConfig(
                    left_ref_end=FlankingConfig(adapter_side="ACGT"),
                ),
                length=4,
            ),
            samtools_backend="python",
        )
        assert result == bam
        # No sidecar should be created
        sidecar = bam.with_suffix(".umi_tags.parquet")
        assert not sidecar.exists()
        # BAM should have no UMI tags
        tags = _read_bam_tags(bam)
        assert "U1" not in tags["r1"]
        assert "U2" not in tags["r1"]
        assert "US" not in tags["r1"]
        assert "UE" not in tags["r1"]
        assert "RX" not in tags["r1"]
        assert "FC" not in tags["r1"]

    # -- Flanking-based extraction -------------------------------------------

    def test_flanking_umi_extraction(self, tmp_path):
        """Flanking-based UMI extraction with UMIKitConfig (forward read, left_only)."""
        umi_kit = UMIKitConfig(
            flanking=PerEndFlankingConfig(
                left_ref_end=FlankingConfig(adapter_side="ACGT"),
            ),
            length=4,
            umi_ends="left_only",
            umi_flank_mode="adapter_only",
        )
        reads = [{"name": "r1", "sequence": "ACGTAAAANNNNNNNNNNNNNNNN"}]
        tags = self._run(
            tmp_path,
            reads,
            umi_kit_config=umi_kit,
            umi_ends="left_only",
            umi_flank_mode="adapter_only",
        )
        # US is delimited: "UMI_seq;slot;flank_seq"
        assert tags["r1"]["US"] == "AAAA;top;ACGT"
        # Forward read: U1=US, U2=UE
        assert tags["r1"]["U1"] == "AAAA"
        assert "U2" not in tags["r1"]
        assert "UE" not in tags["r1"]
        assert tags["r1"]["RX"] == "AAAA"
        assert tags["r1"]["FC"] == "top"

    def test_flanking_top_bottom_across_ends(self, tmp_path):
        """Top flank at read start, bottom flank at read end (forward read)."""
        umi_kit = UMIKitConfig(
            flanking=PerEndFlankingConfig(
                left_ref_end=FlankingConfig(adapter_side="GCTA"),
                right_ref_end=FlankingConfig(adapter_side="CCGA"),
            ),
            length=4,
            umi_ends="both",
            umi_flank_mode="adapter_only",
        )
        # start: GCTA + GGTT
        # end: UMI2_rc (TCGT) + RC(bottom adapter) (TCGG)
        reads = [{"name": "r1", "sequence": "GCTAGGTTNNNNNNNNNNTCGTTCGG"}]
        tags = self._run(
            tmp_path,
            reads,
            umi_kit_config=umi_kit,
            umi_ends="both",
            umi_flank_mode="adapter_only",
        )
        # Delimited US/UE
        assert tags["r1"]["US"] == "GGTT;top;GCTA"
        assert tags["r1"]["UE"] == "ACGA;bottom;CCGA"
        # Forward read: U1=US, U2=UE
        assert tags["r1"]["U1"] == "GGTT"
        assert tags["r1"]["U2"] == "ACGA"
        assert tags["r1"]["RX"] == "GGTT-ACGA"
        assert tags["r1"]["FC"] == "top-bottom"

    # -- Reverse-read orientation swap ----------------------------------------

    def test_reverse_read_swaps_u1_u2(self, tmp_path):
        """Reverse-mapped read: U1=UE, U2=US (swapped from forward)."""
        umi_kit = UMIKitConfig(
            flanking=PerEndFlankingConfig(
                left_ref_end=FlankingConfig(adapter_side="GCTA"),
                right_ref_end=FlankingConfig(adapter_side="CCGA"),
            ),
            length=4,
            umi_ends="both",
            umi_flank_mode="adapter_only",
        )
        # Same sequence as test_flanking_top_bottom_across_ends but reverse-mapped
        reads = [{"name": "r1", "sequence": "GCTAGGTTNNNNNNNNNNTCGTTCGG", "is_reverse": True}]
        tags = self._run(
            tmp_path,
            reads,
            umi_kit_config=umi_kit,
            umi_ends="both",
            umi_flank_mode="adapter_only",
        )
        # US/UE are positional (unchanged by orientation)
        assert tags["r1"]["US"] == "GGTT;top;GCTA"
        assert tags["r1"]["UE"] == "ACGA;bottom;CCGA"
        # Reverse read: U1=UE, U2=US
        assert tags["r1"]["U1"] == "ACGA"  # from UE
        assert tags["r1"]["U2"] == "GGTT"  # from US
        assert tags["r1"]["RX"] == "ACGA-GGTT"
        assert tags["r1"]["FC"] == "bottom-top"

    # -- umi_ends filtering ---------------------------------------------------

    def test_umi_ends_left_only(self, tmp_path):
        """umi_ends='left_only' skips right end."""
        umi_kit = UMIKitConfig(
            flanking=PerEndFlankingConfig(
                left_ref_end=FlankingConfig(adapter_side="ACGT"),
                right_ref_end=FlankingConfig(adapter_side="TGCA"),
            ),
            length=4,
            umi_ends="both",
            umi_flank_mode="adapter_only",
        )
        reads = [{"name": "r1", "sequence": "ACGTAAAANNNNNNNNCCCCTGCA"}]
        tags = self._run(tmp_path, reads, umi_kit_config=umi_kit, umi_ends="left_only")
        assert tags["r1"]["US"] == "AAAA;top;ACGT"
        # Forward: U1=US
        assert tags["r1"]["U1"] == "AAAA"
        assert "U2" not in tags["r1"]
        assert "UE" not in tags["r1"]
        assert tags["r1"]["RX"] == "AAAA"
        assert tags["r1"]["FC"] == "top"

    def test_umi_ends_right_only(self, tmp_path):
        """umi_ends='right_only' skips left end."""
        umi_kit = UMIKitConfig(
            flanking=PerEndFlankingConfig(
                left_ref_end=FlankingConfig(adapter_side="ACGT"),
                right_ref_end=FlankingConfig(adapter_side="TGCA"),
            ),
            length=4,
            umi_ends="both",
            umi_flank_mode="adapter_only",
        )
        reads = [{"name": "r1", "sequence": "ACGTAAAANNNNNNNNCCCCTGCA"}]
        tags = self._run(tmp_path, reads, umi_kit_config=umi_kit, umi_ends="right_only")
        assert "US" not in tags["r1"]
        # bottom flank from read end: RC(TGCA)=TGCA found, UMI before = CCCC, then RC'd = GGGG
        assert tags["r1"]["UE"] == "GGGG;bottom;TGCA"
        # Forward read: U1=US=None, U2=UE
        assert "U1" not in tags["r1"]
        assert tags["r1"]["U2"] == "GGGG"
        assert tags["r1"]["RX"] == "GGGG"
        assert tags["r1"]["FC"] == "bottom"

    # -- No UMI found ---------------------------------------------------------

    def test_no_umi_found(self, tmp_path):
        """No adapter found -> no UMI tags set."""
        reads = [{"name": "r1", "sequence": "TTTTTTTTTTTTTTTTTTTTTTTT"}]
        tags = self._run(tmp_path, reads)
        assert "U1" not in tags["r1"]
        assert "U2" not in tags["r1"]
        assert "US" not in tags["r1"]
        assert "UE" not in tags["r1"]
        assert "RX" not in tags["r1"]
        assert "FC" not in tags["r1"]

    # -- Multiple reads -------------------------------------------------------

    def test_multiple_reads_mixed(self, tmp_path):
        """Multiple reads produce correct per-read UMI tags."""
        reads = [
            {"name": "both", "sequence": "ACGTAAAANNNNNNNNCCCCTGCA"},
            {"name": "left", "sequence": "ACGTGGGGNNNNNNNNNNNNNNNN"},
            {"name": "none", "sequence": "TTTTTTTTTTTTTTTTTTTTTTTT"},
        ]
        umi_kit = UMIKitConfig(
            flanking=PerEndFlankingConfig(
                left_ref_end=FlankingConfig(adapter_side="ACGT"),
                right_ref_end=FlankingConfig(adapter_side="TGCA"),
            ),
            length=4,
            umi_ends="both",
            umi_flank_mode="adapter_only",
        )
        tags = self._run(tmp_path, reads, umi_kit_config=umi_kit)
        # "both" read: top adapter at start, bottom adapter at end
        assert "U1" in tags["both"]
        assert "US" in tags["both"]
        assert "RX" in tags["both"]
        assert "FC" in tags["both"]
        # "left" read: only top adapter at start (forward: U1=US)
        assert tags["left"]["U1"] == "GGGG"
        assert tags["left"]["US"] == "GGGG;top;ACGT"
        assert tags["left"]["RX"] == "GGGG"
        assert tags["left"]["FC"] == "top"
        # "none" read: no adapters
        assert "RX" not in tags["none"]
        assert "US" not in tags["none"]
        assert "FC" not in tags["none"]

    # -- Multiprocessing path -------------------------------------------------

    def test_multiprocessing_produces_same_results(self, tmp_path):
        """threads=2 produces identical tags to single-threaded path."""
        umi_kit = UMIKitConfig(
            flanking=PerEndFlankingConfig(
                left_ref_end=FlankingConfig(adapter_side="GCTA"),
                right_ref_end=FlankingConfig(adapter_side="CCGA"),
            ),
            length=4,
            umi_ends="both",
            umi_flank_mode="adapter_only",
        )
        reads = [
            {"name": "r1", "sequence": "GCTAGGTTNNNNNNNNNNTCGTTCGG"},
            {"name": "r2", "sequence": "GCTAGGTTNNNNNNNNNNTCGTTCGG"},
            {"name": "r3", "sequence": "TTTTTTTTTTTTTTTTTTTTTTTT"},
        ]
        single_dir = tmp_path / "single"
        single_dir.mkdir()
        multi_dir = tmp_path / "multi"
        multi_dir.mkdir()
        tags_single = self._run(
            single_dir,
            reads,
            umi_kit_config=umi_kit,
            umi_ends="both",
            umi_flank_mode="adapter_only",
            threads=1,
        )
        tags_multi = self._run(
            multi_dir,
            reads,
            umi_kit_config=umi_kit,
            umi_ends="both",
            umi_flank_mode="adapter_only",
            threads=2,
        )
        for name in ("r1", "r2", "r3"):
            assert tags_single[name] == tags_multi[name]
