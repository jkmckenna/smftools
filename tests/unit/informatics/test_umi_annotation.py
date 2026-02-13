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
    # Read has two "ACGT" adapters. When searching from end, should use the
    # one nearest to the end (second occurrence) and extract UMI before it.
    # Structure: NNNNNNNN ACGT AAAA TTTT ACGT GGGG
    #                     ^^^1      ^^^^ ^^^2
    #                              UMI  adapter (nearest to end)
    read = "NNNNNNNNACGTAAAATTTTACGTGGGG"
    umi = bam_functions._extract_umi_adjacent_to_adapter_on_read_end(
        read_sequence=read,
        adapter_sequence="ACGT",
        umi_length=4,
        umi_search_window=10,
        search_from_start=False,
    )
    # UMI is extracted BEFORE the adapter when searching from end
    assert umi == "TTTT"


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

    LEFT_ADAPTER = "ACGT"
    RIGHT_ADAPTER = "TGCA"

    def _run(self, tmp_path, reads, **kwargs):
        """Create BAM → run UMI annotation → return {name: {tag: val}}."""
        bam = tmp_path / "test.bam"
        _create_test_bam(bam, reads)
        defaults = dict(
            use_umi=True,
            umi_adapters=[self.LEFT_ADAPTER, self.RIGHT_ADAPTER],
            umi_length=4,
            umi_search_window=200,
            umi_adapter_matcher="exact",
            umi_adapter_max_edits=0,
            samtools_backend="python",
        )
        defaults.update(kwargs)
        bam_functions.annotate_umi_tags_in_bam(bam, **defaults)
        return _read_bam_tags(bam)

    # -- Legacy adapter path --------------------------------------------------

    def test_legacy_both_ends_umi(self, tmp_path):
        """UMI extracted at both ends → U1, U2, and combined RX."""
        # ACGT(0-3) AAAA(4-7) NNNNNNNN(8-15) CCCC(16-19) TGCA(20-23)
        reads = [{"name": "r1", "sequence": "ACGTAAAANNNNNNNNCCCCTGCA"}]
        tags = self._run(tmp_path, reads)
        assert tags["r1"]["U1"] == "AAAA"
        assert tags["r1"]["U2"] == "CCCC"
        assert tags["r1"]["RX"] == "AAAA-CCCC"

    def test_legacy_left_only_umi(self, tmp_path):
        """UMI only at left end → U1 and RX set, no U2."""
        reads = [{"name": "r1", "sequence": "ACGTGGGGNNNNNNNNNNNNNNNN"}]
        tags = self._run(tmp_path, reads)
        assert tags["r1"]["U1"] == "GGGG"
        assert "U2" not in tags["r1"]
        assert tags["r1"]["RX"] == "GGGG"

    def test_legacy_right_only_umi(self, tmp_path):
        """UMI only at right end → U2 and RX set, no U1."""
        reads = [{"name": "r1", "sequence": "NNNNNNNNNNNNNNNNCCCCTGCA"}]
        tags = self._run(tmp_path, reads)
        assert "U1" not in tags["r1"]
        assert tags["r1"]["U2"] == "CCCC"
        assert tags["r1"]["RX"] == "CCCC"

    # -- use_umi=False --------------------------------------------------------

    def test_use_umi_false_returns_early(self, tmp_path):
        """use_umi=False returns input path without modification."""
        bam = tmp_path / "test.bam"
        _create_test_bam(bam, [{"name": "r1", "sequence": "ACGTAAAANNNNNNNN"}])
        result = bam_functions.annotate_umi_tags_in_bam(
            bam,
            use_umi=False,
            umi_adapters=[None, None],
            umi_length=0,
            samtools_backend="python",
        )
        assert result == bam
        tags = _read_bam_tags(bam)
        assert "U1" not in tags["r1"]
        assert "U2" not in tags["r1"]
        assert "RX" not in tags["r1"]

    # -- umi_ends filtering ---------------------------------------------------

    def test_umi_ends_left_only(self, tmp_path):
        """umi_ends='left_only' skips right end."""
        reads = [{"name": "r1", "sequence": "ACGTAAAANNNNNNNNCCCCTGCA"}]
        tags = self._run(tmp_path, reads, umi_ends="left_only")
        assert tags["r1"]["U1"] == "AAAA"
        assert "U2" not in tags["r1"]
        assert tags["r1"]["RX"] == "AAAA"

    def test_umi_ends_right_only(self, tmp_path):
        """umi_ends='right_only' skips left end."""
        reads = [{"name": "r1", "sequence": "ACGTAAAANNNNNNNNCCCCTGCA"}]
        tags = self._run(tmp_path, reads, umi_ends="right_only")
        assert "U1" not in tags["r1"]
        assert tags["r1"]["U2"] == "CCCC"
        assert tags["r1"]["RX"] == "CCCC"

    # -- Flanking-based extraction -------------------------------------------

    def test_flanking_umi_extraction(self, tmp_path):
        """Flanking-based UMI extraction with UMIKitConfig."""
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
        assert tags["r1"]["U1"] == "AAAA"
        assert "U2" not in tags["r1"]
        assert tags["r1"]["RX"] == "AAAA"

    def test_flanking_top_bottom_across_ends(self, tmp_path):
        """Top flank in read start -> U1, bottom flank in read end -> U2 (RC)."""
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
        assert tags["r1"]["U1"] == "GGTT"
        assert tags["r1"]["U2"] == "ACGA"
        assert tags["r1"]["RX"] == "GGTT-ACGA"

    # -- Strand handling ------------------------------------------------------

    def test_reverse_strand_umi(self, tmp_path):
        """Reverse read: left ref → read end, right ref → read start."""
        # TGCA(0-3) GGGG(4-7) NNNNNNNN(8-15) TTTT(16-19) ACGT(20-23)
        reads = [{"name": "r1", "sequence": "TGCAGGGGNNNNNNNNTTTTACGT", "is_reverse": True}]
        tags = self._run(tmp_path, reads)
        # Left ref → search from end → find ACGT at 20-24 → UMI before = TTTT
        assert tags["r1"]["U1"] == "TTTT"
        # Right ref → search from start → find TGCA at 0-4 → UMI after = GGGG
        assert tags["r1"]["U2"] == "GGGG"
        assert tags["r1"]["RX"] == "TTTT-GGGG"

    # -- No UMI found ---------------------------------------------------------

    def test_no_umi_found(self, tmp_path):
        """No adapter found → no UMI tags set."""
        reads = [{"name": "r1", "sequence": "TTTTTTTTTTTTTTTTTTTTTTTT"}]
        tags = self._run(tmp_path, reads)
        assert "U1" not in tags["r1"]
        assert "U2" not in tags["r1"]
        assert "RX" not in tags["r1"]

    # -- Multiple reads -------------------------------------------------------

    def test_multiple_reads_mixed(self, tmp_path):
        """Multiple reads produce correct per-read UMI tags."""
        reads = [
            {"name": "both", "sequence": "ACGTAAAANNNNNNNNCCCCTGCA"},
            {"name": "left", "sequence": "ACGTGGGGNNNNNNNNNNNNNNNN"},
            {"name": "none", "sequence": "TTTTTTTTTTTTTTTTTTTTTTTT"},
        ]
        tags = self._run(tmp_path, reads)
        assert tags["both"]["RX"] == "AAAA-CCCC"
        assert tags["left"]["RX"] == "GGGG"
        assert "RX" not in tags["none"]
