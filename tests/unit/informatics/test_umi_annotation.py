import shutil
import subprocess

import pandas as pd
import pytest

from smftools.constants import UMI_KIT_ALIASES
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

HAS_SAMTOOLS = shutil.which("samtools") is not None

requires_pysam = pytest.mark.skipif(not HAS_PYSAM, reason="pysam not installed")
requires_samtools = pytest.mark.skipif(not HAS_SAMTOOLS, reason="samtools not in PATH")


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
        assert tags["r1"]["FC"] == "top-NA"

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

    def test_reverse_read_same_u1_u2_swapped_us_ue(self, tmp_path):
        """Reverse-mapped read of the same fragment: U1/U2 match forward, US/UE swap."""
        umi_kit = UMIKitConfig(
            flanking=PerEndFlankingConfig(
                left_ref_end=FlankingConfig(adapter_side="GCTA"),
                right_ref_end=FlankingConfig(adapter_side="CCGA"),
            ),
            length=4,
            umi_ends="both",
            umi_flank_mode="adapter_only",
        )
        # Same BAM sequence as the forward test (same fragment, reference orientation).
        # The code will RC it to recover the original read orientation before extraction.
        reads = [{"name": "r1", "sequence": "GCTAGGTTNNNNNNNNNNTCGTTCGG", "is_reverse": True}]
        tags = self._run(
            tmp_path,
            reads,
            umi_kit_config=umi_kit,
            umi_ends="both",
            umi_flank_mode="adapter_only",
        )
        # US/UE are positional (swapped from forward because original read is RC'd)
        assert tags["r1"]["US"] == "ACGA;bottom;CCGA"
        assert tags["r1"]["UE"] == "GGTT;top;GCTA"
        # Genomic U1/U2 are the same as forward (U1=left ref end, U2=right ref end)
        assert tags["r1"]["U1"] == "GGTT"  # same as forward
        assert tags["r1"]["U2"] == "ACGA"  # same as forward
        assert tags["r1"]["RX"] == "GGTT-ACGA"
        assert tags["r1"]["FC"] == "top-bottom"

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
        assert tags["r1"]["FC"] == "top-NA"

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
        assert tags["r1"]["FC"] == "NA-bottom"

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
        assert tags["left"]["FC"] == "top-NA"
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


@requires_pysam
class TestDualNextera12SyntheticFragments:
    """Synthetic fragment tests using the dual-nextera-12 UMI YAML config."""

    @staticmethod
    def _rc(seq: str) -> str:
        trans = str.maketrans("ACGTNacgtn", "TGCANtgcan")
        return seq.translate(trans)[::-1]

    def _run_dual_nextera(self, tmp_path, reads, **overrides):
        """Create BAM, run UMI annotation with dual-nextera-12 config, return sidecar tags."""
        bam = tmp_path / "dual_nextera_test.bam"
        _create_test_bam(bam, reads)
        umi_cfg = bam_functions.load_umi_config_from_yaml(UMI_KIT_ALIASES["dual-nextera-12"])
        defaults = dict(
            use_umi=True,
            umi_kit_config=umi_cfg,
            umi_search_window=200,
            umi_adapter_matcher="exact",
            umi_adapter_max_edits=0,
            samtools_backend="python",
        )
        defaults.update(overrides)
        sidecar = bam_functions.annotate_umi_tags_in_bam(bam, **defaults)
        return _read_parquet_tags(sidecar)

    @pytest.mark.parametrize(
        "label,start_slot,end_slot,is_reverse",
        [
            ("proper_top_bottom_fwd", "top", "bottom", False),
            ("improper_top_top_fwd", "top", "top", False),
            ("improper_bottom_bottom_fwd", "bottom", "bottom", False),
            ("improper_bottom_top_fwd", "bottom", "top", False),
            ("proper_top_bottom_rev", "top", "bottom", True),
            ("improper_top_top_rev", "top", "top", True),
            ("improper_bottom_bottom_rev", "bottom", "bottom", True),
            ("improper_bottom_top_rev", "bottom", "top", True),
        ],
    )
    def test_dual_nextera_fragment_slot_combinations(
        self,
        tmp_path,
        label,
        start_slot,
        end_slot,
        is_reverse,
    ):
        """UMI tags should reflect expected slot pairing for synthetic dual-nextera fragments.

        Genomic tags (U1/U2/RX/FC) are orientation-invariant: they should be
        identical for forward and reverse reads of the same fragment.
        Positional tags (US/UE) swap when the read maps in reverse.
        """
        umi_cfg = bam_functions.load_umi_config_from_yaml(UMI_KIT_ALIASES["dual-nextera-12"])
        top_adapter = umi_cfg.flanking.left_ref_end.adapter_side
        bottom_adapter = umi_cfg.flanking.right_ref_end.adapter_side
        assert top_adapter is not None
        assert bottom_adapter is not None
        assert umi_cfg.length == 12

        # Intentionally distinct UMIs (not reverse/complement mirrors).
        u_start = "AACCGTGATCTA"
        u_end = "GTTCAGACCGTT"

        start_adapter = top_adapter if start_slot == "top" else bottom_adapter
        end_adapter = top_adapter if end_slot == "top" else bottom_adapter
        read_seq = (
            f"{start_adapter}{u_start}"
            f"{'N' * 40}"
            f"{self._rc(u_end)}{self._rc(end_adapter)}"
        )
        # BAM always stores the reference-oriented sequence. For forward and
        # reverse reads of the same fragment, the BAM sequence is the same.
        # The code will RC reverse reads to recover original read orientation.

        tags = self._run_dual_nextera(
            tmp_path,
            reads=[{"name": label, "sequence": read_seq, "is_reverse": is_reverse}],
            umi_adapter_matcher="exact",
            umi_adapter_max_edits=0,
            umi_ends="both",
            umi_flank_mode="adapter_only",
        )

        row = tags[label]

        # Positional tags (US/UE) swap for reverse reads
        if is_reverse:
            assert row["US"] == f"{u_end};{end_slot};{end_adapter}"
            assert row["UE"] == f"{u_start};{start_slot};{start_adapter}"
        else:
            assert row["US"] == f"{u_start};{start_slot};{start_adapter}"
            assert row["UE"] == f"{u_end};{end_slot};{end_adapter}"

        # Genomic tags (U1/U2/RX/FC) are the same regardless of orientation
        assert row["U1"] == u_start
        assert row["U2"] == u_end
        assert row["RX"] == f"{u_start}-{u_end}"
        assert row["FC"] == f"{start_slot}-{end_slot}"


# ---------------------------------------------------------------------------
# Helpers for sequence-orientation tests
# ---------------------------------------------------------------------------
def _rc(seq: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    trans = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(trans)[::-1]


def _read_sequences_pysam(bam_path):
    """Return {read_name: (query_sequence, is_reverse)} using pysam."""
    result = {}
    with _pysam.AlignmentFile(str(bam_path), "rb") as f:
        for read in f.fetch(until_eof=True):
            result[read.query_name] = (read.query_sequence, read.is_reverse)
    return result


def _read_sequences_samtools(bam_path):
    """Return {read_name: (SEQ, is_reverse)} using samtools view."""
    result = {}
    proc = subprocess.run(
        ["samtools", "view", str(bam_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    for line in proc.stdout.strip().split("\n"):
        if not line:
            continue
        fields = line.split("\t")
        name, flag, seq = fields[0], int(fields[1]), fields[9]
        is_reverse = bool(flag & 0x10)
        result[name] = (seq, is_reverse)
    return result


# ---------------------------------------------------------------------------
# Tier: Validate that pysam and samtools return identical sequences
# ---------------------------------------------------------------------------
@requires_pysam
@requires_samtools
class TestBamStorageOrientation:
    """Definitive test: does the BAM file store the sequence as-sequenced
    (original read) or in reference orientation?

    We use a strongly asymmetric sequence so RC is unmistakable.
    The SAM spec says SEQ is stored on the mapped strand (reference orientation).
    pysam's ``query_sequence`` property sets/gets the BAM SEQ field directly.
    pysam's ``get_forward_sequence()`` returns the original-read orientation
    (RC of SEQ for reverse-mapped reads).
    """

    ASYMMETRIC = "AAAAAACCCCCC"  # RC = GGGGGGTTTTTT — clearly different

    def test_pysam_stores_what_you_set(self, tmp_path):
        """pysam query_sequence round-trips the value we set, regardless of flag."""
        bam = tmp_path / "test.bam"
        _create_test_bam(
            bam,
            [
                {"name": "fwd", "sequence": self.ASYMMETRIC},
                {"name": "rev", "sequence": self.ASYMMETRIC, "is_reverse": True},
            ],
        )
        with _pysam.AlignmentFile(str(bam), "rb") as f:
            for read in f.fetch(until_eof=True):
                # Both forward and reverse reads return exactly what we stored
                assert read.query_sequence == self.ASYMMETRIC

    def test_samtools_returns_same_as_pysam(self, tmp_path):
        """samtools view SEQ field matches pysam query_sequence for both orientations."""
        bam = tmp_path / "test.bam"
        _create_test_bam(
            bam,
            [
                {"name": "fwd", "sequence": self.ASYMMETRIC},
                {"name": "rev", "sequence": self.ASYMMETRIC, "is_reverse": True},
            ],
        )
        samtools_seqs = _read_sequences_samtools(bam)
        # samtools view shows the same SEQ as pysam query_sequence
        assert samtools_seqs["fwd"][0] == self.ASYMMETRIC
        assert samtools_seqs["rev"][0] == self.ASYMMETRIC

    def test_get_forward_sequence_returns_rc_for_reverse(self, tmp_path):
        """get_forward_sequence() returns the original-read orientation.
        For reverse-mapped reads this is the RC of the stored SEQ."""
        bam = tmp_path / "test.bam"
        _create_test_bam(
            bam,
            [
                {"name": "fwd", "sequence": self.ASYMMETRIC},
                {"name": "rev", "sequence": self.ASYMMETRIC, "is_reverse": True},
            ],
        )
        with _pysam.AlignmentFile(str(bam), "rb") as f:
            for read in f.fetch(until_eof=True):
                fwd_seq = read.get_forward_sequence()
                if read.query_name == "fwd":
                    # Forward: get_forward_sequence == query_sequence
                    assert fwd_seq == self.ASYMMETRIC
                else:
                    # Reverse: get_forward_sequence == RC(query_sequence)
                    assert fwd_seq == _rc(self.ASYMMETRIC)
                    assert fwd_seq == "GGGGGGTTTTTT"

    def test_interpretation(self, tmp_path):
        """Demonstrate the correct interpretation:

        When we set query_sequence = X with flag 16, the BAM stores X as the
        **reference-oriented** sequence. The original read (as sequenced) was
        RC(X). Therefore to recover the original read from a reverse-mapped
        BAM record, we must RC the stored sequence.
        """
        # Suppose the original read (as sequenced) was "GGGGGGTTTTTT"
        original_read = "GGGGGGTTTTTT"
        # An aligner mapping this to the reverse strand would RC it for storage
        bam_stored = _rc(original_read)  # "AAAAAACCCCCC"
        assert bam_stored == "AAAAAACCCCCC"

        bam = tmp_path / "test.bam"
        _create_test_bam(bam, [{"name": "r1", "sequence": bam_stored, "is_reverse": True}])

        with _pysam.AlignmentFile(str(bam), "rb") as f:
            read = next(f.fetch(until_eof=True))
            # query_sequence = reference orientation (what aligner stored)
            assert read.query_sequence == bam_stored
            # get_forward_sequence = original read (as sequenced)
            assert read.get_forward_sequence() == original_read

        # samtools agrees
        samtools_seqs = _read_sequences_samtools(bam)
        assert samtools_seqs["r1"][0] == bam_stored


@requires_pysam
class TestPysamSequenceOrientation:
    """Verify pysam query_sequence behavior for forward and reverse reads."""

    def test_forward_read_returns_original_sequence(self, tmp_path):
        """Forward-mapped read: query_sequence == the sequence we stored."""
        original = "ACGTACGTAAAA"
        bam = tmp_path / "fwd.bam"
        _create_test_bam(bam, [{"name": "fwd", "sequence": original}])
        seqs = _read_sequences_pysam(bam)
        assert seqs["fwd"] == (original, False)

    def test_reverse_read_returns_stored_sequence(self, tmp_path):
        """Reverse-mapped read: query_sequence == what we set (BAM-stored orientation)."""
        original = "ACGTACGTAAAA"
        bam = tmp_path / "rev.bam"
        _create_test_bam(bam, [{"name": "rev", "sequence": original, "is_reverse": True}])
        seqs = _read_sequences_pysam(bam)
        # pysam query_sequence returns what we set, even for reverse reads
        assert seqs["rev"] == (original, True)

    def test_get_forward_sequence_gives_rc_for_reverse(self, tmp_path):
        """get_forward_sequence returns RC for reverse reads (the 'original read' orientation)."""
        original = "ACGTACGTAAAA"
        bam = tmp_path / "rev.bam"
        _create_test_bam(bam, [{"name": "rev", "sequence": original, "is_reverse": True}])
        with _pysam.AlignmentFile(str(bam), "rb") as f:
            for read in f.fetch(until_eof=True):
                if read.query_name == "rev":
                    assert read.get_forward_sequence() == _rc(original)

    def test_forward_and_reverse_with_distinct_sequences(self, tmp_path):
        """Forward and reverse reads with different sequences are both returned correctly."""
        fwd_seq = "AAACCCGGGTTT"
        rev_seq = "TTTGGGCCCAAA"
        bam = tmp_path / "mixed.bam"
        _create_test_bam(
            bam,
            [
                {"name": "fwd", "sequence": fwd_seq},
                {"name": "rev", "sequence": rev_seq, "is_reverse": True},
            ],
        )
        seqs = _read_sequences_pysam(bam)
        assert seqs["fwd"] == (fwd_seq, False)
        assert seqs["rev"] == (rev_seq, True)


@requires_pysam
@requires_samtools
class TestSamtoolsSequenceOrientation:
    """Verify samtools view SEQ field behavior for forward and reverse reads."""

    def test_forward_read_returns_original_sequence(self, tmp_path):
        """Forward-mapped read: samtools SEQ == stored sequence."""
        original = "ACGTACGTAAAA"
        bam = tmp_path / "fwd.bam"
        _create_test_bam(bam, [{"name": "fwd", "sequence": original}])
        seqs = _read_sequences_samtools(bam)
        assert seqs["fwd"] == (original, False)

    def test_reverse_read_returns_stored_sequence(self, tmp_path):
        """Reverse-mapped read: samtools SEQ == BAM-stored sequence (same as pysam)."""
        original = "ACGTACGTAAAA"
        bam = tmp_path / "rev.bam"
        _create_test_bam(bam, [{"name": "rev", "sequence": original, "is_reverse": True}])
        seqs = _read_sequences_samtools(bam)
        assert seqs["rev"] == (original, True)


@requires_pysam
@requires_samtools
class TestBackendSequenceConsistency:
    """Confirm pysam and samtools return identical sequences for the same BAM."""

    def test_backends_agree_on_forward_read(self, tmp_path):
        """Both backends return the same sequence for a forward-mapped read."""
        seq = "GCTAGCTAGCTA"
        bam = tmp_path / "fwd.bam"
        _create_test_bam(bam, [{"name": "r1", "sequence": seq}])
        pysam_seqs = _read_sequences_pysam(bam)
        samtools_seqs = _read_sequences_samtools(bam)
        assert pysam_seqs["r1"][0] == samtools_seqs["r1"][0]
        assert pysam_seqs["r1"][1] == samtools_seqs["r1"][1]

    def test_backends_agree_on_reverse_read(self, tmp_path):
        """Both backends return the same sequence for a reverse-mapped read."""
        seq = "GCTAGCTAGCTA"
        bam = tmp_path / "rev.bam"
        _create_test_bam(bam, [{"name": "r1", "sequence": seq, "is_reverse": True}])
        pysam_seqs = _read_sequences_pysam(bam)
        samtools_seqs = _read_sequences_samtools(bam)
        assert pysam_seqs["r1"][0] == samtools_seqs["r1"][0]
        assert pysam_seqs["r1"][1] == samtools_seqs["r1"][1]

    def test_backends_agree_mixed_reads(self, tmp_path):
        """Both backends produce identical results for a mix of forward and reverse reads."""
        reads = [
            {"name": "fwd1", "sequence": "AAACCCGGGTTT"},
            {"name": "rev1", "sequence": "TTTGGGCCCAAA", "is_reverse": True},
            {"name": "fwd2", "sequence": "ACGTACGTACGT"},
            {"name": "rev2", "sequence": "GCTAGCTAGCTA", "is_reverse": True},
        ]
        bam = tmp_path / "mixed.bam"
        _create_test_bam(bam, reads)
        pysam_seqs = _read_sequences_pysam(bam)
        samtools_seqs = _read_sequences_samtools(bam)
        for name in ("fwd1", "rev1", "fwd2", "rev2"):
            assert pysam_seqs[name] == samtools_seqs[name], f"Mismatch for {name}"


# ---------------------------------------------------------------------------
# Tier: UMI extraction produces identical results across backends
# ---------------------------------------------------------------------------
@requires_pysam
@requires_samtools
class TestUMIExtractionBackendConsistency:
    """UMI extraction must produce identical tags regardless of samtools_backend."""

    @staticmethod
    def _run_umi(tmp_path, reads, backend, subdir, **kwargs):
        """Run UMI annotation with the specified backend and return sidecar tags."""
        work = tmp_path / subdir
        work.mkdir()
        bam = work / "test.bam"
        _create_test_bam(bam, reads)
        umi_kit = UMIKitConfig(
            flanking=PerEndFlankingConfig(
                left_ref_end=FlankingConfig(adapter_side="GCTA"),
                right_ref_end=FlankingConfig(adapter_side="CCGA"),
            ),
            length=4,
            umi_ends="both",
            umi_flank_mode="adapter_only",
        )
        defaults = dict(
            use_umi=True,
            umi_kit_config=umi_kit,
            umi_search_window=200,
            umi_adapter_matcher="exact",
            umi_adapter_max_edits=0,
            samtools_backend=backend,
        )
        defaults.update(kwargs)
        sidecar = bam_functions.annotate_umi_tags_in_bam(bam, **defaults)
        return _read_parquet_tags(sidecar)

    def test_forward_read_umi_same_across_backends(self, tmp_path):
        """Forward read: pysam and CLI backends produce identical UMI tags."""
        reads = [{"name": "r1", "sequence": "GCTAGGTTNNNNNNNNNNTCGTTCGG"}]
        py_tags = self._run_umi(tmp_path, reads, "python", "py")
        cli_tags = self._run_umi(tmp_path, reads, "cli", "cli")
        assert py_tags["r1"] == cli_tags["r1"]

    def test_reverse_read_umi_same_across_backends(self, tmp_path):
        """Reverse read: pysam and CLI backends produce identical UMI tags."""
        reads = [{"name": "r1", "sequence": "GCTAGGTTNNNNNNNNNNTCGTTCGG", "is_reverse": True}]
        py_tags = self._run_umi(tmp_path, reads, "python", "py")
        cli_tags = self._run_umi(tmp_path, reads, "cli", "cli")
        assert py_tags["r1"] == cli_tags["r1"]

    def test_mixed_fwd_rev_umi_same_across_backends(self, tmp_path):
        """Mix of forward and reverse reads: both backends agree on all UMI tags."""
        reads = [
            {"name": "fwd", "sequence": "GCTAGGTTNNNNNNNNNNTCGTTCGG"},
            {"name": "rev", "sequence": "GCTAGGTTNNNNNNNNNNTCGTTCGG", "is_reverse": True},
            {"name": "no_umi", "sequence": "TTTTTTTTTTTTTTTTTTTTTTTTTT"},
        ]
        py_tags = self._run_umi(tmp_path, reads, "python", "py")
        cli_tags = self._run_umi(tmp_path, reads, "cli", "cli")
        for name in ("fwd", "rev", "no_umi"):
            assert py_tags[name] == cli_tags[name], f"UMI mismatch for {name}"

    def test_reverse_read_u1_u2_consistent_both_backends(self, tmp_path):
        """Fwd and rev reads of the same fragment: U1/U2 (genomic) should be
        identical across orientations and backends."""
        seq = "GCTAGGTTNNNNNNNNNNTCGTTCGG"
        reads_fwd = [{"name": "r1", "sequence": seq}]
        # Same BAM sequence for reverse read (same fragment, reference orientation)
        reads_rev = [{"name": "r1", "sequence": seq, "is_reverse": True}]
        fwd_py = self._run_umi(tmp_path, reads_fwd, "python", "fwd_py")
        fwd_cli = self._run_umi(tmp_path, reads_fwd, "cli", "fwd_cli")
        rev_py = self._run_umi(tmp_path, reads_rev, "python", "rev_py")
        rev_cli = self._run_umi(tmp_path, reads_rev, "cli", "rev_cli")
        # Backends agree on forward read
        assert fwd_py["r1"]["U1"] == fwd_cli["r1"]["U1"]
        assert fwd_py["r1"]["U2"] == fwd_cli["r1"]["U2"]
        # Backends agree on reverse read
        assert rev_py["r1"]["U1"] == rev_cli["r1"]["U1"]
        assert rev_py["r1"]["U2"] == rev_cli["r1"]["U2"]
        # Genomic U1/U2 are the same regardless of mapping orientation
        assert fwd_py["r1"]["U1"] == rev_py["r1"]["U1"]
        assert fwd_py["r1"]["U2"] == rev_py["r1"]["U2"]


# ---------------------------------------------------------------------------
# Tier: Barcode extraction produces identical results across backends
# ---------------------------------------------------------------------------
@requires_pysam
@requires_samtools
class TestBarcodeExtractionBackendConsistency:
    """Barcode extraction must produce identical tags regardless of samtools_backend."""

    @staticmethod
    def _run_barcode(tmp_path, reads, backend, subdir):
        """Run barcode extraction with the specified backend and return sidecar tags."""
        from smftools.informatics.bam_functions import (
            BarcodeKitConfig,
            PerEndFlankingConfig as PEFC,
            extract_and_assign_barcodes_in_bam,
        )

        work = tmp_path / subdir
        work.mkdir()
        bam = work / "test.bam"
        _create_test_bam(bam, reads)

        adapter_left = "GCTA"
        adapter_right = "CCGA"
        bc_kit = BarcodeKitConfig(
            flanking=PEFC(
                left_ref_end=FlankingConfig(adapter_side=adapter_left),
                right_ref_end=FlankingConfig(adapter_side=adapter_right),
            ),
            barcode_ends="both",
        )
        bc_refs = {"NB01": "AAAA", "NB02": "CCCC", "NB03": "GGGG", "NB04": "TTTT"}

        sidecar = extract_and_assign_barcodes_in_bam(
            bam,
            barcode_adapters=[adapter_left, adapter_right],
            barcode_references=bc_refs,
            barcode_length=4,
            barcode_search_window=200,
            barcode_max_edit_distance=1,
            barcode_adapter_matcher="exact",
            barcode_composite_max_edits=4,
            samtools_backend=backend,
            barcode_kit_config=bc_kit,
            barcode_ends="both",
        )
        return _read_parquet_tags(sidecar)

    def test_forward_read_barcode_same_across_backends(self, tmp_path):
        """Forward read: both backends extract the same barcode."""
        # Structure: adapter_left + barcode(AAAA) + body + RC(barcode(AAAA)) + RC(adapter_right)
        reads = [{"name": "r1", "sequence": "GCTAAAAA" + "N" * 20 + "TTTTTCGG"}]
        py_tags = self._run_barcode(tmp_path, reads, "python", "py")
        cli_tags = self._run_barcode(tmp_path, reads, "cli", "cli")
        assert py_tags["r1"] == cli_tags["r1"]

    def test_reverse_read_barcode_same_across_backends(self, tmp_path):
        """Reverse read: both backends extract the same barcode."""
        reads = [{"name": "r1", "sequence": "GCTAAAAA" + "N" * 20 + "TTTTTCGG", "is_reverse": True}]
        py_tags = self._run_barcode(tmp_path, reads, "python", "py")
        cli_tags = self._run_barcode(tmp_path, reads, "cli", "cli")
        assert py_tags["r1"] == cli_tags["r1"]

    def test_mixed_reads_barcode_same_across_backends(self, tmp_path):
        """Mix of forward/reverse reads: both backends agree on barcode assignment."""
        reads = [
            {"name": "fwd_nb01", "sequence": "GCTAAAAA" + "N" * 20 + "TTTTTCGG"},
            {"name": "rev_nb01", "sequence": "GCTAAAAA" + "N" * 20 + "TTTTTCGG", "is_reverse": True},
            {"name": "no_bc", "sequence": "T" * 30},
        ]
        py_tags = self._run_barcode(tmp_path, reads, "python", "py")
        cli_tags = self._run_barcode(tmp_path, reads, "cli", "cli")
        for name in ("fwd_nb01", "rev_nb01", "no_bc"):
            assert py_tags[name] == cli_tags[name], f"Barcode mismatch for {name}"


# ---------------------------------------------------------------------------
# Tier: Forward read and its RC stored as reverse produce identical US/UE
# ---------------------------------------------------------------------------
@requires_pysam
class TestOrientationConsistency:
    """Forward and reverse reads of the same original sequence should produce
    identical positional UMI tags (US/UE) and correctly swapped U1/U2."""

    @staticmethod
    def _run_umi(tmp_path, reads, subdir):
        work = tmp_path / subdir
        work.mkdir()
        bam = work / "test.bam"
        _create_test_bam(bam, reads)
        umi_kit = UMIKitConfig(
            flanking=PerEndFlankingConfig(
                left_ref_end=FlankingConfig(adapter_side="GCTA"),
                right_ref_end=FlankingConfig(adapter_side="CCGA"),
            ),
            length=4,
            umi_ends="both",
            umi_flank_mode="adapter_only",
        )
        sidecar = bam_functions.annotate_umi_tags_in_bam(
            bam,
            use_umi=True,
            umi_kit_config=umi_kit,
            umi_search_window=200,
            umi_adapter_matcher="exact",
            umi_adapter_max_edits=0,
            samtools_backend="python",
        )
        return _read_parquet_tags(sidecar)

    def test_fwd_and_rev_same_fragment_produce_same_u1_u2(self, tmp_path):
        """Forward and reverse reads of the same fragment (same BAM sequence)
        should yield identical genomic UMI tags (U1/U2) but swapped positional tags (US/UE)."""
        bam_seq = "GCTAGGTTNNNNNNNNNNTCGTTCGG"
        fwd_reads = [{"name": "r1", "sequence": bam_seq}]
        # Same BAM sequence — same fragment in reference orientation
        rev_reads = [{"name": "r1", "sequence": bam_seq, "is_reverse": True}]
        fwd_tags = self._run_umi(tmp_path, fwd_reads, "fwd")
        rev_tags = self._run_umi(tmp_path, rev_reads, "rev")
        # Genomic tags (U1=left ref end, U2=right ref end) should be identical
        assert fwd_tags["r1"]["U1"] == rev_tags["r1"]["U1"]
        assert fwd_tags["r1"]["U2"] == rev_tags["r1"]["U2"]
        # Positional tags (US=read start, UE=read end) should be swapped
        assert fwd_tags["r1"]["US"] == rev_tags["r1"]["UE"]
        assert fwd_tags["r1"]["UE"] == rev_tags["r1"]["US"]
