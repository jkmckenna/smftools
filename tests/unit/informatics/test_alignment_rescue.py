import pytest

from smftools.informatics.alignment_rescue import (
    build_record_chromosome_map,
    rescue_secondary_alignments,
)

try:
    import pysam as _pysam

    HAS_PYSAM = True
except ImportError:
    _pysam = None
    HAS_PYSAM = False

requires_pysam = pytest.mark.skipif(not HAS_PYSAM, reason="pysam not installed")


def _write_bam(bam_path, contigs, records):
    """Write a small coordinate-sorted, indexed BAM.

    Args:
        contigs: list of (name, length) tuples, in header SQ order.
        records: list of dicts with keys: name, contig (index into `contigs`),
            start, cigar (list of (op, length) tuples), secondary, supplementary,
            mapping_quality (default 10).
    """
    header = {
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": name, "LN": length} for name, length in contigs],
    }
    with _pysam.AlignmentFile(str(bam_path), "wb", header=header) as outf:
        for info in records:
            a = _pysam.AlignedSegment()
            a.query_name = info["name"]
            query_length = sum(length for op, length in info["cigar"] if op in (0, 1, 4))
            a.query_sequence = "A" * query_length
            a.query_qualities = _pysam.qualitystring_to_array("I" * query_length)
            flag = 0
            if info.get("secondary"):
                flag |= 0x100
            if info.get("supplementary"):
                flag |= 0x800
            a.flag = flag
            a.reference_id = info["contig"]
            a.reference_start = info["start"]
            a.mapping_quality = info.get("mapping_quality", 10)
            a.cigartuples = info["cigar"]
            outf.write(a)
    _pysam.index(str(bam_path))


def _read_records(bam_path):
    out = []
    with _pysam.AlignmentFile(str(bam_path), "rb") as fh:
        for read in fh.fetch(until_eof=True):
            out.append(
                {
                    "name": read.query_name,
                    "reference_name": read.reference_name,
                    "reference_start": read.reference_start,
                    "is_secondary": read.is_secondary,
                    "is_supplementary": read.is_supplementary,
                    "mapping_quality": read.mapping_quality,
                    "query_alignment_length": read.query_alignment_length,
                }
            )
    return out


# pysam CIGAR op codes: M(atch)=0, S(oft clip)=4
CIGAR_M, CIGAR_S = 0, 4


@requires_pysam
def test_rescue_swaps_worse_primary_for_better_secondary(tmp_path):
    bam_path = tmp_path / "in.bam"
    out_path = tmp_path / "out.bam"
    _write_bam(
        bam_path,
        contigs=[("6B6_top", 5000), ("6B6_enh_del_top", 4500)],
        records=[
            # Worse primary: covers only 2000bp of the read (soft-clipped tail).
            {
                "name": "readA",
                "contig": 0,
                "start": 100,
                "cigar": [(CIGAR_M, 2000), (CIGAR_S, 300)],
                "secondary": False,
                "mapping_quality": 1,
            },
            # Better secondary: covers the full 2300bp read, no clip.
            {
                "name": "readA",
                "contig": 1,
                "start": 100,
                "cigar": [(CIGAR_M, 2300)],
                "secondary": True,
                "mapping_quality": 0,
            },
        ],
    )
    record_chromosome = {"6B6_top": "6B6", "6B6_enh_del_top": "6B6_enh_del"}

    summary = rescue_secondary_alignments(bam_path, out_path, record_chromosome)

    assert summary.n_reads_examined == 1
    assert summary.n_reads_rescued == 1
    assert summary.reassignment_counts == {("6B6", "6B6_enh_del"): 1}

    records = {(r["reference_name"], r["reference_start"]): r for r in _read_records(out_path)}
    winner = records[("6B6_enh_del_top", 100)]
    loser = records[("6B6_top", 100)]
    assert winner["name"] == "readA"
    assert not winner["is_secondary"]
    assert winner["mapping_quality"] == 1  # inherited from the old primary
    assert loser["is_secondary"]
    assert loser["mapping_quality"] == 0


@requires_pysam
def test_rescue_leaves_near_tied_candidates_unchanged(tmp_path):
    bam_path = tmp_path / "in.bam"
    out_path = tmp_path / "out.bam"
    _write_bam(
        bam_path,
        contigs=[("6B6_top", 5000), ("6B6_enh_del_top", 4500)],
        records=[
            {
                "name": "readA",
                "contig": 0,
                "start": 100,
                "cigar": [(CIGAR_M, 2290)],
                "secondary": False,
                "mapping_quality": 5,
            },
            {
                "name": "readA",
                "contig": 1,
                "start": 100,
                "cigar": [(CIGAR_M, 2300)],  # only 10bp longer -- below default 20bp margin
                "secondary": True,
                "mapping_quality": 0,
            },
        ],
    )
    record_chromosome = {"6B6_top": "6B6", "6B6_enh_del_top": "6B6_enh_del"}

    summary = rescue_secondary_alignments(bam_path, out_path, record_chromosome)

    assert summary.n_reads_rescued == 0
    records = _read_records(out_path)
    primary = next(r for r in records if not r["is_secondary"])
    assert primary["reference_name"] == "6B6_top"
    assert primary["mapping_quality"] == 5


@requires_pysam
def test_rescue_ignores_secondary_to_same_chromosome(tmp_path):
    bam_path = tmp_path / "in.bam"
    out_path = tmp_path / "out.bam"
    _write_bam(
        bam_path,
        contigs=[("6B6_5mC_top", 5000), ("6B6_unconverted_top", 5000)],
        records=[
            {
                "name": "readA",
                "contig": 0,
                "start": 100,
                "cigar": [(CIGAR_M, 2000), (CIGAR_S, 300)],
                "secondary": False,
                "mapping_quality": 3,
            },
            # Longer alignment, but same chromosome ("6B6") as the primary --
            # just a different conversion-state variant -- so nothing to rescue.
            {
                "name": "readA",
                "contig": 1,
                "start": 100,
                "cigar": [(CIGAR_M, 2300)],
                "secondary": True,
                "mapping_quality": 0,
            },
        ],
    )
    record_chromosome = {"6B6_5mC_top": "6B6", "6B6_unconverted_top": "6B6"}

    summary = rescue_secondary_alignments(bam_path, out_path, record_chromosome)

    assert summary.n_reads_rescued == 0
    primary = next(r for r in _read_records(out_path) if not r["is_secondary"])
    assert primary["reference_name"] == "6B6_5mC_top"


@requires_pysam
def test_rescue_ignores_supplementary_alignments(tmp_path):
    bam_path = tmp_path / "in.bam"
    out_path = tmp_path / "out.bam"
    _write_bam(
        bam_path,
        contigs=[("6B6_top", 5000), ("6B6_enh_del_top", 4500)],
        records=[
            {
                "name": "readA",
                "contig": 0,
                "start": 100,
                "cigar": [(CIGAR_M, 2000), (CIGAR_S, 300)],
                "secondary": False,
                "mapping_quality": 4,
            },
            # A better-covering alignment exists, but as SUPPLEMENTARY (not
            # secondary) -- out of scope, must be left untouched.
            {
                "name": "readA",
                "contig": 1,
                "start": 100,
                "cigar": [(CIGAR_M, 2300)],
                "supplementary": True,
                "mapping_quality": 0,
            },
        ],
    )
    record_chromosome = {"6B6_top": "6B6", "6B6_enh_del_top": "6B6_enh_del"}

    summary = rescue_secondary_alignments(bam_path, out_path, record_chromosome)

    assert summary.n_reads_rescued == 0
    records = _read_records(out_path)
    primary = next(r for r in records if not r["is_secondary"] and not r["is_supplementary"])
    assert primary["reference_name"] == "6B6_top"
    assert primary["mapping_quality"] == 4
    supplementary = next(r for r in records if r["is_supplementary"])
    assert supplementary["reference_name"] == "6B6_enh_del_top"
    assert supplementary["is_secondary"] is False


@requires_pysam
def test_rescue_passthrough_for_single_alignment_reads(tmp_path):
    bam_path = tmp_path / "in.bam"
    out_path = tmp_path / "out.bam"
    _write_bam(
        bam_path,
        contigs=[("6B6_top", 5000)],
        records=[
            {
                "name": "readA",
                "contig": 0,
                "start": 100,
                "cigar": [(CIGAR_M, 2000)],
                "secondary": False,
                "mapping_quality": 42,
            },
        ],
    )
    record_chromosome = {"6B6_top": "6B6"}

    summary = rescue_secondary_alignments(bam_path, out_path, record_chromosome)

    assert summary.n_reads_examined == 1
    assert summary.n_reads_rescued == 0
    records = _read_records(out_path)
    assert len(records) == 1
    assert records[0]["mapping_quality"] == 42
    assert not records[0]["is_secondary"]


@requires_pysam
def test_summary_to_dataframe():
    from smftools.informatics.alignment_rescue import RescueSummary

    summary = RescueSummary(
        n_reads_examined=10,
        n_reads_rescued=2,
        reassignment_counts={("6B6", "6B6_enh_del"): 2},
    )
    df = summary.to_dataframe()
    assert list(df.columns) == ["old_chromosome", "new_chromosome", "n_reads"]
    assert df.iloc[0].to_dict() == {
        "old_chromosome": "6B6",
        "new_chromosome": "6B6_enh_del",
        "n_reads": 2,
    }


@requires_pysam
def test_build_record_chromosome_map_conversion_merges_conversion_states(tmp_path):
    fasta_path = tmp_path / "refs.fasta"
    fasta_path.write_text(
        ">6B6_unconverted_top\n"
        "ACGCGTACGTACGCGTACGTACGCGTACGTACGCGTACGT\n"
        ">6B6_enh_del_unconverted_top\n"
        "ACGCGTACGTACGCGTACGTACGCGTACGT\n"
    )

    record_chromosome = build_record_chromosome_map(
        fasta_path, "conversion", conversion_types=["unconverted", "5mC"]
    )

    # Conversion-state variants of the same allele collapse to one chromosome;
    # distinct alleles stay separate.
    assert record_chromosome["6B6_unconverted_top"] == "6B6"
    assert record_chromosome["6B6_5mC_top"] == "6B6"
    assert record_chromosome["6B6_5mC_bottom"] == "6B6"
    assert record_chromosome["6B6_enh_del_unconverted_top"] == "6B6_enh_del"
    assert record_chromosome["6B6_enh_del_5mC_top"] == "6B6_enh_del"


@requires_pysam
def test_build_record_chromosome_map_deaminase_and_direct_use_identity(tmp_path):
    fasta_path = tmp_path / "refs.fasta"
    fasta_path.write_text(
        ">6B6_top\nACGCGTACGTACGCGTACGT\n>6B6_enh_del_top\nACGCGTACGTAC\n"
    )

    for modality in ("deaminase", "direct"):
        record_chromosome = build_record_chromosome_map(fasta_path, modality)
        assert record_chromosome == {"6B6_top": "6B6_top", "6B6_enh_del_top": "6B6_enh_del_top"}
