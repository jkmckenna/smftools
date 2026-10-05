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
            # minimap2 omits SEQ/QUAL on secondary records; "no_seq" mimics it.
            if not info.get("no_seq"):
                a.query_sequence = info.get("sequence") or "A" * query_length
                a.query_qualities = _pysam.qualitystring_to_array(
                    info.get("qualities") or "I" * query_length
                )
            flag = 0
            if info.get("secondary"):
                flag |= 0x100
            if info.get("supplementary"):
                flag |= 0x800
            if info.get("reverse"):
                flag |= 0x10
            if info.get("mate"):
                flag |= 0x1 | (0x40 if info["mate"] == 1 else 0x80)
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


def _assert_not_rewritten(summary, out_path):
    """Nothing rescued: the input BAM is the result and no output is written."""
    assert summary.output_written is False
    assert not out_path.exists()


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
    assert summary.output_written

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
    _assert_not_rewritten(summary, out_path)
    records = _read_records(bam_path)
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
    _assert_not_rewritten(summary, out_path)
    primary = next(r for r in _read_records(bam_path) if not r["is_secondary"])
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
    _assert_not_rewritten(summary, out_path)
    records = _read_records(bam_path)
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
    _assert_not_rewritten(summary, out_path)
    records = _read_records(bam_path)
    assert len(records) == 1
    assert records[0]["mapping_quality"] == 42
    assert not records[0]["is_secondary"]


@requires_pysam
def test_rescue_threaded_output_matches_unthreaded(tmp_path):
    bam_path = tmp_path / "in.bam"
    records = []
    # Many reads, each with a worse primary and a better secondary, plus
    # single-alignment reads, so the output spans more than one BGZF block.
    for i in range(400):
        start = 100 + (i % 50)
        records.append(
            {
                "name": f"rescued{i}",
                "contig": 0,
                "start": start,
                "cigar": [(CIGAR_M, 2000), (CIGAR_S, 300)],
                "mapping_quality": 7,
            }
        )
        records.append(
            {
                "name": f"rescued{i}",
                "contig": 1,
                "start": start,
                "cigar": [(CIGAR_M, 2300)],
                "secondary": True,
                "mapping_quality": 0,
            }
        )
        records.append(
            {"name": f"single{i}", "contig": 0, "start": start, "cigar": [(CIGAR_M, 2000)]}
        )
    records.sort(key=lambda r: (r["contig"], r["start"]))
    _write_bam(bam_path, [("6B6_top", 5000), ("6B6_enh_del_top", 4500)], records)
    record_chromosome = {"6B6_top": "6B6", "6B6_enh_del_top": "6B6_enh_del"}

    serial = rescue_secondary_alignments(bam_path, tmp_path / "serial.bam", record_chromosome)
    threaded = rescue_secondary_alignments(
        bam_path, tmp_path / "threaded.bam", record_chromosome, threads=4
    )

    assert serial == threaded
    assert threaded.n_reads_rescued == 400

    def full_records(path):
        with _pysam.AlignmentFile(str(path), "rb") as fh:
            return [read.to_string() for read in fh.fetch(until_eof=True)]

    assert full_records(tmp_path / "threaded.bam") == full_records(tmp_path / "serial.bam")
    assert (tmp_path / "threaded.bam.bai").exists()


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
    fasta_path.write_text(">6B6_top\nACGCGTACGTACGCGTACGT\n>6B6_enh_del_top\nACGCGTACGTAC\n")

    for modality in ("deaminase", "direct"):
        record_chromosome = build_record_chromosome_map(fasta_path, modality)
        assert record_chromosome == {"6B6_top": "6B6_top", "6B6_enh_del_top": "6B6_enh_del_top"}


# --- F64: rescued reads must keep their sequence ----------------------------

_READ = "ACGTTGCAACGG" * 25  # 300 bp, asymmetric so reverse-complement is visible
_QUALS = "".join(chr(33 + 10 + (i % 30)) for i in range(300))
_TWO_REFS = [("6B6_top", 5000), ("6B6_enh_del_top", 4500)]
_CHROMS = {"6B6_top": "6B6", "6B6_enh_del_top": "6B6_enh_del"}


def _primary(path):
    with _pysam.AlignmentFile(str(path), "rb") as fh:
        return next(r for r in fh.fetch(until_eof=True) if not r.is_secondary)


@requires_pysam
@pytest.mark.parametrize("opposite_strand", [False, True])
def test_rescued_read_keeps_its_sequence(tmp_path, opposite_strand):
    bam_path = tmp_path / "in.bam"
    _write_bam(
        bam_path,
        contigs=_TWO_REFS,
        records=[
            {  # worse primary, holds the read's SEQ
                "name": "readA",
                "contig": 0,
                "start": 100,
                "cigar": [(CIGAR_M, 250), (CIGAR_S, 50)],
                "sequence": _READ,
                "qualities": _QUALS,
                "mapping_quality": 7,
            },
            {  # better secondary, written without SEQ as minimap2 does
                "name": "readA",
                "contig": 1,
                "start": 100,
                "cigar": [(CIGAR_M, 300)],
                "secondary": True,
                "no_seq": True,
                "reverse": opposite_strand,
                "mapping_quality": 0,
            },
        ],
    )
    out = tmp_path / "out.bam"

    summary = rescue_secondary_alignments(bam_path, out, _CHROMS)

    assert summary.n_reads_rescued == 1
    assert summary.n_sequences_restored == 1
    promoted = _primary(out)
    assert promoted.reference_name == "6B6_enh_del_top"
    expected = _READ.translate(str.maketrans("ACGT", "TGCA"))[::-1] if opposite_strand else _READ
    assert promoted.query_sequence == expected
    quals = [ord(c) - 33 for c in _QUALS]
    assert list(promoted.query_qualities) == (quals[::-1] if opposite_strand else quals)


@requires_pysam
def test_restore_primary_sequences_repairs_an_already_rescued_bam(tmp_path):
    from smftools.informatics.alignment_rescue import restore_primary_sequences

    # The pre-fix state: flags already swapped, promoted record still SEQ-less.
    broken = tmp_path / "broken.bam"
    _write_bam(
        broken,
        contigs=_TWO_REFS,
        records=[
            {
                "name": "readA",
                "contig": 0,
                "start": 100,
                "cigar": [(CIGAR_M, 250), (CIGAR_S, 50)],
                "secondary": True,
                "sequence": _READ,
                "qualities": _QUALS,
            },
            {
                "name": "readA",
                "contig": 1,
                "start": 100,
                "cigar": [(CIGAR_M, 300)],
                "no_seq": True,
                "mapping_quality": 7,
            },
        ],
    )
    out = tmp_path / "restored.bam"

    summary = restore_primary_sequences(broken, out)

    assert (summary.n_sequenceless_primary, summary.n_restored, summary.output_written) == (
        1,
        1,
        True,
    )
    assert _primary(out).query_sequence == _READ
    # Idempotent: the repaired BAM needs nothing and is not rewritten.
    again = restore_primary_sequences(out, tmp_path / "again.bam")
    assert (again.n_sequenceless_primary, again.output_written) == (0, False)
    assert not (tmp_path / "again.bam").exists()


@requires_pysam
def test_restore_refuses_a_donor_that_does_not_fit_the_cigar(tmp_path):
    from smftools.informatics.alignment_rescue import restore_primary_sequences

    broken = tmp_path / "broken.bam"
    _write_bam(
        broken,
        contigs=[("6B6_top", 5000)],
        records=[
            {
                "name": "readA",
                "contig": 0,
                "start": 100,
                "cigar": [(CIGAR_M, 200)],
                "secondary": True,
                "sequence": _READ[:200],
                "qualities": _QUALS[:200],
            },
            {
                "name": "readA",
                "contig": 0,
                "start": 900,
                "cigar": [(CIGAR_M, 300)],
                "no_seq": True,
            },
        ],
    )

    summary = restore_primary_sequences(broken, tmp_path / "out.bam")

    assert summary.n_restored == 0
    assert summary.n_unrestorable == 1


def _primaries(path):
    with _pysam.AlignmentFile(str(path), "rb") as fh:
        return [r for r in fh.fetch(until_eof=True) if not r.is_secondary]


@requires_pysam
def test_rescue_promotes_one_record_when_a_chimeric_read_shares_a_start(tmp_path):
    # Two alignments of one chimeric read at the same start, different extents:
    # rescue must promote only the one it chose (the longer), not both.
    bam_path = tmp_path / "in.bam"
    _write_bam(
        bam_path,
        contigs=_TWO_REFS,
        records=[
            {
                "name": "readA",
                "contig": 0,
                "start": 100,
                "cigar": [(CIGAR_M, 250), (CIGAR_S, 50)],
                "sequence": _READ,
                "qualities": _QUALS,
                "mapping_quality": 7,
            },
            {
                "name": "readA",
                "contig": 1,
                "start": 100,
                "cigar": [(CIGAR_M, 300)],
                "secondary": True,
                "no_seq": True,
            },
            {
                "name": "readA",
                "contig": 1,
                "start": 100,
                "cigar": [(CIGAR_S, 200), (CIGAR_M, 100)],
                "secondary": True,
                "no_seq": True,
            },
        ],
    )
    out = tmp_path / "out.bam"

    rescue_secondary_alignments(bam_path, out, _CHROMS)

    primaries = _primaries(out)
    assert len(primaries) == 1
    assert primaries[0].query_alignment_length == 300
    assert primaries[0].query_sequence == _READ


@requires_pysam
def test_restore_demotes_extra_primaries_left_by_old_rescue(tmp_path):
    from smftools.informatics.alignment_rescue import restore_primary_sequences

    broken = tmp_path / "broken.bam"
    _write_bam(
        broken,
        contigs=_TWO_REFS,
        records=[
            {
                "name": "readA",
                "contig": 0,
                "start": 100,
                "cigar": [(CIGAR_M, 250), (CIGAR_S, 50)],
                "secondary": True,
                "sequence": _READ,
                "qualities": _QUALS,
            },
            {"name": "readA", "contig": 1, "start": 100, "cigar": [(CIGAR_M, 300)], "no_seq": True},
            {
                "name": "readA",
                "contig": 1,
                "start": 100,
                "cigar": [(CIGAR_S, 200), (CIGAR_M, 100)],
                "no_seq": True,
            },
        ],
    )
    out = tmp_path / "restored.bam"

    summary = restore_primary_sequences(broken, out)

    assert (summary.n_restored, summary.n_extra_primaries_demoted) == (1, 1)
    primaries = _primaries(out)
    assert len(primaries) == 1
    assert primaries[0].query_alignment_length == 300
    assert primaries[0].query_sequence == _READ


@requires_pysam
def test_restore_leaves_paired_mates_as_two_primaries(tmp_path):
    from smftools.informatics.alignment_rescue import restore_primary_sequences

    # Mates share a name but are separate reads, each correctly primary.
    bam_path = tmp_path / "pair.bam"
    _write_bam(
        bam_path,
        contigs=[("6B6_top", 5000)],
        records=[
            {"name": "pair", "contig": 0, "start": 100, "cigar": [(CIGAR_M, 150)], "mate": 1},
            {"name": "pair", "contig": 0, "start": 400, "cigar": [(CIGAR_M, 150)], "mate": 2},
        ],
    )

    summary = restore_primary_sequences(bam_path, tmp_path / "out.bam")

    assert (summary.n_extra_primaries_demoted, summary.output_written) == (0, False)
