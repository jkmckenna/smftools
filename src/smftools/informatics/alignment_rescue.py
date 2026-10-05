"""Rescue reads whose primary alignment lost to a worse-covering reference contig.

When the alignment FASTA contains nested/overlapping reference variants for
one locus (e.g. a wild-type contig and a shorter deletion-allele contig),
minimap2's own primary-alignment pick (driven by its affine-gap score) can
prefer a truncated match against the longer/wrong contig over a full-length
match against the correct, shorter one -- even though the correct alignment
covers strictly more of the read with fewer mismatches. minimap2 still
computes that better alignment (via ``-N`` secondary alignments); it's just
tagged secondary and discarded downstream, since the rest of the pipeline
only ever reads primary alignments.

This module re-flags a BAM's primary/secondary bits using read coverage
(``query_alignment_length``) as the selection criterion, grouped by which
biological "chromosome" each contig belongs to (so alignments to different
conversion-state variants of the *same* allele are never treated as
competing candidates), with an ambiguity-rejection margin so near-tied
candidates are left alone. It never touches supplementary alignments (a
different phenomenon -- genuine split/chimeric reads).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

from smftools.logging_utils import get_logger

from .bam_functions import _index_bam_with_pysam, _require_pysam

logger = get_logger(__name__)


@dataclass
class RescueSummary:
    """Outcome of one `rescue_secondary_alignments` pass."""

    n_reads_examined: int = 0
    n_reads_rescued: int = 0
    reassignment_counts: Dict[Tuple[str, str], int] = field(default_factory=dict)
    # False when nothing was rescued and the output BAM was therefore not written.
    output_written: bool = True
    # Promoted records that arrived without SEQ (minimap2 omits it on secondary
    # alignments) and had it carried over from the demoted record (`F64`).
    n_sequences_restored: int = 0

    def to_dataframe(self):
        import pandas as pd

        rows = [
            {"old_chromosome": old, "new_chromosome": new, "n_reads": count}
            for (old, new), count in sorted(self.reassignment_counts.items())
        ]
        return pd.DataFrame(rows, columns=["old_chromosome", "new_chromosome", "n_reads"])


@dataclass
class SequenceRestoreSummary:
    """Outcome of one `restore_primary_sequences` pass (counts are records)."""

    n_sequenceless_primary: int = 0
    n_restored: int = 0
    n_unrestorable: int = 0
    # Extra primaries on one read -- left by the pre-`F64` rescue, which could
    # promote several same-start alignments of a chimeric read -- demoted.
    n_extra_primaries_demoted: int = 0
    output_written: bool = False


#: Per-read tags tied to the read's own bases, carried with SEQ. Base
#: modification positions are relative to the read as sequenced, so they copy
#: unchanged across alignment strands.
_SEQUENCE_TAGS = ("MM", "ML", "MN")
_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")
_QUERY_CONSUMING_OPS = frozenset({0, 1, 4, 7, 8})  # M, I, S, =, X


def _cigar_query_length(read) -> int:
    return sum(length for op, length in (read.cigartuples or ()) if op in _QUERY_CONSUMING_OPS)


def _read_key(read) -> tuple[str, bool, bool]:
    """One read of a template: paired mates share a name but are separate reads."""
    return read.query_name, bool(read.is_read1), bool(read.is_read2)


def _sequence_donor(read) -> tuple:
    """What a sequence-less record of the same read needs, taken from ``read``."""
    qualities = read.query_qualities
    tags = tuple(
        (tag, value, value_type)
        for tag in _SEQUENCE_TAGS
        if read.has_tag(tag)
        for value, value_type in [read.get_tag(tag, with_value_type=True)]
    )
    return (
        read.query_sequence,
        list(qualities) if qualities is not None else None,
        bool(read.is_reverse),
        tags,
    )


def _restore_sequence(read, donor) -> bool:
    """Give a sequence-less record its read's SEQ/QUAL/tags; False if they don't fit.

    SEQ is stored in alignment orientation, so it is reverse-complemented (and
    QUAL reversed) when the donor aligned to the other strand.
    """
    sequence, qualities, donor_reverse, tags = donor
    if bool(read.is_reverse) != donor_reverse:
        sequence = sequence.translate(_COMPLEMENT)[::-1]
        qualities = qualities[::-1] if qualities is not None else None
    if _cigar_query_length(read) != len(sequence):
        return False
    read.query_sequence = sequence
    if qualities is not None:
        read.query_qualities = qualities
    for tag, value, value_type in tags:
        if not read.has_tag(tag):
            read.set_tag(tag, value, value_type=None if value_type == "B" else value_type)
    return True


def _collect_sequence_donors(pysam_mod, bam_path, names, *, primary_only, bgzf_threads) -> dict:
    """One pass: a full-length SEQ donor per wanted read, keyed by `_read_key`.

    Supplementary records are never donors -- they may be hard-clipped.
    """
    donors: dict = {}
    if not names:
        return donors
    with pysam_mod.AlignmentFile(bam_path, "rb", **bgzf_threads) as bam:
        for read in bam.fetch(until_eof=True):
            key = _read_key(read)
            if read.query_name not in names or key in donors:
                continue
            if read.is_unmapped or read.is_supplementary or read.query_sequence is None:
                continue
            if primary_only and read.is_secondary:
                continue
            donors[key] = _sequence_donor(read)
    return donors


def build_record_chromosome_map(
    fasta: str | Path,
    smf_modality: str,
    conversion_types: list[str] | None = None,
) -> Dict[str, str]:
    """Map each alignment-target FASTA record to its biological "chromosome".

    For `conversion` modality, records are further expanded per conversion
    state (e.g. `6B6_5mC_top`, `6B6_unconverted_top`); this reuses the same
    suffix-stripping logic already applied later in the pipeline
    (`converted_BAM_to_adata.process_conversion_sites`) so conversion-state
    variants of one allele collapse to the same chromosome, while distinct
    alleles (`6B6` vs `6B6_enh_del`) remain separate.

    `deaminase`/`direct` modalities have no conversion-state expansion --
    each FASTA record is already its own distinct target, so the identity
    mapping is the correct grouping key (this also correctly separates
    distinct alleles if the FASTA happens to contain them for those
    modalities too).
    """
    from .fasta_functions import get_native_references

    if smf_modality == "conversion":
        from .converted_BAM_to_adata import process_conversion_sites

        _max_len, record_info, _chromosome_sequences = process_conversion_sites(
            fasta, conversion_types, deaminase_footprinting=False
        )
        return {name: info.chromosome for name, info in record_info.items()}

    reference_map = get_native_references(fasta)
    return {name: name for name in reference_map}


def rescue_secondary_alignments(
    bam_path: str | Path,
    output_path: str | Path,
    record_chromosome: Dict[str, str],
    *,
    min_margin_bp: int = 20,
    min_margin_fraction: float = 0.01,
    threads: int | None = None,
) -> RescueSummary:
    """Re-flag primary/secondary alignment bits by read-coverage, not aligner score.

    Args:
        bam_path: Coordinate-sorted, indexed input BAM.
        output_path: Path to write the corrected, coordinate-sorted BAM.
            Re-indexed on completion.
        record_chromosome: Maps each FASTA record/contig name (as it appears
            in the BAM header) to a biological "chromosome" identifier.
            Records sharing a chromosome (e.g. conversion-state variants of
            the same allele) are treated as one candidate, not competing
            ones; records with *different* chromosomes are the actual
            reassignment candidates.
        min_margin_bp: Minimum absolute `query_alignment_length` advantage
            (in bp) the winning chromosome's best record must have over the
            read's current primary before a reassignment is made.
        min_margin_fraction: Minimum relative advantage (as a fraction of the
            winning record's own `query_alignment_length`) required in
            addition to `min_margin_bp`. Both must hold.
        threads: Optional thread count for BGZF decompression/compression in
            both passes and for BAM re-indexing.

    Returns:
        RescueSummary with counts of reads examined/rescued, a breakdown of
        (old_chromosome, new_chromosome) reassignment counts, and
        ``output_written``. When no read is rescued, ``output_path`` is not
        written and ``output_written`` is False: ``bam_path`` is already the
        correct result.

    Notes:
        Supplementary alignments are never inspected or modified -- they
        represent split/chimeric read structure, a different phenomenon from
        a nested-reference scoring ambiguity. Reads with no secondary
        alignments, or where no alternative chromosome clears the ambiguity
        margin, are left completely unchanged (not even rewritten
        byte-for-byte differently beyond the file regeneration itself).
    """
    pysam_mod = _require_pysam()
    bam_path = str(bam_path)
    output_path = str(output_path)
    # BGZF (de)compression is the cost of both passes; without `threads` pysam
    # does it on the calling thread (`F53`).
    bgzf_threads = {"threads": int(threads)} if threads and int(threads) > 1 else {}

    # ------------------------------------------------------------------
    # Pass 1 (read-only): for each read, find the best-covering record per
    # chromosome group, and the read's current primary. A read's primary and
    # a better-covering secondary can be arbitrarily far apart in
    # coordinate-sorted file order, so this can't assume adjacency -- a full
    # `fetch(until_eof=True)` scan is required (matches the existing
    # `extract_secondary_supplementary_alignment_spans` precedent).
    # ------------------------------------------------------------------
    # (query_name, chromosome) -> (query_alignment_length, reference_name, reference_start)
    best_per_read_chromosome: Dict[Tuple[str, str], Tuple[int, str, int]] = {}
    # query_name -> (chromosome, reference_name, reference_start, mapping_quality)
    old_primary: Dict[str, Tuple[str, str, int, int]] = {}
    n_reads_examined = 0
    unknown_records: set[str] = set()

    with pysam_mod.AlignmentFile(bam_path, "rb", **bgzf_threads) as bam:
        for read in bam.fetch(until_eof=True):
            if read.is_unmapped or read.is_supplementary:
                continue
            reference_name = read.reference_name
            chromosome = record_chromosome.get(reference_name)
            if chromosome is None:
                if reference_name not in unknown_records:
                    unknown_records.add(reference_name)
                    logger.warning(
                        "rescue_secondary_alignments: reference '%s' not found in "
                        "record_chromosome map; alignments to it are ignored for "
                        "rescue purposes.",
                        reference_name,
                    )
                continue

            query_alignment_length = int(read.query_alignment_length or 0)
            key = (read.query_name, chromosome)
            current = best_per_read_chromosome.get(key)
            if current is None or query_alignment_length > current[0]:
                best_per_read_chromosome[key] = (
                    query_alignment_length,
                    reference_name,
                    read.reference_start,
                )

            if not read.is_secondary:
                n_reads_examined += 1
                old_primary[read.query_name] = (
                    chromosome,
                    reference_name,
                    read.reference_start,
                    read.mapping_quality,
                )

    # Group best-per-chromosome candidates by read, decide reassignments.
    by_read: Dict[str, list[Tuple[str, int, str, int]]] = {}
    for (query_name, chromosome), (
        query_alignment_length,
        reference_name,
        reference_start,
    ) in best_per_read_chromosome.items():
        by_read.setdefault(query_name, []).append(
            (chromosome, query_alignment_length, reference_name, reference_start)
        )

    # query_name -> (winning_reference_name, winning_reference_start, winning_chromosome,
    #                old_primary_mapq, winning_query_alignment_length)
    promotions: Dict[str, Tuple[str, int, str, int]] = {}
    reassignment_counts: Counter[Tuple[str, str]] = Counter()

    for query_name, candidates in by_read.items():
        if len(candidates) < 2:
            continue
        primary = old_primary.get(query_name)
        if primary is None:
            continue
        primary_chromosome, _primary_rname, _primary_rstart, primary_mapq = primary

        by_chromosome = {c[0]: c for c in candidates}
        primary_entry = by_chromosome.get(primary_chromosome)
        primary_query_alignment_length = primary_entry[1] if primary_entry is not None else 0

        best_chromosome, best_query_alignment_length, best_rname, best_rstart = max(
            candidates, key=lambda c: c[1]
        )
        if best_chromosome == primary_chromosome:
            continue

        advantage = best_query_alignment_length - primary_query_alignment_length
        margin_bp_ok = advantage >= min_margin_bp
        margin_fraction_ok = (
            best_query_alignment_length > 0
            and (advantage / best_query_alignment_length) >= min_margin_fraction
        )
        if not (margin_bp_ok and margin_fraction_ok):
            continue

        promotions[query_name] = (
            best_rname,
            best_rstart,
            best_chromosome,
            primary_mapq,
            best_query_alignment_length,
        )
        reassignment_counts[(primary_chromosome, best_chromosome)] += 1

    logger.info(
        "rescue_secondary_alignments: examined %d reads, rescuing %d (reassignment breakdown: %s).",
        n_reads_examined,
        len(promotions),
        dict(reassignment_counts),
    )

    if not promotions:
        # Nothing to re-flag: the input is already the answer. Rewriting it
        # would cost a full decompress/recompress of the BAM for no change
        # (`F53`), so leave it in place and tell the caller.
        logger.info("rescue_secondary_alignments: no reads rescued; input BAM left unchanged.")
        return RescueSummary(
            n_reads_examined=n_reads_examined,
            n_reads_rescued=0,
            reassignment_counts={},
            output_written=False,
        )

    # ------------------------------------------------------------------
    # Pass 2: rewrite the BAM, flipping the secondary FLAG bit for exactly
    # the winning/demoted record pair per rescued read. Records aren't
    # reordered, so the output stays coordinate-sorted -- only re-indexing
    # is needed, not a re-sort.
    # ------------------------------------------------------------------
    # minimap2 writes secondary alignments without SEQ, so a promoted record
    # has none: flipping flags alone left every rescued read sequence-less and
    # silently dropped at extraction -- the very reads rescue exists for
    # (`F64`). Its old primary, the record being demoted, carries the read's
    # SEQ; take it from there.
    donors = _collect_sequence_donors(
        pysam_mod, bam_path, set(promotions), primary_only=True, bgzf_threads=bgzf_threads
    )
    n_sequences_restored = 0
    promoted: set[str] = set()
    logger.info("rescue_secondary_alignments: rewriting BAM with corrected flags.")
    with (
        pysam_mod.AlignmentFile(bam_path, "rb", **bgzf_threads) as in_bam,
        pysam_mod.AlignmentFile(output_path, "wb", header=in_bam.header, **bgzf_threads) as out_bam,
    ):
        for read in in_bam.fetch(until_eof=True):
            promotion = promotions.get(read.query_name)
            if promotion is not None and not read.is_unmapped and not read.is_supplementary:
                winning_rname, winning_rstart, _chromosome, primary_mapq, winning_length = promotion
                # Match on aligned length too, and promote one record only: a
                # chimeric read can hold several alignments at one start, and
                # matching on (reference, start) alone promoted all of them,
                # leaving one read with several primaries (`F64`).
                is_winning_record = (
                    read.query_name not in promoted
                    and read.reference_name == winning_rname
                    and read.reference_start == winning_rstart
                    and int(read.query_alignment_length or 0) == winning_length
                )
                if is_winning_record:
                    promoted.add(read.query_name)
                    read.is_secondary = False
                    read.mapping_quality = primary_mapq
                    donor = donors.get(_read_key(read))
                    if read.query_sequence is None and donor is not None:
                        n_sequences_restored += int(_restore_sequence(read, donor))
                elif not read.is_secondary:
                    # This record was the old (now-demoted) primary.
                    read.is_secondary = True
                    read.mapping_quality = 0
            out_bam.write(read)

    _index_bam_with_pysam(output_path, threads=threads)
    logger.info("rescue_secondary_alignments: rewrite and re-index complete.")

    logger.info(
        "rescue_secondary_alignments: carried SEQ onto %d of %d promoted records.",
        n_sequences_restored,
        len(promotions),
    )
    return RescueSummary(
        n_reads_examined=n_reads_examined,
        n_reads_rescued=len(promotions),
        reassignment_counts=dict(reassignment_counts),
        n_sequences_restored=n_sequences_restored,
    )


def restore_primary_sequences(
    bam_path: str | Path,
    output_path: str | Path,
    *,
    threads: int | None = None,
) -> SequenceRestoreSummary:
    """Repair alignments rescued before `F64`: one primary per read, with SEQ.

    The pre-fix rescue promoted secondary records, which minimap2 writes without
    SEQ, and could promote several same-start alignments of one chimeric read.
    For every read with a sequence-less primary or more than one primary, the
    longest-aligned primary (rescue's own criterion) is kept and given the
    read's SEQ from a sibling record; any other primaries are demoted to
    secondary. When nothing needs repair the input is left as is and
    ``output_path`` is not written; idempotent on an already-correct BAM.
    """
    pysam_mod = _require_pysam()
    bam_path, output_path = str(bam_path), str(output_path)
    bgzf_threads = {"threads": int(threads)} if threads and int(threads) > 1 else {}

    primaries: dict[tuple[str, bool, bool], list[tuple[str, int, int, bool]]] = {}
    with pysam_mod.AlignmentFile(bam_path, "rb", **bgzf_threads) as bam:
        for read in bam.fetch(until_eof=True):
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            primaries.setdefault(_read_key(read), []).append(
                (
                    read.reference_name,
                    read.reference_start,
                    int(read.query_alignment_length or 0),
                    read.query_sequence is not None,
                )
            )
    summary = SequenceRestoreSummary(
        n_sequenceless_primary=sum(
            1 for records in primaries.values() for record in records if not record[3]
        )
    )
    keep: dict[tuple[str, bool, bool], tuple[str, int, int]] = {}
    for key, records in primaries.items():
        if len(records) > 1 or not records[0][3]:
            best = max(records, key=lambda record: record[2])
            keep[key] = best[:3]
    if not keep:
        logger.info("restore_primary_sequences: every primary has SEQ; input left unchanged.")
        return summary
    donors = _collect_sequence_donors(
        pysam_mod, bam_path, {key[0] for key in keep}, primary_only=False, bgzf_threads=bgzf_threads
    )

    kept: set[tuple[str, bool, bool]] = set()
    with (
        pysam_mod.AlignmentFile(bam_path, "rb", **bgzf_threads) as in_bam,
        pysam_mod.AlignmentFile(output_path, "wb", header=in_bam.header, **bgzf_threads) as out_bam,
    ):
        for read in in_bam.fetch(until_eof=True):
            key = _read_key(read)
            target = keep.get(key)
            if target is not None and not (
                read.is_unmapped or read.is_secondary or read.is_supplementary
            ):
                is_target = key not in kept and target == (
                    read.reference_name,
                    read.reference_start,
                    int(read.query_alignment_length or 0),
                )
                if is_target:
                    kept.add(key)
                    if read.query_sequence is None:
                        donor = donors.get(key)
                        if donor is not None and _restore_sequence(read, donor):
                            summary.n_restored += 1
                        else:
                            summary.n_unrestorable += 1
                else:
                    read.is_secondary = True
                    read.mapping_quality = 0
                    summary.n_extra_primaries_demoted += 1
            out_bam.write(read)
    _index_bam_with_pysam(output_path, threads=threads)
    summary.output_written = True
    logger.info(
        "restore_primary_sequences: restored SEQ on %d record(s), %d unrestorable, "
        "demoted %d extra primary record(s).",
        summary.n_restored,
        summary.n_unrestorable,
        summary.n_extra_primaries_demoted,
    )
    return summary
