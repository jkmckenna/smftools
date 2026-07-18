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

    def to_dataframe(self):
        import pandas as pd

        rows = [
            {"old_chromosome": old, "new_chromosome": new, "n_reads": count}
            for (old, new), count in sorted(self.reassignment_counts.items())
        ]
        return pd.DataFrame(rows, columns=["old_chromosome", "new_chromosome", "n_reads"])


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
        threads: Optional thread count forwarded to BAM re-indexing.

    Returns:
        RescueSummary with counts of reads examined/rescued and a breakdown
        of (old_chromosome, new_chromosome) reassignment counts.

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

    with pysam_mod.AlignmentFile(bam_path, "rb") as bam:
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

    # query_name -> (winning_reference_name, winning_reference_start, winning_chromosome, old_primary_mapq)
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

        promotions[query_name] = (best_rname, best_rstart, best_chromosome, primary_mapq)
        reassignment_counts[(primary_chromosome, best_chromosome)] += 1

    logger.info(
        "rescue_secondary_alignments: examined %d reads, rescuing %d (reassignment breakdown: %s).",
        n_reads_examined,
        len(promotions),
        dict(reassignment_counts),
    )

    # ------------------------------------------------------------------
    # Pass 2: rewrite the BAM, flipping the secondary FLAG bit for exactly
    # the winning/demoted record pair per rescued read. Records aren't
    # reordered, so the output stays coordinate-sorted -- only re-indexing
    # is needed, not a re-sort.
    # ------------------------------------------------------------------
    with (
        pysam_mod.AlignmentFile(bam_path, "rb") as in_bam,
        pysam_mod.AlignmentFile(output_path, "wb", header=in_bam.header) as out_bam,
    ):
        for read in in_bam.fetch(until_eof=True):
            promotion = promotions.get(read.query_name)
            if promotion is not None and not read.is_unmapped and not read.is_supplementary:
                winning_rname, winning_rstart, _winning_chromosome, primary_mapq = promotion
                is_winning_record = (
                    read.reference_name == winning_rname and read.reference_start == winning_rstart
                )
                if is_winning_record:
                    read.is_secondary = False
                    read.mapping_quality = primary_mapq
                elif not read.is_secondary:
                    # This record was the old (now-demoted) primary.
                    read.is_secondary = True
                    read.mapping_quality = 0
            out_bam.write(read)

    _index_bam_with_pysam(output_path, threads=threads)

    return RescueSummary(
        n_reads_examined=n_reads_examined,
        n_reads_rescued=len(promotions),
        reassignment_counts=dict(reassignment_counts),
    )
