"""Validation and owned normalization for existing BAM/CRAM input."""

from __future__ import annotations

import gzip
import hashlib
import shutil
import sqlite3
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ..logging_utils import get_logger
from ..readwrite import atomic_write_json
from .raw_intermediate_manifest import alignment_reference_bundle, artifact_checksum

logger = get_logger(__name__)


class AlignmentValidationError(ValueError):
    """Raised when an existing alignment violates the ingestion contract."""


SEQUENCELESS_PRIMARY_FRACTION_LIMIT = 0.01
"""Largest tolerated share of primary records emitted without SEQ/base qualities.

Aligners occasionally emit a handful of such records -- observed at 43 of 289,209
(0.015%) from dorado's minimap2, on reads carrying extreme soft-clipping. A record
with no sequence carries no data, so it is dropped and counted rather than
ingested. Above this fraction the file is not credibly usable, and the original
hard failure is the right answer.
"""


@dataclass(frozen=True)
class AlignmentValidationSummary:
    """Bounded validation facts collected from one aligned BAM."""

    total_records: int
    primary_records: int
    mapped_primary_records: int
    secondary_records: int
    supplementary_records: int
    paired_primary_records: int
    proper_pair_primary_records: int
    singleton_primary_records: int
    discordant_pair_primary_records: int
    coordinate_sorted: bool
    header_sort_order: str
    source_index_present: bool
    source_index_valid: bool
    mm_ml_primary_records: int
    reference_records: tuple[tuple[str, int], ...]
    program_records: tuple[dict[str, str], ...]
    external_aligner: str
    sequenceless_primary_records: int = 0
    """Primary records dropped because the aligner emitted them without SEQ or
    base qualities. See :data:`SEQUENCELESS_PRIMARY_FRACTION_LIMIT`."""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible validation payload."""
        payload = asdict(self)
        payload["reference_records"] = [
            {"name": name, "length": length} for name, length in self.reference_records
        ]
        payload["program_records"] = list(self.program_records)
        return payload


@dataclass(frozen=True)
class AlignmentPartitionValidation:
    """Validation result for one declared alignment source partition."""

    source_id: str
    namespace: str
    path: str
    summary: AlignmentValidationSummary


def _require_pysam():
    from .bam_functions import _require_pysam

    return _require_pysam()


def _open_alignment(pysam, path: Path, reference_fasta: str | Path):
    """Open BAM or CRAM with the exact declared decoding reference."""
    if path.suffix.lower() == ".cram":
        return pysam.AlignmentFile(
            str(path),
            "rc",
            check_sq=False,
            reference_filename=str(reference_fasta),
        )
    return pysam.AlignmentFile(str(path), "rb", check_sq=False)


def _fasta_records(fasta_path: str | Path) -> tuple[tuple[str, int], ...]:
    """Read ordered FASTA names and lengths without changing the reference."""
    path = Path(fasta_path)
    records: list[tuple[str, int]] = []
    name: str | None = None
    length = 0
    try:
        handle_context = (
            gzip.open(path, "rt", encoding="utf-8")
            if path.suffix.lower() in {".gz", ".gzip"}
            else path.open("r", encoding="utf-8")
        )
        with handle_context as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if name is not None:
                        records.append((name, length))
                    name = line[1:].split(maxsplit=1)[0]
                    if not name:
                        raise AlignmentValidationError(f"FASTA has an empty record name: {path}")
                    length = 0
                elif name is None:
                    raise AlignmentValidationError(f"FASTA sequence precedes its header: {path}")
                else:
                    length += len(line)
    except UnicodeDecodeError as exc:
        raise AlignmentValidationError(f"Reference FASTA is not UTF-8 text: {path}") from exc
    except OSError as exc:
        raise AlignmentValidationError(f"Could not read reference FASTA {path}: {exc}") from exc
    if name is not None:
        records.append((name, length))
    if not records:
        raise AlignmentValidationError(f"Reference FASTA contains no records: {path}")
    if len({item[0] for item in records}) != len(records):
        raise AlignmentValidationError(f"Reference FASTA contains duplicate record names: {path}")
    return tuple(records)


def _fasta_reference_md5s(fasta_path: str | Path) -> dict[str, str]:
    """Return CRAM-compatible MD5 checksums for reference sequences."""
    path = Path(fasta_path)
    checksums: dict[str, str] = {}
    name: str | None = None
    digest = hashlib.md5(usedforsecurity=False)
    handle_context = (
        gzip.open(path, "rt", encoding="utf-8")
        if path.suffix.lower() in {".gz", ".gzip"}
        else path.open("r", encoding="utf-8")
    )
    with handle_context as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    checksums[name] = digest.hexdigest()
                name = line[1:].split(maxsplit=1)[0]
                digest = hashlib.md5(usedforsecurity=False)
            else:
                digest.update(line.upper().encode("ascii"))
    if name is not None:
        checksums[name] = digest.hexdigest()
    return checksums


def _program_provenance(header: dict[str, Any]) -> tuple[tuple[dict[str, str], ...], str]:
    records = tuple(
        {
            key: str(record[key])
            for key in ("ID", "PN", "VN", "CL", "PP")
            if record.get(key) not in (None, "")
        }
        for record in header.get("PG", [])
    )
    ignored = {"samtools", "smftools", "concat-fastq", "concatenate_fastqs_to_bam"}
    candidates = [
        record
        for record in records
        if str(record.get("PN") or record.get("ID") or "").strip().lower() not in ignored
    ]
    selected = candidates[-1] if candidates else None
    aligner = str(selected.get("PN") or selected.get("ID")) if selected else "unknown"
    return records, aligner


def _validate_alignment_bed(bed_path: Path, reference_records: tuple[tuple[str, int], ...]) -> None:
    """Validate BED3 coordinates against the source FASTA before extraction."""
    reference_lengths = dict(reference_records)
    region_count = 0
    try:
        with bed_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line or line.startswith(("#", "track ", "browser ")):
                    continue
                fields = line.split()
                if len(fields) < 3:
                    raise AlignmentValidationError(
                        f"Alignment BED line {line_number} has fewer than three fields."
                    )
                chrom = fields[0]
                try:
                    start, end = int(fields[1]), int(fields[2])
                except ValueError as exc:
                    raise AlignmentValidationError(
                        f"Alignment BED line {line_number} has non-integer coordinates."
                    ) from exc
                if chrom not in reference_lengths:
                    raise AlignmentValidationError(
                        f"Alignment BED line {line_number} references unknown FASTA record "
                        f"{chrom!r}."
                    )
                if start < 0 or end <= start or end > reference_lengths[chrom]:
                    raise AlignmentValidationError(
                        f"Alignment BED line {line_number} has invalid coordinates "
                        f"{chrom}:{start}-{end}."
                    )
                region_count += 1
    except UnicodeDecodeError as exc:
        raise AlignmentValidationError(f"Alignment BED is not UTF-8 text: {bed_path}") from exc
    except OSError as exc:
        raise AlignmentValidationError(f"Could not read alignment BED {bed_path}: {exc}") from exc
    if region_count == 0:
        raise AlignmentValidationError("Alignment BED contains no regions.")


def validate_existing_alignment(
    bam_path: str | Path,
    reference_fasta: str | Path,
    *,
    modality: str,
) -> AlignmentValidationSummary:
    """Validate an aligned BAM or CRAM against the exact prepared reference.

    The source is opened read-only and is never indexed, sorted, or rewritten.

    Args:
        bam_path: Existing BAM or CRAM supplied by the user.
        reference_fasta: Exact prepared FASTA whose records must match ``@SQ``.
        modality: Experiment modality; ``direct`` requires coherent MM/ML tags.

    Returns:
        Bounded validation facts used by the alignment manifest.
    """
    pysam = _require_pysam()
    bam_path = Path(bam_path)
    expected_references = _fasta_records(reference_fasta)
    try:
        with _open_alignment(pysam, bam_path, reference_fasta) as bam:
            header = bam.header.to_dict()
            observed_references = tuple(zip(bam.references, bam.lengths, strict=True))
            if not observed_references:
                raise AlignmentValidationError("Existing alignment has no @SQ reference records.")
            if observed_references != expected_references:
                raise AlignmentValidationError(
                    "Existing alignment @SQ names, lengths, or order do not match the exact "
                    "prepared alignment reference."
                )
            if bam_path.suffix.lower() == ".cram":
                expected_md5s = _fasta_reference_md5s(reference_fasta)
                observed_md5s = {
                    str(record.get("SN", "")): str(record.get("M5", "")).lower()
                    for record in header.get("SQ", [])
                }
                missing_md5 = [name for name, _ in expected_references if not observed_md5s[name]]
                mismatched_md5 = [
                    name
                    for name, _ in expected_references
                    if observed_md5s[name] != expected_md5s[name]
                ]
                if missing_md5 or mismatched_md5:
                    details = []
                    if missing_md5:
                        details.append(f"missing M5 for {missing_md5}")
                    if mismatched_md5:
                        details.append(f"reference checksum mismatch for {mismatched_md5}")
                    raise AlignmentValidationError(
                        "Existing CRAM does not match the exact prepared alignment reference: "
                        + "; ".join(details)
                    )
            source_index_present = bam.has_index()
            source_index_valid = False
            if source_index_present:
                try:
                    bam.check_index()
                    source_index_valid = True
                except (OSError, ValueError):
                    source_index_valid = False

            total = primary = mapped = secondary = supplementary = paired = mm_ml = 0
            proper_pair = singleton = discordant = sequenceless = 0
            coordinate_sorted = True
            last_reference = -1
            last_start = -1
            encountered_unmapped = False
            for read in bam.fetch(until_eof=True):
                total += 1
                if read.is_secondary:
                    secondary += 1
                    continue
                if read.is_supplementary:
                    supplementary += 1
                    continue
                primary += 1
                if read.query_sequence is None or read.query_qualities is None:
                    # No sequence means no data to ingest, so drop the record and
                    # count it. The fraction is checked once at the end: aligner
                    # noise must not abort a run, a wholesale-unusable file still
                    # must.
                    sequenceless += 1
                    continue
                if read.is_paired:
                    paired += 1
                    if read.is_read1 == read.is_read2:
                        raise AlignmentValidationError(
                            f"Paired alignment {read.query_name!r} must set exactly one of "
                            "read1/read2."
                        )
                    if read.is_proper_pair:
                        if read.is_unmapped or read.mate_is_unmapped:
                            raise AlignmentValidationError(
                                f"Proper-pair alignment {read.query_name!r} marks a mate unmapped."
                            )
                        proper_pair += 1
                    elif read.is_unmapped or read.mate_is_unmapped:
                        singleton += 1
                    else:
                        discordant += 1
                    if not read.mate_is_unmapped and (
                        int(read.next_reference_id) < 0 or int(read.next_reference_start) < 0
                    ):
                        raise AlignmentValidationError(
                            f"Paired alignment {read.query_name!r} has a mapped mate but lacks "
                            "mate reference/position fields."
                        )
                elif read.is_read1 or read.is_read2:
                    raise AlignmentValidationError(
                        f"Unpaired alignment {read.query_name!r} has read1/read2 flags."
                    )
                has_mm = read.has_tag("MM") or read.has_tag("Mm")
                has_ml = read.has_tag("ML") or read.has_tag("Ml")
                if has_mm != has_ml:
                    raise AlignmentValidationError(
                        f"Primary alignment {read.query_name!r} has only one of MM/ML."
                    )
                if has_mm:
                    mm_ml += 1
                    try:
                        mm_value = read.get_tag("MM") if read.has_tag("MM") else read.get_tag("Mm")
                        if not isinstance(mm_value, str) or not mm_value:
                            raise ValueError("MM must be a nonempty string")
                        probabilities = (
                            read.get_tag("ML") if read.has_tag("ML") else read.get_tag("Ml")
                        )
                        if any(int(value) < 0 or int(value) > 255 for value in probabilities):
                            raise ValueError("ML value outside byte range")
                        if read.modified_bases is None:
                            raise ValueError("MM/ML could not be decoded")
                    except (KeyError, TypeError, ValueError) as exc:
                        raise AlignmentValidationError(
                            f"Primary alignment {read.query_name!r} has invalid MM/ML tags."
                        ) from exc
                if read.is_unmapped:
                    encountered_unmapped = True
                    continue
                mapped += 1
                if read.cigartuples is None:
                    raise AlignmentValidationError(
                        f"Mapped primary alignment {read.query_name!r} has no CIGAR."
                    )
                reference_id = int(read.reference_id)
                start = int(read.reference_start)
                if (
                    encountered_unmapped
                    or reference_id < last_reference
                    or (reference_id == last_reference and start < last_start)
                ):
                    coordinate_sorted = False
                last_reference = reference_id
                last_start = start
    except AlignmentValidationError:
        raise
    except (OSError, ValueError) as exc:
        source_label = "CRAM" if bam_path.suffix.lower() == ".cram" else "BAM"
        raise AlignmentValidationError(
            f"Could not read existing {source_label} {bam_path}: {exc}"
        ) from exc

    if total == 0 or primary == 0:
        raise AlignmentValidationError("Existing alignment contains no primary records.")
    if sequenceless:
        fraction = sequenceless / primary
        if fraction > SEQUENCELESS_PRIMARY_FRACTION_LIMIT:
            raise AlignmentValidationError(
                f"Existing alignment has {sequenceless} of {primary} primary records "
                f"({fraction:.2%}) without SEQ or base qualities, above the "
                f"{SEQUENCELESS_PRIMARY_FRACTION_LIMIT:.2%} limit."
            )
        logger.warning(
            "Dropping %d of %d primary alignment records (%.3f%%) that carry no SEQ or base "
            "qualities; recorded as sequenceless_primary_records.",
            sequenceless,
            primary,
            fraction * 100,
        )
    if mapped == 0:
        raise AlignmentValidationError("Existing alignment contains no mapped primary records.")
    # Compare against the records actually ingested: dropped sequenceless records
    # never reached the MM/ML check, so counting them here would fail a direct
    # experiment for the wrong reason.
    if str(modality).strip().lower() == "direct" and mm_ml != primary - sequenceless:
        raise AlignmentValidationError(
            "Direct-modification existing alignment requires valid MM/ML tags on every primary read."
        )
    programs, aligner = _program_provenance(header)
    coordinate_sorted = (
        coordinate_sorted and str(header.get("HD", {}).get("SO", "")) == "coordinate"
    )
    return AlignmentValidationSummary(
        total_records=total,
        primary_records=primary,
        mapped_primary_records=mapped,
        secondary_records=secondary,
        supplementary_records=supplementary,
        paired_primary_records=paired,
        proper_pair_primary_records=proper_pair,
        singleton_primary_records=singleton,
        discordant_pair_primary_records=discordant,
        coordinate_sorted=coordinate_sorted,
        header_sort_order=str(header.get("HD", {}).get("SO", "unknown")),
        source_index_present=source_index_present,
        source_index_valid=source_index_valid,
        mm_ml_primary_records=mm_ml,
        reference_records=observed_references,
        program_records=programs,
        external_aligner=aligner,
        sequenceless_primary_records=sequenceless,
    )


def normalize_existing_alignment(
    source_bam: str | Path,
    output_bam: str | Path,
    reference_fasta: str | Path,
    *,
    modality: str,
    threads: int | None = None,
) -> tuple[Path, Path, AlignmentValidationSummary, AlignmentValidationSummary]:
    """Normalize an existing BAM or CRAM into an owned indexed BAM artifact."""
    pysam = _require_pysam()
    source_bam = Path(source_bam)
    output_bam = Path(output_bam)
    output_bam.parent.mkdir(parents=True, exist_ok=True)
    source_summary = validate_existing_alignment(source_bam, reference_fasta, modality=modality)
    if source_summary.coordinate_sorted and source_bam.suffix.lower() == ".bam":
        shutil.copy2(source_bam, output_bam)
    elif source_summary.coordinate_sorted:
        try:
            with _open_alignment(pysam, source_bam, reference_fasta) as source:
                with pysam.AlignmentFile(str(output_bam), "wb", header=source.header) as output:
                    for record in source.fetch(until_eof=True):
                        output.write(record)
        except (OSError, ValueError) as exc:
            raise AlignmentValidationError(
                f"Could not normalize existing CRAM {source_bam}: {exc}"
            ) from exc
    else:
        arguments = ["--reference", str(reference_fasta), "-o", str(output_bam)]
        if threads and int(threads) > 1:
            arguments.extend(["-@", str(int(threads))])
        arguments.append(str(source_bam))
        try:
            pysam.sort(*arguments)
        except (OSError, ValueError) as exc:
            raise AlignmentValidationError(
                f"Could not coordinate-sort existing BAM {source_bam}: {exc}"
            ) from exc
    output_bai = Path(f"{output_bam}.bai")
    try:
        index_arguments = []
        if threads and int(threads) > 1:
            index_arguments.extend(["-@", str(int(threads))])
        index_arguments.append(str(output_bam))
        pysam.index(*index_arguments)
    except (OSError, ValueError) as exc:
        raise AlignmentValidationError(
            f"Could not index normalized BAM {output_bam}: {exc}"
        ) from exc
    normalized_summary = validate_existing_alignment(output_bam, reference_fasta, modality=modality)
    if not normalized_summary.coordinate_sorted or not normalized_summary.source_index_valid:
        raise AlignmentValidationError("Normalized existing alignment is not sorted and indexed.")
    return output_bam, output_bai, source_summary, normalized_summary


def validate_alignment_partitions(
    rows,
    reference_fasta: str | Path,
    *,
    modality: str,
) -> tuple[AlignmentPartitionValidation, ...]:
    """Validate compatible partitions and reject cross-source template collisions.

    Sources are scanned one at a time. Template membership is tracked in a temporary
    SQLite index so validation memory remains bounded by database page caches rather
    than the number of reads in the experiment. Repeated template names are allowed
    across distinct declared namespaces, but never across source shards in the same
    namespace.

    Args:
        rows: Canonically ordered existing-alignment manifest rows.
        reference_fasta: Exact prepared FASTA used for every source partition.
        modality: Experiment modality used for tag-capability validation.

    Returns:
        Per-partition validation summaries in canonical source order.
    """
    pysam = _require_pysam()
    rows = tuple(rows)
    if not rows:
        raise AlignmentValidationError("Alignment partition manifest contains no sources.")
    results: list[AlignmentPartitionValidation] = []
    expected_pair_layout: str | None = None
    with tempfile.TemporaryDirectory(prefix="smftools-alignment-partitions-") as temporary:
        collision_db = Path(temporary) / "templates.sqlite3"
        with sqlite3.connect(collision_db) as connection:
            connection.execute(
                "CREATE TABLE templates ("
                "namespace TEXT NOT NULL, template_id TEXT NOT NULL, "
                "source_id TEXT NOT NULL, PRIMARY KEY(namespace, template_id))"
            )
            for row in rows:
                path = Path(str(row.path))
                source_id = str(row.source_id)
                namespace = str(getattr(row, "namespace", "") or "")
                summary = validate_existing_alignment(path, reference_fasta, modality=modality)
                pair_layout = (
                    "unpaired"
                    if summary.paired_primary_records == 0
                    else "paired"
                    if summary.paired_primary_records == summary.primary_records
                    else "mixed"
                )
                if expected_pair_layout is None:
                    expected_pair_layout = pair_layout
                elif pair_layout != expected_pair_layout:
                    raise AlignmentValidationError(
                        "Alignment source partitions have incompatible pair layouts: "
                        f"expected {expected_pair_layout}, found {pair_layout} in {path}."
                    )
                try:
                    with _open_alignment(pysam, path, reference_fasta) as alignment:
                        batch: list[tuple[str, str, str]] = []
                        for record in alignment.fetch(until_eof=True):
                            if record.is_secondary or record.is_supplementary:
                                continue
                            batch.append((namespace, str(record.query_name), source_id))
                            if len(batch) >= 10_000:
                                _insert_partition_templates(connection, batch, path)
                                batch.clear()
                        if batch:
                            _insert_partition_templates(connection, batch, path)
                except AlignmentValidationError:
                    raise
                except (OSError, ValueError) as exc:
                    raise AlignmentValidationError(
                        f"Could not inspect template identities in {path}: {exc}"
                    ) from exc
                results.append(
                    AlignmentPartitionValidation(
                        source_id=source_id,
                        namespace=namespace,
                        path=str(path),
                        summary=summary,
                    )
                )
    return tuple(results)


def _insert_partition_templates(
    connection: sqlite3.Connection,
    rows: list[tuple[str, str, str]],
    path: Path,
) -> None:
    """Insert one bounded template batch and report cross-source collisions."""
    for namespace, template_id, source_id in rows:
        existing = connection.execute(
            "SELECT source_id FROM templates WHERE namespace = ? AND template_id = ?",
            (namespace, template_id),
        ).fetchone()
        if existing is not None and str(existing[0]) != source_id:
            namespace_label = namespace or "<default>"
            raise AlignmentValidationError(
                "Alignment source partitions contain duplicate template identity "
                f"{template_id!r} in namespace {namespace_label!r}; collision detected at {path}."
            )
        connection.execute(
            "INSERT OR IGNORE INTO templates(namespace, template_id, source_id) VALUES (?, ?, ?)",
            (namespace, template_id, source_id),
        )


def prepare_alignment_reference_bundle(
    source_fasta: str | Path,
    output_directory: str | Path,
    *,
    modality: str,
    conversion_types: list[str] | tuple[str, ...] = (),
    strands: list[str] | tuple[str, ...] = (),
    alignment_regions_bed: str | Path | None = None,
    threads: int = 1,
) -> tuple[Path, Path]:
    """Publish the exact FASTA bundle expected by existing-alignment validation.

    This helper mirrors raw ingestion's reference reduction followed by conversion
    transformation, allowing external workflows to align before running smftools.

    Args:
        source_fasta: Original experiment FASTA.
        output_directory: Destination for the prepared FASTA and JSON manifest.
        modality: Experiment modality.
        conversion_types: Conversion states used for conversion SMF.
        strands: Strand states used for conversion SMF.
        alignment_regions_bed: Optional original-coordinate BED3+ reduction.
        threads: Conversion worker count.

    Returns:
        Prepared FASTA and bundle-manifest paths.
    """
    from .fasta_functions import (
        generate_converted_FASTA,
        get_chromosome_lengths,
        subsample_fasta_from_bed,
    )

    source_fasta = Path(source_fasta).expanduser().resolve(strict=True)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    bed = (
        Path(alignment_regions_bed).expanduser().resolve(strict=True)
        if alignment_regions_bed
        else None
    )
    if str(modality).strip().lower() == "conversion" and (not conversion_types or not strands):
        raise AlignmentValidationError(
            "Conversion reference preparation requires conversion_types and strands."
        )
    cfg = SimpleNamespace(
        fasta=source_fasta,
        alignment_regions_bed=bed,
        smf_modality=str(modality),
        conversion_types=list(conversion_types),
        strands=list(strands),
    )
    bundle = alignment_reference_bundle(cfg)
    stem = f"prepared_alignment_reference_{bundle['digest'][:12]}"
    staged_source = output_directory / f"{stem}.source.fasta"
    reduced_fasta = output_directory / f"{stem}.reduced.fasta"
    prepared_fasta = output_directory / f"{stem}.fasta"
    source_context = (
        gzip.open(source_fasta, "rt", encoding="utf-8")
        if source_fasta.suffix.lower() in {".gz", ".gzip"}
        else source_fasta.open("r", encoding="utf-8")
    )
    with source_context as source_handle, staged_source.open("w", encoding="utf-8") as staged:
        shutil.copyfileobj(source_handle, staged)
    working_fasta = staged_source
    if bed is not None:
        _validate_alignment_bed(bed, _fasta_records(staged_source))
        subsample_fasta_from_bed(staged_source, bed, output_directory, reduced_fasta)
        working_fasta = reduced_fasta
    if str(modality).strip().lower() == "conversion":
        generate_converted_FASTA(
            working_fasta,
            list(conversion_types),
            list(strands),
            prepared_fasta,
            num_threads=max(1, int(threads)),
        )
    else:
        shutil.copy2(working_fasta, prepared_fasta)
    get_chromosome_lengths(prepared_fasta)
    manifest_path = output_directory / f"{stem}.json"
    atomic_write_json(
        manifest_path,
        {
            "schema_version": 1,
            "state": "complete",
            "bundle": bundle,
            "prepared_fasta": {
                "path": prepared_fasta.name,
                "sha256": artifact_checksum(prepared_fasta),
                "reference_records": [
                    {"name": name, "length": length}
                    for name, length in _fasta_records(prepared_fasta)
                ],
            },
        },
    )
    reduced_fasta.unlink(missing_ok=True)
    staged_source.unlink(missing_ok=True)
    Path(f"{staged_source}.fai").unlink(missing_ok=True)
    staged_source.with_suffix(".chrom.sizes").unlink(missing_ok=True)
    return prepared_fasta, manifest_path
