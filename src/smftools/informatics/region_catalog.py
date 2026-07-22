"""Versioned original-coordinate region catalogs and stored-reference mappings."""

from __future__ import annotations

import gzip
import hashlib
import math
import os
from pathlib import Path
from typing import Iterable, Mapping
from uuid import uuid4

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from Bio import SeqIO

from smftools.constants import (
    REFERENCE_INTERVAL_MAP_FILENAME,
    REGION_CATALOG_DIRNAME,
    REGION_CATALOG_FILENAMES,
)

REGION_CATALOG_SCHEMA_VERSION = 1
REFERENCE_INTERVAL_MAP_SCHEMA_VERSION = 1
COORDINATE_SYSTEM = "0-based-half-open-original-fasta"
REGION_SCOPES = ("alignment", "analysis", "plot")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fasta_lengths(path: str | Path) -> dict[str, int]:
    path = Path(path)
    open_func = gzip.open if path.suffix == ".gz" else open
    mode = "rt" if path.suffix == ".gz" else "r"
    lengths: dict[str, int] = {}
    with open_func(path, mode, encoding="utf-8") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            if record.id in lengths:
                raise ValueError(f"FASTA contains duplicate record ID {record.id!r}: {path}")
            lengths[str(record.id)] = len(record.seq)
    if not lengths:
        raise ValueError(f"FASTA contains no records: {path}")
    return lengths


def _stable_id(prefix: str, *parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        encoded = "" if part is None or part is pd.NA else str(part)
        payload = encoded.encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return f"{prefix}_{digest.hexdigest()[:24]}"


def _region_frame(rows: list[dict[str, object]] | None = None) -> pd.DataFrame:
    rows = rows or []
    columns = {
        "schema_version": "int16",
        "scope": "string",
        "region_id": "string",
        "original_reference": "string",
        "original_start": "int64",
        "original_end": "int64",
        "name": "string",
        "score": "Float64",
        "strand": "string",
        "source_row": "int64",
        "source_filename": "string",
        "source_sha256": "string",
        "coordinate_system": "string",
        "overlaps_previous": "bool",
        "adjacent_previous": "bool",
    }
    frame = pd.DataFrame(rows)
    for column, dtype in columns.items():
        if column not in frame:
            frame[column] = pd.Series(dtype=dtype)
        else:
            frame[column] = frame[column].astype(dtype)
    return frame[list(columns)]


def _mapping_frame(rows: list[dict[str, object]] | None = None) -> pd.DataFrame:
    rows = rows or []
    columns = {
        "schema_version": "int16",
        "mapping_id": "string",
        "stored_reference": "string",
        "alignment_reference": "string",
        "original_reference": "string",
        "stored_start": "int64",
        "stored_end": "int64",
        "original_start": "int64",
        "original_end": "int64",
        "strand": "string",
        "conversion": "string",
        "modality": "string",
        "reference_kind": "string",
        "source_region_id": "string",
        "coordinate_orientation": "int8",
        "original_fasta_sha256": "string",
        "alignment_fasta_sha256": "string",
    }
    frame = pd.DataFrame(rows)
    for column, dtype in columns.items():
        if column not in frame:
            frame[column] = pd.Series(dtype=dtype)
        else:
            frame[column] = frame[column].astype(dtype)
    return frame[list(columns)]


def _atomic_write_parquet(
    frame: pd.DataFrame,
    path: Path,
    *,
    metadata: Mapping[str, object] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        table = pa.Table.from_pandas(frame, preserve_index=False)
        parquet_metadata = dict(table.schema.metadata or {})
        parquet_metadata.update(
            {
                str(key).encode("utf-8"): str(value).encode("utf-8")
                for key, value in (metadata or {}).items()
            }
        )
        table = table.replace_schema_metadata(parquet_metadata)
        pq.write_table(table, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def normalize_bed_catalog(
    bed_path: str | Path,
    *,
    scope: str,
    reference_lengths: Mapping[str, int],
) -> pd.DataFrame:
    """Normalize one BED3-BED6 file against original FASTA coordinates.

    Records remain independent and are deterministically sorted. Overlaps and
    adjacency are annotated rather than merged so source-region identity is
    retained for later planning and plotting stages.

    Args:
        bed_path: BED3-BED6 input path.
        scope: One of ``alignment``, ``analysis``, or ``plot``.
        reference_lengths: Original FASTA record lengths.

    Returns:
        A schema-stable normalized region catalog.

    Raises:
        FileNotFoundError: If the BED does not exist.
        ValueError: If a row or coordinate is invalid or ambiguous.
    """
    if scope not in REGION_SCOPES:
        raise ValueError(f"region scope must be one of {REGION_SCOPES}; got {scope!r}")
    path = Path(bed_path)
    if not path.is_file():
        raise FileNotFoundError(f"{scope}_regions_bed not found: {path}")
    source_sha256 = _sha256(path)
    rows: list[dict[str, object]] = []
    seen_names: set[str] = set()
    seen_region_ids: set[str] = set()
    seen_alignment_intervals: set[tuple[str, int, int]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if (
                not line
                or line.startswith("#")
                or line.startswith("track ")
                or line.startswith("browser ")
            ):
                continue
            fields = line.split("\t") if "\t" in line else line.split()
            if not 3 <= len(fields) <= 6:
                raise ValueError(
                    f"{path}:{line_number}: expected BED3-BED6 with 3 to 6 columns; "
                    f"found {len(fields)}"
                )
            reference = fields[0]
            if reference not in reference_lengths:
                raise ValueError(
                    f"{path}:{line_number}: reference {reference!r} is absent from the original FASTA"
                )
            try:
                start = int(fields[1])
                end = int(fields[2])
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number}: BED start and end must be integers"
                ) from exc
            if start < 0 or end <= start:
                raise ValueError(
                    f"{path}:{line_number}: expected 0 <= start < end; got {start}, {end}"
                )
            if end > int(reference_lengths[reference]):
                raise ValueError(
                    f"{path}:{line_number}: interval {reference}:{start}-{end} exceeds original "
                    f"FASTA length {reference_lengths[reference]}"
                )
            name = fields[3] if len(fields) >= 4 and fields[3] != "." else None
            if name is not None:
                if name in seen_names:
                    raise ValueError(f"{path}:{line_number}: duplicate BED name {name!r}")
                seen_names.add(name)
            score: float | None = None
            if len(fields) >= 5 and fields[4] != ".":
                try:
                    score = float(fields[4])
                except ValueError as exc:
                    raise ValueError(
                        f"{path}:{line_number}: BED score must be numeric or '.'"
                    ) from exc
                if not math.isfinite(score) or not 0 <= score <= 1000:
                    raise ValueError(
                        f"{path}:{line_number}: BED score must be finite and within [0, 1000]"
                    )
            strand = fields[5] if len(fields) >= 6 and fields[5] != "." else None
            if strand not in {None, "+", "-"}:
                raise ValueError(f"{path}:{line_number}: BED strand must be '+', '-', or '.'")
            interval_key = (reference, start, end)
            if scope == "alignment" and interval_key in seen_alignment_intervals:
                raise ValueError(
                    f"{path}:{line_number}: duplicate alignment interval {reference}:{start}-{end} "
                    "would create duplicate FASTA record names"
                )
            seen_alignment_intervals.add(interval_key)
            region_id = _stable_id("reg", scope, reference, start, end, name, score, strand)
            if region_id in seen_region_ids:
                raise ValueError(f"{path}:{line_number}: duplicate BED record")
            seen_region_ids.add(region_id)
            rows.append(
                {
                    "schema_version": REGION_CATALOG_SCHEMA_VERSION,
                    "scope": scope,
                    "region_id": region_id,
                    "original_reference": reference,
                    "original_start": start,
                    "original_end": end,
                    "name": name,
                    "score": score,
                    "strand": strand,
                    "source_row": line_number,
                    "source_filename": path.name,
                    "source_sha256": source_sha256,
                    "coordinate_system": COORDINATE_SYSTEM,
                    "overlaps_previous": False,
                    "adjacent_previous": False,
                }
            )
    rows.sort(
        key=lambda row: (
            str(row["original_reference"]),
            int(row["original_start"]),
            int(row["original_end"]),
            "" if row["name"] is None else str(row["name"]),
            str(row["region_id"]),
        )
    )
    state: dict[str, tuple[int, int]] = {}
    for row in rows:
        reference = str(row["original_reference"])
        start = int(row["original_start"])
        end = int(row["original_end"])
        previous = state.get(reference)
        if previous is not None:
            max_end, previous_end = previous
            row["overlaps_previous"] = start < max_end
            row["adjacent_previous"] = start == previous_end
            state[reference] = (max(max_end, end), end)
        else:
            state[reference] = (end, end)
    return _region_frame(rows)


def write_region_catalogs(
    cfg: object,
    *,
    original_fasta: str | Path,
    run_root: str | Path,
) -> dict[str, Path]:
    """Normalize and atomically publish every configured region scope."""
    reference_lengths = _fasta_lengths(original_fasta)
    output_dir = Path(run_root) / REGION_CATALOG_DIRNAME
    outputs: dict[str, Path] = {}
    for scope in REGION_SCOPES:
        configured = getattr(cfg, f"{scope}_regions_bed", None)
        if not configured:
            continue
        catalog = normalize_bed_catalog(
            configured,
            scope=scope,
            reference_lengths=reference_lengths,
        )
        if scope == "alignment" and catalog.empty:
            raise ValueError("alignment_regions_bed contains no alignment intervals")
        outputs[scope] = _atomic_write_parquet(
            catalog,
            output_dir / REGION_CATALOG_FILENAMES[scope],
            metadata={
                "schema_version": REGION_CATALOG_SCHEMA_VERSION,
                "scope": scope,
                "coordinate_system": COORDINATE_SYSTEM,
                "source_filename": Path(configured).name,
                "source_sha256": _sha256(Path(configured)),
            },
        )
    return outputs


def write_normalized_alignment_bed(catalog: pd.DataFrame, path: str | Path) -> Path:
    """Write a validated BED used by the existing FASTA-subsetting backend."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in catalog.itertuples(index=False):
                fields = [
                    str(row.original_reference),
                    str(int(row.original_start)),
                    str(int(row.original_end)),
                ]
                if not pd.isna(row.name):
                    fields.append(str(row.name))
                handle.write("\t".join(fields) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _base_intervals(
    original_fasta: Path,
    alignment_catalog: pd.DataFrame | None,
) -> dict[str, dict[str, object]]:
    if alignment_catalog is not None:
        return {
            f"{row.original_reference}:{int(row.original_start)}-{int(row.original_end)}": {
                "original_reference": str(row.original_reference),
                "original_start": int(row.original_start),
                "original_end": int(row.original_end),
                "source_region_id": str(row.region_id),
                "reference_kind": "alignment_region",
            }
            for row in alignment_catalog.itertuples(index=False)
        }
    return {
        name: {
            "original_reference": name,
            "original_start": 0,
            "original_end": length,
            "source_region_id": None,
            "reference_kind": "full_reference",
        }
        for name, length in _fasta_lengths(original_fasta).items()
    }


def _conversion_records(
    base_intervals: Mapping[str, dict[str, object]],
    conversions: Iterable[str],
    strands: Iterable[str],
) -> dict[str, tuple[str, str, str]]:
    conversions = list(conversions)
    strands = list(strands)
    if not conversions:
        raise ValueError("conversion modality requires at least one conversion label")
    records: dict[str, tuple[str, str, str]] = {}
    unconverted = conversions[0]
    for base_reference in base_intervals:
        records[f"{base_reference}_{unconverted}_top"] = (
            base_reference,
            unconverted,
            "top",
        )
        for conversion in conversions[1:]:
            for strand in strands:
                records[f"{base_reference}_{conversion}_{strand}"] = (
                    base_reference,
                    conversion,
                    strand,
                )
    return records


def build_reference_interval_map(
    *,
    original_fasta: str | Path,
    alignment_fasta: str | Path,
    modality: str,
    conversions: Iterable[str],
    strands: Iterable[str],
    alignment_catalog: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Map raw stored references and alignment records to original coordinates."""
    original_fasta = Path(original_fasta)
    alignment_fasta = Path(alignment_fasta)
    base_intervals = _base_intervals(original_fasta, alignment_catalog)
    alignment_records = _fasta_lengths(alignment_fasta)
    original_sha256 = _sha256(original_fasta)
    alignment_sha256 = _sha256(alignment_fasta)
    rows: list[dict[str, object]] = []
    if modality == "conversion":
        expected = _conversion_records(base_intervals, conversions, strands)
        unexpected = sorted(set(alignment_records).difference(expected))
        missing = sorted(set(expected).difference(alignment_records))
        if unexpected or missing:
            raise ValueError(
                "converted alignment FASTA records do not match the original-coordinate map; "
                f"unexpected={unexpected[:5]}, missing={missing[:5]}"
            )
        record_items = [
            (record, *expected[record], int(alignment_records[record]))
            for record in sorted(alignment_records)
        ]
    elif modality in {"direct", "deaminase"}:
        unexpected = sorted(set(alignment_records).difference(base_intervals))
        missing = sorted(set(base_intervals).difference(alignment_records))
        if unexpected or missing:
            raise ValueError(
                "alignment FASTA records do not match the original-coordinate map; "
                f"unexpected={unexpected[:5]}, missing={missing[:5]}"
            )
        record_items = [
            (record, record, modality, strand, int(alignment_records[record]))
            for record in sorted(alignment_records)
            for strand in ("top", "bottom")
        ]
    else:
        raise ValueError("smf_modality must be one of: conversion, deaminase, direct")
    for alignment_reference, base_reference, conversion, strand, stored_length in record_items:
        interval = base_intervals[base_reference]
        expected_length = int(interval["original_end"]) - int(interval["original_start"])
        if stored_length != expected_length:
            raise ValueError(
                f"alignment FASTA record {alignment_reference!r} has length {stored_length}; "
                f"expected {expected_length} from original coordinates"
            )
        stored_reference = f"{base_reference}_{strand}"
        mapping_id = _stable_id(
            "map",
            stored_reference,
            alignment_reference,
            interval["original_reference"],
            interval["original_start"],
            interval["original_end"],
            conversion,
            strand,
        )
        rows.append(
            {
                "schema_version": REFERENCE_INTERVAL_MAP_SCHEMA_VERSION,
                "mapping_id": mapping_id,
                "stored_reference": stored_reference,
                "alignment_reference": alignment_reference,
                "original_reference": interval["original_reference"],
                "stored_start": 0,
                "stored_end": stored_length,
                "original_start": interval["original_start"],
                "original_end": interval["original_end"],
                "strand": strand,
                "conversion": conversion,
                "modality": modality,
                "reference_kind": interval["reference_kind"],
                "source_region_id": interval["source_region_id"],
                "coordinate_orientation": 1,
                "original_fasta_sha256": original_sha256,
                "alignment_fasta_sha256": alignment_sha256,
            }
        )
    rows.sort(key=lambda row: (str(row["stored_reference"]), str(row["alignment_reference"])))
    return _mapping_frame(rows)


def write_reference_interval_map(
    *,
    run_root: str | Path,
    original_fasta: str | Path,
    alignment_fasta: str | Path,
    modality: str,
    conversions: Iterable[str],
    strands: Iterable[str],
    alignment_catalog: pd.DataFrame | None = None,
) -> Path:
    """Build and atomically publish ``reference_interval_map.parquet``."""
    frame = build_reference_interval_map(
        original_fasta=original_fasta,
        alignment_fasta=alignment_fasta,
        modality=modality,
        conversions=conversions,
        strands=strands,
        alignment_catalog=alignment_catalog,
    )
    return _atomic_write_parquet(
        frame,
        Path(run_root) / REFERENCE_INTERVAL_MAP_FILENAME,
        metadata={
            "schema_version": REFERENCE_INTERVAL_MAP_SCHEMA_VERSION,
            "coordinate_system": COORDINATE_SYSTEM,
            "original_fasta_sha256": _sha256(Path(original_fasta)),
            "alignment_fasta_sha256": _sha256(Path(alignment_fasta)),
        },
    )
