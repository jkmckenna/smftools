"""Canonical, content-addressed input manifests for ingestion workflows."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sqlite3
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from smftools.constants import RAW_DIR

INPUT_MANIFEST_SCHEMA_VERSION = 1
INPUT_MANIFEST_DIRNAME = "input_manifest"
RESOLVED_INPUT_MANIFEST_CSV = "resolved_input_manifest.csv"
RESOLVED_INPUT_MANIFEST_JSON = "resolved_input_manifest.json"
INPUT_RESOLUTION_REPORT_JSON = "input_resolution_report.json"
CHECKSUM_CACHE_FILENAME = "checksum_cache.sqlite3"
HASH_CHUNK_SIZE = 1024 * 1024

_CANONICAL_COLUMNS = (
    "source_id",
    "path",
    "sha256",
    "size_bytes",
    "source_kind",
    "source_role",
    "sample",
    "barcode",
    "read_group",
    "pair_id",
    "mate",
    "namespace",
    "modification_capability",
    "trimmed",
    "inferred_fields",
)
_CSV_COLUMNS = ("schema_version", *_CANONICAL_COLUMNS)
_USER_COLUMNS = frozenset(_CANONICAL_COLUMNS) | {"schema_version"}
_FASTQ_SUFFIXES = (".fastq.gz", ".fq.gz", ".fastq.gzip", ".fq.gzip", ".fastq", ".fq")
_KIND_BY_SUFFIX = {
    ".pod5": "pod5",
    ".fast5": "fast5",
    ".bam": "bam",
    ".cram": "cram",
    ".sam": "sam",
    ".h5ad": "h5ad",
}
_ILLUMINA_RE = re.compile(
    r"^(?P<sample>.+?)_S(?P<sample_number>\d+)_L(?P<lane>\d{3})_"
    r"R(?P<mate>[12])_(?P<chunk>\d{3})$",
    re.IGNORECASE,
)
_GENERIC_MATE_RE = re.compile(r"^(?P<pair>.+?)(?:[._-](?:R|read)?(?P<mate>[12]))$", re.IGNORECASE)
_AMBIGUOUS_MATE_TOKEN_RE = re.compile(r"(?:^|[._-])(?:R|read)?[12](?:[._-]|$)", re.IGNORECASE)


class InputManifestError(ValueError):
    """Raised when input declarations cannot be normalized safely."""


class InputManifestTransitionKind(str, Enum):
    """Relationship between a selected and requested canonical source set."""

    IDENTICAL = "identical"
    APPEND_ONLY = "append_only"
    REMOVED = "removed"
    CONTENT_MUTATED = "content_mutated"
    METADATA_MUTATED = "metadata_mutated"
    REPLACED = "replaced"


@dataclass(frozen=True)
class InputManifestRow:
    """One normalized, content-addressed ingestion source."""

    source_id: str
    path: str
    sha256: str
    size_bytes: int
    source_kind: str
    source_role: str
    sample: str = ""
    barcode: str = ""
    read_group: str = ""
    pair_id: str = ""
    mate: str = "unpaired"
    namespace: str = ""
    modification_capability: str = "sequence_only"
    trimmed: str = "unknown"
    inferred_fields: tuple[str, ...] = ()

    def identity(self) -> dict[str, Any]:
        """Return the relocation-invariant semantic identity payload."""
        payload = asdict(self)
        payload.pop("path")
        payload.pop("source_id")
        payload.pop("inferred_fields")
        return payload

    def csv_record(self) -> dict[str, Any]:
        """Return a stable CSV representation."""
        record = asdict(self)
        record["schema_version"] = INPUT_MANIFEST_SCHEMA_VERSION
        record["inferred_fields"] = ";".join(self.inferred_fields)
        return record


@dataclass(frozen=True)
class ResolvedInputManifest:
    """A validated canonical manifest and its deterministic digest."""

    rows: tuple[InputManifestRow, ...]
    digest: str
    resolution_method: str
    base_directory: str
    warnings: tuple[str, ...] = ()
    cache_hits: int = 0
    cache_misses: int = 0

    @property
    def input_type(self) -> str:
        kinds = {row.source_kind for row in self.rows}
        if kinds and kinds <= {"unaligned_bam", "aligned_bam", "cram"}:
            return "bam"
        if len(kinds) != 1:
            raise InputManifestError(f"Manifest has mixed source kinds: {sorted(kinds)}")
        kind = next(iter(kinds))
        return kind

    def alignment_inputs(self) -> tuple[InputManifestRow, ...]:
        """Return canonical existing-alignment source partitions."""
        if self.input_type != "bam" or any(row.source_role != "alignment" for row in self.rows):
            raise InputManifestError("alignment_inputs() requires an existing-alignment manifest.")
        return self.rows

    def fastq_inputs(self) -> list[Path | tuple[Path, Path]]:
        """Return explicit FASTQ pairs and singles for concatenation."""
        if self.input_type != "fastq":
            raise InputManifestError("fastq_inputs() requires a FASTQ manifest.")
        grouped: dict[str, dict[str, Path]] = {}
        singles: list[Path] = []
        for row in self.rows:
            path = Path(row.path)
            if not row.pair_id:
                singles.append(path)
            else:
                grouped.setdefault(row.pair_id, {})[row.mate] = path
        pairs = [(mates["R1"], mates["R2"]) for _, mates in sorted(grouped.items())]
        return [*pairs, *sorted(singles)]

    def fastq_barcode_map(self) -> dict[str, str]:
        """Return declared FASTQ barcodes keyed by resolved source path."""
        return {
            row.path: row.barcode for row in self.rows if row.source_kind == "fastq" and row.barcode
        }

    def fastq_read_group_map(self) -> dict[str, str]:
        """Return stable FASTQ source/read-pair identifiers for BAM RG tags."""
        return {
            row.path: row.read_group or row.pair_id or row.barcode or row.source_id
            for row in self.rows
            if row.source_kind == "fastq"
        }

    def fastq_sample_map(self) -> dict[str, str]:
        """Return declared FASTQ sample labels keyed by resolved source path."""
        return {
            row.path: row.sample for row in self.rows if row.source_kind == "fastq" and row.sample
        }


@dataclass(frozen=True)
class InputManifestTransition:
    """Deterministic source-set transition used by immutable raw ingestion."""

    kind: InputManifestTransitionKind
    previous_digest: str
    current_digest: str
    reused_source_ids: tuple[str, ...] = ()
    added_source_ids: tuple[str, ...] = ()
    removed_source_ids: tuple[str, ...] = ()
    changed_paths: tuple[str, ...] = ()

    @property
    def permits_incremental_append(self) -> bool:
        """Return whether only complete new sources were added."""
        return self.kind is InputManifestTransitionKind.APPEND_ONLY

    def to_dict(self) -> dict[str, Any]:
        """Return stable schema-1 provenance for generation manifests."""
        return {
            "schema_version": 1,
            "kind": self.kind.value,
            "previous_digest": self.previous_digest,
            "current_digest": self.current_digest,
            "reused_source_ids": list(self.reused_source_ids),
            "added_source_ids": list(self.added_source_ids),
            "removed_source_ids": list(self.removed_source_ids),
            "changed_paths": list(self.changed_paths),
        }


@dataclass(frozen=True)
class InspectedInputManifest:
    """Lightweight structural result used during config loading."""

    path: Path
    source_paths: tuple[Path, ...]
    input_type: str


@dataclass
class _Declaration:
    path: Path
    values: dict[str, str] = field(default_factory=dict)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def input_manifest_digest(rows: Sequence[InputManifestRow]) -> str:
    """Return the relocation-invariant digest for canonical rows."""
    ordered = sorted(rows, key=lambda row: (row.pair_id, row.mate, row.source_id))
    return hashlib.sha256(_json_bytes([row.identity() for row in ordered])).hexdigest()


def subset_input_manifest(
    manifest: ResolvedInputManifest,
    source_ids: Iterable[str],
) -> ResolvedInputManifest:
    """Return a validated canonical subset for source-scoped execution."""
    selected_ids = {str(value) for value in source_ids}
    rows = tuple(row for row in manifest.rows if row.source_id in selected_ids)
    missing = sorted(selected_ids.difference(row.source_id for row in rows))
    if missing:
        raise InputManifestError(f"Input manifest subset contains unknown source IDs: {missing}")
    if not rows:
        raise InputManifestError("Input manifest subset cannot be empty.")
    _validate_pairs(rows)
    return ResolvedInputManifest(
        rows=rows,
        digest=input_manifest_digest(rows),
        resolution_method=f"{manifest.resolution_method}:subset",
        base_directory=manifest.base_directory,
        warnings=manifest.warnings,
        cache_hits=manifest.cache_hits,
        cache_misses=manifest.cache_misses,
    )


def read_resolved_input_manifest(path: str | Path) -> ResolvedInputManifest:
    """Read and verify a previously published canonical manifest JSON."""
    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputManifestError(
            f"Published input manifest is unreadable: {manifest_path}"
        ) from exc
    if int(payload.get("schema_version", -1)) != INPUT_MANIFEST_SCHEMA_VERSION:
        raise InputManifestError("Published input manifest schema is incompatible.")
    source_records = payload.get("sources")
    if not isinstance(source_records, list) or not source_records:
        raise InputManifestError("Published input manifest contains no source rows.")
    rows: list[InputManifestRow] = []
    try:
        for record in source_records:
            if not isinstance(record, Mapping):
                raise TypeError
            rows.append(
                InputManifestRow(
                    source_id=str(record["source_id"]),
                    path=str(record["path"]),
                    sha256=str(record["sha256"]),
                    size_bytes=int(record["size_bytes"]),
                    source_kind=str(record["source_kind"]),
                    source_role=str(record["source_role"]),
                    sample=str(record.get("sample", "")),
                    barcode=str(record.get("barcode", "")),
                    read_group=str(record.get("read_group", "")),
                    pair_id=str(record.get("pair_id", "")),
                    mate=str(record.get("mate", "unpaired")),
                    namespace=str(record.get("namespace", "")),
                    modification_capability=str(
                        record.get("modification_capability", "sequence_only")
                    ),
                    trimmed=str(record.get("trimmed", "unknown")),
                    inferred_fields=tuple(
                        value
                        for value in str(record.get("inferred_fields", "")).split(";")
                        if value
                    ),
                )
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise InputManifestError("Published input manifest source rows are invalid.") from exc
    rows.sort(key=lambda row: (row.pair_id, row.mate, row.source_id))
    if any(
        row.source_id != hashlib.sha256(_json_bytes(row.identity())).hexdigest() for row in rows
    ):
        raise InputManifestError("Published input manifest contains an invalid source identity.")
    _validate_pairs(rows)
    digest = input_manifest_digest(rows)
    if str(payload.get("manifest_digest", "")) != digest:
        raise InputManifestError("Published input manifest digest does not match its source rows.")
    return ResolvedInputManifest(
        rows=tuple(rows),
        digest=digest,
        resolution_method=str(payload.get("resolution_method", "published")),
        base_directory=str(payload.get("base_directory", manifest_path.parent)),
        warnings=tuple(map(str, payload.get("warnings", ()))),
    )


def classify_input_manifest_transition(
    previous: ResolvedInputManifest,
    current: ResolvedInputManifest,
) -> InputManifestTransition:
    """Classify whether a requested source set is a safe append or rebuild."""
    previous_by_id = {row.source_id: row for row in previous.rows}
    current_by_id = {row.source_id: row for row in current.rows}
    previous_ids = set(previous_by_id)
    current_ids = set(current_by_id)
    reused = tuple(sorted(previous_ids & current_ids))
    added = tuple(sorted(current_ids - previous_ids))
    removed = tuple(sorted(previous_ids - current_ids))
    if not added and not removed:
        kind = InputManifestTransitionKind.IDENTICAL
        changed_paths: tuple[str, ...] = ()
    elif not removed:
        kind = InputManifestTransitionKind.APPEND_ONLY
        changed_paths = ()
    else:
        removed_rows = [previous_by_id[source_id] for source_id in removed]
        added_rows = [current_by_id[source_id] for source_id in added]
        current_by_path = {str(Path(row.path).resolve(strict=False)): row for row in added_rows}
        content_paths: set[str] = set()
        metadata_paths: set[str] = set()
        added_by_checksum: dict[str, list[InputManifestRow]] = {}
        for row in added_rows:
            added_by_checksum.setdefault(row.sha256, []).append(row)
        for old in removed_rows:
            normalized_path = str(Path(old.path).resolve(strict=False))
            replacement = current_by_path.get(normalized_path)
            if replacement is not None:
                target = content_paths if replacement.sha256 != old.sha256 else metadata_paths
                target.add(normalized_path)
            elif old.sha256 in added_by_checksum:
                metadata_paths.add(normalized_path)
        changed_paths = tuple(sorted(content_paths | metadata_paths))
        if content_paths:
            kind = InputManifestTransitionKind.CONTENT_MUTATED
        elif metadata_paths:
            kind = InputManifestTransitionKind.METADATA_MUTATED
        elif added:
            kind = InputManifestTransitionKind.REPLACED
        else:
            kind = InputManifestTransitionKind.REMOVED
    return InputManifestTransition(
        kind=kind,
        previous_digest=previous.digest,
        current_digest=current.digest,
        reused_source_ids=reused,
        added_source_ids=added,
        removed_source_ids=removed,
        changed_paths=changed_paths,
    )


def _nonempty(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _source_kind(path: Path, declared: str = "") -> str:
    declared = declared.strip().lower()
    name = path.name.lower()
    inferred = (
        "fastq" if name.endswith(_FASTQ_SUFFIXES) else _KIND_BY_SUFFIX.get(path.suffix.lower())
    )
    if not inferred:
        raise InputManifestError(f"Unsupported input file type: {path}")
    if inferred == "bam":
        if declared not in {"", "bam", "unaligned_bam", "aligned_bam"}:
            raise InputManifestError(
                f"Declared source_kind={declared!r} conflicts with BAM file type: {path}"
            )
        return "unaligned_bam" if declared in {"", "bam"} else declared
    if declared and declared != inferred:
        raise InputManifestError(
            f"Declared source_kind={declared!r} conflicts with file type {inferred!r}: {path}"
        )
    return inferred


def _role_for_kind(kind: str, alignment_mode: str) -> str:
    if kind in {"pod5", "fast5"}:
        return "raw_signal"
    if kind in {"aligned_bam", "cram"} and alignment_mode == "existing":
        return "alignment"
    return "reads"


def _capability_for_kind(kind: str, role: str, modality: str) -> str:
    if role == "raw_signal":
        return "raw_signal"
    if kind in {"unaligned_bam", "aligned_bam", "cram"}:
        if modality == "direct":
            return "mm_ml"
        if modality in {"conversion", "deaminase"}:
            return "conversion_sequence"
    if kind == "fastq":
        return "sequence_only"
    return "sequence_only"


def _read_csv_declarations(manifest_path: Path) -> list[_Declaration]:
    if manifest_path.suffix.lower() == ".json":
        from .export_bundle import ExportBundleError, resolve_bundle_input_manifest

        try:
            manifest_path = resolve_bundle_input_manifest(manifest_path)
        except ExportBundleError as exc:
            raise InputManifestError(str(exc)) from exc
    if manifest_path.suffix.lower() != ".csv":
        raise InputManifestError("Input declarations must be a schema-1 CSV or export bundle JSON.")
    if not manifest_path.is_file():
        raise InputManifestError(f"Input manifest does not exist: {manifest_path}")
    try:
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or ())
            if "path" not in columns:
                raise InputManifestError("Input manifest must contain a 'path' column.")
            unknown = sorted(columns - _USER_COLUMNS)
            if unknown:
                raise InputManifestError(f"Unknown input manifest columns: {', '.join(unknown)}")
            declarations = []
            for line_number, record in enumerate(reader, start=2):
                schema_version = _nonempty(record.get("schema_version"))
                if schema_version and schema_version != str(INPUT_MANIFEST_SCHEMA_VERSION):
                    raise InputManifestError(
                        f"Input manifest row {line_number} declares unsupported "
                        f"schema_version={schema_version!r}."
                    )
                raw_path = _nonempty(record.get("path"))
                if not raw_path:
                    raise InputManifestError(f"Input manifest row {line_number} has an empty path.")
                source_path = Path(raw_path).expanduser()
                if not source_path.is_absolute():
                    source_path = manifest_path.parent / source_path
                values = {key: _nonempty(value) for key, value in record.items() if key}
                declarations.append(_Declaration(source_path.resolve(strict=False), values))
    except OSError as exc:
        raise InputManifestError(f"Could not read input manifest {manifest_path}: {exc}") from exc
    if not declarations:
        raise InputManifestError("Input manifest contains no source rows.")
    return declarations


def _validate_declarations(
    declarations: Sequence[_Declaration],
    alignment_mode: str,
    *,
    explicit_manifest: bool,
) -> tuple[str, tuple[Path, ...]]:
    resolved_paths = tuple(declaration.path for declaration in declarations)
    duplicates = sorted(str(path) for path, count in Counter(resolved_paths).items() if count > 1)
    if duplicates:
        raise InputManifestError(f"Duplicate resolved input paths: {', '.join(duplicates)}")
    kinds = {_source_kind(item.path, item.values.get("source_kind", "")) for item in declarations}
    alignment_kinds = {"unaligned_bam", "aligned_bam", "cram"}
    compatible_alignment_mix = alignment_mode == "existing" and kinds <= alignment_kinds
    if len(kinds) != 1 and not compatible_alignment_mix:
        raise InputManifestError(
            f"Input manifest has mixed source kinds: {', '.join(sorted(kinds))}"
        )
    kind = "aligned_bam" if len(kinds) > 1 else next(iter(kinds))
    if kind == "sam":
        raise InputManifestError(f"{kind.upper()} input is not supported yet.")
    if "cram" in kinds and alignment_mode != "existing":
        raise InputManifestError("CRAM input requires alignment_mode='existing'.")
    if alignment_mode == "existing":
        if not kinds <= alignment_kinds:
            raise InputManifestError(
                "alignment_mode='existing' requires aligned BAM or CRAM input."
            )
        if any(
            _nonempty(item.values.get("source_kind")) == "unaligned_bam" for item in declarations
        ):
            raise InputManifestError(
                "alignment_mode='existing' conflicts with source_kind='unaligned_bam'."
            )
    if kinds <= alignment_kinds and len(declarations) != 1 and not explicit_manifest:
        raise InputManifestError("Multiple alignment sources require an explicit input manifest.")
    return kind, resolved_paths


def inspect_input_manifest(
    manifest_path: str | Path, *, alignment_mode: str = "align"
) -> InspectedInputManifest:
    """Validate manifest structure without hashing or writing task state."""
    path = Path(manifest_path).expanduser().resolve(strict=False)
    declarations = _read_csv_declarations(path)
    kind, source_paths = _validate_declarations(
        declarations, alignment_mode, explicit_manifest=True
    )
    for source_path in source_paths:
        if not source_path.is_file():
            raise InputManifestError(f"Input source is missing or not a file: {source_path}")
    input_type = "bam" if kind in {"unaligned_bam", "aligned_bam", "cram"} else kind
    return InspectedInputManifest(path=path, source_paths=source_paths, input_type=input_type)


def _stat_signature(stat_result: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
        stat_result.st_ctime_ns,
    )


def _initialize_cache(connection: sqlite3.Connection) -> None:
    connection.execute(
        """CREATE TABLE IF NOT EXISTS checksums (
        path TEXT PRIMARY KEY, device INTEGER NOT NULL, inode INTEGER NOT NULL,
        size_bytes INTEGER NOT NULL, mtime_ns INTEGER NOT NULL, ctime_ns INTEGER NOT NULL,
        sha256 TEXT NOT NULL)"""
    )


def _checksum(path: Path, connection: sqlite3.Connection) -> tuple[str, int, bool]:
    try:
        before = path.stat()
    except OSError as exc:
        raise InputManifestError(f"Could not stat input source {path}: {exc}") from exc
    signature = _stat_signature(before)
    cached = connection.execute(
        "SELECT device, inode, size_bytes, mtime_ns, ctime_ns, sha256 FROM checksums WHERE path = ?",
        (str(path),),
    ).fetchone()
    if cached and tuple(cached[:5]) == signature:
        return str(cached[5]), before.st_size, True

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if _stat_signature(opened) != signature:
                raise InputManifestError(f"Input source changed before hashing began: {path}")
            while chunk := handle.read(HASH_CHUNK_SIZE):
                digest.update(chunk)
            after_handle = os.fstat(handle.fileno())
        after_path = path.stat()
    except InputManifestError:
        raise
    except OSError as exc:
        raise InputManifestError(f"Could not read input source {path}: {exc}") from exc
    if _stat_signature(after_handle) != signature or _stat_signature(after_path) != signature:
        raise InputManifestError(f"Input source changed while it was being hashed: {path}")
    hexdigest = digest.hexdigest()
    connection.execute(
        """INSERT OR REPLACE INTO checksums
        (path, device, inode, size_bytes, mtime_ns, ctime_ns, sha256)
        VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (str(path), *signature, hexdigest),
    )
    return hexdigest, before.st_size, False


def _fastq_stem(path: Path) -> str:
    lower = path.name.lower()
    for suffix in _FASTQ_SUFFIXES:
        if lower.endswith(suffix):
            return path.name[: -len(suffix)]
    return path.stem


def _infer_fastq_metadata(path: Path) -> tuple[dict[str, str], tuple[str, ...]]:
    stem = _fastq_stem(path)
    illumina = _ILLUMINA_RE.match(stem)
    if illumina:
        match = illumina.groupdict()
        pair_id = f"{match['sample']}_S{match['sample_number']}_L{match['lane']}_{match['chunk']}"
        return (
            {"sample": match["sample"], "pair_id": pair_id, "mate": f"R{match['mate']}"},
            ("sample", "pair_id", "mate"),
        )
    generic = _GENERIC_MATE_RE.match(stem)
    if generic:
        return (
            {"pair_id": generic.group("pair"), "mate": f"R{generic.group('mate')}"},
            ("pair_id", "mate"),
        )
    if _AMBIGUOUS_MATE_TOKEN_RE.search(stem):
        raise InputManifestError(
            f"Ambiguous FASTQ mate pattern in {path.name}; declare pair_id and mate explicitly."
        )
    return ({"mate": "unpaired"}, ("mate",))


def _normalized_row(
    declaration: _Declaration,
    *,
    sha256: str,
    size_bytes: int,
    alignment_mode: str,
    modality: str,
    barcode_map: Mapping[str, str],
    auto_pair: bool,
) -> InputManifestRow:
    values = declaration.values
    kind = _source_kind(declaration.path, values.get("source_kind", ""))
    inferred: list[str] = []
    if alignment_mode == "existing" and kind == "unaligned_bam":
        kind = "aligned_bam"
        inferred.append("source_kind")
    metadata = {key: values.get(key, "") for key in _CANONICAL_COLUMNS}
    if kind == "fastq" and auto_pair and not (metadata["pair_id"] and metadata["mate"]):
        inferred_values, inferred_names = _infer_fastq_metadata(declaration.path)
        for name in inferred_names:
            if not metadata[name]:
                metadata[name] = inferred_values[name]
                inferred.append(name)
    if kind == "fastq":
        mapped_barcode = (
            barcode_map.get(str(declaration.path))
            or barcode_map.get(declaration.path.name)
            or barcode_map.get(_fastq_stem(declaration.path))
        )
        if mapped_barcode and not metadata["barcode"]:
            metadata["barcode"] = str(mapped_barcode)
            inferred.append("barcode")
    role = metadata["source_role"] or _role_for_kind(kind, alignment_mode)
    expected_role = _role_for_kind(kind, alignment_mode)
    if role != expected_role:
        raise InputManifestError(
            f"source_role={role!r} conflicts with source_kind={kind!r}: {declaration.path}"
        )
    capability = metadata["modification_capability"] or _capability_for_kind(kind, role, modality)
    allowed_capabilities = {
        "pod5": {"raw_signal"},
        "fast5": {"raw_signal"},
        "fastq": {"sequence_only", "conversion_sequence"},
        "unaligned_bam": {"mm_ml", "sequence_only", "conversion_sequence"},
        "aligned_bam": {"mm_ml", "sequence_only", "conversion_sequence"},
        "cram": {"mm_ml", "sequence_only", "conversion_sequence"},
        "h5ad": {"sequence_only", "conversion_sequence", "mm_ml"},
    }.get(kind, {"sequence_only"})
    if capability not in allowed_capabilities:
        raise InputManifestError(
            f"modification_capability={capability!r} conflicts with source_kind={kind!r}: "
            f"{declaration.path}"
        )
    if modality == "direct" and capability not in {"raw_signal", "mm_ml"}:
        raise InputManifestError(
            "Direct-modification analysis requires raw-signal or MM/ML-capable input; "
            f"got {capability!r}: {declaration.path}"
        )
    trimmed = (metadata["trimmed"] or "unknown").lower()
    if trimmed not in {"true", "false", "unknown"}:
        raise InputManifestError(f"Invalid trimmed value {trimmed!r}: {declaration.path}")
    mate = metadata["mate"] or "unpaired"
    mate = mate.upper() if mate.lower() != "unpaired" else "unpaired"
    if mate not in {"R1", "R2", "unpaired"}:
        raise InputManifestError(f"Invalid mate value {mate!r}: {declaration.path}")
    identity = {
        "sha256": sha256,
        "size_bytes": size_bytes,
        "source_kind": kind,
        "source_role": role,
        "sample": metadata["sample"],
        "barcode": metadata["barcode"],
        "read_group": metadata["read_group"],
        "pair_id": metadata["pair_id"],
        "mate": mate,
        "namespace": metadata["namespace"],
        "modification_capability": capability,
        "trimmed": trimmed,
    }
    source_id = hashlib.sha256(_json_bytes(identity)).hexdigest()
    for field_name in ("sha256", "size_bytes", "source_id"):
        declared = values.get(field_name, "")
        actual = str(identity.get(field_name, source_id))
        if declared and declared != actual:
            raise InputManifestError(
                f"Declared {field_name} does not match resolved source {declaration.path}."
            )
    return InputManifestRow(
        source_id=source_id,
        path=str(declaration.path),
        inferred_fields=tuple(sorted(set(inferred))),
        **identity,
    )


def _validate_pairs(rows: Sequence[InputManifestRow]) -> None:
    pairs: dict[str, list[InputManifestRow]] = {}
    for row in rows:
        if row.pair_id:
            if row.source_kind != "fastq":
                raise InputManifestError("pair_id is supported only for FASTQ sources.")
            pairs.setdefault(row.pair_id, []).append(row)
        elif row.mate != "unpaired":
            raise InputManifestError(f"FASTQ mate {row.mate} is missing pair_id: {row.path}")
    for pair_id, members in sorted(pairs.items()):
        mates = [member.mate for member in members]
        if len(members) != 2 or sorted(mates) != ["R1", "R2"]:
            raise InputManifestError(
                f"FASTQ pair {pair_id!r} must contain exactly one R1 and one R2; got {mates}."
            )
        for field_name in ("sample", "barcode", "read_group", "namespace", "source_role"):
            values = {getattr(member, field_name) for member in members}
            if len(values) != 1:
                raise InputManifestError(
                    f"FASTQ pair {pair_id!r} has conflicting {field_name} metadata."
                )


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as tmp:
        json.dump(payload, tmp, indent=2, sort_keys=True)
        tmp.write("\n")
        temporary = Path(tmp.name)
    os.replace(temporary, path)


def _atomic_write_csv(path: Path, rows: Iterable[InputManifestRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf-8", newline=""
    ) as tmp:
        writer = csv.DictWriter(tmp, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(row.csv_record() for row in rows)
        temporary = Path(tmp.name)
    os.replace(temporary, path)


def resolve_input_manifest(
    *,
    output_directory: str | Path | None,
    input_manifest_path: str | Path | None = None,
    input_paths: Sequence[str | Path] | None = None,
    alignment_mode: str = "align",
    modality: str = "",
    barcode_map: Mapping[str, str] | None = None,
    auto_pair: bool = True,
    publish: bool = True,
) -> ResolvedInputManifest:
    """Resolve declarations, hash sources, validate metadata, and publish schema 1."""
    if bool(input_manifest_path) == bool(input_paths):
        raise InputManifestError("Provide exactly one of input_manifest_path or input_paths.")
    if publish and output_directory is None:
        raise InputManifestError("output_directory is required when publishing a manifest.")
    staging = (
        Path(output_directory) / RAW_DIR / INPUT_MANIFEST_DIRNAME
        if output_directory is not None
        else None
    )
    if staging is not None and publish:
        staging.mkdir(parents=True, exist_ok=True)
    if input_manifest_path:
        manifest_path = Path(input_manifest_path).expanduser().resolve(strict=False)
        declarations = _read_csv_declarations(manifest_path)
        method = "user_manifest"
        base_directory = manifest_path.parent
    else:
        declarations = [
            _Declaration(Path(path).expanduser().resolve(strict=False))
            for path in input_paths or ()
        ]
        if not declarations:
            raise InputManifestError("No input sources were provided.")
        method = "path_discovery"
        base_directory = Path(
            os.path.commonpath([str(declaration.path.parent) for declaration in declarations])
        )
    _validate_declarations(
        declarations,
        alignment_mode,
        explicit_manifest=bool(input_manifest_path),
    )

    cache_path = (
        staging / CHECKSUM_CACHE_FILENAME if staging is not None and publish else ":memory:"
    )
    rows: list[InputManifestRow] = []
    hits = misses = 0
    with sqlite3.connect(cache_path) as connection:
        _initialize_cache(connection)
        for declaration in declarations:
            checksum, size_bytes, cache_hit = _checksum(declaration.path, connection)
            hits += int(cache_hit)
            misses += int(not cache_hit)
            rows.append(
                _normalized_row(
                    declaration,
                    sha256=checksum,
                    size_bytes=size_bytes,
                    alignment_mode=alignment_mode,
                    modality=str(modality or "").strip().lower(),
                    barcode_map=barcode_map or {},
                    auto_pair=auto_pair,
                )
            )
        connection.commit()

    rows.sort(key=lambda row: (row.pair_id, row.mate, row.source_id))
    duplicate_content = sorted(
        digest for digest, count in Counter(row.sha256 for row in rows).items() if count > 1
    )
    if duplicate_content:
        raise InputManifestError(
            "Duplicate input content was declared more than once: " + ", ".join(duplicate_content)
        )
    duplicate_ids = sorted(
        source_id
        for source_id, count in Counter(row.source_id for row in rows).items()
        if count > 1
    )
    if duplicate_ids:
        raise InputManifestError(
            "Duplicate metadata declarations resolve to the same source identity: "
            + ", ".join(duplicate_ids)
        )
    _validate_pairs(rows)
    digest = input_manifest_digest(rows)
    result = ResolvedInputManifest(
        rows=tuple(rows),
        digest=digest,
        resolution_method=method,
        base_directory=str(base_directory),
        cache_hits=hits,
        cache_misses=misses,
    )
    if publish:
        assert staging is not None
        _atomic_write_csv(staging / RESOLVED_INPUT_MANIFEST_CSV, result.rows)
        _atomic_write_json(
            staging / RESOLVED_INPUT_MANIFEST_JSON,
            {
                "schema_version": INPUT_MANIFEST_SCHEMA_VERSION,
                "manifest_digest": digest,
                "resolution_method": method,
                "base_directory": str(base_directory),
                "source_count": len(rows),
                "warnings": list(result.warnings),
                "sources": [row.csv_record() for row in rows],
            },
        )
        _atomic_write_json(
            staging / INPUT_RESOLUTION_REPORT_JSON,
            {
                "schema_version": INPUT_MANIFEST_SCHEMA_VERSION,
                "manifest_digest": digest,
                "resolution_method": method,
                "base_directory": str(base_directory),
                "source_count": len(rows),
                "cache_hits": hits,
                "cache_misses": misses,
                "warnings": list(result.warnings),
            },
        )
    return result


def resolve_input_manifest_readonly(
    *,
    input_manifest_path: str | Path | None = None,
    input_paths: Sequence[str | Path] | None = None,
    alignment_mode: str = "align",
    modality: str = "",
    barcode_map: Mapping[str, str] | None = None,
    auto_pair: bool = True,
) -> ResolvedInputManifest:
    """Resolve current source identity without writing cache or publication artifacts."""
    return resolve_input_manifest(
        output_directory=None,
        input_manifest_path=input_manifest_path,
        input_paths=input_paths,
        alignment_mode=alignment_mode,
        modality=modality,
        barcode_map=barcode_map,
        auto_pair=auto_pair,
        publish=False,
    )


def input_manifest_artifact_paths(output_directory: str | Path) -> dict[str, Path]:
    """Return paths to the published canonical manifest artifacts."""
    root = Path(output_directory) / RAW_DIR / INPUT_MANIFEST_DIRNAME
    return {
        "input_manifest_csv": root / RESOLVED_INPUT_MANIFEST_CSV,
        "input_manifest_json": root / RESOLVED_INPUT_MANIFEST_JSON,
        "input_resolution_report": root / INPUT_RESOLUTION_REPORT_JSON,
    }


def materialize_input_view(manifest: ResolvedInputManifest, output_directory: str | Path) -> Path:
    """Create a digest-scoped, read-only symlink view for directory-based tools."""
    view = (
        Path(output_directory) / RAW_DIR / INPUT_MANIFEST_DIRNAME / "source_views" / manifest.digest
    )
    view.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(manifest.rows):
        source = Path(row.path)
        link = view / f"{index:06d}_{source.name}"
        if link.exists() or link.is_symlink():
            if link.resolve(strict=False) != source.resolve(strict=False):
                raise InputManifestError(f"Input view collision at {link}")
            continue
        try:
            link.symlink_to(source)
        except OSError as exc:
            raise InputManifestError(
                f"Could not create task-local input view {link}: {exc}"
            ) from exc
    return view
