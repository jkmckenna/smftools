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
        if len(kinds) != 1:
            raise InputManifestError(f"Manifest has mixed source kinds: {sorted(kinds)}")
        kind = next(iter(kinds))
        return "bam" if kind in {"unaligned_bam", "aligned_bam"} else kind

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
        """Return filename-to-barcode overrides for declared FASTQ metadata."""
        return {Path(row.path).name: row.barcode for row in self.rows if row.barcode}


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
    if kind in {"unaligned_bam", "aligned_bam"}:
        if modality == "direct":
            return "mm_ml"
        if modality in {"conversion", "deaminase"}:
            return "conversion_sequence"
    if kind == "fastq":
        return "sequence_only"
    return "sequence_only"


def _read_csv_declarations(manifest_path: Path) -> list[_Declaration]:
    if manifest_path.suffix.lower() != ".csv":
        raise InputManifestError("Input manifest schema 1 supports CSV files only.")
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
    declarations: Sequence[_Declaration], alignment_mode: str
) -> tuple[str, tuple[Path, ...]]:
    resolved_paths = tuple(declaration.path for declaration in declarations)
    duplicates = sorted(str(path) for path, count in Counter(resolved_paths).items() if count > 1)
    if duplicates:
        raise InputManifestError(f"Duplicate resolved input paths: {', '.join(duplicates)}")
    kinds = {_source_kind(item.path, item.values.get("source_kind", "")) for item in declarations}
    if len(kinds) != 1:
        raise InputManifestError(
            f"Input manifest has mixed source kinds: {', '.join(sorted(kinds))}"
        )
    kind = next(iter(kinds))
    if kind in {"sam", "cram"}:
        raise InputManifestError(f"{kind.upper()} input is not supported yet.")
    if alignment_mode == "existing":
        raise InputManifestError("alignment_mode='existing' is reserved but not implemented yet.")
    if kind in {"unaligned_bam", "aligned_bam"} and len(declarations) != 1:
        raise InputManifestError("Multiple BAM input sources are not supported yet.")
    return kind, resolved_paths


def inspect_input_manifest(
    manifest_path: str | Path, *, alignment_mode: str = "align"
) -> InspectedInputManifest:
    """Validate manifest structure without hashing or writing task state."""
    path = Path(manifest_path).expanduser().resolve(strict=False)
    declarations = _read_csv_declarations(path)
    kind, source_paths = _validate_declarations(declarations, alignment_mode)
    for source_path in source_paths:
        if not source_path.is_file():
            raise InputManifestError(f"Input source is missing or not a file: {source_path}")
    input_type = "bam" if kind in {"unaligned_bam", "aligned_bam"} else kind
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
    output_directory: str | Path,
    input_manifest_path: str | Path | None = None,
    input_paths: Sequence[str | Path] | None = None,
    alignment_mode: str = "align",
    modality: str = "",
    barcode_map: Mapping[str, str] | None = None,
    auto_pair: bool = True,
) -> ResolvedInputManifest:
    """Resolve declarations, hash sources, validate metadata, and publish schema 1."""
    if bool(input_manifest_path) == bool(input_paths):
        raise InputManifestError("Provide exactly one of input_manifest_path or input_paths.")
    staging = Path(output_directory) / RAW_DIR / INPUT_MANIFEST_DIRNAME
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
    _validate_declarations(declarations, alignment_mode)

    cache_path = staging / CHECKSUM_CACHE_FILENAME
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
    digest = hashlib.sha256(_json_bytes([row.identity() for row in rows])).hexdigest()
    result = ResolvedInputManifest(
        rows=tuple(rows),
        digest=digest,
        resolution_method=method,
        base_directory=str(base_directory),
        cache_hits=hits,
        cache_misses=misses,
    )
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
