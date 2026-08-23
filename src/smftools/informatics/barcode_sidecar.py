"""Canonical barcode and sample identity sidecars."""

from __future__ import annotations

import json
import os
import re
import tempfile
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from ..logging_utils import get_logger
from ..readwrite import atomic_write_json
from .molecule_identity import alignment_segment_id

logger = get_logger(__name__)

BARCODE_IDENTITY_SCHEMA_VERSION = 2
BARCODE_IDENTITY_REPORT_SUFFIX = ".identity_report.json"
BARCODE_IDENTITY_COLUMNS = (
    "identity_schema_version",
    "read_name",
    "barcode",
    "barcode_source",
    "barcode_confidence",
    "sample",
    "sample_source",
    "sample_confidence",
    "read_group",
    "namespace",
    "identity_status",
    "identity_conflicts",
    # `F35`: the independent assignments, kept apart rather than collapsed into
    # one resolved value. `barcode` remains the authority downstream groups on;
    # these are the evidence it was chosen from, and the only way to tell a
    # directory assignment from a sequence re-derivation after the fact.
    "barcode_assigned",
    "barcode_rederived",
    "barcode_front",
    "barcode_rear",
)
_UNCLASSIFIED = frozenset({"unclassified", "unassigned"})
_UNKNOWN = frozenset({"", "unknown", "none", "nan", "null"})
_FASTQ_SUFFIXES = (".fastq.gz", ".fq.gz", ".fastq.gzip", ".fq.gzip", ".fastq", ".fq")
_MATE_SUFFIX_RE = re.compile(r"(?:[._-](?:R|read)?[12](?:[._-]\d{3})?)$", re.IGNORECASE)
_BARCODE_TOKEN_RE = re.compile(r"barcode([0-9A-Za-z-]+)", re.IGNORECASE)
#: An explicit "no barcode assigned" marker, as its own path component or token.
_UNCLASSIFIED_TOKEN_RE = re.compile(
    r"(?:^|[._-])(?:unclassified|unassigned)(?:$|[._-])", re.IGNORECASE
)


class BarcodeIdentityError(ValueError):
    """Raised when barcode/sample evidence cannot be normalized safely."""


def barcode_identity_report_path(sidecar_path: str | Path) -> Path:
    """Return the validation-report path paired with a canonical sidecar."""
    path = Path(sidecar_path)
    return path.with_name(f"{path.stem}{BARCODE_IDENTITY_REPORT_SUFFIX}")


def _value(value: Any) -> str:
    if value is None or (not isinstance(value, (list, tuple, dict)) and pd.isna(value)):
        return ""
    return str(value).strip()


def _classified(value: str) -> bool:
    return value.lower() not in _UNKNOWN | _UNCLASSIFIED


def _evidence(value: Any) -> str:
    """A per-end evidence value, with placeholder spellings flattened to empty."""
    text = _value(value)
    return "" if text.lower() in _UNKNOWN else text


def _barcode_key(value: str) -> str:
    """Comparison key that survives the naming differences between sources (`F35`).

    A demultiplexed directory yields the bare token `11`, while a sequence
    classifier emits the kit-qualified `NB11`. Compared literally these look
    like disagreeing assignments, which flagged **every** barcoded read as
    `conflicting` and left `classified` at zero across 1.75M reads.

    Only the discriminating part is compared: a recognised prefix is stripped
    and a purely numeric remainder is zero-padded, so `11`, `NB11`, `bc11` and
    `barcode011` share a key. Anything that does not reduce to digits is
    compared case-insensitively as-is rather than being forced into a shape it
    does not have.
    """
    text = value.strip().lower()
    if not text:
        return ""
    for prefix in ("barcode", "bc", "nb", "rb", "bp"):
        if text.startswith(prefix) and text[len(prefix) :]:
            text = text[len(prefix) :]
            break
    text = text.lstrip("-_")
    return f"{int(text):04d}" if text.isdigit() else text


def _confidence(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return min(1.0, max(0.0, result))


def _demuxed_stem(path: Path) -> str:
    """The filename with its FASTQ suffix removed."""
    name = path.name
    lower = name.lower()
    for suffix in _FASTQ_SUFFIXES:
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def _legacy_filename_barcode(path: str | Path) -> str:
    """The barcode a demultiplexed tree assigns to a file, or `""` if it assigns none.

    Both the parent directory and the filename are consulted, directory first:
    in a MinKNOW tree the directory *is* the assignment and the filename merely
    repeats it, and only the directory is present for every layout.

    `unclassified` is a real answer, not a missing one. Reads the demultiplexer
    declined to assign belong to no barcode, and saying so is what makes a
    contamination denominator possible -- silently dropping them leaves only the
    errors, which measures 100% contamination on any input (`F36`).

    The stem fallback below is kept: a `sample_R1.fastq.gz` layout really does
    name its barcode that way. What it must never swallow is MinKNOW's
    `<flowcell>_pass_unclassified_<runid>_<hash>_<chunk>.fastq.gz`, which it
    used to return whole -- handing 163,232 unclassified reads 16 distinct
    filename-shaped "barcodes", one per chunk file. That was harmless only
    while this tier always lost; `F35` made it authoritative and the garbage
    won. Recognising the marker, rather than second-guessing the stem's shape,
    is what fixes it (`F36`).
    """
    path = Path(path)
    for candidate in (path.parent.name, _demuxed_stem(path)):
        if not candidate:
            continue
        if _UNCLASSIFIED_TOKEN_RE.search(candidate):
            return "unclassified"
        match = _BARCODE_TOKEN_RE.search(candidate)
        if match:
            return match.group(1)
    stem = _MATE_SUFFIX_RE.sub("", _demuxed_stem(path))
    return stem or "unknown"


def _manifest_rows(input_manifest: Any) -> tuple[Any, ...]:
    if input_manifest is None:
        return ()
    rows = getattr(input_manifest, "rows", input_manifest)
    return tuple(rows or ())


def _row_value(row: Any, key: str) -> str:
    if isinstance(row, Mapping):
        return _value(row.get(key))
    return _value(getattr(row, key, ""))


def _matching_manifest_rows(
    rows: Sequence[Any],
    *,
    bam_barcode: str,
    read_group: str,
    classifier_barcode: str,
) -> tuple[Any, ...]:
    if len(rows) <= 1:
        return tuple(rows)
    matches = []
    observed = {value for value in (bam_barcode, read_group, classifier_barcode) if value}
    for row in rows:
        declared = {
            _row_value(row, "barcode"),
            _row_value(row, "read_group"),
            _row_value(row, "pair_id"),
            _row_value(row, "source_id"),
        }
        if observed.intersection(declared - {""}):
            matches.append(row)
    return tuple(matches)


def _common_manifest_value(rows: Sequence[Any], matched: Sequence[Any], key: str) -> list[str]:
    selected = matched or rows
    values = {_row_value(row, key) for row in selected} - {""}
    if len(values) == 1:
        return list(values)
    return sorted(values) if matched else []


def _classifier_records(
    sidecar_path: str | Path | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    if sidecar_path is None or not Path(sidecar_path).is_file():
        return {}, {}
    frame = pd.read_parquet(sidecar_path)
    read_column = "read_name" if "read_name" in frame else "read_id"
    if read_column not in frame:
        raise BarcodeIdentityError(
            f"barcode evidence sidecar lacks read_name/read_id: {sidecar_path}"
        )
    barcode_column = "barcode" if "barcode" in frame else "BC" if "BC" in frame else None
    records: dict[str, dict[str, Any]] = {}
    conflicts: dict[str, list[str]] = {}
    for read_name, group in frame.groupby(read_column, sort=False, dropna=False):
        key = str(read_name)
        if barcode_column is not None:
            values = sorted(
                {_value(value) for value in group[barcode_column] if _classified(_value(value))}
            )
            if len(values) > 1:
                conflicts[key] = values
        records[key] = group.iloc[0].to_dict()
    return records, conflicts


#: Reads between progress lines while reconciling identity (`F23`).
_IDENTITY_PROGRESS_INTERVAL = 250_000


def _bam_records(bam_path: str | Path) -> dict[str, dict[str, Any]]:
    """Collect per-read identity evidence from a BAM.

    Emits progress periodically: this is a single-threaded pass over every read
    and took **27.7 minutes** on a 1.75M-read run, during which it previously
    logged nothing at all. A silent half-hour is indistinguishable from a hang,
    and both neighbouring phases report progress (`F23`).
    """
    import time

    from .bam_functions import _require_pysam

    pysam = _require_pysam()
    records: dict[str, dict[str, Any]] = {}
    scanned = 0
    started = time.perf_counter()
    logger.info("Reconciling read identity from %s", Path(bam_path).name)
    with pysam.AlignmentFile(str(bam_path), "rb", check_sq=False) as bam:
        header = bam.header.to_dict()
        read_groups = {
            str(record.get("ID")): record
            for record in header.get("RG", [])
            if record.get("ID") is not None
        }
        for read in bam.fetch(until_eof=True):
            scanned += 1
            if scanned % _IDENTITY_PROGRESS_INTERVAL == 0:
                elapsed = time.perf_counter() - started
                logger.info(
                    "Identity reconciliation: %s reads scanned, %s retained, %.0f reads/s "
                    "(%.1f min elapsed)",
                    f"{scanned:,}",
                    f"{len(records):,}",
                    scanned / max(elapsed, 1e-9),
                    elapsed / 60.0,
                )
            if read.is_secondary or read.is_supplementary:
                continue
            read_group = _value(read.get_tag("RG")) if read.has_tag("RG") else ""
            rg_record = read_groups.get(read_group, {})
            record = {
                "bam_barcode": _value(read.get_tag("BC")) if read.has_tag("BC") else "",
                "read_group": read_group,
                "bam_sample": _value(rg_record.get("SM")),
                "bam_bm": _value(read.get_tag("BM")) if read.has_tag("BM") else "",
                "bam_bi": list(read.get_tag("bi")) if read.has_tag("bi") else None,
            }
            segment_id = alignment_segment_id(read)
            existing = records.get(segment_id)
            if existing is not None and any(
                _classified(_value(existing.get(key)))
                and _classified(_value(record.get(key)))
                and _value(existing.get(key)) != _value(record.get(key))
                for key in ("bam_barcode", "read_group", "bam_sample")
            ):
                existing["bam_record_conflict"] = True
                continue
            records.setdefault(segment_id, record)
    elapsed = time.perf_counter() - started
    logger.info(
        "Identity reconciliation complete: %s reads scanned, %s segments retained in %.1f min "
        "(%.0f reads/s)",
        f"{scanned:,}",
        f"{len(records):,}",
        elapsed / 60.0,
        scanned / max(elapsed, 1e-9),
    )
    return records


def _select(
    tiers: Sequence[tuple[str, float, Sequence[str]]],
    *,
    field: str,
) -> tuple[str, str, float, list[dict[str, str]]]:
    selected = ""
    selected_source = "none"
    selected_confidence = 0.0
    conflicts: list[dict[str, str]] = []
    for source, confidence, raw_values in tiers:
        values = []
        for raw_value in raw_values:
            value = _value(raw_value)
            if value and value.lower() not in _UNKNOWN and value not in values:
                values.append(value)
        classified = [value for value in values if _classified(value)]
        # Comparisons are on the normalized key, never the raw text (`F35`).
        if len({_barcode_key(value) for value in classified}) > 1:
            conflicts.append({"field": field, "source": source, "values": "|".join(classified)})
        candidate = classified[0] if classified else values[0] if values else ""
        if not selected and candidate:
            selected = candidate
            selected_source = source
            selected_confidence = confidence
        elif (
            _classified(selected)
            and _classified(candidate)
            and _barcode_key(candidate) != _barcode_key(selected)
        ):
            conflicts.append(
                {
                    "field": field,
                    "source": source,
                    "values": f"selected={selected}|observed={candidate}",
                }
            )
    if not selected:
        selected = "unknown"
    return selected, selected_source, selected_confidence, conflicts


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".parquet"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def publish_barcode_identity_sidecar(
    bam_path: str | Path,
    output_path: str | Path,
    *,
    input_manifest: Any = None,
    classifier_sidecar: str | Path | None = None,
    classifier_source: str = "sequence",
    directory_authoritative: bool = False,
) -> tuple[Path, Path]:
    """Resolve all barcode/sample authorities and publish canonical schema 1.

    Args:
        bam_path: Prepared BAM whose primary reads define sidecar membership.
        output_path: Destination Parquet path.
        input_manifest: Resolved input manifest or its row sequence.
        classifier_sidecar: Optional route-specific barcode evidence.
        classifier_source: Evidence label, normally ``sequence`` or ``filename``.
            It must describe what the classifier sidecar actually contains. A
            sequence classifier labelled ``filename`` makes the resolved source
            unreadable and silently changes which tier wins (`F35`).
        directory_authoritative: Set for an already-demultiplexed input tree,
            where the directory a read arrived in *is* its assignment and must
            outrank sequence re-derivation. The re-derived call is still kept,
            as ``barcode_rederived``, so the two can be compared.

    Returns:
        The canonical sidecar and validation-report paths.
    """
    rows = _manifest_rows(input_manifest)
    bam_records = _bam_records(bam_path)
    classifier_records, classifier_conflicts = _classifier_records(classifier_sidecar)
    classifier_expected = classifier_sidecar is not None and Path(classifier_sidecar).is_file()
    output_rows: list[dict[str, Any]] = []
    filename_authority_used = False

    # Manifest resolution depends only on the three observed identity values, and
    # `rows` is fixed for the whole loop -- so it is memoized rather than redone
    # per read. Uncached this is O(reads x manifest rows): on a MinKNOW FASTQ
    # tree that is 1.75M reads x 575 files ~= 1e9 row comparisons, plus four
    # more passes for the `_common_manifest_value` calls, which measured **27.7
    # minutes** of silence. The distinct key count is the number of barcodes
    # (~41 here), so the cache turns it into a few dozen computations.
    manifest_cache: dict[tuple[str, str, str], tuple[Any, ...]] = {}

    def _resolve_manifest(bam_barcode: str, read_group: str, classifier_barcode: str):
        key = (bam_barcode, read_group, classifier_barcode)
        cached = manifest_cache.get(key)
        if cached is None:
            matched = _matching_manifest_rows(
                rows,
                bam_barcode=bam_barcode,
                read_group=read_group,
                classifier_barcode=classifier_barcode,
            )
            cached = (
                matched,
                _common_manifest_value(rows, matched, "barcode"),
                _common_manifest_value(rows, matched, "sample"),
                _common_manifest_value(rows, matched, "read_group"),
                _common_manifest_value(rows, matched, "namespace"),
                tuple(row for row in matched if _row_value(row, "source_kind") == "fastq"),
            )
            manifest_cache[key] = cached
        return cached

    for read_name, bam_record in bam_records.items():
        classifier = classifier_records.get(read_name, {})
        classifier_barcode = _value(classifier.get("barcode", classifier.get("BC", "")))
        if (
            not classifier_barcode
            and classifier_expected
            and classifier_source.startswith("sequence")
        ):
            classifier_barcode = "unclassified"
        (
            matched,
            manifest_barcodes,
            manifest_samples,
            manifest_read_groups,
            manifest_namespaces,
            matched_fastq_rows,
        ) = _resolve_manifest(
            _value(bam_record.get("bam_barcode")),
            _value(bam_record.get("read_group")),
            classifier_barcode,
        )
        bam_barcode_evidence = _value(bam_record.get("bam_barcode"))
        bam_read_group_evidence = _value(bam_record.get("read_group"))
        if matched_fastq_rows:
            # FASTQ normalization materializes inferred identities into BC/RG. Do not
            # promote those implementation tags into independent BAM authorities.
            if not manifest_barcodes:
                bam_barcode_evidence = ""
            generated_read_groups = {
                _row_value(row, "pair_id") or _row_value(row, "source_id")
                for row in matched_fastq_rows
                if not _row_value(row, "read_group")
            } - {""}
            if bam_read_group_evidence in generated_read_groups:
                bam_read_group_evidence = ""
        # The directory assignment is gathered unconditionally. It used to be
        # collected only when the classifier was *labelled* `filename`, which
        # tied "is there a directory assignment" to an unrelated label and left
        # the real assignment recoverable only from a conflict string (`F35`).
        directory_barcodes = [
            _legacy_filename_barcode(_row_value(row, "path")) for row in (matched or rows)
        ]
        if len({_barcode_key(value) for value in directory_barcodes if value}) > 1 and not matched:
            directory_barcodes = []
        directory_barcode = next((value for value in directory_barcodes if _value(value)), "")

        classifier_confidence = _confidence(classifier.get("barcode_confidence"), 0.75)
        if directory_authoritative:
            barcode_tiers = (
                ("manifest", 1.0, manifest_barcodes),
                ("demux_directory", 0.97, directory_barcodes),
                ("bam:BC", 0.95, [bam_barcode_evidence]),
                ("bam:RG", 0.9, [bam_read_group_evidence]),
                (classifier_source, classifier_confidence, [classifier_barcode]),
            )
        else:
            barcode_tiers = (
                ("manifest", 1.0, manifest_barcodes),
                ("bam:BC", 0.95, [bam_barcode_evidence]),
                ("bam:RG", 0.9, [bam_read_group_evidence]),
                (classifier_source, classifier_confidence, [classifier_barcode]),
                ("filename", 0.25, directory_barcodes),
            )
        barcode, barcode_source, barcode_confidence, conflicts = _select(
            barcode_tiers,
            field="barcode",
        )
        sample, sample_source, sample_confidence, sample_conflicts = _select(
            (
                ("manifest", 1.0, manifest_samples),
                ("bam:SM", 0.95, [_value(bam_record.get("bam_sample"))]),
                (
                    classifier_source,
                    classifier_confidence,
                    [_value(classifier.get("sample", ""))],
                ),
            ),
            field="sample",
        )
        conflicts.extend(sample_conflicts)
        if read_name in classifier_conflicts:
            conflicts.append(
                {
                    "field": "barcode",
                    "source": classifier_source,
                    "values": "|".join(classifier_conflicts[read_name]),
                }
            )
        if bam_record.get("bam_record_conflict"):
            conflicts.append({"field": "identity", "source": "bam", "values": "duplicate"})
        if sample == "unknown" and _classified(barcode):
            sample = barcode
            sample_source = f"{barcode_source}:barcode"
            sample_confidence = barcode_confidence
        if barcode_source == "filename":
            filename_authority_used = True
        status = (
            "conflicting"
            if conflicts
            else "classified"
            if _classified(barcode)
            else "unclassified"
            if barcode.lower() in _UNCLASSIFIED
            else "unknown"
        )
        read_group = _value(bam_record.get("read_group")) or (
            manifest_read_groups[0] if len(manifest_read_groups) == 1 else ""
        )
        namespace = manifest_namespaces[0] if len(manifest_namespaces) == 1 else ""
        output = {
            "identity_schema_version": BARCODE_IDENTITY_SCHEMA_VERSION,
            "read_name": read_name,
            "barcode": barcode,
            "barcode_source": barcode_source,
            "barcode_confidence": barcode_confidence,
            "sample": sample,
            "sample_source": sample_source,
            "sample_confidence": sample_confidence,
            "read_group": read_group,
            "namespace": namespace,
            "identity_status": status,
            "identity_conflicts": json.dumps(conflicts, sort_keys=True, separators=(",", ":")),
            # `F35`: the independent assignments, preserved side by side.
            "barcode_assigned": directory_barcode,
            "barcode_rederived": classifier_barcode,
            "barcode_front": _evidence(classifier.get("B5")),
            "barcode_rear": _evidence(classifier.get("B6")),
            # Backward-compatible aliases/evidence consumed by split and dense paths.
            "BC": barcode,
            "BM": _value(classifier.get("BM")) or _value(bam_record.get("bam_bm")),
            "bi": classifier.get("bi", bam_record.get("bam_bi")),
        }
        for key in ("B1", "B2", "B3", "B4", "B5", "B6"):
            output[key] = classifier.get(key)
        output_rows.append(output)

    if filename_authority_used:
        warnings.warn(
            "Barcode identity used legacy filename fallback; declare barcode/sample metadata "
            "in the input manifest or BAM read groups.",
            UserWarning,
            stacklevel=2,
        )
    frame = pd.DataFrame(output_rows)
    if frame.empty:
        frame = pd.DataFrame(columns=(*BARCODE_IDENTITY_COLUMNS, "BC", "BM", "bi"))
    frame = frame.sort_values("read_name", kind="stable").reset_index(drop=True)
    output_path = _atomic_write_parquet(frame, Path(output_path))
    report_path = barcode_identity_report_path(output_path)
    statuses = Counter(frame["identity_status"].astype(str))
    for status in ("classified", "unclassified", "unknown", "conflicting"):
        statuses.setdefault(status, 0)
    total = len(frame)
    atomic_write_json(
        report_path,
        {
            "schema_version": BARCODE_IDENTITY_SCHEMA_VERSION,
            "total_reads": total,
            "status_counts": dict(sorted(statuses.items())),
            "status_fractions": {
                key: count / total if total else 0.0 for key, count in sorted(statuses.items())
            },
            "barcode_source_counts": dict(
                sorted(Counter(frame["barcode_source"].astype(str)).items())
            ),
            "sample_source_counts": dict(
                sorted(Counter(frame["sample_source"].astype(str)).items())
            ),
        },
    )
    return output_path, report_path


def read_barcode_identity_sidecar(path: str | Path) -> pd.DataFrame:
    """Read canonical schema 1, upgrading legacy ``read_name``/``BC`` files."""
    frame = pd.read_parquet(path)
    if set(BARCODE_IDENTITY_COLUMNS).issubset(frame.columns):
        versions = set(pd.to_numeric(frame["identity_schema_version"], errors="coerce").dropna())
        if versions and versions != {BARCODE_IDENTITY_SCHEMA_VERSION}:
            raise BarcodeIdentityError(f"unsupported barcode identity schema versions: {versions}")
        return frame
    read_column = "read_name" if "read_name" in frame else "read_id" if "read_id" in frame else None
    if read_column is None or "BC" not in frame:
        raise BarcodeIdentityError(
            "barcode sidecar lacks canonical columns and legacy read_name/BC"
        )
    warnings.warn(
        "Reading a legacy barcode sidecar; rerun raw ingestion to publish canonical identity schema 1.",
        UserWarning,
        stacklevel=2,
    )
    result = frame.copy()
    result["read_name"] = result[read_column].astype(str)
    result["barcode"] = result["BC"].fillna("unclassified").astype(str)
    result["barcode_source"] = "legacy_sidecar"
    result["barcode_confidence"] = 0.5
    result["sample"] = result["barcode"]
    result["sample_source"] = "legacy_sidecar:barcode"
    result["sample_confidence"] = 0.5
    result["read_group"] = ""
    result["namespace"] = ""
    result["identity_status"] = result["barcode"].map(
        lambda value: "classified" if _classified(str(value)) else "unclassified"
    )
    result["identity_conflicts"] = "[]"
    result["identity_schema_version"] = BARCODE_IDENTITY_SCHEMA_VERSION
    return result
