"""Versioned request and safe selection-predicate contracts for re-basecalling."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

REBASECALL_REQUEST_SCHEMA_VERSION = 1

SELECTION_MODES = (
    "all-signal",
    "all-parent-molecules",
    "qc",
    "ids",
)
# D2 in the generation-lifecycle plan: the selection determines which reads a
# basecall contains, so the kind is a property of the selection mode and is
# stamped on the basecall generation. A descendant raw generation derives it.
SELECTION_GENERATION_KINDS = {
    "all-signal": "full_source",
    "all-parent-molecules": "parent_universe",
    "qc": "selected_cohort",
    "ids": "selected_cohort",
}
ID_KINDS = ("molecule_uid", "read_id", "pod5_read_id")
READ_SPLITTING_POLICIES = ("preserve", "disable")
TRIM_POLICIES = ("none", "all")
DOWNSTREAM_TARGETS = ("raw", "preprocess", "spatial", "hmm", "latent", "full")

QC_PREDICATE_COLUMNS = frozenset(
    {
        "passes_read_qc",
        "passes_modification_qc",
        "passes_nonvariant_qc",
        "passes_variant_qc",
        "passes_qc",
        "is_duplicate",
        "passes_dedup",
    }
)
PREDICATE_OPERATORS = frozenset(
    {"eq", "ne", "lt", "le", "gt", "ge", "in", "not_in", "is_null", "not_null"}
)
MISSING_POLICIES = frozenset({"fail", "false", "true"})

_MAX_PREDICATE_DEPTH = 16
_MAX_PREDICATE_NODES = 256
_MISSING = object()


class RebasecallRequestError(ValueError):
    """Raised when a re-basecall request is malformed or unsafe."""


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RebasecallRequestError(f"{field_name} must be an object")
    return value


def _reject_unknown(value: Mapping[str, Any], allowed: set[str], field_name: str) -> None:
    unknown = sorted(map(str, set(value).difference(allowed)))
    if unknown:
        raise RebasecallRequestError(
            f"{field_name} contains unknown field(s): {', '.join(unknown)}"
        )


def _string(value: Any, field_name: str, *, default: str | None = None) -> str:
    if value is None and default is not None:
        value = default
    if not isinstance(value, str):
        raise RebasecallRequestError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise RebasecallRequestError(f"{field_name} must not be empty")
    return normalized


def _boolean(value: Any, field_name: str, *, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise RebasecallRequestError(f"{field_name} must be true or false")
    return value


def _number(value: Any, field_name: str, *, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RebasecallRequestError(f"{field_name} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise RebasecallRequestError(f"{field_name} must be finite and nonnegative")
    return normalized


def _json_value(value: Any, field_name: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RebasecallRequestError(f"{field_name} must be finite")
        return value
    if isinstance(value, (list, tuple)):
        return tuple(_json_value(item, field_name) for item in value)
    raise RebasecallRequestError(f"{field_name} must be a JSON scalar or list")


@dataclass(frozen=True)
class SelectionPredicate:
    """One bounded structured predicate over canonical QC-mask columns."""

    kind: str
    column: str | None = None
    operator: str | None = None
    value: Any = None
    missing: str = "fail"
    children: tuple["SelectionPredicate", ...] = ()

    @property
    def columns(self) -> tuple[str, ...]:
        if self.kind == "leaf":
            assert self.column is not None
            return (self.column,)
        return tuple(sorted({column for child in self.children for column in child.columns}))

    def to_dict(self) -> dict[str, Any]:
        if self.kind == "leaf":
            payload: dict[str, Any] = {
                "column": self.column,
                "op": self.operator,
                "missing": self.missing,
            }
            if self.operator not in {"is_null", "not_null"}:
                payload["value"] = list(self.value) if isinstance(self.value, tuple) else self.value
            return payload
        if self.kind == "not":
            return {"not": self.children[0].to_dict()}
        return {self.kind: [child.to_dict() for child in self.children]}

    def evaluate(self, frame: pd.DataFrame) -> pd.Series:
        """Evaluate without executing user-provided code or query text."""
        if self.kind == "all":
            result = pd.Series(True, index=frame.index, dtype=bool)
            for child in self.children:
                result &= child.evaluate(frame)
            return result
        if self.kind == "any":
            result = pd.Series(False, index=frame.index, dtype=bool)
            for child in self.children:
                result |= child.evaluate(frame)
            return result
        if self.kind == "not":
            return ~self.children[0].evaluate(frame)

        assert self.column is not None
        assert self.operator is not None
        if self.column not in frame.columns:
            if self.missing == "fail":
                raise RebasecallRequestError(
                    f"selection predicate column {self.column!r} is unavailable"
                )
            return pd.Series(self.missing == "true", index=frame.index, dtype=bool)

        series = frame[self.column]
        nulls = series.isna()
        if self.operator == "is_null":
            return nulls.astype(bool)
        if self.operator == "not_null":
            return (~nulls).astype(bool)
        if nulls.any() and self.missing == "fail":
            raise RebasecallRequestError(
                f"selection predicate column {self.column!r} contains missing values"
            )
        try:
            if self.operator == "eq":
                result = series.eq(self.value)
            elif self.operator == "ne":
                result = series.ne(self.value)
            elif self.operator == "lt":
                result = series.lt(self.value)
            elif self.operator == "le":
                result = series.le(self.value)
            elif self.operator == "gt":
                result = series.gt(self.value)
            elif self.operator == "ge":
                result = series.ge(self.value)
            elif self.operator == "in":
                result = series.isin(self.value)
            else:
                result = ~series.isin(self.value)
        except (TypeError, ValueError) as exc:
            raise RebasecallRequestError(
                f"selection predicate cannot compare column {self.column!r} with its value"
            ) from exc
        if nulls.any():
            result = result.mask(nulls, self.missing == "true")
        return result.fillna(False).astype(bool)


def parse_selection_predicate(value: Any) -> SelectionPredicate:
    """Parse one safe predicate with bounded nesting and node count."""
    node_count = 0

    def parse(raw: Any, depth: int) -> SelectionPredicate:
        nonlocal node_count
        node_count += 1
        if node_count > _MAX_PREDICATE_NODES:
            raise RebasecallRequestError("selection predicate contains too many nodes")
        if depth > _MAX_PREDICATE_DEPTH:
            raise RebasecallRequestError("selection predicate nesting is too deep")
        payload = _mapping(raw, "selection.predicate")
        logical = set(payload).intersection({"all", "any", "not"})
        if logical:
            if len(logical) != 1 or len(payload) != 1:
                raise RebasecallRequestError(
                    "logical predicates must contain exactly one of 'all', 'any', or 'not'"
                )
            kind = next(iter(logical))
            child_payload = payload[kind]
            if kind == "not":
                return SelectionPredicate(kind="not", children=(parse(child_payload, depth + 1),))
            if not isinstance(child_payload, (list, tuple)) or not child_payload:
                raise RebasecallRequestError(f"selection.predicate.{kind} must be a nonempty list")
            return SelectionPredicate(
                kind=kind,
                children=tuple(parse(child, depth + 1) for child in child_payload),
            )

        _reject_unknown(payload, {"column", "op", "value", "missing"}, "selection.predicate")
        column = _string(payload.get("column"), "selection.predicate.column")
        if column not in QC_PREDICATE_COLUMNS:
            raise RebasecallRequestError(
                f"selection predicate column {column!r} is not allowlisted"
            )
        operator = _string(payload.get("op"), "selection.predicate.op")
        if operator not in PREDICATE_OPERATORS:
            raise RebasecallRequestError(f"unsupported selection predicate operator {operator!r}")
        missing = _string(payload.get("missing"), "selection.predicate.missing", default="fail")
        if missing not in MISSING_POLICIES:
            raise RebasecallRequestError(f"unsupported missing-value policy {missing!r}")
        raw_value = payload.get("value", _MISSING)
        if operator in {"is_null", "not_null"}:
            if raw_value is not _MISSING:
                raise RebasecallRequestError(f"operator {operator!r} does not accept a value")
            normalized_value = None
        else:
            if raw_value is _MISSING:
                raise RebasecallRequestError(f"operator {operator!r} requires a value")
            normalized_value = _json_value(raw_value, "selection.predicate.value")
            if operator in {"in", "not_in"} and (
                not isinstance(normalized_value, tuple) or not normalized_value
            ):
                raise RebasecallRequestError(f"operator {operator!r} requires a nonempty list")
            if operator in {"in", "not_in"} and any(
                isinstance(item, tuple) for item in normalized_value
            ):
                raise RebasecallRequestError(
                    f"operator {operator!r} requires a list of scalar values"
                )
            if operator not in {"in", "not_in"} and isinstance(normalized_value, tuple):
                raise RebasecallRequestError(f"operator {operator!r} requires a scalar value")
        return SelectionPredicate(
            kind="leaf",
            column=column,
            operator=operator,
            value=normalized_value,
            missing=missing,
        )

    return parse(value, 1)


@dataclass(frozen=True)
class RebasecallSource:
    raw_generation: str = "current"
    preprocess_generation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_generation": self.raw_generation,
            "preprocess_generation": self.preprocess_generation,
        }


@dataclass(frozen=True)
class RebasecallSelection:
    mode: str
    predicate: SelectionPredicate | None = None
    id_kind: str | None = None
    ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"mode": self.mode}
        if self.predicate is not None:
            payload["predicate"] = self.predicate.to_dict()
        if self.id_kind is not None:
            payload["id_kind"] = self.id_kind
            payload["ids"] = list(self.ids)
        return payload


@dataclass(frozen=True)
class RebasecallBasecall:
    model: str
    modified_bases: tuple[str, ...] = ()
    read_splitting: str = "preserve"
    trim: str = "none"
    emit_moves: bool = True
    min_qscore: float = 0.0
    barcode_kit: str | None = None
    barcode_both_ends: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "modified_bases": list(self.modified_bases),
            "read_splitting": self.read_splitting,
            "trim": self.trim,
            "emit_moves": self.emit_moves,
            "min_qscore": self.min_qscore,
            "barcode_kit": self.barcode_kit,
            "barcode_both_ends": self.barcode_both_ends,
        }


@dataclass(frozen=True)
class SourceRelocation:
    path: str
    source_id: str | None = None
    sha256: str | None = None

    def to_dict(self, *, include_path: bool = True) -> dict[str, Any]:
        payload = {"source_id": self.source_id, "sha256": self.sha256}
        if include_path:
            payload["path"] = self.path
        return payload


@dataclass(frozen=True)
class RebasecallSignal:
    materialize: bool = False
    relocations: tuple[SourceRelocation, ...] = ()

    def to_dict(self, *, include_paths: bool = True) -> dict[str, Any]:
        return {
            "materialize": self.materialize,
            "relocations": [
                relocation.to_dict(include_path=include_paths) for relocation in self.relocations
            ],
        }


@dataclass(frozen=True)
class RebasecallRequest:
    name: str
    source: RebasecallSource
    selection: RebasecallSelection
    basecall: RebasecallBasecall
    signal: RebasecallSignal
    downstream_target: str
    schema_version: int = REBASECALL_REQUEST_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "source": self.source.to_dict(),
            "selection": self.selection.to_dict(),
            "basecall": self.basecall.to_dict(),
            "signal": self.signal.to_dict(),
            "downstream": {"target": self.downstream_target},
            "promotion": {"activate": False},
        }

    def semantic_payload(self) -> dict[str, Any]:
        """Return relocation-independent fields that define the requested computation."""
        return {
            "schema_version": self.schema_version,
            "source": self.source.to_dict(),
            "selection": self.selection.to_dict(),
            "basecall": self.basecall.to_dict(),
            "signal": self.signal.to_dict(include_paths=False),
            "downstream": {"target": self.downstream_target},
        }

    @property
    def request_id(self) -> str:
        encoded = json.dumps(self.semantic_payload(), sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(encoded).hexdigest()

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), indent=indent)


def rebasecall_request_from_dict(
    value: Mapping[str, Any],
    *,
    base_directory: str | Path | None = None,
) -> RebasecallRequest:
    """Validate and normalize one request mapping."""
    payload = _mapping(value, "request")
    _reject_unknown(
        payload,
        {
            "schema_version",
            "name",
            "source",
            "selection",
            "basecall",
            "signal",
            "downstream",
            "promotion",
        },
        "request",
    )
    schema_version = payload.get("schema_version", -1)
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise RebasecallRequestError("request.schema_version must be an integer")
    if schema_version != REBASECALL_REQUEST_SCHEMA_VERSION:
        raise RebasecallRequestError(
            f"unsupported request.schema_version={schema_version!r}; expected 1"
        )
    name = _string(payload.get("name"), "request.name")

    source_payload = _mapping(payload.get("source", {}), "request.source")
    _reject_unknown(source_payload, {"raw_generation", "preprocess_generation"}, "request.source")
    raw_generation = _string(
        source_payload.get("raw_generation"), "request.source.raw_generation", default="current"
    )
    preprocess_raw = source_payload.get("preprocess_generation")

    selection_payload = _mapping(payload.get("selection"), "request.selection")
    _reject_unknown(selection_payload, {"mode", "predicate", "id_kind", "ids"}, "request.selection")
    mode = _string(selection_payload.get("mode"), "request.selection.mode")
    if mode not in SELECTION_MODES:
        raise RebasecallRequestError(f"unsupported selection mode {mode!r}")
    predicate = None
    id_kind = None
    ids: tuple[str, ...] = ()
    if mode == "qc":
        if "predicate" not in selection_payload:
            raise RebasecallRequestError("qc selection requires selection.predicate")
        predicate = parse_selection_predicate(selection_payload["predicate"])
        if "id_kind" in selection_payload or "ids" in selection_payload:
            raise RebasecallRequestError("qc selection does not accept id_kind or ids")
    elif mode == "ids":
        if "predicate" in selection_payload:
            raise RebasecallRequestError("ids selection does not accept a predicate")
        id_kind = _string(selection_payload.get("id_kind"), "request.selection.id_kind")
        if id_kind not in ID_KINDS:
            raise RebasecallRequestError(f"unsupported selection id_kind {id_kind!r}")
        raw_ids = selection_payload.get("ids")
        if not isinstance(raw_ids, (list, tuple)) or not raw_ids:
            raise RebasecallRequestError("ids selection requires a nonempty ids list")
        ids = tuple(sorted(_string(item, "request.selection.ids") for item in raw_ids))
        if len(set(ids)) != len(ids):
            raise RebasecallRequestError("request.selection.ids contains duplicates")
    elif set(selection_payload).difference({"mode"}):
        raise RebasecallRequestError(f"{mode} selection accepts only the mode field")
    preprocess_generation = (
        _string(preprocess_raw, "request.source.preprocess_generation")
        if preprocess_raw is not None
        else ("current" if mode == "qc" else None)
    )

    basecall_payload = _mapping(payload.get("basecall"), "request.basecall")
    _reject_unknown(
        basecall_payload,
        {
            "model",
            "modified_bases",
            "read_splitting",
            "trim",
            "emit_moves",
            "min_qscore",
            "barcode_kit",
            "barcode_both_ends",
        },
        "request.basecall",
    )
    model = _string(basecall_payload.get("model"), "request.basecall.model")
    raw_modifications = basecall_payload.get("modified_bases", ())
    if not isinstance(raw_modifications, (list, tuple)):
        raise RebasecallRequestError("request.basecall.modified_bases must be a list")
    modified_bases = tuple(
        _string(item, "request.basecall.modified_bases") for item in raw_modifications
    )
    if len(set(modified_bases)) != len(modified_bases):
        raise RebasecallRequestError("request.basecall.modified_bases contains duplicates")
    read_splitting = _string(
        basecall_payload.get("read_splitting"),
        "request.basecall.read_splitting",
        default="preserve",
    )
    if read_splitting not in READ_SPLITTING_POLICIES:
        raise RebasecallRequestError(f"unsupported read-splitting policy {read_splitting!r}")
    trim = _string(basecall_payload.get("trim"), "request.basecall.trim", default="none")
    if trim not in TRIM_POLICIES:
        raise RebasecallRequestError(f"unsupported trim policy {trim!r}")
    barcode_raw = basecall_payload.get("barcode_kit")
    barcode_kit = (
        None if barcode_raw is None else _string(barcode_raw, "request.basecall.barcode_kit")
    )

    signal_payload = _mapping(payload.get("signal", {}), "request.signal")
    _reject_unknown(signal_payload, {"materialize", "relocations"}, "request.signal")
    raw_relocations = signal_payload.get("relocations", ())
    if not isinstance(raw_relocations, (list, tuple)):
        raise RebasecallRequestError("request.signal.relocations must be a list")
    base = Path(base_directory or ".")
    relocations: list[SourceRelocation] = []
    for index, item in enumerate(raw_relocations):
        relocation = _mapping(item, f"request.signal.relocations[{index}]")
        _reject_unknown(
            relocation, {"path", "source_id", "sha256"}, f"request.signal.relocations[{index}]"
        )
        source_id_raw = relocation.get("source_id")
        sha_raw = relocation.get("sha256")
        source_id = None if source_id_raw is None else _string(source_id_raw, "source_id")
        sha256 = None if sha_raw is None else _string(sha_raw, "sha256").lower()
        if source_id is None and sha256 is None:
            raise RebasecallRequestError(
                "each source relocation requires source_id, sha256, or both"
            )
        if sha256 is not None and (
            len(sha256) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in sha256)
        ):
            raise RebasecallRequestError(
                "source relocation sha256 must be 64 hexadecimal characters"
            )
        path = Path(_string(relocation.get("path"), "source relocation path")).expanduser()
        if not path.is_absolute():
            path = base / path
        relocations.append(
            SourceRelocation(
                path=str(path.resolve(strict=False)), source_id=source_id, sha256=sha256
            )
        )
    relocation_keys = [(item.source_id, item.sha256) for item in relocations]
    if len(set(relocation_keys)) != len(relocation_keys):
        raise RebasecallRequestError("request.signal.relocations contains duplicate identities")

    downstream_payload = _mapping(payload.get("downstream", {}), "request.downstream")
    _reject_unknown(downstream_payload, {"target"}, "request.downstream")
    downstream_target = _string(
        downstream_payload.get("target"), "request.downstream.target", default="full"
    )
    if downstream_target not in DOWNSTREAM_TARGETS:
        raise RebasecallRequestError(f"unsupported downstream target {downstream_target!r}")

    promotion_payload = _mapping(payload.get("promotion", {}), "request.promotion")
    _reject_unknown(promotion_payload, {"activate"}, "request.promotion")
    if _boolean(promotion_payload.get("activate"), "request.promotion.activate", default=False):
        raise RebasecallRequestError(
            "request.promotion.activate must be false; promotion is a separate explicit operation"
        )

    return RebasecallRequest(
        name=name,
        source=RebasecallSource(
            raw_generation=raw_generation,
            preprocess_generation=preprocess_generation,
        ),
        selection=RebasecallSelection(
            mode=mode,
            predicate=predicate,
            id_kind=id_kind,
            ids=ids,
        ),
        basecall=RebasecallBasecall(
            model=model,
            modified_bases=modified_bases,
            read_splitting=read_splitting,
            trim=trim,
            emit_moves=_boolean(
                basecall_payload.get("emit_moves"), "request.basecall.emit_moves", default=True
            ),
            min_qscore=_number(
                basecall_payload.get("min_qscore"), "request.basecall.min_qscore", default=0.0
            ),
            barcode_kit=barcode_kit,
            barcode_both_ends=_boolean(
                basecall_payload.get("barcode_both_ends"),
                "request.basecall.barcode_both_ends",
                default=False,
            ),
        ),
        signal=RebasecallSignal(
            materialize=_boolean(
                signal_payload.get("materialize"), "request.signal.materialize", default=False
            ),
            relocations=tuple(
                sorted(
                    relocations,
                    key=lambda item: (item.source_id or "", item.sha256 or ""),
                )
            ),
        ),
        downstream_target=downstream_target,
    )


def load_rebasecall_request(path: str | Path) -> RebasecallRequest:
    """Load a strict schema-1 JSON or YAML request."""
    request_path = Path(path)
    try:
        text = request_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RebasecallRequestError(f"could not read re-basecall request: {request_path}") from exc
    suffix = request_path.suffix.lower()
    if suffix == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RebasecallRequestError(
                f"re-basecall request is unreadable: {request_path}"
            ) from exc
    elif suffix in {".yaml", ".yml"}:
        import yaml

        try:
            payload = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise RebasecallRequestError(
                f"re-basecall request is unreadable: {request_path}"
            ) from exc
    else:
        raise RebasecallRequestError("re-basecall requests must use .json, .yaml, or .yml")
    return rebasecall_request_from_dict(
        _mapping(payload, "request"), base_directory=request_path.parent
    )
