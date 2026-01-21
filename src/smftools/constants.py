from __future__ import annotations

from types import MappingProxyType
from typing import Any, Dict, Final, Mapping


## Helpers ##
def _deep_freeze(obj: Any) -> Any:
    """Recursively freeze common containers. Use for constant exports."""
    if isinstance(obj, dict):
        return MappingProxyType({k: _deep_freeze(v) for k, v in obj.items()})
    if isinstance(obj, (list, tuple)):
        return tuple(_deep_freeze(v) for v in obj)
    if isinstance(obj, set):
        return frozenset(_deep_freeze(v) for v in obj)
    return obj  # ints/strs/tuples (already immutable)


## Constants ##
BAM_SUFFIX: Final[str] = ".bam"
BARCODE_BOTH_ENDS: Final[bool] = False
REF_COL: Final[str] = "Reference_strand"
SAMPLE_COL: Final[str] = "Experiment_name_and_barcode"
SPLIT_DIR: Final[str] = "demultiplexed_BAMs"
TRIM: Final[bool] = False

_private_conversions = ["unconverted"]
CONVERSIONS: Final[list[str]] = _deep_freeze(_private_conversions)

_private_mod_list = ("5mC_5hmC", "6mA")
MOD_LIST: Final[tuple[str, ...]] = _deep_freeze(_private_mod_list)

_private_mod_map: Dict[str, str] = {"6mA": "6mA", "5mC_5hmC": "5mC"}
MOD_MAP: Final[Mapping[str, str]] = _deep_freeze(_private_mod_map)

_private_strands = ("bottom", "top")
STRANDS: Final[tuple[str, ...]] = _deep_freeze(_private_strands)

MODKIT_EXTRACT_TSV_COLUMN_CHROM: Final[str] = "chrom"
MODKIT_EXTRACT_TSV_COLUMN_REF_POSITION: Final[str] = "ref_position"
MODKIT_EXTRACT_TSV_COLUMN_MODIFIED_PRIMARY_BASE: Final[str] = "modified_primary_base"
MODKIT_EXTRACT_TSV_COLUMN_REF_STRAND: Final[str] = "ref_strand"
MODKIT_EXTRACT_TSV_COLUMN_READ_ID: Final[str] = "read_id"
MODKIT_EXTRACT_TSV_COLUMN_CALL_CODE: Final[str] = "call_code"
MODKIT_EXTRACT_TSV_COLUMN_CALL_PROB: Final[str] = "call_prob"

MODKIT_EXTRACT_MODIFIED_BASE_A: Final[str] = "A"
MODKIT_EXTRACT_MODIFIED_BASE_C: Final[str] = "C"
MODKIT_EXTRACT_REF_STRAND_PLUS: Final[str] = "+"
MODKIT_EXTRACT_REF_STRAND_MINUS: Final[str] = "-"

_private_modkit_extract_call_code_modified = ("a", "h", "m")
MODKIT_EXTRACT_CALL_CODE_MODIFIED: Final[tuple[str, ...]] = _deep_freeze(
    _private_modkit_extract_call_code_modified
)
_private_modkit_extract_call_code_canonical = ("-",)
MODKIT_EXTRACT_CALL_CODE_CANONICAL: Final[tuple[str, ...]] = _deep_freeze(
    _private_modkit_extract_call_code_canonical
)

MODKIT_EXTRACT_SEQUENCE_BASES: Final[tuple[str, ...]] = _deep_freeze(
    ("A", "C", "G", "T", "N")
)
MODKIT_EXTRACT_SEQUENCE_PADDING_BASE: Final[str] = "PAD"
_private_modkit_extract_base_to_int: Dict[str, int] = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
    "N": 4,
    "PAD": 5,
}
MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT: Final[Mapping[str, int]] = _deep_freeze(
    _private_modkit_extract_base_to_int
)
_private_modkit_extract_int_to_base: Dict[int, str] = {
    value: key for key, value in _private_modkit_extract_base_to_int.items()
}
MODKIT_EXTRACT_SEQUENCE_INT_TO_BASE: Final[Mapping[int, str]] = _deep_freeze(
    _private_modkit_extract_int_to_base
)
