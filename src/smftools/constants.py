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
