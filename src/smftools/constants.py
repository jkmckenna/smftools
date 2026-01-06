from __future__ import annotations
from typing import Final, Mapping, Any, Dict
from types import MappingProxyType


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
SPLIT_DIR: Final[str] = "demultiplexed_BAMs"

_private_mod_map: Dict[str, str] = {"6mA": "6mA", "5mC_5hmC": "5mC"}
MOD_MAP = Final[Mapping[str, str]] = _deep_freeze(_private_mod_map)
