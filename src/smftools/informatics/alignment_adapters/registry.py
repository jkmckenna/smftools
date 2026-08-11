"""Registry for validated alignment adapter selection."""

from __future__ import annotations

from types import MappingProxyType

from .base import AlignmentAdapter, AlignmentAdapterError
from .builtin import DoradoAdapter, Minimap2Adapter
from .short_read import Bowtie2Adapter, BwaMem2Adapter

_ADAPTERS = MappingProxyType(
    {
        "bowtie2": Bowtie2Adapter(),
        "bwa-mem2": BwaMem2Adapter(),
        "dorado": DoradoAdapter(),
        "minimap2": Minimap2Adapter(),
    }
)


def adapter_names() -> tuple[str, ...]:
    """Return registered public adapter names in deterministic order."""
    return tuple(sorted(_ADAPTERS))


def get_alignment_adapter(name: str) -> AlignmentAdapter:
    """Resolve one adapter or fail before external execution."""
    normalized = str(name).strip().lower()
    try:
        return _ADAPTERS[normalized]
    except KeyError as exc:
        raise AlignmentAdapterError(
            f"Unknown alignment adapter {name!r}; choose one of: {', '.join(adapter_names())}."
        ) from exc
