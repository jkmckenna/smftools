"""Structured alignment adapter registry."""

from .base import (
    AlignmentAdapterError,
    AlignmentCapabilities,
    AlignmentEnvironment,
    AlignmentExecutionResult,
    AlignmentRequest,
)
from .registry import adapter_names, get_alignment_adapter

__all__ = [
    "AlignmentAdapterError",
    "AlignmentCapabilities",
    "AlignmentEnvironment",
    "AlignmentExecutionResult",
    "AlignmentRequest",
    "adapter_names",
    "get_alignment_adapter",
]
