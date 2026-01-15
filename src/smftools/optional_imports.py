"""Utilities for optional dependency handling."""

from __future__ import annotations

from importlib import import_module
from typing import Any


def require(package: str, *, extra: str, purpose: str | None = None) -> Any:
    """Import an optional dependency with a helpful error message.

    Args:
        package: Importable module name (e.g., "torch", "scanpy").
        extra: Extra name users should install (e.g., "ml", "omics").
        purpose: Optional context describing the feature needing the dependency.

    Returns:
        The imported module.

    Raises:
        ModuleNotFoundError: If the package is not installed.
    """
    try:
        return import_module(package)
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on env
        reason = f" for {purpose}" if purpose else ""
        message = (
            f"Optional dependency '{package}' is required{reason}. "
            f"Install it with: pip install 'smftools[{extra}]'"
        )
        raise ModuleNotFoundError(message) from exc