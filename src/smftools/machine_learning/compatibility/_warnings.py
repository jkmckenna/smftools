"""Shared warnings for legacy machine-learning compatibility adapters."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

LEGACY_ML_REMOVAL_VERSION = "3.0.0"

_P = ParamSpec("_P")
_R = TypeVar("_R")


def warn_legacy_ml_api(symbol: str, replacement: str, *, stacklevel: int = 2) -> None:
    """Warn that a legacy ML symbol has a canonical replacement.

    Args:
        symbol: Fully qualified legacy symbol.
        replacement: Canonical API or migration action.
        stacklevel: Warning stack level relative to this helper.
    """
    warnings.warn(
        f"{symbol} is deprecated and will be removed in smftools "
        f"{LEGACY_ML_REMOVAL_VERSION}; use {replacement}.",
        FutureWarning,
        stacklevel=stacklevel,
    )


def deprecated_ml_alias(
    symbol: str,
    replacement: str,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Decorate a compatibility callable with a standardized warning."""

    def decorate(function: Callable[_P, _R]) -> Callable[_P, _R]:
        @wraps(function)
        def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            warn_legacy_ml_api(symbol, replacement, stacklevel=3)
            return function(*args, **kwargs)

        return wrapped

    return decorate
