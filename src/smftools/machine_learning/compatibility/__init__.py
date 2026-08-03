"""Internal implementations supporting the staged legacy ML API transition."""

from ._warnings import LEGACY_ML_REMOVAL_VERSION, deprecated_ml_alias, warn_legacy_ml_api

__all__ = [
    "LEGACY_ML_REMOVAL_VERSION",
    "deprecated_ml_alias",
    "warn_legacy_ml_api",
]
