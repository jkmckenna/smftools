from __future__ import annotations

from importlib import resources
from pathlib import Path

SCHEMA_REGISTRY_VERSION = "1"
SCHEMA_REGISTRY_RESOURCE = "anndata_schema_v1.yaml"


def get_schema_registry_path() -> Path:
    return resources.files(__package__).joinpath(SCHEMA_REGISTRY_RESOURCE)
