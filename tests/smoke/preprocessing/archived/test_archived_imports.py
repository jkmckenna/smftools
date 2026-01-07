"""Import smoke tests for smftools modules."""

from __future__ import annotations

import pytest

from tests.smoke.import_helpers import import_module_or_skip


MODULES = [
    "smftools.preprocessing.archived.add_read_length_and_mapping_qc",
    "smftools.preprocessing.archived.calculate_complexity",
    "smftools.preprocessing.archived.mark_duplicates",
    "smftools.preprocessing.archived.preprocessing",
    "smftools.preprocessing.archived.remove_duplicates",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_imports(module_name: str) -> None:
    import_module_or_skip(module_name)
