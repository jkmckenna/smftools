"""Smoke tests for framework-independent ML orchestration imports."""

import importlib

import pytest

pytestmark = pytest.mark.smoke


@pytest.mark.parametrize(
    "module_name",
    [
        "smftools.machine_learning.orchestration",
        "smftools.machine_learning.orchestration.actions",
        "smftools.machine_learning.orchestration.contracts",
        "smftools.machine_learning.orchestration.resolution",
        "smftools.machine_learning.orchestration.service",
    ],
)
def test_orchestration_modules_import(module_name: str) -> None:
    importlib.import_module(module_name)
