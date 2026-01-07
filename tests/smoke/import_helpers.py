"""Helpers for import smoke tests."""

from __future__ import annotations

import importlib

import pytest


def import_module_or_skip(module_name: str) -> None:
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        pytest.skip(f"Optional dependency missing for {module_name}: {exc.name}")
    except ImportError as exc:
        pytest.skip(f"Import error for {module_name}: {exc}")

