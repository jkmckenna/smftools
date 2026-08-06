"""The documented public surface and the API-page exclusions, asserted.

Two claims are pinned here.

``docs/source/api/machine_learning.md`` documents 61 of 66 modules and states
that the five it omits are excluded because they cannot import under the docs
build's mocked dependencies, *and* that all five are deprecated or gated behind
an unbuilt integration. That second half is what makes the omission acceptable.
If a module that is neither ever joins the list, the justification is false.

``machine_learning.__all__`` declares the contract modules a caller actually
needs. It previously omitted ``plan``, ``contracts``, and ``manifests`` while
the quick start told readers to import them.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import smftools.machine_learning as ml

pytestmark = pytest.mark.unit

API_PAGE = Path("docs/source/api/machine_learning.md")
PACKAGE_ROOT = Path("src/smftools/machine_learning")

EXPECTED_EXCLUSIONS = {
    "smftools.machine_learning.data.anndata_data_module",
    "smftools.machine_learning.inference.sliding_window_inference",
    "smftools.machine_learning.models.lightning_base",
    "smftools.machine_learning.training.train_lightning_model",
    "smftools.machine_learning.training.train_sklearn_model",
}

REQUIRED_EXPORTS = {
    "artifacts",
    "contracts",
    "data",
    "evaluation",
    "inference",
    "interpretability",
    "manifests",
    "models",
    "orchestration",
    "plan",
    "selection",
    "splitting",
    "training",
    "utils",
    "workspace",
}


def _documentable_modules() -> set[str]:
    return {
        ".".join(path.relative_to(Path("src")).with_suffix("").parts)
        for path in PACKAGE_ROOT.rglob("*.py")
        if "__pycache__" not in str(path)
        and path.name != "__init__.py"
        and not path.name.startswith("_")
    }


def _documented_modules() -> set[str]:
    text = API_PAGE.read_text(encoding="utf-8")
    return set(re.findall(r"^\s+(smftools\.machine_learning[\w.]*)$", text, re.M))


def test_every_module_is_either_documented_or_a_known_exclusion() -> None:
    # A new module added to the package must be documented or deliberately
    # excluded. Silently missing from both is the failure this catches.
    undocumented = _documentable_modules() - _documented_modules() - EXPECTED_EXCLUSIONS

    assert not undocumented, (
        f"new modules are neither documented nor listed as exclusions: {sorted(undocumented)}. "
        f"Add them to {API_PAGE} or justify the exclusion."
    )


def test_the_exclusion_list_has_not_grown() -> None:
    # The page justifies its five omissions by saying each is deprecated or
    # gated behind an unbuilt integration. A sixth exclusion for some other
    # reason breaks that justification, so it has to be a deliberate edit.
    documented = _documented_modules()
    excluded = _documentable_modules() - documented

    assert excluded == EXPECTED_EXCLUSIONS


def test_the_page_states_why_each_exclusion_is_acceptable() -> None:
    text = API_PAGE.read_text(encoding="utf-8")

    assert "Not listed here" in text
    for module in EXPECTED_EXCLUSIONS:
        leaf = module.rsplit(".", 1)[1]
        assert leaf in text, f"{leaf} is excluded but not explained on the page"
    assert "deprecated" in text.lower()


def test_public_exports_include_the_contract_modules() -> None:
    assert set(ml.__all__) == REQUIRED_EXPORTS


def test_every_declared_export_actually_resolves() -> None:
    # __all__ is lazy, so a typo in the module map would only surface on first
    # access. Force every one.
    for name in ml.__all__:
        assert getattr(ml, name) is not None
