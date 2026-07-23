"""Validate the public dependency-profile contract."""

import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).parents[2]


def _configuration() -> dict:
    """Load project metadata from the repository configuration."""
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def _requirement_names(requirements: list[str]) -> set[str]:
    """Return normalized distribution names from requirement strings."""
    return {Requirement(requirement).name.lower() for requirement in requirements}


def test_default_install_contains_experiment_full_dependencies() -> None:
    """Keep unconditional basecalled-BAM workflow imports in the default install."""
    dependencies = _requirement_names(_configuration()["project"]["dependencies"])

    assert {
        "joblib",
        "matplotlib",
        "pysam",
        "scikit-learn",
        "seaborn",
        "torch",
    } <= dependencies


def test_all_extra_contains_every_feature_profile() -> None:
    """Ensure the aggregate extra covers every documented optional capability."""
    extras = _configuration()["project"]["optional-dependencies"]
    feature_profiles = (
        "ont",
        "umi",
        "genome-io",
        "project",
        "analysis",
        "ml-extended",
        "qc",
    )
    expected = set().union(*(_requirement_names(extras[name]) for name in feature_profiles))

    assert expected <= _requirement_names(extras["all"])


def test_all_2_remains_an_exact_compatibility_alias() -> None:
    """Prevent the legacy aggregate extra from drifting during migration."""
    extras = _configuration()["project"]["optional-dependencies"]

    assert extras["all_2"] == extras["all"]


def test_compatibility_tooling_extras_match_dependency_groups() -> None:
    """Keep legacy contributor extras aligned with their dependency groups."""
    configuration = _configuration()
    extras = configuration["project"]["optional-dependencies"]
    groups = configuration["dependency-groups"]

    expected_dev = _requirement_names(groups["test"] + groups["lint"] + groups["release"])

    assert _requirement_names(extras["dev"]) == expected_dev
    assert {"build", "hatch", "hatch-vcs", "twine"} <= expected_dev
    assert _requirement_names(extras["docs"]) == _requirement_names(groups["docs"])
