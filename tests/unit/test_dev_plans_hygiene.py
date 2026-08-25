"""Design records must not leak machine-local paths or unpublished datasets.

`dev/plans/` is tracked and public. Every rule in this repo that relied on
discipline rather than CI has failed at least once, and four absolute paths
survived in a plan for months before anyone noticed.

Every pattern here is **structural**. An earlier version of this file listed the
operator's name and lab directory as regex literals, and used a real sequencing
run as the worked example -- publishing, in the guard itself, exactly what the
guard exists to keep out. Site-specific terms belong in `dev/.private-terms`,
which is untracked; this file must stay readable by anyone without telling them
anything.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
PLANS = REPO / "dev" / "plans"

#: A machine-local path. `<repo>` and `<outputs>` placeholders are the substitute.
#: This also covers operator identity, which travels inside home directories.
ABSOLUTE_PATH = re.compile(r"(?:/Users/|/home/|[A-Za-z]:\\\\)")

#: A sequencing-run identifier: a six-digit date, an underscore, a descriptive
#: name -- `<YYMMDD>_<description>`. Runs are unpublished experiments, and a
#: design document needs a run's scale and modality, never its identity.
RUN_IDENTIFIER = re.compile(r"\b\d{6}_[A-Za-z][A-Za-z0-9_]{3,}")

#: Optional newline-separated terms specific to this checkout -- a lab folder
#: name, a project codename. Lives outside the tracked tree precisely so that
#: naming them does not publish them.
PRIVATE_TERMS_FILE = REPO / "dev" / ".private-terms"


def _tracked_plans() -> list[Path]:
    """Every design record git actually publishes.

    `logs/` is gitignored -- it is where measurements from unpublished
    experiments land first -- so it is deliberately not checked here.
    """
    if not PLANS.is_dir():
        return []
    return sorted(p for p in PLANS.rglob("*.md") if "logs" not in p.relative_to(PLANS).parts)


def _private_terms() -> list[str]:
    if not PRIVATE_TERMS_FILE.is_file():
        return []
    return [
        line.strip()
        for line in PRIVATE_TERMS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def test_there_are_plans_to_check():
    """Guard the guard: a silent zero-file pass would defeat the point."""
    if not PLANS.is_dir():
        pytest.skip("dev/plans is absent in this checkout")
    assert _tracked_plans(), "dev/plans exists but no tracked documents were found"


@pytest.mark.parametrize("path", _tracked_plans(), ids=lambda p: p.name)
def test_no_absolute_paths(path):
    """Repo-relative paths or `<repo>`/`<outputs>` placeholders only."""
    offenders = [
        line for line in path.read_text(encoding="utf-8").splitlines() if ABSOLUTE_PATH.search(line)
    ]
    assert not offenders, f"{path.name} contains machine-local paths: {offenders[:2]}"


@pytest.mark.parametrize("path", _tracked_plans(), ids=lambda p: p.name)
def test_no_sequencing_run_identifiers(path):
    """Cite a run's scale and modality, never its name.

    "a 1.3M-read deaminase run" carries everything a design document needs. The
    run name is unpublished research data and belongs in the analyses
    repository, or in `logs/`, which is not tracked.
    """
    offenders = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if RUN_IDENTIFIER.search(line)
    ]
    assert not offenders, f"{path.name} names a sequencing run: {offenders[:2]}"


@pytest.mark.parametrize("path", _tracked_plans(), ids=lambda p: p.name)
def test_no_private_terms(path):
    """Check this checkout's own sensitive terms, without naming them here."""
    terms = _private_terms()
    if not terms:
        pytest.skip("no dev/.private-terms in this checkout")
    text = path.read_text(encoding="utf-8").lower()
    hits = [term for term in terms if term.lower() in text]
    assert not hits, f"{path.name} contains {len(hits)} private term(s) from dev/.private-terms"


def test_logs_are_excluded_from_the_check():
    """The exclusion is intentional and must stay visible, not be assumed."""
    if not (PLANS / "logs").is_dir():
        pytest.skip("no logs directory in this checkout")
    assert all("logs" not in p.relative_to(PLANS).parts for p in _tracked_plans())
