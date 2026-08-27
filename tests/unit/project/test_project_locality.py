"""A registered experiment is not necessarily a readable one (`PSR-18`).

A project references run directories it does not own, so those directories
routinely sit on an external SSD or another machine's disk. Before this, nothing
asked whether they answered: `ProjectCatalog`'s union methods dropped an absent
path and returned the union of what remained, and pooling failed mid-stream from
deep inside a file open rather than up front.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from smftools.project.locality import (
    MISSING,
    OFFLINE,
    REACHABLE,
    UnreachableExperimentsError,
    locality_for_entries,
    require_reachable,
    resolve_experiment_locality,
)

pytestmark = pytest.mark.unit

DETACHED = "/Volumes/ProjectSSDForTests/runs/expA"


def test_reachable_experiment(tmp_path):
    assert resolve_experiment_locality("expA", tmp_path).state == REACHABLE


def test_experiment_on_a_detached_volume_is_offline():
    locality = resolve_experiment_locality("expA", DETACHED)
    assert locality.state == OFFLINE
    assert locality.volume == Path("/Volumes/ProjectSSDForTests")
    assert "detached volume" in locality.describe()


def test_deleted_experiment_is_missing_not_offline(tmp_path):
    """The two must read differently, as they do for raw input."""
    locality = resolve_experiment_locality("expA", tmp_path / "gone")
    assert locality.state == MISSING
    assert locality.volume is None
    assert "path missing" in locality.describe()


def test_require_reachable_refuses_by_default(tmp_path):
    entries = [
        {"id": "expA", "path": str(tmp_path)},
        {"id": "expB", "path": DETACHED},
    ]
    with pytest.raises(UnreachableExperimentsError) as excinfo:
        require_reachable(entries, operation="project selection")
    message = str(excinfo.value)
    assert "expB" in message
    assert "ProjectSSDForTests" in message
    # The remedy has to be actionable, not just a diagnosis.
    assert "Attach" in message
    assert "allow_unreachable" in message


def test_require_reachable_reports_what_it_skipped_when_allowed(tmp_path):
    """Proceeding is permitted; proceeding *silently* is the defect."""
    entries = [
        {"id": "expA", "path": str(tmp_path)},
        {"id": "expB", "path": DETACHED},
    ]
    skipped = require_reachable(entries, operation="project selection", allow_unreachable=True)
    assert [item.experiment for item in skipped] == ["expB"]


def test_require_reachable_is_silent_when_everything_answers(tmp_path):
    assert require_reachable([{"id": "expA", "path": str(tmp_path)}], operation="x") == []


def test_locality_for_entries_keys_by_id(tmp_path):
    entries = [{"id": "expA", "path": str(tmp_path)}, {"id": "expB", "path": DETACHED}]
    localities = locality_for_entries(entries)
    assert set(localities) == {"expA", "expB"}
    assert localities["expA"].is_reachable
    assert not localities["expB"].is_reachable


def test_missing_experiments_are_named_individually(tmp_path):
    """A count alone does not tell you which drive to plug in."""
    entries = [
        {"id": "expA", "path": "/Volumes/DriveOne/runs/a"},
        {"id": "expB", "path": "/Volumes/DriveTwo/runs/b"},
    ]
    with pytest.raises(UnreachableExperimentsError) as excinfo:
        require_reachable(entries, operation="project materialize")
    message = str(excinfo.value)
    assert "DriveOne" in message and "DriveTwo" in message
    assert "expA" in message and "expB" in message
