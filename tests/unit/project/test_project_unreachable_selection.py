"""A selection covering unreachable experiments must refuse, not quietly shrink.

`PSR-18`. Built on the same two-experiment fixture shape the catalog tests use,
then one experiment's directory is renamed to a detached-volume path so it is
registered but unreadable — the state produced by unplugging the SSD the runs
live on.
"""

from __future__ import annotations

import json

import pytest

from smftools.project import registry as reg
from smftools.project.catalog import ProjectCatalog, resolve_set_members
from smftools.project.locality import UnreachableExperimentsError

pytestmark = pytest.mark.unit

pytest.importorskip("anndata")

from tests.unit.test_project_catalog import (  # noqa: E402
    SEQUENCE,
    _make_raw_experiment,
    reference_uid,
)

DETACHED_ROOT = "/Volumes/ProjectSSDForTests/runs/expB"


@pytest.fixture
def project_with_one_detached(tmp_path):
    """Two registered experiments, one of which now points at a detached volume."""
    uid = reference_uid(SEQUENCE, 12)
    _make_raw_experiment(tmp_path / "expA", reference_strand="geneA_top", uid=uid, n=4)
    _make_raw_experiment(tmp_path / "expB", reference_strand="geneB_top", uid=uid, n=3)
    proj = tmp_path / "project"
    reg.init_project(proj)
    reg.add_experiment(proj, tmp_path / "expA")
    reg.add_experiment(proj, tmp_path / "expB")

    # Repoint every path expB recorded -- run root, spines and catalogs -- at a
    # volume that is not attached, which is what unplugging the drive does.
    # Rewriting only the run root left the catalogs pointing at files that still
    # existed, so the union methods read them happily and the fixture proved
    # nothing.
    registry_path = proj / "registry.json"
    payload = json.loads(registry_path.read_text())
    entry = json.dumps(payload["experiments"]["expB"])
    # Spine and catalog pointers are stored *relative to the project dir*
    # (``../expB/...``), so rewriting only the absolute run root left them
    # resolving to files that still existed and the fixture proved nothing.
    entry = entry.replace(str(tmp_path / "expB"), DETACHED_ROOT)
    entry = entry.replace("../expB", DETACHED_ROOT)
    payload["experiments"]["expB"] = json.loads(entry)
    payload["experiments"]["expB"]["path"] = DETACHED_ROOT
    registry_path.write_text(json.dumps(payload))
    return proj, uid


def test_selection_refuses_when_an_experiment_is_unreachable(project_with_one_detached):
    proj, uid = project_with_one_detached
    catalog = ProjectCatalog.open(proj)
    with pytest.raises(UnreachableExperimentsError) as excinfo:
        resolve_set_members(catalog, uid)
    message = str(excinfo.value)
    assert "expB" in message
    assert "ProjectSSDForTests" in message


def test_selection_proceeds_when_explicitly_allowed(project_with_one_detached):
    """Partial is permitted when asked for; the members returned are the reachable ones."""
    proj, uid = project_with_one_detached
    catalog = ProjectCatalog.open(proj)
    members = resolve_set_members(catalog, uid, allow_unreachable=True)
    assert [member["experiment"] for member in members] == ["expA"]


def test_fully_reachable_project_is_unaffected(tmp_path):
    """The guard must not change behaviour when every experiment answers."""
    uid = reference_uid(SEQUENCE, 12)
    _make_raw_experiment(tmp_path / "expA", reference_strand="geneA_top", uid=uid, n=4)
    _make_raw_experiment(tmp_path / "expB", reference_strand="geneB_top", uid=uid, n=3)
    proj = tmp_path / "project"
    reg.init_project(proj)
    reg.add_experiment(proj, tmp_path / "expA")
    reg.add_experiment(proj, tmp_path / "expB")
    members = resolve_set_members(ProjectCatalog.open(proj), uid)
    assert sorted(member["experiment"] for member in members) == ["expA", "expB"]


def test_union_catalogs_report_the_experiments_they_omit(project_with_one_detached, caplog):
    """These used to drop an unreachable experiment with no signal at all."""
    import logging

    proj, _ = project_with_one_detached
    catalog = ProjectCatalog.open(proj)
    with caplog.at_level(logging.WARNING, logger="smftools.project.catalog"):
        catalog.interval_catalog()
    assert any("unreachable" in record.message for record in caplog.records)
    assert any("expB" in record.message for record in caplog.records)


def test_project_list_labels_locality(project_with_one_detached):
    from smftools.cli.project_cmd import project_list

    proj, _ = project_with_one_detached
    experiments, _refs = project_list(proj)
    by_id = {entry["id"]: entry for entry in experiments}
    assert by_id["expA"]["locality"] == "present"
    assert by_id["expB"]["locality"] == "offline"
    assert by_id["expB"]["locality_volume"] == "/Volumes/ProjectSSDForTests"
