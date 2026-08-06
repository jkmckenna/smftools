"""Security boundaries for the ML package (ML-702).

Two acceptance criteria are asserted adversarially rather than by reading the
implementation:

- untrusted pickle/joblib loading is prohibited or requires an explicit opt-in;
- manifest and path inputs cannot escape the active workspace.

These are regression tests for properties that are cheap to lose. A refactor
that swaps ``resolve()`` for ``absolute()``, or relaxes a run-id check to allow
a nested path, would leave every functional test passing and quietly reopen a
traversal hole.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from smftools.machine_learning.artifacts.common import (
    MLArtifactManifestError,
    SerializationPolicy,
)
from smftools.machine_learning.workspace import MLWorkspaceError, resolve_ml_workspace

pytestmark = pytest.mark.unit


def _workspace(tmp_path: Path):
    owner = tmp_path / "experiment"
    owner.mkdir()
    return resolve_ml_workspace(
        experiment_config=SimpleNamespace(
            output_directory=str(owner), experiment_name="experiment-a"
        )
    )


# --------------------------------------------------------------------------
# Deserialization trust
# --------------------------------------------------------------------------


@pytest.mark.parametrize("unsafe_format", ["pickle", "joblib"])
def test_pickle_and_joblib_policies_require_an_explicit_unsafe_flag(
    unsafe_format: str,
) -> None:
    with pytest.raises(MLArtifactManifestError, match="requires_unsafe_load"):
        SerializationPolicy(
            format=unsafe_format,
            loader=f"{unsafe_format}.load",
            requires_unsafe_load=False,
            allowed_types=(),
            package_versions={},
        )


def test_skops_policies_require_a_reviewed_type_allowlist() -> None:
    # An empty allowlist would mean "trust whatever is in the payload".
    with pytest.raises(MLArtifactManifestError, match="allowed_types"):
        SerializationPolicy(
            format="skops",
            loader="skops.io.load",
            requires_unsafe_load=False,
            allowed_types=(),
            package_versions={},
        )


def test_the_canonical_policies_declare_themselves_safe() -> None:
    torch_policy = SerializationPolicy(
        format="torch-state-dict",
        loader="torch.load",
        requires_unsafe_load=False,
        allowed_types=(),
        package_versions={"torch": "2.13.0"},
    )
    sklearn_policy = SerializationPolicy(
        format="skops",
        loader="skops.io.load",
        requires_unsafe_load=False,
        allowed_types=("sklearn.naive_bayes.BernoulliNB",),
        package_versions={"scikit-learn": "1.9.0"},
    )

    assert not torch_policy.requires_unsafe_load
    assert not sklearn_policy.requires_unsafe_load


# --------------------------------------------------------------------------
# Path containment
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "run_id",
    ["../../outside", "a/../../../outside", "/etc", "..", "nested/run", ""],
)
def test_run_ids_must_be_a_single_safe_path_component(tmp_path: Path, run_id: str) -> None:
    workspace = _workspace(tmp_path)

    with pytest.raises(MLWorkspaceError):
        workspace.run_paths(run_id)


def test_a_legitimate_run_id_still_resolves_inside_the_workspace(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)

    paths = workspace.run_paths("run-1")

    assert paths.root.is_relative_to(workspace.root)


def test_portable_references_reject_paths_outside_the_workspace(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "loot.txt").write_text("x", encoding="utf-8")

    with pytest.raises(MLWorkspaceError, match="escapes"):
        workspace.portable_reference(outside / "loot.txt")

    with pytest.raises(MLWorkspaceError, match="escapes"):
        workspace.portable_reference(Path("/etc/passwd"))


def test_a_symlink_inside_the_workspace_cannot_smuggle_a_path_out(tmp_path: Path) -> None:
    # The containment check resolves before comparing, so a link that lives
    # inside the workspace but points outside is caught. If resolution is ever
    # replaced with a non-following equivalent, this is the test that fails.
    workspace = _workspace(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "loot.txt").write_text("x", encoding="utf-8")
    workspace.root.mkdir(parents=True, exist_ok=True)
    os.symlink(outside, workspace.root / "sneaky")

    with pytest.raises(MLWorkspaceError, match="escapes"):
        workspace.portable_reference(workspace.root / "sneaky" / "loot.txt")


@pytest.mark.parametrize("reference", ["../../../etc/passwd", "/etc/passwd", "a/../../b"])
def test_stored_references_cannot_traverse_out_on_the_way_back_in(
    tmp_path: Path, reference: str
) -> None:
    # Resolution is the reverse direction: a manifest written elsewhere, or
    # edited by hand, must not be able to point a reader outside the workspace.
    workspace = _workspace(tmp_path)

    with pytest.raises(MLWorkspaceError):
        workspace.resolve_reference(reference)


def test_a_contained_reference_round_trips(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    target = workspace.root / "runs" / "run-1" / "result.json"

    reference = workspace.portable_reference(target)

    assert workspace.resolve_reference(reference) == target
