"""Directory checksums ignore operating-system metadata (`F10`).

A published generation is validated by re-hashing its artifact directories.
Opening one of those directories in macOS Finder writes a `.DS_Store` inside it,
which changed the digest and made a validated, immutable artifact report as
"missing or corrupt" for a reason having nothing to do with its contents. This
happened on the `241213` pilot and blocked a re-run.

The digest must describe the artifact, not which platform last browsed it.
"""

from __future__ import annotations

import pytest

from smftools.cli.workflow_contract import _sha256 as workflow_sha256
from smftools.constants import is_os_metadata
from smftools.informatics.experiment_manifest import _sha256 as manifest_sha256
from smftools.informatics.raw_intermediate_manifest import artifact_checksum

pytestmark = pytest.mark.unit

_HASHERS = (
    pytest.param(manifest_sha256, id="generation_manifest"),
    pytest.param(artifact_checksum, id="raw_intermediate"),
    pytest.param(workflow_sha256, id="workflow_contract"),
)


def _artifact(tmp_path):
    directory = tmp_path / "plots"
    (directory / "read_span_quality").mkdir(parents=True)
    (directory / "catalog.parquet").write_bytes(b"catalog-bytes")
    (directory / "read_span_quality" / "ref__bc01.png").write_bytes(b"png-bytes")
    return directory


@pytest.mark.parametrize("hasher", _HASHERS)
def test_finder_visiting_a_published_artifact_does_not_invalidate_it(tmp_path, hasher):
    directory = _artifact(tmp_path)
    published = hasher(directory)

    # Exactly what macOS writes when the folder is opened.
    (directory / ".DS_Store").write_bytes(b"\x00\x00\x00\x01Bud1")

    assert hasher(directory) == published


@pytest.mark.parametrize("hasher", _HASHERS)
def test_other_platforms_metadata_is_ignored_too(tmp_path, hasher):
    directory = _artifact(tmp_path)
    published = hasher(directory)

    (directory / "Thumbs.db").write_bytes(b"thumbs")
    (directory / "desktop.ini").write_bytes(b"[.ShellClassInfo]")
    (directory / "._catalog.parquet").write_bytes(b"resource-fork")
    (directory / "__MACOSX").mkdir()
    (directory / "__MACOSX" / "._archive").write_bytes(b"junk")

    assert hasher(directory) == published


@pytest.mark.parametrize("hasher", _HASHERS)
def test_real_content_still_changes_the_digest(tmp_path, hasher):
    """The ignore must not blunt the check it is protecting."""
    directory = _artifact(tmp_path)
    published = hasher(directory)

    (directory / "read_span_quality" / "ref__bc01.png").write_bytes(b"tampered")

    assert hasher(directory) != published


@pytest.mark.parametrize("hasher", _HASHERS)
def test_an_added_real_file_still_changes_the_digest(tmp_path, hasher):
    directory = _artifact(tmp_path)
    published = hasher(directory)

    (directory / "read_span_quality" / "ref__bc02.png").write_bytes(b"new-plot")

    assert hasher(directory) != published


def test_the_predicate_names_metadata_and_nothing_else():
    from pathlib import Path

    assert is_os_metadata(Path("plots/.DS_Store"))
    assert is_os_metadata(Path("plots/Thumbs.db"))
    assert is_os_metadata(Path("plots/._catalog.parquet"))
    assert is_os_metadata(Path("plots/__MACOSX/._archive"))
    # A dotfile is not automatically metadata: smftools writes its own.
    assert not is_os_metadata(Path("plots/.smftools_marker"))
    assert not is_os_metadata(Path("plots/catalog.parquet"))
    assert not is_os_metadata(Path("plots/read_span_quality/ref__bc01.png"))
