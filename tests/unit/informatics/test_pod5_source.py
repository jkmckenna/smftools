from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest

from smftools.informatics.input_manifest import (
    InputManifestError,
    InputManifestRow,
    checksum_input_source,
)
from smftools.informatics.pod5_identity import build_pod5_dataset_index
from smftools.informatics.pod5_source import (
    Pod5SourceCandidate,
    resolve_pod5_sources,
)

pytestmark = pytest.mark.unit


def _write(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _row(source_id: str, path: Path, content: bytes) -> InputManifestRow:
    return InputManifestRow(
        source_id=source_id,
        path=str(path),
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        source_kind="pod5",
        source_role="raw_signal",
    )


def test_resolves_mixed_recorded_and_relocated_exact_sources(tmp_path):
    first_content = b"first-pod5"
    second_content = b"second-pod5"
    first = _write(tmp_path / "original" / "first.pod5", first_content)
    missing = tmp_path / "original" / "second.pod5"
    relocated = _write(tmp_path / "archive" / "renamed.pod5", second_content)
    rows = (_row("source-a", first, first_content), _row("source-b", missing, second_content))

    result = resolve_pod5_sources(
        rows,
        candidates=(
            Pod5SourceCandidate(
                path=relocated,
                evidence="explicit_relocation",
                source_id="source-b",
                sha256=hashlib.sha256(second_content).hexdigest(),
            ),
        ),
    )

    assert result.complete
    assert result.recorded_path_count == 1
    assert result.relocated_path_count == 1
    assert result.evidence_counts == {"explicit_relocation": 1, "recorded_path": 1}
    assert result.resolved_sources == (
        ("source-a", first.resolve()),
        ("source-b", relocated.resolve()),
    )


def test_same_named_wrong_content_is_not_accepted(tmp_path):
    expected = b"expected-pod5"
    missing = tmp_path / "original" / "reads.pod5"
    wrong = _write(tmp_path / "relocated" / "reads.pod5", b"different-pod5")
    row = _row("source-a", missing, expected)

    result = resolve_pod5_sources(
        (row,),
        candidates=(
            Pod5SourceCandidate(
                path=wrong,
                evidence="explicit_relocation",
                source_id=row.source_id,
            ),
        ),
    )

    assert not result.complete
    assert result.checksum_mismatch_count == 1
    assert result.rows[0].resolved_path is None
    assert result.rows[0].observed_sha256s == (hashlib.sha256(wrong.read_bytes()).hexdigest(),)


def test_missing_and_unreadable_sources_remain_distinct(tmp_path):
    content = b"pod5"
    missing = tmp_path / "missing.pod5"
    missing_row = _row("source-missing", missing, content)
    missing_result = resolve_pod5_sources((missing_row,))

    unreadable = _write(tmp_path / "unreadable.pod5", content)
    unreadable_row = _row("source-unreadable", unreadable, content)

    def fail_checksum(_path):
        raise InputManifestError("denied")

    unreadable_result = resolve_pod5_sources(
        (unreadable_row,),
        checksum_reader=fail_checksum,
    )

    assert missing_result.missing_count == 1
    assert missing_result.unreadable_count == 0
    assert unreadable_result.missing_count == 0
    assert unreadable_result.unreadable_count == 1


def test_redundant_exact_candidates_choose_deterministically(tmp_path):
    content = b"pod5"
    missing = tmp_path / "missing.pod5"
    second = _write(tmp_path / "z" / "reads.pod5", content)
    first = _write(tmp_path / "a" / "reads.pod5", content)
    row = _row("source-a", missing, content)
    candidates = (
        Pod5SourceCandidate(second, "explicit_relocation", source_id=row.source_id),
        Pod5SourceCandidate(first, "explicit_relocation", sha256=row.sha256),
    )

    forward = resolve_pod5_sources((row,), candidates=candidates)
    reverse = resolve_pod5_sources((row,), candidates=reversed(candidates))

    assert forward.rows[0].resolved_path == first.resolve()
    assert reverse.rows[0].resolved_path == first.resolve()
    assert forward.duplicate_valid_candidate_count == 1
    assert forward.digest == reverse.digest


def test_unmatched_candidate_identity_is_counted_without_guessing(tmp_path):
    content = b"pod5"
    source = _write(tmp_path / "reads.pod5", content)
    row = _row("source-a", source, content)

    result = resolve_pod5_sources(
        (row,),
        candidates=(
            Pod5SourceCandidate(
                tmp_path / "other.pod5",
                "explicit_relocation",
                source_id="unknown-source",
            ),
        ),
    )

    assert result.complete
    assert result.unmatched_candidate_count == 1
    assert result.resolved_sources == (("source-a", source.resolve()),)


def test_resolution_digest_is_invariant_to_exact_byte_location(tmp_path):
    content = b"pod5"
    first = _write(tmp_path / "one" / "reads.pod5", content)
    second = _write(tmp_path / "two" / "renamed.pod5", content)
    missing = tmp_path / "missing.pod5"
    row = _row("source-a", missing, content)

    first_result = resolve_pod5_sources(
        (row,),
        candidates=(Pod5SourceCandidate(first, "explicit_relocation", source_id=row.source_id),),
    )
    second_result = resolve_pod5_sources(
        (row,),
        candidates=(Pod5SourceCandidate(second, "explicit_relocation", source_id=row.source_id),),
    )

    assert first_result.digest == second_result.digest
    assert first_result.rows[0].to_dict() != second_result.rows[0].to_dict()


def test_checked_in_pod5_can_be_indexed_through_relocated_resolution(tmp_path):
    fixture = Path(__file__).parents[2] / "_test_inputs" / "_test_pod5_I.pod5"
    sha256, size_bytes = checksum_input_source(fixture)
    relocated = tmp_path / "archive" / "renamed.pod5"
    relocated.parent.mkdir()
    shutil.copyfile(fixture, relocated)
    row = InputManifestRow(
        source_id="fixture-source",
        path=str(tmp_path / "original-is-gone.pod5"),
        sha256=sha256,
        size_bytes=size_bytes,
        source_kind="pod5",
        source_role="raw_signal",
    )

    resolution = resolve_pod5_sources(
        (row,),
        candidates=(
            Pod5SourceCandidate(
                relocated,
                "explicit_relocation",
                source_id=row.source_id,
                sha256=row.sha256,
            ),
        ),
    )
    index = build_pod5_dataset_index(resolution.resolved_sources)

    assert resolution.complete
    assert resolution.rows[0].resolved_path == relocated.resolve()
    assert index.unique_read_count == 4
    assert index.duplicate_read_id_count == 0


def test_rejects_non_pod5_manifest_rows(tmp_path):
    source = _write(tmp_path / "reads.bam", b"bam")
    row = InputManifestRow(
        source_id="source-a",
        path=str(source),
        sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        size_bytes=source.stat().st_size,
        source_kind="aligned_bam",
        source_role="alignment",
    )

    with pytest.raises(ValueError, match="only POD5"):
        resolve_pod5_sources((row,))
