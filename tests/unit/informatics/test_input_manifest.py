import csv
import json
from pathlib import Path

import pytest

from smftools.config.discover_input_files import discover_input_files
from smftools.informatics import input_manifest as manifest_module
from smftools.informatics.input_manifest import (
    HASH_CHUNK_SIZE,
    InputManifestError,
    inspect_input_manifest,
    resolve_input_manifest,
)


def _write(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _manifest(path: Path, rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_path_and_user_manifest_resolve_to_same_digest(tmp_path):
    source = _write(tmp_path / "reads" / "sample.fastq", b"@one\nAC\n+\n!!\n")
    user_manifest = _manifest(
        tmp_path / "manifest.csv",
        [{"path": "reads/sample.fastq"}],
    )

    single_file = resolve_input_manifest(
        output_directory=tmp_path / "out-file", input_paths=[source]
    )
    directory = resolve_input_manifest(
        output_directory=tmp_path / "out-directory",
        input_paths=discover_input_files(tmp_path / "reads")["fastq_paths"],
    )
    declared = resolve_input_manifest(
        output_directory=tmp_path / "out-declared", input_manifest_path=user_manifest
    )

    assert single_file.digest == directory.digest == declared.digest
    assert single_file.rows[0].source_id == declared.rows[0].source_id


def test_user_row_reordering_is_deterministic(tmp_path):
    _write(tmp_path / "a.fastq", b"a")
    _write(tmp_path / "b.fastq", b"b")
    forward = _manifest(tmp_path / "forward.csv", [{"path": "a.fastq"}, {"path": "b.fastq"}])
    reverse = _manifest(tmp_path / "reverse.csv", [{"path": "b.fastq"}, {"path": "a.fastq"}])

    first = resolve_input_manifest(
        output_directory=tmp_path / "out-forward", input_manifest_path=forward
    )
    second = resolve_input_manifest(
        output_directory=tmp_path / "out-reverse", input_manifest_path=reverse
    )

    assert first.digest == second.digest
    assert [row.source_id for row in first.rows] == [row.source_id for row in second.rows]


def test_relative_manifest_is_relocation_invariant(tmp_path):
    digests = []
    for location in (tmp_path / "one", tmp_path / "two"):
        _write(location / "inputs" / "sample.fastq", b"@one\nAC\n+\n!!\n")
        manifest = _manifest(location / "manifest.csv", [{"path": "inputs/sample.fastq"}])
        result = resolve_input_manifest(
            output_directory=location / "output", input_manifest_path=manifest
        )
        digests.append(result.digest)

    assert digests[0] == digests[1]


def test_changed_bytes_at_same_path_change_identity(tmp_path):
    source = _write(tmp_path / "sample.fastq", b"first")
    first = resolve_input_manifest(output_directory=tmp_path / "output", input_paths=[source])
    source.write_bytes(b"second")
    second = resolve_input_manifest(output_directory=tmp_path / "output", input_paths=[source])

    assert first.digest != second.digest
    assert first.rows[0].source_id != second.rows[0].source_id


def test_casava_pairs_keep_lanes_distinct_and_explicit_metadata_wins(tmp_path):
    paths = [
        _write(tmp_path / f"tumor_S1_L{lane}_R{mate}_001.fastq.gz", f"{lane}-{mate}".encode())
        for lane in ("001", "002")
        for mate in (1, 2)
    ]
    user_manifest = _manifest(
        tmp_path / "manifest.csv",
        [
            {"path": path.name, "sample": "declared-sample" if index < 2 else ""}
            for index, path in enumerate(paths)
        ],
    )

    result = resolve_input_manifest(
        output_directory=tmp_path / "output", input_manifest_path=user_manifest
    )

    assert {row.pair_id for row in result.rows} == {
        "tumor_S1_L001_001",
        "tumor_S1_L002_001",
    }
    assert next(row for row in result.rows if row.path == str(paths[0])).sample == "declared-sample"


def test_fastq_identity_maps_use_resolved_paths(tmp_path):
    source = _write(tmp_path / "inputs" / "sample.fastq", b"@one\nAC\n+\n!!\n")
    user_manifest = _manifest(
        tmp_path / "manifest.csv",
        [
            {
                "path": "inputs/sample.fastq",
                "barcode": "barcode01",
                "sample": "sample-a",
                "read_group": "group-a",
            }
        ],
    )

    result = resolve_input_manifest(
        output_directory=tmp_path / "output", input_manifest_path=user_manifest
    )

    assert result.fastq_barcode_map() == {str(source): "barcode01"}
    assert result.fastq_read_group_map() == {str(source): "group-a"}
    assert result.fastq_sample_map() == {str(source): "sample-a"}


def test_existing_bam_resolves_as_aligned_source(tmp_path):
    source = _write(tmp_path / "aligned.bam", b"bam-placeholder")

    result = resolve_input_manifest(
        output_directory=tmp_path / "output",
        input_paths=[source],
        alignment_mode="existing",
        modality="conversion",
    )

    assert result.input_type == "bam"
    assert result.rows[0].source_kind == "aligned_bam"
    assert result.rows[0].source_role == "alignment"
    assert "source_kind" in result.rows[0].inferred_fields


def test_existing_mode_rejects_explicit_unaligned_bam(tmp_path):
    source = _write(tmp_path / "reads.bam", b"bam-placeholder")
    user_manifest = _manifest(
        tmp_path / "manifest.csv",
        [{"path": source.name, "source_kind": "unaligned_bam"}],
    )

    with pytest.raises(InputManifestError, match="conflicts with source_kind='unaligned_bam'"):
        resolve_input_manifest(
            output_directory=tmp_path / "output",
            input_manifest_path=user_manifest,
            alignment_mode="existing",
        )


def test_incomplete_pair_fails_cleanly(tmp_path):
    source = _write(tmp_path / "sample_R1.fastq", b"one")

    with pytest.raises(InputManifestError, match="exactly one R1 and one R2"):
        resolve_input_manifest(output_directory=tmp_path / "output", input_paths=[source])


def test_ambiguous_pair_pattern_requires_explicit_metadata(tmp_path):
    source = _write(tmp_path / "sample_R1_extra.fastq", b"one")

    with pytest.raises(InputManifestError, match="declare pair_id and mate explicitly"):
        resolve_input_manifest(output_directory=tmp_path / "output", input_paths=[source])


def test_missing_and_unreadable_sources_fail_cleanly(tmp_path, monkeypatch):
    missing = tmp_path / "missing.fastq"
    with pytest.raises(InputManifestError, match="Could not stat input source"):
        resolve_input_manifest(output_directory=tmp_path / "missing-output", input_paths=[missing])

    source = _write(tmp_path / "unreadable.fastq", b"one")
    real_open = Path.open

    def permission_denied(path, mode="r", *args, **kwargs):
        if path == source and mode == "rb":
            raise PermissionError("denied for test")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", permission_denied)
    with pytest.raises(InputManifestError, match="Could not read input source"):
        resolve_input_manifest(
            output_directory=tmp_path / "unreadable-output", input_paths=[source]
        )


def test_duplicate_paths_and_duplicate_content_fail(tmp_path):
    first = _write(tmp_path / "first.fastq", b"same")
    second = _write(tmp_path / "second.fastq", b"same")

    with pytest.raises(InputManifestError, match="Duplicate resolved input paths"):
        resolve_input_manifest(
            output_directory=tmp_path / "path-output", input_paths=[first, first]
        )
    with pytest.raises(InputManifestError, match="Duplicate input content"):
        resolve_input_manifest(
            output_directory=tmp_path / "content-output", input_paths=[first, second]
        )


def test_hashing_is_bounded_and_cache_is_task_local(tmp_path, monkeypatch):
    source = _write(tmp_path / "sample.fastq", b"x" * (HASH_CHUNK_SIZE * 2 + 17))
    read_sizes = []
    real_open = Path.open

    class RecordingReader:
        def __init__(self, handle):
            self._handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self._handle.__exit__(*args)

        def fileno(self):
            return self._handle.fileno()

        def read(self, size=-1):
            read_sizes.append(size)
            return self._handle.read(size)

    def recording_open(path, mode="r", *args, **kwargs):
        handle = real_open(path, mode, *args, **kwargs)
        return RecordingReader(handle) if mode == "rb" and path == source else handle

    monkeypatch.setattr(Path, "open", recording_open)
    first = resolve_input_manifest(output_directory=tmp_path / "output", input_paths=[source])
    second = resolve_input_manifest(output_directory=tmp_path / "output", input_paths=[source])

    assert read_sizes and set(read_sizes) == {HASH_CHUNK_SIZE}
    assert first.cache_misses == 1
    assert second.cache_hits == 1


def test_mutation_during_hashing_fails_cleanly(tmp_path, monkeypatch):
    source = _write(tmp_path / "sample.fastq", b"content")
    original = manifest_module._stat_signature
    calls = 0

    def changing_signature(stat_result):
        nonlocal calls
        calls += 1
        signature = original(stat_result)
        if calls == 4:
            return (*signature[:-1], signature[-1] + 1)
        return signature

    monkeypatch.setattr(manifest_module, "_stat_signature", changing_signature)
    with pytest.raises(InputManifestError, match="changed while it was being hashed"):
        resolve_input_manifest(output_directory=tmp_path / "output", input_paths=[source])


def test_published_manifest_records_digest_and_report(tmp_path):
    source = _write(tmp_path / "sample.fastq", b"content")
    result = resolve_input_manifest(output_directory=tmp_path / "output", input_paths=[source])
    root = tmp_path / "output" / "raw_outputs" / "input_manifest"

    metadata = json.loads((root / "resolved_input_manifest.json").read_text())
    report = json.loads((root / "input_resolution_report.json").read_text())
    assert metadata["schema_version"] == 1
    assert metadata["manifest_digest"] == result.digest
    assert metadata["resolution_method"] == "path_discovery"
    assert metadata["base_directory"] == str(tmp_path)
    assert (root / "resolved_input_manifest.csv").read_text().startswith("schema_version,")
    assert report["resolution_method"] == "path_discovery"


def test_inspection_rejects_unknown_columns(tmp_path):
    source = _write(tmp_path / "sample.fastq", b"content")
    manifest = _manifest(tmp_path / "manifest.csv", [{"path": source.name, "surprise": "x"}])

    with pytest.raises(InputManifestError, match="Unknown input manifest columns: surprise"):
        inspect_input_manifest(manifest)
