import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from smftools.informatics.raw_intermediate_manifest import (
    IntermediateSpec,
    RawIntermediateError,
    alignment_reference_bundle,
    artifact_checksum,
    commit_intermediate,
    committed_output,
    executable_version,
    prepare_intermediate,
    validate_intermediate_commit,
)


def test_executable_version_replaces_invalid_utf8(monkeypatch):
    def fake_run(command, **kwargs):
        assert command == ["samtools", "--version"]
        assert kwargs["encoding"] == "utf-8"
        assert kwargs["errors"] == "replace"
        return SimpleNamespace(stdout="samtools 1.16.1\ufffdbuild\nmetadata", stderr="")

    monkeypatch.setattr("smftools.informatics.raw_intermediate_manifest.subprocess.run", fake_run)

    assert executable_version("samtools") == "samtools 1.16.1\ufffdbuild"


def _spec(checksum="source-v1", *, version="1.0", policy="strict"):
    return IntermediateSpec(
        operation="fastq-normalization",
        input_artifacts=(("input-manifest", checksum),),
        operation_config={"barcode_tag": "BC"},
        tool_versions={"samtools": version},
        tool_version_policy=policy,
    )


def _commit(output, spec, content=b"bam"):
    workspace = prepare_intermediate(output, spec)
    artifact = workspace.root / "canonical.bam"
    artifact.write_bytes(content)
    commit_intermediate(workspace, {"bam": artifact})
    return workspace, artifact


def test_same_semantics_reuse_only_a_valid_complete_commit(tmp_path):
    spec = _spec()
    committed, artifact = _commit(tmp_path, spec)

    reused = prepare_intermediate(tmp_path, spec)

    assert reused.reusable is True
    assert reused.root == committed.root
    assert committed_output(reused, "bam") == artifact


def test_changed_source_or_corrupt_output_forces_new_revision(tmp_path):
    spec = _spec()
    committed, artifact = _commit(tmp_path, spec)
    changed_source = prepare_intermediate(tmp_path, _spec("source-v2"))
    assert changed_source.reusable is False
    assert changed_source.root != committed.root

    artifact.write_bytes(b"corrupt")
    assert validate_intermediate_commit(committed.manifest_path, spec) is None
    repaired = prepare_intermediate(tmp_path, spec)
    assert repaired.reusable is False
    assert repaired.root != committed.root


def test_missing_or_corrupt_commit_metadata_is_never_reused(tmp_path):
    spec = _spec()
    incomplete = prepare_intermediate(tmp_path, spec)
    (incomplete.root / "canonical.bam").write_bytes(b"uncommitted")

    replacement = prepare_intermediate(tmp_path, spec)
    assert replacement.reusable is False
    assert replacement.root != incomplete.root
    replacement.manifest_path.write_text("not-json", encoding="utf-8")

    next_attempt = prepare_intermediate(tmp_path, spec)
    assert next_attempt.reusable is False
    assert next_attempt.root != replacement.root


def test_legacy_fixed_output_without_commit_is_never_reused(tmp_path):
    legacy = tmp_path / "bam_outputs" / "canonical_basecalls.bam"
    legacy.parent.mkdir()
    legacy.write_bytes(b"stale")

    workspace = prepare_intermediate(tmp_path, _spec())

    assert workspace.reusable is False
    assert workspace.root != legacy.parent
    assert not (workspace.root / legacy.name).exists()


def test_tool_version_policy_controls_compatibility(tmp_path):
    strict = _spec(version="1.0", policy="strict")
    strict_workspace, _ = _commit(tmp_path / "strict", strict)
    strict_changed = prepare_intermediate(
        tmp_path / "strict", _spec(version="2.0", policy="strict")
    )
    assert strict_changed.root != strict_workspace.root
    assert strict_changed.reusable is False

    provenance = _spec(version="1.0", policy="provenance_only")
    provenance_workspace, _ = _commit(tmp_path / "provenance", provenance)
    provenance_changed = prepare_intermediate(
        tmp_path / "provenance", _spec(version="2.0", policy="provenance_only")
    )
    assert provenance_changed.root == provenance_workspace.root
    assert provenance_changed.reusable is True


def test_force_redo_allocates_revision_without_mutating_commit(tmp_path):
    spec = _spec()
    committed, artifact = _commit(tmp_path, spec, content=b"original")

    forced = prepare_intermediate(tmp_path, spec, force_redo=True)
    forced_artifact = forced.root / "canonical.bam"
    forced_artifact.write_bytes(b"replacement")
    commit_intermediate(forced, {"bam": forced_artifact})

    assert forced.root != committed.root
    assert artifact.read_bytes() == b"original"
    assert forced_artifact.read_bytes() == b"replacement"


def test_empty_output_is_never_committed(tmp_path):
    """A silently failed tool leaves an empty artifact that must not be committed."""
    spec = _spec()
    workspace = prepare_intermediate(tmp_path, spec)
    empty_file = workspace.root / "canonical.bam"
    empty_file.write_bytes(b"")

    with pytest.raises(RawIntermediateError, match="is empty"):
        commit_intermediate(workspace, {"bam": empty_file})

    assert not workspace.manifest_path.exists()
    assert prepare_intermediate(tmp_path, spec).reusable is False


def test_empty_directory_output_is_never_committed(tmp_path):
    spec = _spec()
    workspace = prepare_intermediate(tmp_path, spec)
    empty_directory = workspace.root / "mod_beds"
    empty_directory.mkdir()

    with pytest.raises(RawIntermediateError, match="has no files"):
        commit_intermediate(workspace, {"mod-beds": empty_directory})

    populated = workspace.root / "mod_tsvs"
    populated.mkdir()
    (populated / "ref.tsv").write_text("read_id\n", encoding="utf-8")
    assert commit_intermediate(workspace, {"mod-tsvs": populated}).exists()


def test_previously_committed_empty_artifact_is_not_reused(tmp_path):
    """A cache poisoned before this check existed must heal, not be reused.

    Reproduces the dorado failure mode: a zero-byte BAM has a perfectly stable
    checksum, so checksum validation alone accepted the commit and every later
    run reused it instead of rerunning the basecaller.
    """
    spec = _spec()
    workspace = prepare_intermediate(tmp_path, spec)
    poisoned = workspace.root / "canonical.bam"
    poisoned.write_bytes(b"")
    # Hand-write the manifest that the pre-fix commit path would have published.
    workspace.manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state": "complete",
                "operation": spec.operation,
                "revision_id": workspace.revision_id,
                "compatibility_key": spec.compatibility_key,
                "compatibility": spec.compatibility_payload(),
                "tool_versions": dict(spec.tool_versions),
                "outputs": [
                    {
                        "artifact_id": "bam",
                        "path": "canonical.bam",
                        "kind": "file",
                        "size_bytes": 0,
                        "sha256": artifact_checksum(poisoned),
                    }
                ],
                "committed_at": "2026-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    assert validate_intermediate_commit(workspace.manifest_path, spec) is None

    replacement = prepare_intermediate(tmp_path, spec)
    assert replacement.reusable is False
    assert replacement.root != workspace.root


def test_reference_bundle_is_relocation_invariant_and_content_sensitive(tmp_path):
    digests = []
    for name in ("one", "two"):
        root = tmp_path / name
        root.mkdir()
        fasta = root / "reference.fa"
        fasta.write_text(">ref\nACGT\n", encoding="utf-8")
        bed = root / "regions.bed"
        bed.write_text("ref\t0\t4\n", encoding="utf-8")
        cfg = SimpleNamespace(
            fasta=fasta,
            alignment_regions_bed=bed,
            smf_modality="conversion",
            conversion_types=["unconverted"],
            strands=["top", "bottom"],
        )
        digests.append(alignment_reference_bundle(cfg)["digest"])

    assert digests[0] == digests[1]

    changed = Path(tmp_path / "two" / "reference.fa")
    changed.write_text(">ref\nTGCA\n", encoding="utf-8")
    changed_cfg = SimpleNamespace(
        fasta=changed,
        alignment_regions_bed=tmp_path / "two" / "regions.bed",
        smf_modality="conversion",
        conversion_types=["unconverted"],
        strands=["top", "bottom"],
    )
    assert alignment_reference_bundle(changed_cfg)["digest"] != digests[1]
