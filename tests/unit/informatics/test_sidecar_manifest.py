import json
from pathlib import Path

from smftools.informatics.sidecar_manifest import (
    register_sidecar,
    resolve_sidecar,
    sidecar_manifest_path,
)


def test_register_and_resolve_sidecar(tmp_path):
    manifest = sidecar_manifest_path(tmp_path)
    sidecar = tmp_path / "a.parquet"
    sidecar.write_text("x", encoding="utf-8")

    register_sidecar(manifest, "umi_oriented", sidecar, metadata={"source_bam": "aligned.bam"})
    resolved = resolve_sidecar(manifest, "umi_oriented")

    assert resolved == sidecar
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["version"] == 2
    assert payload["sidecars"]["umi_oriented"]["path"] == "a.parquet"
    assert payload["sidecars"]["umi_oriented"]["path_kind"] == "relative"
    assert payload["sidecars"]["umi_oriented"]["anchor"] == "manifest_parent"


def test_resolve_missing_or_nonexistent_sidecar(tmp_path):
    manifest = sidecar_manifest_path(tmp_path)
    missing = resolve_sidecar(manifest, "barcode")
    assert missing is None

    sidecar = tmp_path / "b.parquet"
    register_sidecar(manifest, "barcode", sidecar)
    missing_after_register = resolve_sidecar(manifest, "barcode")
    assert missing_after_register is None


def test_sidecar_manifest_resolves_after_directory_move(tmp_path):
    original = tmp_path / "original" / "spatial_outputs"
    original.mkdir(parents=True)
    sidecar = original / "metrics.parquet"
    sidecar.write_text("metrics", encoding="utf-8")
    manifest = sidecar_manifest_path(original)
    register_sidecar(manifest, "metrics", sidecar)

    moved = tmp_path / "moved" / "spatial_outputs"
    moved.parent.mkdir()
    original.rename(moved)

    assert resolve_sidecar(sidecar_manifest_path(moved), "metrics") == moved / "metrics.parquet"


def test_resolve_legacy_absolute_and_relative_sidecars(tmp_path):
    absolute = tmp_path / "absolute.parquet"
    relative = tmp_path / "relative.parquet"
    absolute.write_text("absolute", encoding="utf-8")
    relative.write_text("relative", encoding="utf-8")
    manifest = sidecar_manifest_path(tmp_path)
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "sidecars": {
                    "absolute": {"path": str(absolute)},
                    "relative": {"path": "relative.parquet"},
                },
            }
        ),
        encoding="utf-8",
    )

    assert resolve_sidecar(manifest, "absolute") == absolute
    assert resolve_sidecar(manifest, "relative") == relative
