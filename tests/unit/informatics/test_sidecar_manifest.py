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


def test_resolve_missing_or_nonexistent_sidecar(tmp_path):
    manifest = sidecar_manifest_path(tmp_path)
    missing = resolve_sidecar(manifest, "barcode")
    assert missing is None

    sidecar = tmp_path / "b.parquet"
    register_sidecar(manifest, "barcode", sidecar)
    missing_after_register = resolve_sidecar(manifest, "barcode")
    assert missing_after_register is None
