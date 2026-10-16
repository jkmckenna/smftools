"""`SLC-11`: the read-only storage usage report."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from smftools.cli_entry import cli
from smftools.constants import RAW_DIR, SPATIAL_DIR
from smftools.data.storage_usage import render_usage, storage_usage
from smftools.informatics.generation_listing import (
    CURRENT_FILENAME,
    GENERATION_MANIFEST,
    GENERATIONS_SUBDIR,
)

pytestmark = pytest.mark.unit


def _generation(container: Path, gid: str, *, size: int, current: bool = False) -> Path:
    path = container / GENERATIONS_SUBDIR / gid
    path.mkdir(parents=True)
    (path / "store.bin").write_bytes(b"x" * size)
    (path / GENERATION_MANIFEST).write_text(
        json.dumps(
            {"status": "complete", "generation_id": gid, "created_at": "2026-01-01T00:00:00Z"}
        )
    )
    if current:
        (container / CURRENT_FILENAME).write_text(
            json.dumps({"generation_id": gid, "generation_path": f"{GENERATIONS_SUBDIR}/{gid}"})
        )
    return path


def _model(workspace: Path, model_id: str, *, compressed: bool, size: int) -> None:
    bundle = workspace / "models" / model_id
    (bundle / "payload").mkdir(parents=True)
    method = zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED
    with zipfile.ZipFile(bundle / "payload" / "model.skops", "w", method) as archive:
        archive.writestr("schema.json", "0" * size)
    (bundle / "model_manifest.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "architecture": {"name": "random_forest_v1"},
                "artifact": {
                    "relative_path": "payload/model.skops",
                    "media_type": "application/vnd.skops+zip",
                },
            }
        )
    )


def _drive(tmp_path: Path) -> Path:
    drive = tmp_path / "drive"
    run_dir = drive / "runs" / "run_a"
    root = run_dir / "exp_a"
    root.mkdir(parents=True)
    (root / "experiment_manifest.json").write_text("{}")
    (root / "bams").mkdir()
    (root / "bams" / "reads.bam").write_bytes(b"b" * 50)
    _generation(root / RAW_DIR, "raw-read", size=1000)
    _generation(root / RAW_DIR, "raw-stale", size=2000)
    _generation(root / RAW_DIR, "raw-current", size=300, current=True)
    spatial = _generation(root / SPATIAL_DIR, "sp-current", size=400, current=True)
    (spatial / "sidecar.json").write_text(
        json.dumps({"molecules": f"../../{RAW_DIR}/{GENERATIONS_SUBDIR}/raw-read/molecules"})
    )
    legacy = run_dir / "outputs_old"
    legacy.mkdir()
    (legacy / "old.h5ad").write_bytes(b"o" * 5000)

    project = drive / "projects" / "proj"
    (project / "project_outputs").mkdir(parents=True)
    (project / "registry.json").write_text(
        json.dumps(
            {
                "experiments": {
                    "exp_a": {
                        "spines": {
                            "raw": f"../../runs/run_a/exp_a/{RAW_DIR}/{GENERATIONS_SUBDIR}/raw-current/spine.h5ad"
                        }
                    }
                }
            }
        )
    )
    workspace = project / "project_outputs" / "ml"
    (workspace / "runs" / "r1").mkdir(parents=True)
    (workspace / "runs" / "r1" / "run_manifest.json").write_text(
        json.dumps({"source_model_ids": ["m-read"]})
    )
    _model(workspace, "m-read", compressed=True, size=4000)
    _model(workspace, "m-unread", compressed=False, size=4000)
    return drive


def test_usage_buckets_generations_unmanaged_and_models(tmp_path: Path) -> None:
    drive = _drive(tmp_path)
    paths_before = sorted(p.relative_to(drive) for p in drive.rglob("*"))
    report = storage_usage(drive)

    (experiment,) = report["experiments"]
    assert experiment["registered"] is True
    g = experiment["generations"]
    assert g["current"]["count"] == 2
    assert g["referenced"]["count"] == 1  # raw-read, read by the current spatial generation
    assert g["prune_candidate"] == {"count": 1, "bytes": g["prune_candidate"]["bytes"]}
    assert g["prune_candidate"]["bytes"] >= 2000
    assert experiment["other_bytes"] >= 50  # the BAM, outside generations

    (legacy,) = report["unmanaged"]
    assert legacy["path"].endswith("run_a/outputs_old") and legacy["bytes"] == 5000

    (workspace,) = report["ml_workspaces"]
    assert workspace["models"]["count"] == 2
    assert workspace["never_read"]["count"] == 1
    assert workspace["uncompressed"]["count"] == 1
    assert workspace["by_architecture"]["random_forest_v1"]["count"] == 2

    reclaim = {item["bucket"]: item["bytes"] for item in report["reclaim"]}
    assert reclaim["generations nothing live reads"] == g["prune_candidate"]["bytes"]
    assert reclaim["unmanaged directories beside experiment roots"] == 5000
    text = render_usage(report)
    assert "outputs_old" in text and "ML WORKSPACE" in text
    # read-only
    assert sorted(p.relative_to(drive) for p in drive.rglob("*")) == paths_before


def test_unregistered_experiment_is_flagged(tmp_path: Path) -> None:
    drive = _drive(tmp_path)
    other = drive / "runs" / "run_b" / "exp_b"
    other.mkdir(parents=True)
    (other / "experiment_manifest.json").write_text("{}")
    report = storage_usage(drive)
    flags = {Path(e["root"]).name: e["registered"] for e in report["experiments"]}
    assert flags == {"exp_a": True, "exp_b": False}
    assert "[not registered]" in render_usage(report)


def test_cli_usage(tmp_path: Path) -> None:
    drive = _drive(tmp_path)
    result = CliRunner().invoke(cli, ["data", "usage", str(drive), "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["dry_run"] is True
    table = CliRunner().invoke(cli, ["data", "usage", str(drive)])
    assert table.exit_code == 0 and "RECLAIMABLE" in table.output
