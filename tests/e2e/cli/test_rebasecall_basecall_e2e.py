"""Opt-in end-to-end selective basecalling against a real installed Dorado.

Everything else covering `SRB-04b` runs Dorado through an injected fake. This
profile is the only place the executed argv, the emitted BAM header, and the
`pi` parent tags are produced by the real basecaller, which is the class of
problem a fake cannot surface.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from smftools.informatics.input_manifest import InputManifestRow, ResolvedInputManifest
from smftools.informatics.pod5_identity import build_pod5_dataset_index
from smftools.pipeline import rebasecall_plan
from smftools.pipeline.rebasecall_basecall import (
    BASECALL_ORIGIN_FILENAME,
    execute_rebasecall_basecall,
    read_published_rebasecall_basecall,
)
from smftools.pipeline.rebasecall_plan import ParentGeneration
from smftools.pipeline.rebasecall_request import rebasecall_request_from_dict
from smftools.pipeline.rebasecall_selection import freeze_rebasecall_selection

FIXTURE = Path(__file__).parents[2] / "_test_inputs" / "_test_pod5_I.pod5"


def _model_directory() -> Path | None:
    configured = os.environ.get("SMFTOOLS_DORADO_MODEL_DIR")
    candidate = Path(configured) if configured else Path.home() / "dorado_models"
    return candidate if candidate.is_dir() else None


def _skip_without_prerequisites() -> Path:
    missing = []
    if shutil.which("dorado") is None:
        missing.append("the dorado executable")
    model_directory = _model_directory()
    if model_directory is None:
        missing.append("a dorado model directory (SMFTOOLS_DORADO_MODEL_DIR or ~/dorado_models)")
    if not FIXTURE.is_file():
        missing.append(f"the POD5 fixture at {FIXTURE}")
    if missing:
        pytest.skip(f"real selective basecalling requires {', '.join(missing)}")
    assert model_directory is not None
    return model_directory


def _install_real_source(tmp_path, monkeypatch):
    """Install a plan whose signal is the checked-in POD5 fixture."""
    raw_dir = tmp_path / "raw_outputs" / "generations" / "raw-a"
    raw_dir.mkdir(parents=True)
    source = tmp_path / "source" / "reads.pod5"
    source.parent.mkdir()
    shutil.copyfile(FIXTURE, source)
    index = build_pod5_dataset_index((("source-a", source),))
    read_ids = sorted(index.sources_by_read_id)
    pd.DataFrame(
        {
            "read_id": read_ids,
            "molecule_uid": [f"m{ordinal}" for ordinal in range(len(read_ids))],
            "pod5_read_id": read_ids,
            "pod5_source_id": ["source-a"] * len(read_ids),
        }
    ).to_parquet(raw_dir / "obs.parquet", index=False)
    row = InputManifestRow(
        source_id="source-a",
        path=str(source),
        sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        size_bytes=source.stat().st_size,
        source_kind="pod5",
        source_role="raw_signal",
    )
    manifest = ResolvedInputManifest(
        rows=(row,),
        digest="manifest-a",
        resolution_method="published",
        base_directory=str(source.parent),
    )
    raw_parent = ParentGeneration(
        stage="raw",
        selector="raw-a",
        generation_id="raw-a",
        generation_dir=raw_dir,
        manifest={"generation_id": "raw-a"},
    )
    monkeypatch.setattr(rebasecall_plan, "_resolve_raw_parent", lambda *_args: raw_parent)
    monkeypatch.setattr(rebasecall_plan, "_resolve_preprocess_parent", lambda *_args: None)
    monkeypatch.setattr(rebasecall_plan, "_read_input_manifest", lambda _parent: manifest)
    monkeypatch.setattr(
        rebasecall_plan,
        "read_experiment_manifest",
        lambda _root: {"experiment_uid": "uid-a", "experiment_id": "experiment-a"},
    )
    return (
        SimpleNamespace(
            output_directory=tmp_path,
            experiment_id="experiment-a",
            experiment_name="experiment-a",
            model_dir=str(_model_directory()),
            device="cpu",
        ),
        read_ids,
    )


@pytest.mark.e2e
def test_real_dorado_publishes_a_validated_selective_basecall(tmp_path, monkeypatch):
    model_directory = _skip_without_prerequisites()
    cfg, read_ids = _install_real_source(tmp_path, monkeypatch)
    request = rebasecall_request_from_dict(
        {
            "schema_version": 1,
            "name": "e2e-selective-basecall",
            "source": {"raw_generation": "raw-a"},
            "selection": {
                "mode": "ids",
                "id_kind": "pod5_read_id",
                "ids": read_ids[:2],
            },
            "basecall": {"model": "hac@latest", "emit_moves": False},
            "signal": {"materialize": False},
            "downstream": {"target": "raw"},
            "promotion": {"activate": False},
        }
    )

    plan = rebasecall_plan.build_rebasecall_plan(cfg, request)
    if plan.status != "ready":
        pytest.skip(f"the real fixture plan is blocked: {[r.code for r in plan.blockers]}")
    frozen = freeze_rebasecall_selection(
        plan,
        tmp_path / "selection-results",
        accepted_plan_id=plan.plan_id,
        # The parent here is a synthetic stand-in; this profile exists to exercise
        # the real basecaller, and SRB-01b already covers parent revalidation.
        parent_validator=lambda _plan: None,
    )

    published = execute_rebasecall_basecall(
        plan,
        frozen,
        tmp_path / "basecalls",
        accepted_plan_id=plan.plan_id,
        model_directory=model_directory,
    )

    counts = published.manifest["counts"]
    assert counts["requested_unique_read_count"] == 2
    assert counts["duplicate_output_read_id_count"] == 0
    assert counts["source_parent_observed_count"] + counts["missing_read_count"] == 2
    # The exit gate: the exact selected UUID set was basecalled, so a manifest
    # that merely validated an empty BAM would not satisfy this test.
    assert counts["source_parent_observed_count"] == 2
    assert counts["output_record_count"] >= 2
    assert published.generation_kind == "selected_cohort"
    # The header agreement is the real assertion here: these values came from
    # Dorado's own @PG/@RG records, not from a fake.
    assert (
        published.manifest["dorado"]["header"]["dorado_version"]
        == (published.manifest["identity"]["model"]["dorado_version"])
    )
    origin = pd.read_csv(published.directory / BASECALL_ORIGIN_FILENAME)
    assert set(origin["pod5_read_id"]).issubset(set(read_ids[:2]))
    assert set(origin["pod5_source_id"]) <= {"source-a"}
    read_published_rebasecall_basecall(
        published.directory,
        expected_basecall_id=published.basecall_id,
    )
