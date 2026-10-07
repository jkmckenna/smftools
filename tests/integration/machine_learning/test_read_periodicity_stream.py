"""RPG-02: per-read periodograms streamed from a plan dataset."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from smftools.informatics.molecule_identity import molecule_uid
from smftools.informatics.partition_store import write_experiment_store
from smftools.machine_learning.plan import parse_ml_plan
from smftools.project.reference_registry import ReferenceRegistry
from smftools.project.registry import init_project, load_registry, save_registry
from smftools.tools.read_periodicity import compute_read_periodicity, windows

pytestmark = pytest.mark.integration

LENGTH = 1600
READS = 8
PERIOD = 190.0
C_SITES = np.arange(LENGTH) % 3 == 0


@pytest.fixture
def project(tmp_path: Path):
    rng = np.random.default_rng(1)
    run_root = tmp_path / "runs" / "exp"
    preprocess = run_root / "preprocess_adata_outputs"
    barcodes = ["barcode01"] * READS + ["barcode02"] * READS
    read_ids = [f"r{index}" for index in range(2 * READS)]
    probability = 0.5 + 0.45 * np.sin(2 * np.pi * np.arange(LENGTH) / PERIOD)
    calls = (rng.random((len(read_ids), LENGTH)) < probability).astype(np.float32)
    calls[:, ~C_SITES] = np.nan
    obs = pd.DataFrame(
        {
            "Reference_strand": pd.Categorical(["locus_top"] * len(read_ids)),
            "Sample": pd.Categorical(barcodes),
        },
        index=read_ids,
    )
    source = ad.AnnData(X=calls, obs=obs)
    source.var_names = [str(position) for position in range(LENGTH)]
    source.var["locus_top_C_site"] = C_SITES
    paths = write_experiment_store(source, preprocess, experiment="exp", modality="deaminase")
    experiment_uid = str(uuid4())
    uids = [molecule_uid(experiment_uid, read_id) for read_id in read_ids]
    (run_root / "molecule_index").mkdir(parents=True)
    pd.DataFrame(
        {
            "molecule_uid": uids,
            "experiment_uid": experiment_uid,
            "read_id": read_ids,
            "Reference_strand": "locus_top",
            "Sample": barcodes,
            "Barcode": barcodes,
            "reference_start": 0,
            "reference_end": LENGTH,
        }
    ).to_parquet(run_root / "molecule_index" / "part.parquet", index=False)
    (preprocess / "read_index").mkdir()
    pd.DataFrame(
        {
            "molecule_uid": uids,
            "group_path": [f"store/{barcode}" for barcode in barcodes],
            "group_row": list(range(READS)) * 2,
        }
    ).to_parquet(preprocess / "read_index" / "part.parquet", index=False)
    pd.DataFrame(
        {"task_id": ["t"], "reference": ["locus_top"], "layers": [[]], "has_x": [True]}
    ).to_parquet(preprocess / "catalog.parquet", index=False)
    raw = run_root / "raw_outputs"
    raw.mkdir()
    (raw / "spine.h5ad").touch()
    pd.DataFrame({"reference": ["locus_top"], "max_end": [LENGTH]}).to_parquet(
        raw / "interval_catalog.parquet", index=False
    )
    root = tmp_path / "project"
    init_project(root)
    registry = load_registry(root)
    registry["experiments"] = {
        "exp": {
            "path": str(run_root),
            "name": "exp",
            "experiment_uid": experiment_uid,
            "modality": "deaminase",
            "schema_version": 1,
            "spines": {"raw": str(raw / "spine.h5ad"), "preprocess": str(paths["spine"])},
            "references": {"locus_top": "uid"},
            "n_reads": len(read_ids),
            "status": "active",
            "catalogs": {
                "interval_catalog.parquet": str(raw / "interval_catalog.parquet"),
                "molecule_index": str(run_root / "molecule_index"),
                "preprocess_read_index": str(preprocess / "read_index"),
            },
        }
    }
    save_registry(root, registry)
    ReferenceRegistry(canonical_names={"uid": "locus"}).save(root / "reference_registry.yaml")
    return root


def _plan():
    return parse_ml_plan(
        {
            "schema_version": 1,
            "scope": {"kind": "project"},
            "datasets": {
                "reads": {
                    "modalities": ["deaminase"],
                    "references": ["locus"],
                    "channels": [
                        {
                            "name": "C",
                            "biological_role": "accessibility",
                            "sources": [
                                {
                                    "modality": "deaminase",
                                    "stage": "preprocess",
                                    "layer": "X",
                                    "site_context": "C",
                                }
                            ],
                        }
                    ],
                }
            },
            "splits": {},
            "models": {},
            "jobs": {},
        }
    )


def test_windows():
    assert windows([0, 1, 2, 5, 6, 9]) == [(0, 3), (5, 7), (9, 10)]
    assert windows([]) == []


def test_default_region_recovers_the_period_per_group(project) -> None:
    result = compute_read_periodicity(_plan(), "reads", project_dir=project, group_by="Barcode")
    assert list(result.grids) == [f"0-{LENGTH}"]
    stats = result.stats
    assert len(stats) == 2 * READS and set(stats["group"]) == {"barcode01", "barcode02"}
    assert (stats["status"] == "ok").all()
    assert np.all(np.abs(stats["peak_period_bp"] - PERIOD) <= 8)
    assert result.power[f"0-{LENGTH}"].shape == (2 * READS, 321)
    assert result.regions["status"].tolist() == ["ok"]


def test_explicit_regions_narrow_or_skip(project) -> None:
    result = compute_read_periodicity(
        _plan(), "reads", project_dir=project, regions=[(0, 1600), (1000, 1600), (0, 300)]
    )
    regions = result.regions.set_index("region")
    assert regions.loc["1000-1600", "period_max_bp"] == 200 and regions.loc["1000-1600", "narrowed"]
    assert regions.loc["0-300", "status"] == "region_too_short"
    assert result.power["1000-1600"].shape == (2 * READS, 121)
    assert (result.stats.query("region == '0-300'")["status"] == "region_too_short").all()
    assert set(result.stats["group"]) == {"all"}


def test_workers_give_identical_results(project) -> None:
    one = compute_read_periodicity(_plan(), "reads", project_dir=project)
    two = compute_read_periodicity(_plan(), "reads", project_dir=project, workers=2)
    pd.testing.assert_frame_equal(one.stats, two.stats)
    for name in one.power:
        np.testing.assert_array_equal(one.power[name], two.power[name])


def test_bad_channel_is_an_error(project) -> None:
    with pytest.raises(KeyError, match="channel"):
        compute_read_periodicity(_plan(), "reads", project_dir=project, channel="nope")


def _labelled_plan(root: Path, names: dict[int, str]):
    """The plan, grouped by a label table written into the project (`F72`)."""
    pd.DataFrame(
        {
            "experiment_id": "exp",
            "barcode": list(names),
            "physical_reference": "locus_top",
            "group": list(names.values()),
        }
    ).to_parquet(root / "labels.parquet", index=False)
    document = _plan().to_dict()
    document["datasets"]["reads"]["labels"] = {
        "source": "table",
        "table": "labels.parquet",
        "keys": ["experiment_id", "barcode", "physical_reference"],
        "column": "group",
        "classes": {name: index for index, name in enumerate(sorted(set(names.values())))},
    }
    return parse_ml_plan(document)


def _invoke(*args):
    from click.testing import CliRunner

    from smftools.cli_entry import cli

    result = CliRunner().invoke(cli, [str(a) for a in args])
    assert result.exit_code == 0, result.output
    return result


def test_cli_writes_outputs_and_reuses_them(project, tmp_path) -> None:
    import json

    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_plan().to_dict()))
    regions = tmp_path / "regions.bed"
    regions.write_text("# name start end\nfull 0 1600\ntail 1000 1600\nshort 0 300\n")
    out = tmp_path / "out"
    args = [
        "project",
        "periodicity",
        project,
        "--plan",
        plan_path,
        "--dataset",
        "reads",
        "--group-by",
        "Barcode",
        "--regions-file",
        regions,
        "--max-reads-per-plot",
        5,
        "--output",
        out,
    ]
    first = _invoke(*args)
    assert "computed" in first.output and "narrowed for: 1000-1600" in first.output
    for name in (
        "read_periodicity.parquet",
        "regions.parquet",
        "power_0-1600.npy",
        "periods_1000-1600.npy",
        "plot_values_0-1600.npz",
        "periodicity_key.json",
        "run.json",
        "figures/0-1600/barcode01.png",
        "figures/0-1600/all_groups.png",
        "figures/1000-1600/barcode02.png",
    ):
        assert (out / name).exists(), name
    assert not (out / "figures" / "0-300").exists()  # too short: no figure
    record = json.loads((out / "run.json").read_text())
    assert record["groups"] == ["barcode01", "barcode02"] and record["molecules"] == 2 * READS
    again = _invoke(*args, "--no-figures")
    assert "reused cached results" in again.output
    redrawn = _invoke(*args, "--coordinate-origin", 800, "--coordinate-reverse", "--ascending")
    assert "reused cached results" in redrawn.output  # figure options are not in the key
    changed = _invoke(*args, "--min-coverage", 0.5)
    assert "computed" in changed.output


def test_editing_the_label_table_invalidates_cached_results(project, tmp_path) -> None:
    """`F72`, for periodicity and context-bias alike."""
    from smftools.tools.read_periodicity import run_periodicity
    from smftools.tools.site_context_bias import load_or_count

    out = tmp_path / "out"
    plan = _labelled_plan(project, {1: "low", 2: "high"})
    first = run_periodicity(
        plan, "reads", out, project_dir=project, group_by="group", figures=False
    )
    assert first["groups"] == ["high", "low"] and not first["results_reused"]
    assert run_periodicity(
        plan, "reads", out, project_dir=project, group_by="group", figures=False
    )["results_reused"]
    _, reused = load_or_count(
        plan, "reads", tmp_path / "counts", project_dir=project, group_by="group"
    )
    assert not reused

    plan = _labelled_plan(project, {1: "dose_a", 2: "dose_b"})  # same plan hash, new labels
    second = run_periodicity(
        plan, "reads", out, project_dir=project, group_by="group", figures=False
    )
    assert not second["results_reused"] and second["groups"] == ["dose_a", "dose_b"]
    counts, reused = load_or_count(
        plan, "reads", tmp_path / "counts", project_dir=project, group_by="group"
    )
    assert not reused and set(counts.sites["group"]) == {"dose_a", "dose_b"}


def test_cache_key_hashes_referenced_files(project) -> None:
    from smftools.tools.analysis_cache import cache_key, referenced_files

    plan = _labelled_plan(project, {1: "low", 2: "high"})
    files = referenced_files(plan, "reads", project)
    assert list(files) == ["labels.parquet"] and files["labels.parquet"] != "missing"
    assert referenced_files(plan, "reads", project / "elsewhere") == {"labels.parquet": "missing"}
    a = cache_key(plan, "reads", base_dir=project, parameters={"x": 1})
    assert a == cache_key(plan, "reads", base_dir=project, parameters={"x": 1})
    assert a != cache_key(plan, "reads", base_dir=project, parameters={"x": 2})
