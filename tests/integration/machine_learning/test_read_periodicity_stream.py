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
