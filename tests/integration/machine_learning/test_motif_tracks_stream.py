"""MOT-03: bulk class tracks streamed from a plan dataset, with motif lanes."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from smftools.machine_learning.plan import parse_ml_plan
from smftools.tools.motif_tracks import compute_tracks, run_motif_tracks

from .test_read_periodicity_stream import C_SITES, LENGTH, PERIOD, READS, project  # noqa: F401

pytestmark = pytest.mark.integration


def _plan():
    channel = {
        "name": "C",
        "biological_role": "accessibility",
        "sources": [
            {"modality": "deaminase", "stage": "preprocess", "layer": "X", "site_context": "C"}
        ],
    }
    return parse_ml_plan(
        {
            "schema_version": 1,
            "scope": {"kind": "project"},
            "datasets": {
                "reads": {
                    "modalities": ["deaminase"],
                    "references": ["locus"],
                    "channels": [channel, {**channel, "name": "C_again"}],
                }
            },
            "splits": {},
            "models": {},
            "jobs": {},
        }
    )


def _calls():
    """The fixture's calls (same seed and construction)."""
    rng = np.random.default_rng(1)
    probability = 0.5 + 0.45 * np.sin(2 * np.pi * np.arange(LENGTH) / PERIOD)
    calls = (rng.random((2 * READS, LENGTH)) < probability).astype(np.float32)
    calls[:, ~C_SITES] = np.nan
    return calls


def test_tracks_equal_a_direct_count_per_group(project) -> None:  # noqa: F811
    tracks, frame, molecules = compute_tracks(
        _plan(), "reads", project_dir=project, group_by="Barcode"
    )
    assert frame == "locus" and molecules == 2 * READS
    calls = _calls()
    for g_index, group in enumerate(["barcode01", "barcode02"]):
        rows = calls[g_index * READS : (g_index + 1) * READS]
        got = tracks[(tracks.group == group) & (tracks.track == "C")].set_index("position")
        sites = np.flatnonzero(C_SITES)
        np.testing.assert_array_equal(
            got.loc[sites, "spanning"], (~np.isnan(rows[:, sites])).sum(0)
        )
        np.testing.assert_array_equal(got.loc[sites, "in_class"], (rows[:, sites] > 0).sum(0))
    assert set(tracks["track"]) == {"C", "C_again"}


def test_workers_give_identical_tracks(project) -> None:  # noqa: F811
    one, _, _ = compute_tracks(_plan(), "reads", project_dir=project, group_by="Barcode")
    two, _, _ = compute_tracks(_plan(), "reads", project_dir=project, group_by="Barcode", workers=2)
    pd.testing.assert_frame_equal(one, two)


def test_run_writes_tables_figures_and_reuses_counts(project, tmp_path) -> None:  # noqa: F811
    hits = pd.DataFrame(
        {
            "motif_id": ["M1", "M2", "M3"],
            "motif_alt_id": "",
            "motif_name": ["one", "two", "three"],
            "family": ["Sox", "bZIP", "Sox"],
            "reference": ["locus", "locus", "elsewhere"],
            "start": [100, 400, 10],
            "end": [112, 410, 20],
            "motif_strand": ["+", "-", "+"],
            "score": 10.0,
            "pvalue": [1e-6, 5e-5, 1e-6],
            "matched_sequence": "",
        }
    )
    out = tmp_path / "tracks"
    record = run_motif_tracks(
        _plan(),
        "reads",
        out,
        hits,
        project_dir=project,
        group_by=["Barcode"],
        channels=["C"],
        regions={"zoom": (50, 500)},
        min_spanning=1,
    )
    assert record["motif_instances"] == 2 and not record["tracks_reused"]
    assert sorted(record["figures"]) == [
        "figures/Barcode/span_groups.png",
        "figures/Barcode/zoom_groups.png",
    ]
    contrast = pd.read_parquet(out / "motif_contrast.parquet")
    assert set(contrast["motif_id"]) == {"M1", "M2"} and set(contrast["group"]) == {
        "barcode01",
        "barcode02",
    }
    again = run_motif_tracks(
        _plan(),
        "reads",
        out,
        hits,
        project_dir=project,
        group_by=["Barcode"],
        channels=["C"],
        max_pvalue=1e-5,
        layout="tracks",
        min_spanning=1,
    )
    assert again["tracks_reused"] and again["motif_instances"] == 1
    assert json.loads((out / "run.json").read_text())["filters"]["max_pvalue"] == 1e-5
    with pytest.raises(KeyError, match="no motif instances"):
        run_motif_tracks(
            _plan(),
            "reads",
            tmp_path / "x",
            hits.assign(reference="nope"),
            project_dir=project,
            figures=False,
        )
