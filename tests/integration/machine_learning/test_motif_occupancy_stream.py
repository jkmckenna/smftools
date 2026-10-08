"""MOT-04: per-molecule motif occupancy streamed from a plan dataset."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from smftools.analysis.compute.motif_occupancy import STATES, OccupancyRules
from smftools.machine_learning.plan import parse_ml_plan
from smftools.tools.motif_occupancy import compute_occupancy, run_motif_occupancy

from .test_motif_tracks_stream import _calls
from .test_read_periodicity_stream import C_SITES, READS, project  # noqa: F401

pytestmark = pytest.mark.integration
SITES = np.flatnonzero(C_SITES)[[10, 40, 200]]  # three C sites
RULES = OccupancyRules(flank=0, min_sites=1, min_cover=0.5)
ROLES = {"accessible": ["C"]}


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
                    "channels": [channel],
                }
            },
            "splits": {},
            "models": {},
            "jobs": {},
        }
    )


def _hits(sites=SITES):
    # One-base "motifs" at C sites: a read spans one when it has a call there.
    return pd.DataFrame(
        {
            "motif_id": [f"M{i}" for i in range(len(sites))],
            "motif_alt_id": "",
            "motif_name": [f"m{i}" for i in range(len(sites))],
            "family": "fam",
            "reference": "locus",
            "start": sites,
            "end": sites + 1,
            "motif_strand": "+",
            "score": 1.0,
            "pvalue": 1e-6,
            "matched_sequence": "C",
        }
    )


def test_states_equal_the_calls_at_each_site(project) -> None:  # noqa: F811
    result = compute_occupancy(
        _plan(), "reads", _hits(), roles=ROLES, sites="C", project_dir=project, rules=RULES
    )
    # States follow the dataset's molecule order; the fixture's reads are r0..r15.
    order = [int(read_id[1:]) for read_id in result["reads"]["read_id"]]
    calls = _calls()[order][:, SITES]
    expected = np.where(
        np.isnan(calls),
        STATES.index("uninformative"),
        np.where(calls > 0, STATES.index("accessible"), STATES.index("other")),
    )
    np.testing.assert_array_equal(result["states"], expected)
    assert result["instances"]["motif_id"].tolist() == ["M0", "M1", "M2"]


def test_workers_give_identical_states(project) -> None:  # noqa: F811
    kwargs = dict(roles=ROLES, sites="C", project_dir=project, rules=RULES, group_by="Barcode")
    one = compute_occupancy(_plan(), "reads", _hits(), **kwargs)
    two = compute_occupancy(_plan(), "reads", _hits(), workers=2, **kwargs)
    np.testing.assert_array_equal(one["states"], two["states"])
    assert one["molecule_uids"] == two["molecule_uids"]
    pd.testing.assert_frame_equal(one["reads"], two["reads"])


def test_run_writes_tables_and_reuses_states(project, tmp_path) -> None:  # noqa: F811
    out = tmp_path / "occ"
    kwargs = dict(roles=ROLES, sites="C", project_dir=project, rules=RULES, group_by=["Barcode"])
    record = run_motif_occupancy(_plan(), "reads", out, _hits(), **kwargs)
    assert not record["states_reused"] and record["instances"] == 3
    assert record["molecules"] == 2 * READS
    occupancy = pd.read_parquet(out / "occupancy.parquet")
    assert set(occupancy["group"]) == {"barcode01", "barcode02"}
    row = occupancy[(occupancy.group == "barcode01") & (occupancy.instance == 0)].iloc[0]
    calls = _calls()[:READS, SITES[0]]
    assert row.n_informative == (~np.isnan(calls)).sum()
    assert row.frac_accessible == pytest.approx((calls > 0).sum() / (~np.isnan(calls)).sum())
    again = run_motif_occupancy(_plan(), "reads", out, _hits(), **kwargs)
    assert again["states_reused"]
    changed = run_motif_occupancy(_plan(), "reads", out, _hits(SITES[:2]), **kwargs)
    assert not changed["states_reused"] and changed["instances"] == 2  # instances in the key
    assert json.loads((out / "run.json").read_text())["rules"]["min_sites"] == 1
    with pytest.raises(KeyError, match="not in dataset channels"):
        run_motif_occupancy(
            _plan(),
            "reads",
            tmp_path / "x",
            _hits(),
            roles={"tf_bound": ["nope"]},
            sites="C",
            project_dir=project,
        )
    with pytest.raises(KeyError, match="no motif instances"):
        run_motif_occupancy(
            _plan(),
            "reads",
            tmp_path / "y",
            _hits().assign(reference="z"),
            roles=ROLES,
            sites="C",
            project_dir=project,
        )
