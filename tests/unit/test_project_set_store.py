import pandas as pd
import pytest

from smftools.informatics.raw_store import write_raw_store
from smftools.informatics.reference_identity import reference_uid
from smftools.project.registry import add_experiment, init_project
from smftools.project.set_store import materialize_set, set_cache_dir
from smftools.readwrite import safe_read_h5ad, safe_write_h5ad

SEQUENCE = "ACGTACGTACGT"


def _make_raw_experiment(out_dir, *, reference_strand, uid, npos=12, n=4, sample="bc01"):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "read_id": f"{reference_strand}_r{i}",
            "reference": reference_strand.rsplit("_", 1)[0],
            "Reference_strand": reference_strand,
            "sample": sample,
            "barcode": sample,
            "strand": "top",
            "mapping_direction": "fwd",
            "reference_start": 0,
            "cigar": f"{npos}M",
            "aligned_length": npos,
            "sequence": [i % 4 for _ in range(npos)],
            "quality": [30] * npos,
            "mismatch": [4] * npos,
            "modification_signal": [float(i % 2)] * npos,
        }
        for i in range(n)
    ]
    write_raw_store(
        pd.DataFrame(rows),
        out_dir,
        reference_lengths={reference_strand: npos},
        extra_uns={
            "reference_uids": {reference_strand: uid},
            "modality": "direct",
            "experiment": out_dir.name,
        },
    )
    return out_dir


def _make_project(tmp_path, n_expA=4, n_expB=3):
    uid = reference_uid(SEQUENCE, 12)
    _make_raw_experiment(tmp_path / "expA", reference_strand="geneA_top", uid=uid, n=n_expA)
    _make_raw_experiment(tmp_path / "expB", reference_strand="geneB_top", uid=uid, n=n_expB)
    proj = tmp_path / "project"
    init_project(proj)
    add_experiment(proj, tmp_path / "expA")
    add_experiment(proj, tmp_path / "expB")
    return proj, uid


def test_materialize_set_writes_cache_and_matches_project_adata(tmp_path):
    from smftools.project.catalog import project_adata

    proj, uid = _make_project(tmp_path)
    cached = materialize_set(proj, uid)
    direct = project_adata(proj, uid)

    assert cached.n_obs == direct.n_obs == 7
    cache_dir = set_cache_dir(proj, uid)
    assert (cache_dir / "base.h5ad").exists()
    assert (cache_dir / "composition.json").exists()


def test_materialize_set_second_call_hits_cache(tmp_path):
    proj, uid = _make_project(tmp_path)
    first = materialize_set(proj, uid)
    assert first.n_obs == 7

    # Tamper with the cached file directly -- a second call that re-materialized
    # would overwrite this; a cache hit returns it as-is.
    cache_dir = set_cache_dir(proj, uid)
    cached, _ = safe_read_h5ad(cache_dir / "base.h5ad", verbose=False)
    cached.obs["marker"] = "from_cache"
    safe_write_h5ad(cached, cache_dir / "base.h5ad", backup=False, verbose=False)

    second = materialize_set(proj, uid)
    assert "marker" in second.obs.columns
    assert (second.obs["marker"] == "from_cache").all()


def test_materialize_set_force_recompute_bypasses_cache(tmp_path):
    proj, uid = _make_project(tmp_path)
    materialize_set(proj, uid)

    cache_dir = set_cache_dir(proj, uid)
    cached, _ = safe_read_h5ad(cache_dir / "base.h5ad", verbose=False)
    cached.obs["marker"] = "stale"
    safe_write_h5ad(cached, cache_dir / "base.h5ad", backup=False, verbose=False)

    fresh = materialize_set(proj, uid, force_recompute=True)
    assert "marker" not in fresh.obs.columns


def test_materialize_set_cache_invalidates_when_new_experiment_registered(tmp_path):
    proj, uid = _make_project(tmp_path)
    first = materialize_set(proj, uid)
    assert first.n_obs == 7
    first_cache_dir = set_cache_dir(proj, uid)

    _make_raw_experiment(tmp_path / "expC", reference_strand="geneC_top", uid=uid, n=2)
    add_experiment(proj, tmp_path / "expC")

    second_cache_dir = set_cache_dir(proj, uid)
    assert second_cache_dir != first_cache_dir

    second = materialize_set(proj, uid)
    assert second.n_obs == 9
    assert set(second.obs["experiment"]) == {"expA", "expB", "expC"}
    # The old cache is left behind (no GC yet), but is no longer what gets served.
    assert first_cache_dir.exists()


def test_materialize_set_raises_for_unmatched_reference(tmp_path):
    proj, _ = _make_project(tmp_path)
    with pytest.raises(ValueError):
        materialize_set(proj, "does_not_exist")


def test_set_cache_dir_is_cheap_and_does_not_create_anything(tmp_path):
    proj, uid = _make_project(tmp_path)
    cache_dir = set_cache_dir(proj, uid)
    assert not cache_dir.exists()
