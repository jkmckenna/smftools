import anndata as ad
import numpy as np
import pandas as pd
import pytest

from smftools.informatics.partition_store import write_experiment_store
from smftools.project.registry import add_experiment, init_project
from smftools.project.sample_analysis import compute_periodicity, join_periodicity
from smftools.project.sample_store import backfill_per_sample_store
from smftools.readwrite import safe_write_h5ad

# Matches tests/unit/analysis/test_autocorrelation.py's proven synthetic periodic
# fixture -- reliably produces non-empty compute_single_molecule_periodicity_direct
# results with default coverage/LS parameters.


def _make_periodic_mat(n_reads=30, n_sites=300, period_bp=185.0, locus_len=4000, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    positions = np.sort(rng.choice(locus_len, size=n_sites, replace=False))
    prob = 0.5 + 0.4 * np.cos(2 * np.pi * positions / period_bp)
    mat = (rng.random((n_reads, n_sites)) < prob).astype(np.float32)
    return mat, positions


def _modern_periodic_spine(run_dir, *, reference_strand="ref0_top", sample="bc00", **kwargs):
    mat, positions = _make_periodic_mat(**kwargs)
    obs = pd.DataFrame(
        {"Reference_strand": [reference_strand] * mat.shape[0], "Sample": [sample] * mat.shape[0]},
        index=[f"read{i:04d}" for i in range(mat.shape[0])],
    )
    a = ad.AnnData(X=mat, obs=obs)
    a.var_names = [str(int(p)) for p in positions]
    write_experiment_store(a, run_dir, experiment="e", modality="conversion")
    return run_dir / "spine.h5ad"


def _register_modern(project_dir, experiment_id, spine_path):
    """Modern (pointer-kind) partitions resolve their spine through the registry at
    read time (see sample_analysis._load_partition_adata), so the registry has to
    know about the experiment too -- exactly like real usage, where per-sample store
    backfill only ever happens alongside project add's add_experiment() call."""
    init_project(project_dir)
    add_experiment(project_dir, spine_path.parent, experiment_id=experiment_id)


def _legacy_periodic_spine(path, *, reference_strand="ref0_top", sample="bc00", **kwargs):
    mat, positions = _make_periodic_mat(**kwargs)
    obs = pd.DataFrame(
        {"Reference_strand": [reference_strand] * mat.shape[0], "Sample": [sample] * mat.shape[0]},
        index=[f"read{i:04d}" for i in range(mat.shape[0])],
    )
    a = ad.AnnData(X=mat, obs=obs)
    a.var_names = [str(int(p)) for p in positions]
    safe_write_h5ad(a, path, backup=False, verbose=False)
    assert "is_spine" not in a.uns
    return path


def test_compute_periodicity_modern_experiment_returns_and_caches_result(tmp_path):
    project_dir = tmp_path / "project"
    spine_path = _modern_periodic_spine(tmp_path / "run")
    backfill_per_sample_store(project_dir, "expA", spine_path)
    _register_modern(project_dir, "expA", spine_path)

    result = compute_periodicity(project_dir, "expA", "ref0_top", "bc00")

    assert not result.empty
    assert result.index.name == "read_id"
    assert {"ls_nrl_bp", "ls_snr", "ls_peak_power", "ls_fwhm_bp", "n_sites"}.issubset(
        result.columns
    )
    assert "ls_freqs" not in result.columns and "ls_power" not in result.columns
    assert set(result.index).issubset({f"read{i:04d}" for i in range(30)})


def test_compute_periodicity_second_call_hits_cache(tmp_path):
    project_dir = tmp_path / "project"
    spine_path = _modern_periodic_spine(tmp_path / "run")
    backfill_per_sample_store(project_dir, "expA", spine_path)
    _register_modern(project_dir, "expA", spine_path)

    first = compute_periodicity(project_dir, "expA", "ref0_top", "bc00")
    assert not first.empty

    # Tamper with the cached parquet directly -- a second call that re-computed
    # would overwrite this.
    from smftools.project.sample_analysis import (
        _analysis_dir,
        _definition_hash,
        _periodicity_definition,
    )

    definition = _periodicity_definition(
        layer=None, start=None, end=None, method="direct", kwargs={}
    )
    analysis_dir = _analysis_dir(
        project_dir, "expA", "ref0_top", "bc00", "periodicity", _definition_hash(definition)
    )
    cached = pd.read_parquet(analysis_dir / "result.parquet")
    cached["ls_nrl_bp"] = -999.0
    cached.to_parquet(analysis_dir / "result.parquet")

    second = compute_periodicity(project_dir, "expA", "ref0_top", "bc00")
    assert (second["ls_nrl_bp"] == -999.0).all()


def test_compute_periodicity_force_recompute_bypasses_cache(tmp_path):
    project_dir = tmp_path / "project"
    spine_path = _modern_periodic_spine(tmp_path / "run")
    backfill_per_sample_store(project_dir, "expA", spine_path)
    _register_modern(project_dir, "expA", spine_path)
    compute_periodicity(project_dir, "expA", "ref0_top", "bc00")

    from smftools.project.sample_analysis import (
        _analysis_dir,
        _definition_hash,
        _periodicity_definition,
    )

    definition = _periodicity_definition(
        layer=None, start=None, end=None, method="direct", kwargs={}
    )
    analysis_dir = _analysis_dir(
        project_dir, "expA", "ref0_top", "bc00", "periodicity", _definition_hash(definition)
    )
    cached = pd.read_parquet(analysis_dir / "result.parquet")
    cached["ls_nrl_bp"] = -999.0
    cached.to_parquet(analysis_dir / "result.parquet")

    fresh = compute_periodicity(project_dir, "expA", "ref0_top", "bc00", force_recompute=True)
    assert not (fresh["ls_nrl_bp"] == -999.0).all()


def test_compute_periodicity_different_definitions_do_not_collide(tmp_path):
    project_dir = tmp_path / "project"
    spine_path = _modern_periodic_spine(tmp_path / "run")
    backfill_per_sample_store(project_dir, "expA", spine_path)
    _register_modern(project_dir, "expA", spine_path)

    whole = compute_periodicity(project_dir, "expA", "ref0_top", "bc00")
    windowed = compute_periodicity(project_dir, "expA", "ref0_top", "bc00", start=0, end=2000)

    assert not whole.equals(windowed)
    partition = (
        project_dir
        / "project_outputs"
        / "per_sample"
        / "expA"
        / "ref0_top"
        / "bc00"
        / "analyses"
        / "periodicity"
    )
    assert len(list(partition.iterdir())) == 2


def test_compute_periodicity_legacy_experiment_uses_cached_partition(tmp_path):
    project_dir = tmp_path / "project"
    spine_path = _legacy_periodic_spine(tmp_path / "run" / "legacy.h5ad")
    backfill_per_sample_store(project_dir, "expLegacy", spine_path)

    result = compute_periodicity(project_dir, "expLegacy", "ref0_top", "bc00")

    assert not result.empty


def test_compute_periodicity_missing_partition_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        compute_periodicity(tmp_path / "project", "nope", "ref0_top", "bc00")


def test_join_periodicity_attaches_columns_by_read_id(tmp_path):
    project_dir = tmp_path / "project"
    spine_path = _modern_periodic_spine(tmp_path / "run")
    backfill_per_sample_store(project_dir, "expA", spine_path)
    _register_modern(project_dir, "expA", spine_path)
    computed = compute_periodicity(project_dir, "expA", "ref0_top", "bc00")

    obs = pd.DataFrame(
        {
            "experiment": ["expA"] * 30,
            "Reference_strand": ["ref0_top"] * 30,
            "Sample": ["bc00"] * 30,
        },
        index=[f"read{i:04d}" for i in range(30)],
    )
    adata = ad.AnnData(X=np.zeros((30, 1)), obs=obs)

    joined = join_periodicity(adata, project_dir)

    assert "periodicity_ls_nrl_bp" in joined.obs.columns
    for read_id in computed.index:
        assert joined.obs.loc[read_id, "periodicity_ls_nrl_bp"] == pytest.approx(
            computed.loc[read_id, "ls_nrl_bp"]
        )
    # Reads that didn't survive the analysis's coverage filtering get NaN, not dropped.
    missing = set(adata.obs_names) - set(computed.index)
    if missing:
        assert joined.obs.loc[list(missing), "periodicity_ls_nrl_bp"].isna().all()


def test_join_periodicity_noop_without_required_obs_columns(tmp_path):
    adata = ad.AnnData(X=np.zeros((2, 1)), obs=pd.DataFrame(index=["a", "b"]))
    joined = join_periodicity(adata, tmp_path / "project")
    assert "periodicity_ls_nrl_bp" not in joined.obs.columns


def test_join_periodicity_returns_unchanged_when_nothing_cached(tmp_path):
    obs = pd.DataFrame(
        {"experiment": ["expA"], "Reference_strand": ["ref0_top"], "Sample": ["bc00"]}, index=["r0"]
    )
    adata = ad.AnnData(X=np.zeros((1, 1)), obs=obs)
    joined = join_periodicity(adata, tmp_path / "project")
    assert list(joined.obs.columns) == ["experiment", "Reference_strand", "Sample"]
