import anndata as ad
import numpy as np
import pandas as pd

from smftools.plotting.plotting_utils import subsample_reads_for_plot


def _adata(n_obs: int) -> ad.AnnData:
    x = np.arange(n_obs * 3, dtype=float).reshape(n_obs, 3)
    obs = pd.DataFrame({"read_id": [f"r{i}" for i in range(n_obs)]}).set_index("read_id")
    return ad.AnnData(X=x, obs=obs)


def test_returns_unchanged_when_at_or_under_cap():
    a = _adata(100)
    assert subsample_reads_for_plot(a, 100) is a
    assert subsample_reads_for_plot(a, 200) is a


def test_disabled_when_max_none_or_nonpositive():
    a = _adata(100)
    assert subsample_reads_for_plot(a, None) is a
    assert subsample_reads_for_plot(a, 0) is a
    assert subsample_reads_for_plot(a, -5) is a


def test_subsamples_to_exact_cap_and_preserves_order():
    a = _adata(1000)
    out = subsample_reads_for_plot(a, 250)
    assert out.n_obs == 250
    # kept rows are a subset, in original order (indices strictly increasing)
    kept = [int(name[1:]) for name in out.obs_names]
    assert kept == sorted(kept)
    assert set(kept).issubset(set(range(1000)))
    # X rows still match their original obs (row i of X == i*3 .. i*3+2)
    for row, name in zip(out.X, out.obs_names):
        i = int(name[1:])
        np.testing.assert_array_equal(row, [i * 3, i * 3 + 1, i * 3 + 2])


def test_reproducible_across_calls():
    a = _adata(1000)
    first = list(subsample_reads_for_plot(a, 100).obs_names)
    second = list(subsample_reads_for_plot(a, 100).obs_names)
    assert first == second


def test_config_default_is_5000_and_int_override_applies():
    from smftools.config.experiment_config import ExperimentConfig

    cfg, _ = ExperimentConfig.from_var_dict({})
    assert cfg.clustermap_max_reads_per_plot == 5000
    cfg_set, _ = ExperimentConfig.from_var_dict({"clustermap_max_reads_per_plot": "1200"})
    assert cfg_set.clustermap_max_reads_per_plot == 1200
