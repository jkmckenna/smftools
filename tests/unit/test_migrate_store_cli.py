import types

import anndata as ad
import numpy as np
import pandas as pd
from click.testing import CliRunner

from smftools import cli_entry
from smftools.cli.helpers import get_adata_paths
from smftools.constants import LOAD_DIR


def _cfg(tmp_path):
    return types.SimpleNamespace(
        output_directory=str(tmp_path),
        experiment_name="EXP",
        smf_modality="conversion",
        from_adata_stage=None,
    )


def _synth(n=60, n_pos=20):
    rng = np.random.default_rng(0)
    x = rng.integers(0, 2, (n, n_pos)).astype(np.float32)
    obs = pd.DataFrame(
        {
            "Reference_strand": pd.Categorical(np.repeat(["r1_top", "r2_top"], n // 2)),
            "Sample": pd.Categorical(np.tile(["bc01", "bc02"], n // 2)),
        },
        index=[f"read{i:04d}" for i in range(n)],
    )
    a = ad.AnnData(X=x, obs=obs, layers={"read_span_mask": rng.integers(0, 2, (n, n_pos), np.int8)})
    a.var_names = [str(i) for i in range(n_pos)]
    return a


def test_get_adata_paths_includes_store_artifacts(tmp_path):
    p = get_adata_paths(_cfg(tmp_path))
    load_dir = tmp_path / LOAD_DIR
    assert p.store == load_dir / "store"
    assert p.spine == load_dir / "spine.h5ad"
    assert p.catalog == load_dir / "catalog.parquet"
    # store artifacts sit beside the monolithic raw h5ad's load directory
    assert p.raw.parent.parent == load_dir


def test_migrate_store_command_builds_store(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr("smftools.cli.helpers.load_experiment_config", lambda _p: cfg)

    paths = get_adata_paths(cfg)
    paths.raw.parent.mkdir(parents=True, exist_ok=True)
    _synth().write_h5ad(paths.raw)

    config_file = tmp_path / "config.csv"
    config_file.write_text("placeholder\n")  # only needs to exist for click.Path

    res = CliRunner().invoke(cli_entry.cli, ["migrate-store", str(config_file)])
    assert res.exit_code == 0, res.output

    assert paths.store.is_dir()
    assert paths.spine.exists()
    assert paths.catalog.exists()
    cat = pd.read_parquet(paths.catalog)
    # 2 references x 2 samples = 4 partitions
    assert len(cat) == 4
    assert cat["n_reads"].sum() == 60
