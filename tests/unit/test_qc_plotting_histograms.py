from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")

from smftools.plotting.qc_plotting import plot_read_qc_histograms


def test_plot_read_qc_histograms_numeric_stable(tmp_path):
    n = 300
    obs = pd.DataFrame(
        {
            "Experiment_name_and_barcode": pd.Categorical(["NB01"] * (n // 2) + ["NB02"] * (n // 2)),
            "read_length": np.concatenate(
                [
                    np.random.randint(100, 5000, size=n // 2).astype(float),
                    np.full(n // 2, 1200.0),
                ]
            ),
            "mapping_quality": np.concatenate(
                [
                    np.linspace(0, 60, n // 2),
                    np.full(n // 2, 60.0),
                ]
            ),
        }
    )
    # Include non-finite values to exercise sanitization path.
    obs.loc[0, "read_length"] = np.nan
    obs.loc[1, "mapping_quality"] = np.inf

    adata = SimpleNamespace(obs=obs)
    outdir = tmp_path / "qc_plots"

    plot_read_qc_histograms(
        adata=adata,
        outdir=outdir,
        obs_keys=["read_length", "mapping_quality"],
        sample_key="Experiment_name_and_barcode",
        bins=60,
    )

    outputs = list(outdir.glob("qc_grid_*.png"))
    assert outputs, "Expected QC histogram PNG output."
