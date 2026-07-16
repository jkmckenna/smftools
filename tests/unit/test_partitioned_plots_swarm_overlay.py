from __future__ import annotations

import numpy as np
import pandas as pd

from smftools.cli.stage_artifacts import prepare_analysis_plot_layout
from smftools.constants import REFERENCE_STRAND
from smftools.preprocessing.partitioned_plots import _barcode_distribution_plots


def test_barcode_distribution_plots_renders_swarm_overlay_without_crashing(tmp_path):
    rng = np.random.default_rng(0)
    # One barcode with more reads than swarm_max_points, to exercise the
    # per-box subsampling path, plus a couple of small barcodes.
    rows = []
    for barcode, n in (("bc1", 50), ("bc2", 5), ("bc3", 0)):
        for _ in range(n):
            rows.append(
                {
                    REFERENCE_STRAND: "ref_top",
                    "barcode": barcode,
                    "read_length": float(rng.integers(100, 500)),
                    "read_quality": float(rng.uniform(10, 30)),
                }
            )
    obs = pd.DataFrame(rows)
    layout = prepare_analysis_plot_layout(tmp_path / "preprocess_outputs", stage="preprocess")

    _barcode_distribution_plots(
        obs,
        "barcode",
        ["read_length", "read_quality"],
        layout,
        category="read_qc",
        plot_type="read_qc_metric_dashboard",
        swarm_max_points=10,
    )

    catalog = pd.read_parquet(layout.catalog)
    matching = catalog[catalog["plot_type"] == "read_qc_metric_dashboard"]
    assert len(matching) == 1
    plot_path = layout.root.parent / matching.iloc[0]["path"]
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0
