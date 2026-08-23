"""Position matrices stream to disk instead of accumulating (`F32`).

`compute_positionwise_statistics` retained every barcode's P-by-P matrix in
`adata.uns` until sidecars were published. A matrix is 168 MiB at a 4,690 bp
locus, so 41 barcodes estimated ~41 GB and the budget guard refused the run
outright -- correctly, but it made full-width matrices unusable on a real
barcode panel.

Reducing threads would not have helped: `max_threads` parallelises *within* one
(sample, reference) cell, and the accumulation is what scaled.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from smftools.tools.partitioned_spatial import (
    _plot_position_matrix_sidecars,
    _position_matrix_estimated_bytes,
    _position_matrix_region_dir,
    _position_matrix_sink,
    _write_position_matrix_provenance,
)

pytestmark = pytest.mark.unit

REGION = {"reference": "6B6_top", "start": 0, "end": 4690}


def _matrix(n=4, label="bc1"):
    idx = [str(i) for i in range(n)]
    return pd.DataFrame(np.eye(n), index=idx, columns=idx)


# --- the sink -----------------------------------------------------------------


def test_sink_writes_each_matrix_immediately(tmp_path):
    sink, written = _position_matrix_sink(tmp_path, REGION)

    sink("pearson", ("bc1", "6B6_top"), _matrix(), 100)
    sink("pearson", ("bc2", "6B6_top"), _matrix(), 100)

    assert len(written) == 2
    for _method, _barcode, path in written:
        assert path.is_file(), "each matrix must reach disk as it completes"


def test_sink_records_method_and_barcode(tmp_path):
    sink, written = _position_matrix_sink(tmp_path, REGION)
    sink("spearman", ("bc7", "6B6_top"), _matrix(), 10)

    method, barcode, path = written[0]
    assert (method, barcode) == ("spearman", "bc7")
    assert "method=spearman" in str(path)


def test_sink_round_trips_the_matrix(tmp_path):
    """Streaming must not change the numbers."""
    sink, written = _position_matrix_sink(tmp_path, REGION)
    original = _matrix(5)
    sink("pearson", "bc1", original, 10)

    restored = pd.read_parquet(written[0][2])
    np.testing.assert_allclose(restored.to_numpy(dtype=float), original.to_numpy(dtype=float))


def test_sink_skips_empty_matrices(tmp_path):
    sink, written = _position_matrix_sink(tmp_path, REGION)
    sink("pearson", "bc1", pd.DataFrame(), 0)
    assert written == []


def test_sink_accepts_a_plain_label(tmp_path):
    sink, written = _position_matrix_sink(tmp_path, REGION)
    sink("pearson", "bc9", _matrix(), 10)
    assert written[0][1] == "bc9"


# --- the budget ---------------------------------------------------------------


def test_budget_no_longer_scales_with_barcode_count():
    """The whole point: one matrix resident, not one per barcode."""
    one = _position_matrix_estimated_bytes(4690, n_methods=1, n_barcodes=1)
    forty_one = _position_matrix_estimated_bytes(4690, n_methods=1, n_barcodes=41)

    assert forty_one == one * 41, "the estimator itself is still linear in barcodes"
    assert one / 1024**3 < 1.0, "a single 4,690 bp matrix must fit comfortably"


def test_single_matrix_estimate_matches_the_arithmetic():
    width = 4690
    expected = width * width * 8 * 3  # float64, result + two work arrays
    assert _position_matrix_estimated_bytes(width, n_methods=1, n_barcodes=1) == expected


# --- provenance ---------------------------------------------------------------


def test_provenance_records_the_selection_and_matrix_count(tmp_path):
    sink, written = _position_matrix_sink(tmp_path, REGION)
    sink("pearson", "bc1", _matrix(), 100)

    paths = _write_position_matrix_provenance(
        tmp_path,
        REGION,
        "key",
        written=written,
        provenance={"reads_used": 1000, "reads_available": 2740, "selection_seed": 0},
    )

    recorded = json.loads(paths[0].read_text())
    assert recorded["matrices"] == 1
    assert recorded["reads_used"] == 1000
    assert recorded["reads_available"] == 2740


def test_no_provenance_written_without_matrices(tmp_path):
    assert (
        _write_position_matrix_provenance(
            tmp_path, REGION, "key", written=[], provenance={"reads_used": 1}
        )
        == []
    )


# --- plotting from disk -------------------------------------------------------


def test_plots_render_from_streamed_paths(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    sink, written = _position_matrix_sink(tmp_path / "store", REGION)
    sink("pearson", ("bc1", "6B6_top"), _matrix(6), 100)
    sink("pearson", ("bc2", "6B6_top"), _matrix(6), 100)

    cfg = SimpleNamespace(correlation_matrix_types=["pearson"])
    plotted = _plot_position_matrix_sidecars(written, REGION, tmp_path / "plots", cfg)

    assert len(plotted) == 2
    assert {barcode for _path, barcode in plotted} == {"bc1", "bc2"}
    for path, _barcode in plotted:
        assert path.is_file()


def test_region_dir_is_stable(tmp_path):
    a = _position_matrix_region_dir(tmp_path, REGION)
    b = _position_matrix_region_dir(tmp_path, dict(REGION))
    assert a == b
    assert "reference=6B6_top" in str(a)
