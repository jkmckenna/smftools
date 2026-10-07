"""RPG-03: paired input / periodogram clustermaps."""

import numpy as np
import pytest

from smftools.analysis.compute.read_periodicity import period_grid, read_periodograms
from smftools.analysis.plot.read_periodicity import (
    periodicity_row_order,
    plot_read_periodicity_clustermap,
)


def test_row_order_by_peak_and_bins():
    peaks = np.array([200, np.nan, 150, 180, 170])
    assert periodicity_row_order(peaks).tolist() == [0, 3, 4, 2]  # largest first
    assert periodicity_row_order(peaks, descending=False).tolist() == [2, 4, 3, 0]
    bins = ["b", "a", "a", "b", "a"]
    assert periodicity_row_order(peaks, bins=bins).tolist() == [0, 3, 4, 2]  # first appearance
    assert periodicity_row_order(peaks, bins=bins, bin_order=["a", "b"]).tolist() == [4, 2, 0, 3]


@pytest.fixture
def scored():
    rng = np.random.default_rng(0)
    positions = np.arange(1500)
    reads = []
    for period in (160, 180, 200, 220):
        probability = 0.5 + 0.45 * np.sin(2 * np.pi * positions / period)
        reads.append((rng.random((5, positions.size)) < probability).astype(float))
    values = np.vstack(reads)
    observed = np.broadcast_to(positions % 3 == 0, values.shape).copy()
    values[~observed] = np.nan
    grid = period_grid(0, 1500)
    power, stats = read_periodograms(positions, values, observed, observed.copy(), grid)
    return positions, values, power, grid, stats["peak_period_bp"].to_numpy()


def _written(path):
    assert path.exists() and path.stat().st_size > 0


def test_binary_input_with_bins(scored, tmp_path):
    positions, values, power, grid, peaks = scored
    bins = ["low"] * 10 + ["high"] * 10
    peaks = peaks.copy()
    peaks[3] = np.nan  # one read without a peak
    plot_read_periodicity_clustermap(
        values,
        positions,
        power,
        grid.periods,
        peaks,
        tmp_path / "binned.png",
        bins=bins,
        bin_order=["high", "low"],
        peak_range=grid.peak_range,
        title="t",
    )
    _written(tmp_path / "binned.png")


def test_dense_input_explicit_order_and_subsampling(scored, tmp_path):
    positions, values, power, grid, peaks = scored
    lengths = np.where(np.isfinite(values), values * 120.0, np.nan)  # a continuous layer
    plot_read_periodicity_clustermap(
        lengths,
        positions,
        power,
        grid.periods,
        peaks,
        tmp_path / "ordered.png",
        order=np.arange(20)[::-1],
        max_reads=8,
        input_label="lengths",
    )
    _written(tmp_path / "ordered.png")


def test_no_scored_reads_writes_a_note(scored, tmp_path):
    positions, values, power, grid, peaks = scored
    plot_read_periodicity_clustermap(
        values, positions, power, grid.periods, np.full_like(peaks, np.nan), tmp_path / "empty.png"
    )
    _written(tmp_path / "empty.png")


def test_shape_errors(scored, tmp_path):
    positions, values, power, grid, peaks = scored
    with pytest.raises(ValueError, match="same rows"):
        plot_read_periodicity_clustermap(
            values[:5], positions, power, grid.periods, peaks, tmp_path / "x.png"
        )
    with pytest.raises(ValueError, match="one label per row"):
        plot_read_periodicity_clustermap(
            values, positions, power, grid.periods, peaks, tmp_path / "x.png", bins=["a"]
        )


def test_display_coordinates():
    from smftools.analysis.plot.read_periodicity import _display_coordinates

    positions = np.array([0, 10, 20])
    key, shown, label = _display_coordinates(positions, 15, True)
    assert shown.tolist() == [15, 5, -5] and np.argsort(key).tolist() == [2, 1, 0]
    assert "relative to 15" in label and "reversed" in label
    key, shown, label = _display_coordinates(positions, None, True)
    assert shown.tolist() == [0, 10, 20] and np.argsort(key).tolist() == [2, 1, 0]
    key, shown, _ = _display_coordinates(positions, 5, False)
    assert shown.tolist() == [-5, 5, 15] and np.argsort(key).tolist() == [0, 1, 2]


def test_tss_relative_figure_with_input_colorbar(scored, tmp_path):
    positions, values, power, grid, peaks = scored
    plot_read_periodicity_clustermap(
        values,
        positions,
        power,
        grid.periods,
        peaks,
        tmp_path / "tss.png",
        coordinate_origin=1000,
        coordinate_reverse=True,
        input_colorbar=True,
        descending=False,
    )
    _written(tmp_path / "tss.png")


def test_grid_with_an_empty_cell_and_errors(scored, tmp_path):
    from smftools.analysis.plot.read_periodicity import plot_read_periodicity_grid

    positions, values, power, grid, peaks = scored
    cells = {
        name: {
            "values": values[rows],
            "positions": positions,
            "power": power[rows],
            "periods": grid.periods,
            "peak_period": peaks[rows],
        }
        for name, rows in (("a", slice(0, 5)), ("b", slice(5, 10)), ("c", slice(10, 20)))
    }
    cells["none"] = {**cells["a"], "peak_period": np.full(5, np.nan)}
    layout = [["a", "b", None], ["c", "none", "a"]]
    plot_read_periodicity_grid(
        cells,
        layout,
        tmp_path / "grid.png",
        row_labels=["low", "high"],
        col_labels=["x", "y", "z"],
        coordinate_origin=750,
        coordinate_reverse=True,
        title="t",
    )
    _written(tmp_path / "grid.png")
    with pytest.raises(KeyError, match="not given"):
        plot_read_periodicity_grid(cells, [["missing"]], tmp_path / "x.png")
    with pytest.raises(ValueError, match="col_labels"):
        plot_read_periodicity_grid(cells, layout, tmp_path / "x.png", col_labels=["x"])


def test_mean_spectra_figure(scored, tmp_path):
    from smftools.analysis.plot.read_periodicity import plot_mean_spectra

    positions, values, power, grid, peaks = scored
    panels = {
        "NK": [
            {"label": "WT", "power": power[:10], "color": "#1565C0"},
            {"label": "enh-del", "power": power[10:], "color": "#C62828"},
        ],
        "B": [{"label": "WT", "power": power[:5], "color": "#1565C0"}],
    }
    plot_mean_spectra(
        panels,
        grid.periods,
        tmp_path / "spectra.png",
        layout=[["NK", "B"], ["B", None]],
        row_labels=["0 mM", "0.5 mM"],
        col_labels=["NK", "B"],
        peak_range=grid.peak_range,
        n_boot=50,
    )
    _written(tmp_path / "spectra.png")
