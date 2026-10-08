"""MLR-04: attribution clustermaps -- one row order for inputs, attributions and strips."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smftools.analysis.plot import ml_results
from smftools.analysis.plot.ml_results import (
    attribution_row_layout,
    plot_attribution_clustermap,
)

pytestmark = pytest.mark.unit

N, POSITIONS = 60, 12


def _molecules(n: int = N) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    truth = np.where(np.arange(n) % 3 == 0, "active", "inactive")
    return pd.DataFrame(
        {
            "molecule_uid": [f"m{i:03d}" for i in range(n)],
            "fold": np.where(np.arange(n) < n // 2, "fold_a", "fold_b"),
            "truth": truth,
            "score": rng.random(n),
        }
    )


def _tagged(n: int = N) -> tuple[np.ndarray, np.ndarray]:
    """Inputs and attributions whose first position holds the molecule index."""
    rng = np.random.default_rng(1)
    attributions = rng.normal(size=(n, 1, POSITIONS))
    inputs = (rng.random((n, 1, POSITIONS)) < 0.5).astype(float)
    attributions[:, 0, 0] = np.arange(n)
    inputs[:, 0, 0] = np.arange(n)
    return inputs, attributions


@pytest.fixture
def captured(monkeypatch):
    calls = {}

    def capture(panels, **kwargs):
        calls.update(panels=panels, **kwargs)
        return {"output_path": None}

    monkeypatch.setattr("smftools.plotting.latent_plotting.plot_latent_ordered_clustermap", capture)
    return calls


def test_label_order_puts_the_positive_class_first_in_contiguous_blocks() -> None:
    molecules = _molecules()
    _inputs, attributions = _tagged()
    row_order, blocks, labels = attribution_row_layout(
        molecules, attributions, order="label", positive_class="active"
    )
    assert sorted(row_order) == list(range(N))
    assert [block[0] for block in blocks] == ["active", "inactive"]
    for label, start, stop in blocks:
        assert set(labels[row_order[start:stop]]) == {label}


def test_score_order_is_highest_first() -> None:
    molecules = _molecules()
    row_order, blocks, _labels = attribution_row_layout(molecules, _tagged()[1], order="score")
    scores = molecules["score"].to_numpy()[row_order]
    assert np.all(np.diff(scores) <= 0) and len(blocks) == 1


def test_bins_follow_the_given_order() -> None:
    molecules = _molecules()
    bins = np.where(np.arange(N) % 2 == 0, "both open", "neither open")
    row_order, blocks, labels = attribution_row_layout(
        molecules, _tagged()[1], order="bins", bins=bins, bin_order=["neither open", "both open"]
    )
    assert [block[0] for block in blocks] == ["neither open", "both open"]
    assert set(labels[row_order[: blocks[0][2]]]) == {"neither open"}


def test_every_panel_and_strip_shares_one_row_order(captured) -> None:
    molecules = _molecules()
    inputs, attributions = _tagged()
    result = plot_attribution_clustermap(
        molecules,
        attributions,
        inputs=inputs,
        channels=["C"],
        coordinates=list(range(POSITIONS)),
        positive_class="active",
        extra_panels=[{"name": "hmm", "matrix": inputs[:, 0] * 2, "cmap": "Greys"}],
        extra_strips=[{"name": "experiment", "values": molecules["fold"].str.upper()}],
    )
    order = captured["row_order"]
    panels = {panel["name"]: panel["matrix"][order] for panel in captured["panels"]}
    drawn = panels["C attribution"][:, 0]
    np.testing.assert_array_equal(panels["C (input)"][:, 0], drawn)
    np.testing.assert_array_equal(panels["hmm"][:, 0], 2 * drawn)
    # The returned row identities are the drawn molecules, in order.
    assert result["row_uids"] == [f"m{int(i):03d}" for i in drawn]
    strips = {
        strip["name"]: np.asarray(strip["values"])[order] for strip in captured["extra_strips"]
    }
    expected = molecules.set_index("molecule_uid").loc[result["row_uids"]]
    assert list(strips["held out"]) == list(expected["fold"])
    assert list(strips["experiment"]) == list(expected["fold"].str.upper())
    np.testing.assert_allclose(strips["score"].astype(float), expected["score"])
    assert np.asarray(captured["labels"])[order].tolist() == list(expected["truth"])
    assert captured["cluster_colors"]["active"] == ml_results._POSITIVE_COLOR


def test_attribution_colour_scale_is_symmetric_about_zero(captured) -> None:
    molecules = _molecules()
    _inputs, attributions = _tagged()
    result = plot_attribution_clustermap(
        molecules, attributions, channels=["C"], coordinates=list(range(POSITIONS))
    )
    (panel,) = captured["panels"]
    assert panel["vmin"] == -panel["vmax"] == -result["attribution_limit"]
    assert result["attribution_limit"] == pytest.approx(np.percentile(np.abs(attributions), 99))
    assert captured["extra_strips"][-1]["kind"] == "continuous"


def test_rows_are_sampled_by_class_and_seeded(captured) -> None:
    molecules = _molecules(300)
    inputs, attributions = _tagged(300)
    first = plot_attribution_clustermap(
        molecules, attributions, channels=["C"], coordinates=list(range(POSITIONS)), max_rows=30
    )
    second = plot_attribution_clustermap(
        molecules, attributions, channels=["C"], coordinates=list(range(POSITIONS)), max_rows=30
    )
    assert first["row_uids"] == second["row_uids"] and len(first["row_uids"]) == 30
    drawn = molecules.set_index("molecule_uid").loc[first["row_uids"], "truth"]
    assert (drawn == "active").sum() == 10


def test_shapes_are_checked() -> None:
    molecules = _molecules()
    with pytest.raises(ValueError, match="molecules x channels x positions"):
        plot_attribution_clustermap(
            molecules, np.zeros((N - 1, 1, POSITIONS)), channels=["C"], coordinates=[]
        )
    with pytest.raises(ValueError, match="one bin value per molecule"):
        attribution_row_layout(molecules, np.zeros((N, 1, POSITIONS)), order="bins")


def test_the_figure_renders_with_a_continuous_strip(tmp_path) -> None:
    molecules = _molecules()
    inputs, attributions = _tagged()
    path = tmp_path / "figure.png"
    result = plot_attribution_clustermap(
        molecules,
        attributions,
        inputs=inputs,
        channels=["C"],
        coordinates=list(range(POSITIONS)),
        coordinate_labels=[f"{p - 6:+d}" for p in range(POSITIONS)],
        output_path=path,
    )
    assert path.is_file() and result["output_path"] == str(path)
    assert result["panels"] == ["C (input)", "C attribution"]
    assert ml_results.ATTRIBUTION_ORDERS == ("label", "score", "bins")


def test_columns_follow_numeric_labels_with_breaks_between_windows(captured) -> None:
    # Two windows (10-12 and 50-51) labelled TSS-relative, decreasing with the coordinate.
    coordinates = [10, 11, 12, 50, 51]
    labels = [100 - c for c in coordinates]  # 90, 89, 88, 50, 49
    n = 6
    molecules = _molecules(n)
    attributions = np.tile(np.asarray(coordinates, dtype=float), (n, 1, 1))
    result = plot_attribution_clustermap(
        molecules,
        attributions,
        inputs=attributions.copy(),
        channels=["C"],
        coordinates=coordinates,
        coordinate_labels=labels,
    )
    assert result["column_coordinates"] == [51, 50, 12, 11, 10]  # labels ascending
    assert result["column_separators"] == [2]  # the 50 -> 12 jump
    for panel in captured["panels"]:
        assert panel["positions"] == [49, 50, 88, 89, 90]
        assert panel["matrix"][0].tolist() == [51, 50, 12, 11, 10]
        assert panel["column_separators"] == [2]
