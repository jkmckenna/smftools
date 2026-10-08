"""Block-strip colours and extra strips in latent clustermaps."""

import matplotlib.axes
import numpy as np
import pytest
from matplotlib.colors import to_hex

from smftools.plotting.latent_plotting import plot_latent_ordered_clustermap

pytestmark = pytest.mark.unit


def _strips(monkeypatch):
    """Record the strips drawn: (colours in category order, vmin, vmax, codes)."""
    drawn = []
    original = matplotlib.axes.Axes.imshow

    def imshow(self, data, **kwargs):
        data = np.asarray(data)
        if data.ndim == 2 and data.shape[1] == 1:
            colors = [to_hex(c) for c in kwargs["cmap"].colors]
            drawn.append((colors, kwargs.get("vmin"), kwargs.get("vmax"), data[:, 0].tolist()))
        return original(self, data, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "imshow", imshow)
    return drawn


def _figure(tmp_path, **kwargs):
    labels = np.array(["2", "2", "5", "5", "5"])
    return plot_latent_ordered_clustermap(
        [{"name": "x", "matrix": np.arange(15).reshape(5, 3), "cmap": "viridis"}],
        row_order=np.arange(5),
        blocks=[("2", 0, 2), ("5", 2, 5)],
        labels=labels,
        save_path=tmp_path / "f.png",
        **kwargs,
    )


def test_given_cluster_colours_are_used_with_a_fixed_range(tmp_path, monkeypatch):
    drawn = _strips(monkeypatch)
    colors = {"1": "#111111", "2": "#222222", "5": "#555555"}  # "1" absent here
    _figure(tmp_path, cluster_colors=colors)
    (strip,) = drawn
    assert strip[0] == ["#111111", "#222222", "#555555"]
    assert (strip[1], strip[2]) == (-0.5, 2.5)  # the full range, not the codes present
    assert strip[3] == [1, 1, 2, 2, 2]


def test_extra_strips_are_drawn_with_their_colours(tmp_path, monkeypatch):
    drawn = _strips(monkeypatch)
    result = _figure(
        tmp_path,
        cluster_name="NDR state",
        extra_strips=[
            {
                "name": "leiden",
                "values": ["0", "3", "3", "0", "7"],
                "colors": {"0": "#aa0000", "3": "#00aa00", "7": "#0000aa", "9": "#999999"},
                "order": ["3", "0", "7", "9"],
            }
        ],
    )
    assert len(drawn) == 2 and result["n_molecules"] == 5
    leiden = drawn[1]
    assert leiden[0] == ["#aa0000", "#00aa00", "#0000aa", "#999999"]
    assert leiden[3] == [0, 1, 1, 0, 2] and leiden[2] == 3.5
    assert (tmp_path / "f.png").exists()


def test_default_colours_still_work(tmp_path, monkeypatch):
    drawn = _strips(monkeypatch)
    _figure(tmp_path)
    assert len(drawn) == 1 and drawn[0][1] == -0.5


def test_block_legend_is_optional(tmp_path, monkeypatch):
    import matplotlib.figure

    legends = []
    original = matplotlib.figure.Figure.legend

    def legend(self, *args, **kwargs):
        legends.append(kwargs.get("title"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "legend", legend)
    _figure(tmp_path)
    assert legends == []
    _figure(tmp_path, cluster_name="NDR state", cluster_legend=True)
    assert legends == ["NDR state"]
