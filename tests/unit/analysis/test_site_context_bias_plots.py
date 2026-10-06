"""SCB-02: site-context bias figures (smoke)."""

import numpy as np
import pandas as pd
import pytest

from smftools.analysis.compute.site_context_bias import (
    group_differences,
    kmer_rates,
    offset_enrichment,
)
from smftools.analysis.plot.site_context_bias import (
    plot_enrichment_logo,
    plot_group_differences,
    plot_kmer_rates,
    plot_offset_enrichment_heatmap,
)


@pytest.fixture
def sites():
    rng = np.random.default_rng(0)
    rows = []
    for group in ("enzyme_a", "enzyme_b", "enzyme_c"):
        for position in range(60):
            context = (
                "".join(rng.choice(list("ACGT"), 3)) + "C" + "".join(rng.choice(list("ACGT"), 3))
            )
            if position == 0:
                context = "NN" + context[2:]  # reference-end padding
            observed = int(rng.integers(20, 100))
            rate = 0.8 if context[4] == "T" and group == "enzyme_a" else 0.3
            rows.append(
                {
                    "group": group,
                    "physical_reference": "ref_top",
                    "position": position,
                    "observed": observed,
                    "modified": int(rng.binomial(observed, rate)),
                    "context": context,
                }
            )
    return pd.DataFrame(rows)


def _written(path):
    assert path.exists() and path.stat().st_size > 0


def test_enrichment_figures(sites, tmp_path):
    table = offset_enrichment(sites, flank=3)
    plot_offset_enrichment_heatmap(table, tmp_path / "heatmap.png", title="t")
    plot_enrichment_logo(table, tmp_path / "logo.png", groups=["enzyme_b", "enzyme_a"])
    plot_group_differences(
        group_differences(table, reference_group="enzyme_b"), tmp_path / "differences.png"
    )
    for name in ("heatmap", "logo", "differences"):
        _written(tmp_path / f"{name}.png")


def test_kmer_figure_limits_rows(sites, tmp_path):
    rates = kmer_rates(sites, flank=3, k=5)
    plot_kmer_rates(rates, tmp_path / "kmers.png", max_kmers=10)
    _written(tmp_path / "kmers.png")


def test_unknown_group_is_an_error(sites, tmp_path):
    with pytest.raises(KeyError, match="groups not in table"):
        plot_offset_enrichment_heatmap(
            offset_enrichment(sites, flank=3), tmp_path / "x.png", groups=["missing"]
        )


def test_a_single_group_with_no_signal(tmp_path):
    flat = pd.DataFrame(
        {
            "group": "g",
            "physical_reference": "ref_top",
            "position": [0, 1],
            "observed": [10, 10],
            "modified": [5, 5],
            "context": ["ACA", "ACA"],
        }
    )
    table = offset_enrichment(flat, flank=1)
    plot_offset_enrichment_heatmap(table, tmp_path / "h.png")
    plot_enrichment_logo(table, tmp_path / "l.png")
    _written(tmp_path / "h.png")
    _written(tmp_path / "l.png")


def test_logo_grid_layout(sites, tmp_path):
    table = offset_enrichment(sites, flank=3)
    layout = [["enzyme_a", None], [None, "enzyme_b"], ["enzyme_c", "enzyme_a"]]
    plot_enrichment_logo(
        table,
        tmp_path / "grid.png",
        layout=layout,
        row_labels=["a", "b", "c"],
        col_labels=["low", "high"],
    )
    _written(tmp_path / "grid.png")
    plot_enrichment_logo(
        table, tmp_path / "ragged.png", layout=[["enzyme_a"], ["enzyme_b", "enzyme_c"]]
    )
    _written(tmp_path / "ragged.png")
    with pytest.raises(ValueError, match="row_labels"):
        plot_enrichment_logo(table, tmp_path / "x.png", layout=layout, row_labels=["a"])
    with pytest.raises(KeyError, match="groups not in table"):
        plot_enrichment_logo(table, tmp_path / "x.png", layout=[["missing"]])


def test_kmer_series_single_and_ranged(sites):
    from smftools.analysis.plot.site_context_bias import _series_values

    rates = kmer_rates(sites, flank=3, k=3)
    one = _series_values(rates, ["enzyme_a"], "absolute")
    a = rates[rates["group"] == "enzyme_a"].set_index("kmer")
    pd.testing.assert_series_equal(one["low"], a["rate_low"], check_names=False)
    both = _series_values(rates, ["enzyme_a", "enzyme_b"], "relative")
    table = rates.pivot_table(index="kmer", columns="group", values="log2_relative_rate")
    kmer = both.index[0]
    assert both.loc[kmer, "low"] == pytest.approx(table.loc[kmer, ["enzyme_a", "enzyme_b"]].min())
    assert both.loc[kmer, "high"] == pytest.approx(table.loc[kmer, ["enzyme_a", "enzyme_b"]].max())


def test_kmer_series_figures(sites, tmp_path):
    from smftools.analysis.plot.site_context_bias import plot_kmer_rate_series

    rates = kmer_rates(sites, flank=3, k=3)
    by_enzyme = {
        "A": [{"label": "low", "groups": ["enzyme_a"], "color": "#90CAF9"}],
        "B": [
            {"label": "low", "groups": ["enzyme_b"], "color": "#A5D6A7"},
            {"label": "high", "groups": ["enzyme_c"], "color": "#2E7D32"},
        ],
    }
    merged = {
        "all": [
            {"label": "A", "groups": ["enzyme_a"], "color": "#1565C0"},
            {"label": "B", "groups": ["enzyme_b", "enzyme_c"], "color": "#2E7D32"},
        ]
    }
    for scale in ("absolute", "relative"):
        plot_kmer_rate_series(rates, tmp_path / f"d_{scale}.png", panels=by_enzyme, scale=scale)
        plot_kmer_rate_series(
            rates, tmp_path / f"m_{scale}.png", panels=merged, scale=scale, sort="spread"
        )
        _written(tmp_path / f"d_{scale}.png")
        _written(tmp_path / f"m_{scale}.png")
    with pytest.raises(ValueError, match="scale"):
        plot_kmer_rate_series(rates, tmp_path / "x.png", panels=merged, scale="log")
    with pytest.raises(KeyError):
        plot_kmer_rate_series(
            rates,
            tmp_path / "x.png",
            panels={"p": [{"label": "x", "groups": ["missing"], "color": "k"}]},
        )
