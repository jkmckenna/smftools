from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from smftools.plotting.qc_plotting import plot_reference_barcode_chimera_rate


def _row(reference, barcode, ct, ga, purity):
    return {
        "Reference_strand": reference,
        "barcode": barcode,
        "ct_event_count": ct,
        "ga_event_count": ga,
        "strand_segment_purity": purity,
    }


def _obs():
    return pd.DataFrame(
        [
            # refA / bc1 -> 2 chimeras out of 4 (two one-sided reads not flagged)
            _row("refA_top", "bc1", 6, 6, 1.0),
            _row("refA_top", "bc1", 6, 6, 1.0),
            _row("refA_top", "bc1", 10, 0, 1.0),
            _row("refA_top", "bc1", 0, 10, 1.0),
            # refA / bc2 -> pure top, 0 chimeras
            _row("refA_top", "bc2", 10, 0, 1.0),
            _row("refA_top", "bc2", 9, 0, 1.0),
            # refB / bc1 -> 1 chimera out of 2 (one heavily one-sided read excluded)
            _row("refB_top", "bc1", 5, 5, 1.0),
            _row("refB_top", "bc1", 50, 4, 1.0),
        ]
    )


def test_writes_png_and_summary_csv(tmp_path):
    out_png = plot_reference_barcode_chimera_rate(_obs(), tmp_path)
    assert out_png is not None
    png_path = tmp_path / "reference_barcode_chimera_rate.png"
    csv_path = tmp_path / "reference_barcode_chimera_rate.csv"
    assert png_path.exists()
    assert csv_path.exists()

    summary = pd.read_csv(csv_path).set_index(["Reference_strand", "barcode"])
    assert summary.loc[("refA_top", "bc1"), "n_reads"] == 4
    assert summary.loc[("refA_top", "bc1"), "n_chimera"] == 2
    assert summary.loc[("refA_top", "bc1"), "chimera_rate"] == 0.5
    assert summary.loc[("refA_top", "bc2"), "chimera_rate"] == 0.0
    assert summary.loc[("refB_top", "bc1"), "chimera_rate"] == 0.5


def test_returns_none_when_columns_missing(tmp_path):
    out = plot_reference_barcode_chimera_rate(
        pd.DataFrame({"barcode": ["x"], "Reference_strand": ["r"]}), tmp_path
    )
    assert out is None
    assert not (tmp_path / "reference_barcode_chimera_rate.png").exists()
