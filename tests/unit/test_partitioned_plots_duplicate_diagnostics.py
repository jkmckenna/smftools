from __future__ import annotations

import pandas as pd

from smftools.cli.stage_artifacts import prepare_analysis_plot_layout
from smftools.constants import REFERENCE_STRAND
from smftools.preprocessing.partitioned_plots import _duplicate_diagnostics


def test_duplicate_diagnostics_no_reads_pass_qc_returns_without_crashing(tmp_path):
    # All reads fail QC (e.g. a dataset-wide modification-QC threshold
    # miscalibration): obs filters down to zero rows, so `references` is
    # empty. Must return cleanly instead of dividing by zero when computing
    # the histogram grid's row/column layout.
    obs = pd.DataFrame(
        {
            "passes_qc": [False, False, False],
            REFERENCE_STRAND: ["ref_top", "ref_bottom", "ref_top"],
            "sequence__min_hamming_to_pair": [0.1, 0.2, 0.3],
        }
    )
    layout = prepare_analysis_plot_layout(tmp_path / "preprocess_outputs", stage="preprocess")

    result = _duplicate_diagnostics(obs, "barcode", layout, 0.1)

    assert result is False


def test_duplicate_diagnostics_some_reads_pass_qc_still_plots(tmp_path):
    obs = pd.DataFrame(
        {
            "passes_qc": [True, True, False],
            REFERENCE_STRAND: ["ref_top", "ref_top", "ref_bottom"],
            "sequence__min_hamming_to_pair": [0.1, 0.2, 0.3],
            "barcode": ["bc1", "bc2", "bc1"],
            "is_duplicate": [False, True, False],
        }
    )
    layout = prepare_analysis_plot_layout(tmp_path / "preprocess_outputs", stage="preprocess")

    _duplicate_diagnostics(obs, "barcode", layout, 0.1)

    catalog = pd.read_parquet(layout.catalog)
    matching = catalog[catalog["plot_type"] == "hamming_distance_by_reference"]
    assert len(matching) == 1
