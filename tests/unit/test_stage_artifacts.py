import json

import pandas as pd
import pytest

from smftools.cli.stage_artifacts import (
    prepare_analysis_plot_layout,
    prepare_stage_plot_layout,
    register_plot_artifact,
)


def test_stage_plot_layout_and_catalog_registration(tmp_path):
    source_spine = tmp_path / "raw" / "spine.h5ad"
    layout = prepare_stage_plot_layout(
        tmp_path / "preprocess",
        stage="preprocess",
        source_spine=source_spine,
        categories=("read_qc", "coverage"),
    )
    plot = layout.categories["coverage"] / "ref_top_0_100.png"
    plot.touch()
    register_plot_artifact(
        layout,
        plot,
        stage="preprocess",
        category="coverage",
        plot_type="valid_fraction",
        reference="ref_top",
        core_start=0,
        core_end=100,
    )

    assert set(layout.categories) == {"read_qc", "coverage"}
    context = json.loads(layout.context.read_text())
    assert context["stage"] == "preprocess"
    assert context["source_spine"] == str(source_spine.resolve())
    catalog = pd.read_parquet(layout.catalog)
    assert len(catalog) == 1
    assert catalog.iloc[0]["path"] == "plots/coverage/ref_top_0_100.png"
    assert catalog.iloc[0]["reference"] == "ref_top"


def test_plot_registration_rejects_external_path(tmp_path):
    layout = prepare_stage_plot_layout(tmp_path / "stage", stage="variant")
    external = tmp_path / "outside.png"
    external.touch()

    with pytest.raises(ValueError, match="inside the stage"):
        register_plot_artifact(
            layout,
            external,
            stage="variant",
            category="global",
            plot_type="summary",
        )


def test_analysis_stage_uses_specific_plot_categories(tmp_path):
    layout = prepare_analysis_plot_layout(tmp_path / "spatial", stage="spatial")

    assert set(layout.categories) == {
        "clustermaps",
        "autocorrelation",
        "rolling_metrics",
        "position_correlation",
        "embeddings",
        "diagnostics",
    }
