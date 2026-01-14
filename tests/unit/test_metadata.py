from pathlib import Path

import anndata as ad
import numpy as np

from smftools.cli.helpers import write_gz_h5ad

from smftools.config.experiment_config import ExperimentConfig
from smftools.metadata import record_smftools_metadata


def test_record_smftools_metadata(tmp_path):
    adata = ad.AnnData(np.ones((2, 2)))
    cfg = ExperimentConfig(output_directory=str(tmp_path))

    config_path = tmp_path / "experiment_config.csv"
    config_path.write_text("variable,value\nexperiment_name,example\n", encoding="utf-8")
    input_path = tmp_path / "input.h5ad"
    input_path.write_text("input", encoding="utf-8")
    output_path = tmp_path / "output.h5ad.gz"

    output_path = write_gz_h5ad(adata, output_path)
    record_smftools_metadata(
        adata,
        step_name="load",
        cfg=cfg,
        config_path=config_path,
        input_paths=[input_path],
        output_path=output_path,
    )

    smftools_uns = adata.uns["smftools"]
    assert smftools_uns["created_by"]["version"]
    assert smftools_uns["history"]
    assert smftools_uns["schema_registry_version"] == "1"

    step = smftools_uns["history"][-1]
    assert step["step"] == "load"
    assert step["outputs"]["h5ad_path"] == str(output_path)
    assert Path(step["outputs"]["schema_yaml_path"]).exists()

    inputs = smftools_uns["provenance"]["inputs"]
    assert any(item.get("path") == str(config_path) for item in inputs)
