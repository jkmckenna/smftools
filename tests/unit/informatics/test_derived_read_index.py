from types import SimpleNamespace

import pandas as pd
import pytest

from smftools.informatics.derived_read_index import write_derived_read_index
from smftools.informatics.molecule_identity import molecule_uid, new_experiment_uid


@pytest.mark.unit
def test_derived_read_index_preserves_rows_and_hmm_models(tmp_path):
    experiment_uid = new_experiment_uid()
    obs = pd.DataFrame(
        {
            "read_id": ["r1", "r2"],
            "experiment_uid": [experiment_uid, experiment_uid],
            "molecule_uid": [
                molecule_uid(experiment_uid, "r1"),
                molecule_uid(experiment_uid, "r2"),
            ],
        },
        index=["r1", "r2"],
    )
    task = SimpleNamespace(
        task_id="ref|bc|0-10|00000",
        reference="ref",
        barcode="bc",
        chunk_index=0,
        core_start=0,
        core_end=10,
        load_start=0,
        load_end=12,
    )

    path = write_derived_read_index(
        tmp_path,
        stage="hmm",
        task=task,
        obs=obs,
        group_path="partials/task.zarr",
        stage_schema_version=2,
        model_artifacts=[
            {"model_id": "model-a", "model_checksum": "checksum-a"},
            {"model_id": "model-b", "model_checksum": "checksum-b"},
        ],
    )

    index = pd.read_parquet(path)
    assert len(index) == 4
    assert set(index["model_id"]) == {"model-a", "model-b"}
    assert set(index.loc[index["read_id"] == "r1", "group_row"]) == {0}
    assert set(index.loc[index["read_id"] == "r2", "group_row"]) == {1}
    assert set(index["group_path"]) == {"partials/task.zarr"}
