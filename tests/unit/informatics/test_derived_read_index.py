from types import SimpleNamespace

import pandas as pd
import pytest

from smftools.informatics.derived_read_index import (
    LATENT_READ_INDEX_SCHEMA_VERSION,
    molecule_index_bucket,
    write_derived_read_index,
    write_latent_read_index,
)
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


@pytest.mark.unit
def test_latent_read_index_preserves_scope_and_prunes_molecule_buckets(tmp_path):
    import pyarrow.dataset as ds

    experiment_uid = new_experiment_uid()
    read_ids = [f"r{index}" for index in range(64)]
    molecule_uids = [molecule_uid(experiment_uid, read_id) for read_id in read_ids]
    obs = pd.DataFrame(
        {
            "read_id": read_ids,
            "experiment_uid": experiment_uid,
            "molecule_uid": molecule_uids,
        },
        index=read_ids,
    )
    record = {
        "reference": "ref_top",
        "core_start": 0,
        "core_end": 10,
        "analysis_core_id": "core-a",
        "group_sha256": "a" * 64,
        "model_id": "model-a",
        "model_checksum": "b" * 64,
        "obsm_keys": ["X_pca_signal"],
        "varm_keys": ["PCs_signal"],
        "obs_columns": ["sample", "leiden_signal"],
    }

    paths = write_latent_read_index(
        tmp_path,
        obs=obs,
        record=record,
        generation_id="generation-a",
        group_path="latent_adata_outputs/generations/generation-a/store/task.zarr",
        reference_uid="reference-a",
        stage_schema_version=3,
    )

    dataset = ds.dataset(tmp_path / "read_index", format="parquet", partitioning="hive")
    index = dataset.to_table().to_pandas()
    assert len(index) == len(obs)
    assert set(index["index_schema_version"]) == {LATENT_READ_INDEX_SCHEMA_VERSION}
    assert set(index["analysis_core_id"]) == {"core-a"}
    assert list(index.loc[0, "representation_keys"]) == ["X_pca_signal"]
    assert list(index.loc[0, "label_keys"]) == ["leiden_signal"]

    wanted_bucket = molecule_index_bucket(molecule_uids[0])
    fragments = list(
        dataset.get_fragments(
            filter=ds.field("molecule_bucket") == wanted_bucket,
        )
    )
    assert len(paths) > 1
    assert 0 < len(fragments) < len(paths)
    assert all(f"molecule_bucket={wanted_bucket}" in fragment.path for fragment in fragments)
