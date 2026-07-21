import pandas as pd

from smftools.readwrite import safe_read_h5ad
from tests.fixtures.partitioned_pipeline import REFERENCE_LENGTHS


def test_partitioned_pipeline_store_has_queryable_reference_barcode_strata(
    partitioned_pipeline_store,
):
    fixture = partitioned_pipeline_store
    spine, _ = safe_read_h5ad(fixture.paths["spine"])
    molecules = pd.read_parquet(fixture.paths["molecules"])
    barcode_index = pd.read_parquet(fixture.paths["barcode_index"])

    assert spine.n_obs == 8
    assert set(spine.obs_names) == set(fixture.read_ids)
    assert spine.uns["fixture_modality"] == "conversion"
    assert spine.uns["reference_lengths"] == REFERENCE_LENGTHS
    assert set(molecules["read_id"]) == set(fixture.read_ids)
    assert set(zip(barcode_index["reference"], barcode_index["sample"], strict=True)) == {
        (reference, barcode)
        for reference in REFERENCE_LENGTHS
        for barcode in ("barcode01", "barcode02")
    }
    assert all(not str(path).startswith("/") for path in spine.obs["ragged_shard"])
    assert all(not str(path).startswith("/") for path in spine.obs["bam_path"])
