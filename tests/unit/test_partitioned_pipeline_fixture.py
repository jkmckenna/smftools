import shutil
from pathlib import Path

import pandas as pd
import pytest

from smftools.informatics.partition_read import materialize
from smftools.readwrite import safe_read_h5ad
from tests.fixtures.partitioned_pipeline import (
    ACCEPTANCE_REFERENCE_LENGTHS,
    REFERENCE_LENGTHS,
    build_partitioned_pipeline_fixture,
)


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


@pytest.mark.parametrize("modality", ["conversion", "deaminase", "direct"])
@pytest.mark.parametrize("analysis_mode", ["locus", "genome"])
def test_acceptance_fixture_covers_modality_mode_matrix_and_multiple_chunks(
    tmp_path, modality, analysis_mode
):
    fixture = build_partitioned_pipeline_fixture(
        tmp_path / f"{modality}_{analysis_mode}",
        modality=modality,
        analysis_mode=analysis_mode,
        reads_per_stratum=4,
        include_strand_derivatives=True,
    )
    spine, _ = safe_read_h5ad(fixture.paths["spine"])
    molecules = pd.read_parquet(fixture.paths["molecules"])

    assert spine.n_obs == 32
    assert spine.uns["reference_lengths"] == ACCEPTANCE_REFERENCE_LENGTHS
    assert set(spine.obs["Strand"]) == {"top", "bottom"}
    assert set(spine.obs["Dataset"]) == {modality}
    assert spine.uns["analysis_mode"] == analysis_mode
    chunks_per_stratum = molecules.groupby(["Reference_strand", "Barcode"], observed=True)[
        "group_path"
    ].nunique()
    assert chunks_per_stratum.min() >= 2


def test_acceptance_fixture_remains_queryable_after_tree_relocation(tmp_path):
    fixture = build_partitioned_pipeline_fixture(
        tmp_path / "producer",
        modality="direct",
        analysis_mode="genome",
        reads_per_stratum=4,
        include_strand_derivatives=True,
    )
    relocated = tmp_path / "consumer" / fixture.run_dir.name
    shutil.copytree(fixture.run_dir, relocated)
    copied_spine = relocated / Path(fixture.paths["spine"]).relative_to(fixture.run_dir)

    selected = materialize(
        copied_spine,
        references="fixture_ref_a_top",
        barcodes="barcode01",
        read_ids=[fixture.read_ids[0]],
        start=2,
        end=6,
        lazy=False,
    )

    assert selected.obs["read_id"].tolist() == [fixture.read_ids[0]]
    assert selected.var_names.tolist() == ["2", "3", "4", "5"]
