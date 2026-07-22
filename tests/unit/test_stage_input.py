import numpy as np
import pandas as pd

from smftools.cli.stage_input import iter_stage_slices
from smftools.informatics.raw_store import write_raw_store
from smftools.readwrite import safe_read_h5ad, safe_write_h5ad


def _frame():
    return pd.DataFrame(
        [
            {
                "read_id": "keep",
                "reference": "chr1",
                "Reference_strand": "chr1_top",
                "barcode": "bc1",
                "sample": "bc1",
                "reference_start": 1,
                "cigar": "3M",
                "aligned_length": 3,
                "sequence": [0, 1, 2],
                "modification_signal": [0.0, 1.0, 0.0],
            },
            {
                "read_id": "drop",
                "reference": "chr1",
                "Reference_strand": "chr1_top",
                "barcode": "bc1",
                "sample": "bc1",
                "reference_start": 8,
                "cigar": "3M",
                "aligned_length": 3,
                "sequence": [0, 1, 2],
                "modification_signal": [1.0, 0.0, 1.0],
            },
        ]
    )


def test_stage_iterator_filters_reads_and_skips_empty_genome_cores(tmp_path):
    paths = write_raw_store(
        _frame(),
        tmp_path,
        reference_lengths={"chr1_top": 20},
        analysis_mode="genome",
        genome_tile_size=5,
        genome_tile_halo=1,
    )
    spine, _ = safe_read_h5ad(paths["spine"])
    spine.obs["passes_dedup"] = [True, False]
    safe_write_h5ad(spine, paths["spine"], backup=False, verbose=False)

    slices = list(iter_stage_slices(paths["spine"], layers=[]))

    assert len(slices) == 1
    stage_slice = slices[0]
    assert (stage_slice.core_start, stage_slice.core_end) == (0, 5)
    assert (stage_slice.load_start, stage_slice.load_end) == (0, 6)
    assert list(stage_slice.adata.obs_names) == ["keep"]
    assert list(stage_slice.adata.var_names) == [str(position) for position in range(6)]
    assert list(stage_slice.core().var_names) == [str(position) for position in range(5)]


def test_stage_iterator_yields_full_locus_reference(tmp_path):
    paths = write_raw_store(
        _frame().iloc[[0]],
        tmp_path,
        reference_lengths={"chr1_top": 20},
        analysis_mode="locus",
    )

    stage_slice = next(iter(iter_stage_slices(paths["spine"], filter_mask=None)))

    assert stage_slice.analysis_mode == "locus"
    assert stage_slice.adata.shape == (1, 20)
    np.testing.assert_array_equal(stage_slice.core().X, stage_slice.adata.X)


def test_stage_iterator_inherits_analysis_catalog(tmp_path):
    paths = write_raw_store(
        _frame().iloc[[0]],
        tmp_path / "raw_outputs",
        reference_lengths={"chr1_top": 20},
        analysis_mode="genome",
        genome_tile_size=5,
        genome_tile_halo=1,
    )
    catalog = tmp_path / "analysis.parquet"
    mapping = tmp_path / "mapping.parquet"
    pd.DataFrame(
        {
            "region_id": ["reg_a"],
            "original_reference": ["chr1"],
            "original_start": [2],
            "original_end": [3],
        }
    ).to_parquet(catalog, index=False)
    pd.DataFrame(
        {
            "stored_reference": ["chr1_top"],
            "stored_start": [0],
            "stored_end": [20],
            "original_reference": ["chr1"],
            "original_start": [0],
            "original_end": [20],
            "coordinate_orientation": [1],
        }
    ).to_parquet(mapping, index=False)
    spine, _ = safe_read_h5ad(paths["spine"])
    spine.uns["region_catalogs"] = {"analysis": str(catalog)}
    spine.uns["reference_interval_map"] = str(mapping)
    safe_write_h5ad(spine, paths["spine"], backup=False, verbose=False)

    stage_slice = next(iter(iter_stage_slices(paths["spine"], filter_mask=None)))

    assert (stage_slice.core_start, stage_slice.core_end) == (2, 3)
    assert stage_slice.analysis_region_ids == ("reg_a",)
    assert list(stage_slice.core().var_names) == ["2"]


def _multi_sample_frame(reads_per_barcode: int) -> pd.DataFrame:
    rows = []
    for barcode in ("bc1", "bc2"):
        for i in range(reads_per_barcode):
            rows.append(
                {
                    "read_id": f"{barcode}_read{i}",
                    "reference": "chr1",
                    "Reference_strand": "chr1_top",
                    "barcode": barcode,
                    "sample": barcode,
                    "reference_start": 0,
                    "cigar": "3M",
                    "aligned_length": 3,
                    "sequence": [0, 1, 2],
                    "modification_signal": [0.0, 1.0, 0.0],
                }
            )
    return pd.DataFrame(rows)


def test_stage_iterator_caps_reads_per_sample_before_materializing(tmp_path):
    # Regression test for a real ~108s materialize() call on one 43,735-read
    # reference in a real experiment, where only 300 reads/barcode were ever
    # used downstream (a preprocess QC plot) -- max_reads_per_sample must cap
    # read_ids *before* materialize() runs, independently per sample, not as
    # a single flat cap across the whole reference (which could starve a
    # low-abundance barcode of any reads at all).
    paths = write_raw_store(
        _multi_sample_frame(5),
        tmp_path,
        reference_lengths={"chr1_top": 20},
        analysis_mode="locus",
    )

    stage_slice = next(
        iter(iter_stage_slices(paths["spine"], filter_mask=None, max_reads_per_sample=2))
    )

    assert stage_slice.adata.n_obs == 4  # 2 samples x 2 capped reads each
    obs_names = set(map(str, stage_slice.adata.obs_names))
    bc1_kept = {name for name in obs_names if name.startswith("bc1_")}
    bc2_kept = {name for name in obs_names if name.startswith("bc2_")}
    assert len(bc1_kept) == 2
    assert len(bc2_kept) == 2


def test_stage_iterator_uncapped_by_default(tmp_path):
    paths = write_raw_store(
        _multi_sample_frame(5),
        tmp_path,
        reference_lengths={"chr1_top": 20},
        analysis_mode="locus",
    )

    stage_slice = next(iter(iter_stage_slices(paths["spine"], filter_mask=None)))

    assert stage_slice.adata.n_obs == 10  # 2 samples x 5 reads each, uncapped
