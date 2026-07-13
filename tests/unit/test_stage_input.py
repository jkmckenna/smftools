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
