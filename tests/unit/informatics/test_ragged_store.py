import numpy as np
import pandas as pd
import pytest

from smftools.constants import (
    BASE_QUALITY_SCORES,
    MISMATCH_INTEGER_ENCODING,
    MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT,
    READ_SPAN_MASK,
    SEQUENCE_INTEGER_ENCODING,
)
from smftools.informatics.ragged_store import (
    ALIGNED_LENGTH,
    CIGAR,
    MISMATCH,
    MODIFICATION_SIGNAL,
    QUALITY,
    READ_ID,
    REFERENCE,
    REFERENCE_START,
    SEQUENCE,
    alignment_to_ragged_record,
    cigar_query_length,
    cigar_reference_length,
    iter_cigar_aligned_pairs,
    materialize_ragged,
    read_ragged_parquet,
    write_ragged_parquet,
)


class FakeRead:
    is_unmapped = False
    is_secondary = False
    is_supplementary = False
    is_reverse = False
    query_name = "read1"
    reference_name = "chr1"
    reference_start = 1
    cigarstring = "1S3M1I2M1D2M"
    query_sequence = "NAGCTGGTA"
    query_qualities = list(range(20, 29))


def _ragged_frame() -> pd.DataFrame:
    record = alignment_to_ragged_record(FakeRead(), "AACCGGTTAACC")
    record[MODIFICATION_SIGNAL] = [float(value) for value in range(9)]
    return pd.DataFrame([record])


def test_cigar_lengths_and_aligned_pairs():
    cigar = FakeRead.cigarstring
    assert cigar_query_length(cigar) == 9
    assert cigar_reference_length(cigar) == 8
    assert list(iter_cigar_aligned_pairs(cigar, 1)) == [
        (1, 1),
        (2, 2),
        (3, 3),
        (5, 4),
        (6, 5),
        (7, 7),
        (8, 8),
    ]


def test_ragged_parquet_round_trip(tmp_path):
    path = write_ragged_parquet(_ragged_frame(), tmp_path / "reads.parquet")
    result = read_ragged_parquet(path)

    assert list(result[READ_ID]) == ["read1"]
    assert result.at[0, CIGAR] == FakeRead.cigarstring
    assert result.at[0, ALIGNED_LENGTH] == 8
    # Ragged array columns round-trip as narrow numpy dtypes (int8/float32), not
    # the original Python lists -- see _narrow_ragged_values.
    np.testing.assert_array_equal(result.at[0, SEQUENCE], _ragged_frame().at[0, SEQUENCE])
    assert result.at[0, SEQUENCE].dtype == np.int8
    np.testing.assert_array_equal(result.at[0, QUALITY], list(range(20, 29)))
    assert result.at[0, QUALITY].dtype == np.int8


def test_ragged_validation_rejects_array_length_mismatch(tmp_path):
    frame = _ragged_frame()
    frame.at[0, QUALITY] = [1, 2]
    with pytest.raises(ValueError, match="expected 9 from CIGAR"):
        write_ragged_parquet(frame, tmp_path / "invalid.parquet")


def test_ragged_parquet_round_trip_narrows_modification_signal_to_float32(tmp_path):
    # Regression test for a real ~27x memory balloon: ragged arrays were
    # stored/read as int64/float64 (pandas' default for plain Python
    # lists), ~8x wider than the int8/float32 the values actually need
    # (small integer codes, Phred scores, probabilities) -- see
    # dev/pipeline_scaling_audit.md.
    path = write_ragged_parquet(_ragged_frame(), tmp_path / "reads.parquet")
    result = read_ragged_parquet(path)
    np.testing.assert_array_equal(result.at[0, MODIFICATION_SIGNAL], [float(v) for v in range(9)])
    assert result.at[0, MODIFICATION_SIGNAL].dtype == np.float32
    assert result.at[0, MISMATCH].dtype == np.int8


def test_validate_ragged_frame_rejects_quality_outside_int8_range(tmp_path):
    frame = _ragged_frame()
    frame.at[0, QUALITY] = [200] + [20] * 8  # 200 > int8 max (127)
    with pytest.raises(ValueError, match="outside int8 range"):
        write_ragged_parquet(frame, tmp_path / "overflow.parquet")


def test_read_ragged_parquet_pyarrow_filter_matches_pandas_filter(tmp_path):
    # The read_ids filter is pushed down to pyarrow instead of reading the
    # whole shard and filtering in pandas afterward -- functional result
    # must be identical either way (only the memory/IO cost differs).
    rows = []
    for i in range(5):
        record = alignment_to_ragged_record(FakeRead(), "AACCGGTTAACC")
        record[READ_ID] = f"read{i}"
        record[MODIFICATION_SIGNAL] = [float(i)] * 9
        rows.append(record)
    frame = pd.DataFrame(rows)
    path = write_ragged_parquet(frame, tmp_path / "multi.parquet")

    selected = read_ragged_parquet(path, read_ids=["read1", "read3"])
    assert sorted(selected[READ_ID]) == ["read1", "read3"]

    whole = read_ragged_parquet(path)
    assert sorted(whole[READ_ID]) == [f"read{i}" for i in range(5)]

    none_selected = read_ragged_parquet(path, read_ids=["read_missing"])
    assert len(none_selected) == 0


def test_materialize_ragged_places_indels_and_soft_clips():
    frame = _ragged_frame()
    obs = pd.DataFrame(
        {"Reference_strand": ["chr1_top"], "Sample": ["bc01"]},
        index=["read1"],
    )
    result = materialize_ragged(
        frame,
        obs=obs,
        reference_lengths={"chr1_top": 12},
    )

    unknown = MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT["N"]
    expected_sequence = np.array(
        [unknown, 0, 2, 1, 2, 2, unknown, 3, 0, unknown, unknown, unknown],
        dtype=np.int8,
    )
    np.testing.assert_array_equal(result.layers[SEQUENCE_INTEGER_ENCODING][0], expected_sequence)

    expected_mismatch = np.full(12, unknown, dtype=np.int8)
    expected_mismatch[2] = MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT["G"]
    np.testing.assert_array_equal(result.layers[MISMATCH_INTEGER_ENCODING][0], expected_mismatch)

    expected_quality = np.full(12, -1, dtype=np.int8)
    expected_quality[[1, 2, 3, 4, 5, 7, 8]] = [21, 22, 23, 25, 26, 27, 28]
    np.testing.assert_array_equal(result.layers[BASE_QUALITY_SCORES][0], expected_quality)

    expected_span = np.zeros(12, dtype=np.int8)
    expected_span[1:9] = 1
    np.testing.assert_array_equal(result.layers[READ_SPAN_MASK][0], expected_span)

    expected_signal = np.full(12, np.nan, dtype=np.float32)
    expected_signal[[1, 2, 3, 4, 5, 7, 8]] = [1, 2, 3, 5, 6, 7, 8]
    np.testing.assert_array_equal(result.X[0], expected_signal)


def test_materialize_ragged_pads_shorter_references():
    first = _ragged_frame().iloc[0].to_dict()
    first[CIGAR] = "2M"
    first[REFERENCE_START] = 0
    first[ALIGNED_LENGTH] = 2
    first[SEQUENCE] = [0, 1]
    first[QUALITY] = [30, 31]
    first[MISMATCH] = [4, 4]
    first[MODIFICATION_SIGNAL] = [0.1, 0.2]
    second = dict(first)
    second[READ_ID] = "read2"
    second[REFERENCE] = "chr2"
    second["Reference_strand"] = "chr2_top"
    frame = pd.DataFrame([first, second])
    obs = pd.DataFrame(
        {"Reference_strand": ["chr1_top", "chr2_top"]},
        index=["read1", "read2"],
    )

    result = materialize_ragged(
        frame,
        obs=obs,
        reference_lengths={"chr1_top": 4, "chr2_top": 6},
    )
    padding = MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT["PAD"]
    np.testing.assert_array_equal(
        result.layers[SEQUENCE_INTEGER_ENCODING][0, 4:], [padding, padding]
    )


def _multi_read_frame() -> pd.DataFrame:
    """Three reads on one reference, distinct read_ids, for streaming tests."""
    base = _ragged_frame().iloc[0].to_dict()
    rows = []
    for i, read_id in enumerate(("read1", "read2", "read3")):
        row = dict(base)
        row[READ_ID] = read_id
        row["Reference_strand"] = "chr1_top"
        row[MODIFICATION_SIGNAL] = [float(i * 10 + value) for value in range(9)]
        rows.append(row)
    return pd.DataFrame(rows)


def test_materialize_ragged_streaming_matches_whole_frame():
    # The memory fix: streaming shard-by-shard must produce byte-identical
    # output to reading the whole selection at once. Chunks are split so a
    # read's output row (obs order) differs from its chunk order, and one
    # chunk carries a read not in the selection (shards hold many barcodes'
    # reads) -- both must be handled: scatter to the right row, ignore the
    # extra read.
    from smftools.informatics.ragged_store import materialize_ragged_streaming

    frame = _multi_read_frame()
    obs = pd.DataFrame(
        {"Reference_strand": ["chr1_top", "chr1_top", "chr1_top"]},
        index=["read3", "read1", "read2"],  # deliberately not chunk order
    )
    reference_lengths = {"chr1_top": 12}

    whole = materialize_ragged(frame, obs=obs, reference_lengths=reference_lengths)

    # A stray read ("read9") in a chunk must be ignored, not scattered.
    stray = frame.iloc[[0]].copy()
    stray[READ_ID] = ["read9"]
    chunk_a = pd.concat([frame.iloc[[1]], stray], ignore_index=True)  # read2 + read9
    chunk_b = frame.iloc[[0, 2]].reset_index(drop=True)  # read1, read3
    stream = materialize_ragged_streaming(
        [chunk_a, chunk_b], obs=obs, reference_lengths=reference_lengths
    )

    assert list(map(str, stream.obs_names)) == ["read3", "read1", "read2"]
    assert list(stream.var_names) == list(whole.var_names)
    np.testing.assert_array_equal(stream.X, whole.X)
    assert set(stream.layers) == set(whole.layers)
    for layer in whole.layers:
        np.testing.assert_array_equal(stream.layers[layer], whole.layers[layer])


def test_materialize_ragged_streaming_raises_on_missing_read():
    from smftools.informatics.ragged_store import materialize_ragged_streaming

    frame = _multi_read_frame()
    obs = pd.DataFrame(
        {"Reference_strand": ["chr1_top", "chr1_top", "chr1_top"]},
        index=["read1", "read2", "read3"],
    )
    # Chunks omit read3 entirely -- must raise, not silently drop a selected read.
    with pytest.raises(KeyError, match="lacks 1 selected read"):
        materialize_ragged_streaming(
            [frame.iloc[[0, 1]]], obs=obs, reference_lengths={"chr1_top": 12}
        )
