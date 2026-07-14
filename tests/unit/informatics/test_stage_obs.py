import pandas as pd

from smftools.informatics.stage_obs import (
    obs_parquet_path,
    read_joined_obs,
    read_stage_obs,
    write_stage_obs,
)


def _obs(rows: dict) -> pd.DataFrame:
    return pd.DataFrame(rows).set_index("read_id")


def test_write_stage_obs_with_no_columns_filter_keeps_everything(tmp_path):
    obs = _obs(
        {
            "read_id": ["r0", "r1"],
            "Reference_strand": ["ref0_top", "ref0_top"],
            "Sample": ["bc00", "bc01"],
        }
    )
    path = write_stage_obs(tmp_path, obs)

    assert path == obs_parquet_path(tmp_path)
    reloaded = read_stage_obs(tmp_path)
    assert list(reloaded.index) == ["r0", "r1"]
    assert list(reloaded.columns) == ["Reference_strand", "Sample"]
    assert list(reloaded["Sample"]) == ["bc00", "bc01"]


def test_write_stage_obs_with_columns_filter_keeps_only_those(tmp_path):
    obs = _obs(
        {
            "read_id": ["r0", "r1"],
            "Reference_strand": ["ref0_top", "ref0_top"],
            "Sample": ["bc00", "bc01"],
            "passes_read_qc": [True, False],
        }
    )
    write_stage_obs(tmp_path, obs, columns=["passes_read_qc"])

    reloaded = read_stage_obs(tmp_path)
    assert list(reloaded.columns) == ["passes_read_qc"]
    assert list(reloaded["passes_read_qc"]) == [True, False]
    # Not re-stored -- those live in an earlier stage's obs.parquet.
    assert "Reference_strand" not in reloaded.columns
    assert "Sample" not in reloaded.columns


def test_read_joined_obs_combines_stages_by_read_id(tmp_path):
    raw_dir = tmp_path / "raw_outputs"
    preprocess_dir = tmp_path / "preprocess_adata_outputs"
    raw_obs = _obs(
        {
            "read_id": ["r0", "r1", "r2"],
            "Reference_strand": ["ref0_top", "ref0_top", "ref0_top"],
            "Sample": ["bc00", "bc00", "bc01"],
        }
    )
    write_stage_obs(raw_dir, raw_obs)

    # preprocess only kept r0, r1 (r2 filtered out -- a later stage's row set is a
    # subset), and only newly adds passes_read_qc.
    preprocess_obs = _obs(
        {"read_id": ["r0", "r1"], "passes_read_qc": [True, True]}
    )
    write_stage_obs(preprocess_dir, preprocess_obs, columns=["passes_read_qc"])

    joined = read_joined_obs([raw_dir, preprocess_dir])

    assert list(joined.index) == ["r0", "r1"]  # narrowed to preprocess's row set
    assert set(joined.columns) == {"Reference_strand", "Sample", "passes_read_qc"}
    assert list(joined["Sample"]) == ["bc00", "bc00"]
    assert list(joined["passes_read_qc"]) == [True, True]


def test_read_joined_obs_single_stage_matches_read_stage_obs(tmp_path):
    obs = _obs({"read_id": ["r0"], "Reference_strand": ["ref0_top"]})
    write_stage_obs(tmp_path, obs)

    assert read_joined_obs([tmp_path]).equals(read_stage_obs(tmp_path))


def test_read_joined_obs_empty_list_returns_empty_frame():
    assert read_joined_obs([]).empty
