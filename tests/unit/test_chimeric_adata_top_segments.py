import numpy as np
import pandas as pd

from smftools.cli.chimeric_adata import _build_top_segments_obs_tuples


def test_build_top_segments_obs_tuples_uses_partner_name_and_ints():
    obs_names = pd.Index(["read0", "read1"])
    read_df = pd.DataFrame(
        {
            "segment_start_label": ["2"],
            "segment_end_label": ["5"],
            "partner_id": [1],
            "partner_name": ["partner_read"],
        }
    )

    tuples = _build_top_segments_obs_tuples(read_df, obs_names)

    assert tuples == [(2, 5, "partner_read")]


def test_build_top_segments_obs_tuples_falls_back_to_obs_name():
    obs_names = pd.Index(["readA"])
    read_df = pd.DataFrame(
        {
            "segment_start_label": [1],
            "segment_end_label": [3],
            "partner_id": [0],
            "partner_name": [np.nan],
        }
    )

    tuples = _build_top_segments_obs_tuples(read_df, obs_names)

    assert tuples == [(1, 3, "readA")]
