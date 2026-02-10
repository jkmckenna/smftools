from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from smftools.preprocessing import load_sample_sheet


def test_load_sample_sheet_raises_when_no_keys_match(tmp_path):
    adata = ad.AnnData(X=np.zeros((2, 3)))
    adata.obs_names = ["read1", "read2"]

    sheet = pd.DataFrame(
        {
            "obs_names": ["other1", "other2"],
            "Sample_Names": ["S1", "S2"],
        }
    )
    sheet_path = Path(tmp_path) / "sample_sheet.csv"
    sheet.to_csv(sheet_path, index=False)

    with pytest.raises(ValueError, match="no keys from adata matched"):
        load_sample_sheet(adata, sheet_path)
