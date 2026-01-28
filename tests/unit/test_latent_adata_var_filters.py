from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd

from smftools.cli.latent_adata import _build_mod_sites_var_filter_mask


def test_build_mod_sites_var_filter_mask_and_across_refs() -> None:
    var = pd.DataFrame(
        {
            "ref1_A_site": [True, False, False, True],
            "ref1_C_site": [False, True, False, False],
            "position_in_ref1": [True, True, True, False],
            "ref2_A_site": [True, False, True, False],
            "ref2_C_site": [False, True, False, False],
            "position_in_ref2": [True, False, True, True],
        }
    )
    adata = ad.AnnData(np.zeros((1, var.shape[0])), var=var)
    cfg = SimpleNamespace(mod_target_bases=["A", "C"])

    mask = _build_mod_sites_var_filter_mask(
        adata=adata,
        references=["ref1", "ref2"],
        cfg=cfg,
        smf_modality="direct",
        deaminase=False,
    )

    expected = np.array([True, False, False, False])
    np.testing.assert_array_equal(mask, expected)
