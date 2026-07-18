from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from smftools.preprocessing.label_deaminase_pcr_chimeras import (
    label_deaminase_pcr_chimeras,
)


def _make_adata() -> ad.AnnData:
    obs = pd.DataFrame(
        {
            "ct_event_count": [10, 0, 6, 5, 1],
            "ga_event_count": [0, 10, 5, 5, 8],
            "strand_segment_purity": [1.0, 1.0, 1.0, 0.6, 1.0],
        },
        index=["pure_ct", "pure_ga", "chimera", "noisy", "mostly_ga"],
    )
    return ad.AnnData(X=np.zeros((5, 1), dtype=float), obs=obs)


def test_labels_only_clean_chimera_and_keeps_all_reads():
    adata = _make_adata()
    out = label_deaminase_pcr_chimeras(adata)
    flags = out.obs["deaminase_PCR_chimera"].to_dict()
    assert flags == {
        "pure_ct": False,  # ga below min events
        "pure_ga": False,  # ct below min events
        "chimera": True,  # both spans present, pure, balanced
        "noisy": False,  # segment purity too low
        "mostly_ga": False,  # one-sided (ct below min events)
    }
    # label-only: no reads removed
    assert out.n_obs == 5


def test_single_strand_fraction_gate_rejects_one_sided_reads():
    # both counts high enough and pure, but heavily one-sided -> not a chimera
    obs = pd.DataFrame(
        {
            "ct_event_count": [50, 6],
            "ga_event_count": [4, 6],
            "strand_segment_purity": [1.0, 1.0],
        },
        index=["one_sided", "balanced"],
    )
    adata = ad.AnnData(X=np.zeros((2, 1), dtype=float), obs=obs)
    out = label_deaminase_pcr_chimeras(adata, max_single_strand_fraction=0.8)
    assert out.obs["deaminase_PCR_chimera"].to_dict() == {
        "one_sided": False,  # 50/54 = 0.93 > 0.8
        "balanced": True,
    }


def test_thresholds_are_configurable():
    adata = _make_adata()
    # raise the min events per span so the 6/5 chimera no longer qualifies
    out = label_deaminase_pcr_chimeras(adata, min_events_per_span=6)
    assert not out.obs["deaminase_PCR_chimera"].any()


def test_bypass_returns_input_without_column():
    adata = _make_adata()
    out = label_deaminase_pcr_chimeras(adata, bypass=True)
    assert out is adata
    assert "deaminase_PCR_chimera" not in out.obs


def test_missing_columns_skips_gracefully():
    adata = ad.AnnData(X=np.zeros((2, 1), dtype=float), obs=pd.DataFrame(index=["a", "b"]))
    out = label_deaminase_pcr_chimeras(adata)
    assert out.obs["deaminase_PCR_chimera"].to_dict() == {"a": False, "b": False}
