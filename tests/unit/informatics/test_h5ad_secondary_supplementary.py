import anndata as ad
import numpy as np

from smftools.informatics import bam_functions, h5ad_functions


def test_add_secondary_supplementary_alignment_spans(monkeypatch, tmp_path):
    adata = ad.AnnData(X=np.zeros((2, 2)))
    adata.obs_names = ["read1", "read2"]

    monkeypatch.setattr(
        bam_functions,
        "find_secondary_supplementary_read_names",
        lambda *args, **kwargs: ({"read1"}, {"read2"}),
    )
    monkeypatch.setattr(
        bam_functions,
        "extract_secondary_supplementary_alignment_spans",
        lambda *args, **kwargs: (
            {"read1": [(5.0, 15.0, 10.0)]},
            {"read2": [(20.0, 35.0, 15.0)]},
        ),
    )

    bam_path = tmp_path / "sample.bam"
    bam_path.write_text("stub")

    h5ad_functions.add_secondary_supplementary_alignment_flags(adata, bam_path)

    assert adata.obs["has_secondary_alignment"].tolist() == [True, False]
    assert adata.obs["has_supplementary_alignment"].tolist() == [False, True]
    assert adata.obs["secondary_alignment_spans"].tolist() == [[(5.0, 15.0, 10.0)], None]
    assert adata.obs["supplementary_alignment_spans"].tolist() == [None, [(20.0, 35.0, 15.0)]]
