from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd

from smftools.cli.helpers import AdataPaths
from smftools.cli.variant_adata import variant_adata_core


def test_variant_adata_applies_reindexing_offsets_and_invert(tmp_path, monkeypatch):
    seq1_col = "refA_top_strand_FASTA_base"
    seq2_col = "refB_top_strand_FASTA_base"
    segment_layer = f"{seq1_col}__{seq2_col}_variant_segments"

    adata = ad.AnnData(X=np.zeros((2, 4)))
    adata.obs["Sample_Names"] = pd.Categorical(["S1", "S1"])
    adata.obs["Reference_strand"] = pd.Categorical(["refA_top", "refA_top"])
    adata.var_names = ["10", "11", "12", "13"]
    adata.layers[segment_layer] = np.ones((2, 4), dtype=np.int8)

    captured: dict = {}

    def _no_op(*args, **kwargs):
        return None

    def _capture_plot(adata_arg, *args, index_col_suffix=None, **kwargs):
        if "var" not in captured:
            captured["var"] = adata_arg.var.copy()
            captured["index_col_suffix"] = index_col_suffix
        return []

    import smftools.plotting as plotting
    import smftools.preprocessing as preprocessing

    monkeypatch.setattr(plotting, "plot_mismatch_base_frequency_by_position", _no_op)
    monkeypatch.setattr(plotting, "plot_sequence_integer_encoding_clustermaps", _no_op)
    monkeypatch.setattr(plotting, "plot_variant_segment_clustermaps", _capture_plot)
    monkeypatch.setattr(preprocessing, "append_sequence_mismatch_annotations", _no_op)
    monkeypatch.setattr(preprocessing, "append_mismatch_frequency_sites", _no_op)
    monkeypatch.setattr(preprocessing, "append_variant_call_layer", _no_op)
    monkeypatch.setattr(preprocessing, "append_variant_segment_layer", _no_op)
    monkeypatch.setattr(preprocessing, "load_sample_sheet", _no_op)

    cfg = SimpleNamespace(
        log_level="INFO",
        output_directory=str(tmp_path / "outputs"),
        emit_log_file=False,
        smf_modality="conversion",
        sample_sheet_path=None,
        sample_sheet_mapping_column=None,
        force_reload_sample_sheet=False,
        references_to_align_for_variant_annotation=[seq1_col, seq2_col],
        reference_column="Reference_strand",
        mismatch_frequency_layer="mismatch_integer_encoding",
        mismatch_frequency_read_span_layer="read_span_mask",
        mismatch_frequency_range=[0.01, 0.99],
        bypass_append_mismatch_frequency_sites=False,
        force_redo_append_mismatch_frequency_sites=False,
        sample_name_col_for_plotting="Sample_Names",
        mod_target_bases=["C"],
        clustermap_demux_types_to_plot=["single"],
        force_redo_preprocessing=False,
        variant_overlay_seq1_color="gold",
        variant_overlay_seq2_color="navy",
        variant_overlay_marker_size=7.0,
        reindexing_offsets={"refA_top": 1000},
        reindexed_var_suffix="reindexed",
        reindexing_invert={"refA_top": True},
    )

    variant_path = tmp_path / "existing_variant.h5ad.gz"
    variant_path.parent.mkdir(parents=True, exist_ok=True)
    variant_path.touch()
    paths = AdataPaths(
        raw=tmp_path / "raw.h5ad.gz",
        pp=tmp_path / "pp.h5ad.gz",
        pp_dedup=tmp_path / "pp_dedup.h5ad.gz",
        spatial=tmp_path / "spatial.h5ad.gz",
        hmm=tmp_path / "hmm.h5ad.gz",
        latent=tmp_path / "latent.h5ad.gz",
        variant=variant_path,
        chimeric=tmp_path / "chimeric.h5ad.gz",
    )

    variant_adata_core(
        adata=adata,
        cfg=cfg,
        paths=paths,
        source_adata_path=Path("source.h5ad.gz"),
        config_path="config.csv",
    )

    assert captured["index_col_suffix"] == "reindexed"
    var_coords = captured["var"].index.astype(int)
    reindexed_col = captured["var"]["refA_top_reindexed"].astype(int)
    # invert=True -> sign flipped: -(var_coords + 1000)
    assert (reindexed_col == -(var_coords + 1000)).all()
