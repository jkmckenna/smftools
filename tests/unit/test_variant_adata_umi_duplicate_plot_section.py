from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd

from smftools.cli.helpers import AdataPaths
from smftools.cli.variant_adata import variant_adata_core


def test_variant_adata_adds_umi_cluster_duplicate_plot_section(tmp_path, monkeypatch):
    seq1_col = "refA_top_strand_FASTA_base"
    seq2_col = "refB_top_strand_FASTA_base"
    segment_layer = f"{seq1_col}__{seq2_col}_variant_segments"

    adata = ad.AnnData(X=np.zeros((4, 6)))
    adata.obs["Sample_Names"] = pd.Categorical(["S1", "S1", "S1", "S1"])
    adata.obs["Reference_strand"] = pd.Categorical(["refA_top", "refA_top", "refA_top", "refA_top"])
    adata.obs["U1_valid"] = [True, True, False, False]
    adata.obs["U2_valid"] = [True, False, True, False]
    adata.obs["U1_cluster"] = ["AAAA", "BBBB", None, None]
    adata.obs["U2_cluster"] = ["TTTT", None, "CCCC", None]
    adata.obs["U1_cluster_is_duplicate"] = [True, False, False, False]
    adata.obs["U2_cluster_is_duplicate"] = [True, False, False, False]
    adata.obs["RX_is_dominant_pair"] = [True, False, False, False]
    adata.obs["RX_edge_count"] = [1, 1, 1, 0]
    adata.layers[segment_layer] = np.ones((4, 6), dtype=np.int8)

    captured_calls: list[dict] = []
    captured_multi_calls: list[dict] = []

    def _no_op(*args, **kwargs):
        return None

    def _capture_plot(*args, **kwargs):
        captured_calls.append(kwargs)
        return []

    def _capture_plot_multi(*args, **kwargs):
        captured_multi_calls.append(kwargs)
        return []

    import smftools.plotting as plotting
    import smftools.preprocessing as preprocessing

    monkeypatch.setattr(plotting, "plot_mismatch_base_frequency_by_position", _no_op)
    monkeypatch.setattr(plotting, "plot_sequence_integer_encoding_clustermaps", _no_op)
    monkeypatch.setattr(plotting, "plot_variant_segment_clustermaps", _capture_plot)
    monkeypatch.setattr(plotting, "plot_variant_segment_clustermaps_multi_obs", _capture_plot_multi)
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
        use_umi=True,
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

    dup_calls = [
        k
        for k in captured_calls
        if k.get("mismatch_type_obs_col") == "UMI_cluster_duplicate_status"
    ]
    assert len(dup_calls) == 1
    assert dup_calls[0]["mismatch_type_legend_prefix"] == "UMI cluster duplicate"

    assert len(captured_multi_calls) == 1
    multi_specs = captured_multi_calls[0]["annotation_specs"]
    assert len(multi_specs) == 2
    assert multi_specs[0]["obs_col"] == "UMI_pass_status"
    assert multi_specs[1]["obs_col"] == "UMI_cluster_duplicate_status"
