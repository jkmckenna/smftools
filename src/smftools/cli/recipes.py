from __future__ import annotations


def raw_adata(config_path: str):
    from ..cli.raw_adata import raw_adata as _raw_adata

    return _raw_adata(config_path)


def preprocess_adata(config_path: str):
    from ..cli.preprocess_adata import preprocess_adata as _preprocess_adata

    return _preprocess_adata(config_path)


def spatial_adata(config_path: str):
    from ..cli.spatial_adata import spatial_adata as _spatial_adata

    return _spatial_adata(config_path)


def hmm_adata(config_path: str):
    from ..cli.hmm_adata import hmm_adata as _hmm_adata

    return _hmm_adata(config_path)


def full_flow(config_path: str):
    """Run the standard raw-to-HMM workflow with stage-level restart semantics."""
    raw_adata(config_path)
    preprocess_adata(config_path)
    spatial_adata(config_path)
    return hmm_adata(config_path)
