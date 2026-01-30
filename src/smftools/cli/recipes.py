from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import anndata as ad

from ..cli.chimeric_adata import chimeric_adata
from ..cli.hmm_adata import hmm_adata
from ..cli.latent_adata import latent_adata
from ..cli.load_adata import load_adata
from ..cli.preprocess_adata import preprocess_adata
from ..cli.spatial_adata import spatial_adata
from ..cli.variant_adata import variant_adata


def full_flow(
    config_path: str,
) -> Tuple[Optional[ad.AnnData], Optional[Path]]:
    load_adata(config_path)
    preprocess_adata(config_path)
    spatial_adata(config_path)
    variant_adata(config_path)
    chimeric_adata(config_path)
    hmm_adata(config_path)
    latent_adata(config_path)
