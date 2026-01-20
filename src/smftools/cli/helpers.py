from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import anndata as ad

from ..metadata import write_runtime_schema_yaml
from ..readwrite import safe_write_h5ad


@dataclass
class AdataPaths:
    raw: Path
    pp: Path
    pp_dedup: Path
    spatial: Path
    hmm: Path


def get_adata_paths(cfg) -> AdataPaths:
    """
    Central helper: given cfg, compute all standard AnnData paths.
    """
    h5_dir = Path(cfg.output_directory) / "h5ads"

    raw = h5_dir / f"{cfg.experiment_name}.h5ad.gz"

    pp = h5_dir / f"{cfg.experiment_name}_preprocessed.h5ad.gz"

    if cfg.smf_modality == "direct":
        # direct SMF: duplicate-removed path is just preprocessed path
        pp_dedup = pp
    else:
        pp_dedup = h5_dir / f"{cfg.experiment_name}_preprocessed_duplicates_removed.h5ad.gz"

    pp_dedup_base = pp_dedup.name.removesuffix(".h5ad.gz")

    spatial = h5_dir / f"{pp_dedup_base}_spatial.h5ad.gz"
    hmm = h5_dir / f"{pp_dedup_base}_spatial_hmm.h5ad.gz"

    return AdataPaths(
        raw=raw,
        pp=pp,
        pp_dedup=pp_dedup,
        spatial=spatial,
        hmm=hmm,
    )


def write_gz_h5ad(adata: ad.AnnData, path: Path) -> Path:
    if path.suffix != ".gz":
        path = path.with_name(path.name + ".gz")
    safe_write_h5ad(adata, path, compression="gzip", backup=True)
    write_runtime_schema_yaml(adata, path, step_name="runtime")
    return path
