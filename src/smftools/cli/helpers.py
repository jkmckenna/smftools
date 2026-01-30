from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import anndata as ad

from smftools.constants import H5_DIR, HMM_DIR, LATENT_DIR, LOAD_DIR, PREPROCESS_DIR, SPATIAL_DIR, VARIANT_DIR, CHIMERIC_DIR

from ..metadata import write_runtime_schema_yaml
from ..readwrite import safe_write_h5ad


@dataclass
class AdataPaths:
    raw: Path
    pp: Path
    pp_dedup: Path
    spatial: Path
    hmm: Path
    latent: Path
    variant: Path
    chimeric: Path


def get_adata_paths(cfg) -> AdataPaths:
    """
    Central helper: given cfg, compute all standard AnnData paths.
    """
    output_directory = Path(cfg.output_directory)

    # Raw and Preprocessed adata file pathes will have set names.
    raw = output_directory / LOAD_DIR / H5_DIR / f"{cfg.experiment_name}.h5ad.gz"
    pp = output_directory / PREPROCESS_DIR / H5_DIR / f"{cfg.experiment_name}_preprocessed.h5ad.gz"

    if cfg.smf_modality == "direct":
        # direct SMF: duplicate-removed path is just preprocessed path
        pp_dedup = pp
    else:
        pp_dedup = (
            output_directory
            / PREPROCESS_DIR
            / H5_DIR
            / f"{cfg.experiment_name}_preprocessed_duplicates_removed.h5ad.gz"
        )

    pp_dedup_base = pp_dedup.name.removesuffix(".h5ad.gz")

    # All of the following just append a new suffix to the preprocessesed_deduplicated base name
    spatial = output_directory / SPATIAL_DIR / H5_DIR / f"{pp_dedup_base}_spatial.h5ad.gz"
    hmm = output_directory / HMM_DIR / H5_DIR / f"{pp_dedup_base}_hmm.h5ad.gz"
    latent = output_directory / LATENT_DIR / H5_DIR / f"{pp_dedup_base}_latent.h5ad.gz"
    variant = output_directory / VARIANT_DIR / H5_DIR / f"{pp_dedup_base}_variant.h5ad.gz"
    chimeric = output_directory / CHIMERIC_DIR / H5_DIR / f"{pp_dedup_base}_chimeric.h5ad.gz"

    return AdataPaths(
        raw=raw,
        pp=pp,
        pp_dedup=pp_dedup,
        spatial=spatial,
        hmm=hmm,
        latent=latent,
        variant=variant,
        chimeric=chimeric,
    )


def load_experiment_config(config_path: str):
    """Load ExperimentConfig without invoking any pipeline stages."""
    from datetime import datetime
    from importlib import resources

    from ..config import ExperimentConfig, LoadExperimentConfig

    date_str = datetime.today().strftime("%y%m%d")
    loader = LoadExperimentConfig(config_path)
    defaults_dir = resources.files("smftools").joinpath("config")
    cfg, _ = ExperimentConfig.from_var_dict(
        loader.var_dict, date_str=date_str, defaults_dir=defaults_dir
    )
    return cfg


def write_gz_h5ad(adata: ad.AnnData, path: Path) -> Path:
    if path.suffix != ".gz":
        path = path.with_name(path.name + ".gz")
    safe_write_h5ad(adata, path, compression="gzip", backup=True)
    write_runtime_schema_yaml(adata, path, step_name="runtime")
    return path
