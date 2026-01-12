"""Dataset helpers for bundled SMF datasets."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import anndata as ad


def import_HERE() -> Path:
    """Resolve the local dataset directory.

    Returns:
        Path: Path to the datasets directory.
    """
    return Path(__file__).parent


def dCas9_kinetics() -> "ad.AnnData":
    """Load the in vitro Hia5 dCas9 kinetics SMF dataset.

    Returns:
        anndata.AnnData: Annotated dataset with Nanopore HAC m6A modcalls.
    """
    import anndata as ad

    filepath = import_HERE() / "dCas9_m6A_invitro_kinetics.h5ad.gz"
    return ad.read_h5ad(filepath)


def Kissiov_and_McKenna_2025() -> "ad.AnnData":
    """Load the F1 Hybrid M.CviPI natural killer cell SMF dataset.

    Returns:
        anndata.AnnData: Annotated dataset with canonical calls of NEB EMseq converted SMF gDNA.
    """
    import anndata as ad

    filepath = import_HERE() / "F1_hybrid_NKG2A_enhander_promoter_GpC_conversion_SMF.h5ad.gz"
    return ad.read_h5ad(filepath)
