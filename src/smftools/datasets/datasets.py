## datasets

def import_HERE():
    """
    Imports HERE for loading datasets
    """
    from pathlib import Path
    from .._settings import settings
    HERE = Path(__file__).parent
    return HERE

def dCas9_kinetics():
    """
    in vitro Hia5 dCas9 kinetics SMF dataset. Nanopore HAC m6A modcalls.
    """
    import anndata as ad
    HERE = import_HERE()
    filepath = HERE / "dCas9_m6A_invitro_kinetics.h5ad.gz"
    return ad.read_h5ad(filepath)

def Kissiov_and_McKenna_2025():
    """
    F1 Hybrid M.CviPI natural killer cell SMF. Nanopore canonical calls of NEB EMseq converted SMF gDNA.
    """
    import anndata as ad
    HERE = import_HERE()
    filepath = HERE / "F1_hybrid_NKG2A_enhander_promoter_GpC_conversion_SMF.h5ad.gz"
    return ad.read_h5ad(filepath)
