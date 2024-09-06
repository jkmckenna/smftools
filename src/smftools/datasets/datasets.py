## datasets

def import_deps():
    """
    
    """
    import anndata as ad
    from pathlib import Path
    from .._settings import settings
    HERE = Path(__file__).parent
    return HERE

def dCas9_kinetics():
    """
    
    """
    HERE = import_deps()
    filepath = HERE / "dCas9_m6A_invitro_kinetics.h5ad.gz"
    return ad.read_h5ad(filepath)

def Kissiov_and_McKenna_2025():
    """
    
    """
    HERE = import_deps()
    filepath = HERE / "F1_hybrid_NKG2A_enhander_promoter_GpC_conversion_SMF.h5ad.gz"
    return ad.read_h5ad(filepath)
