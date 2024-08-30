## datasets

import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path

from .._settings import settings

HERE = Path(__file__).parent


def dCas9_kinetics():
    """
    
    """
    filepath = HERE / "dCas9_m6A_invitro_kinetics.h5ad.gz"
    return ad.read_h5ad(filepath)

def Kissiov_and_McKenna_2025():
    """
    
    """
    filepath = HERE / "F1_hybrid_NKG2A_enhander_promoter_GpC_conversion_SMF.h5ad.gz"
    return ad.read_h5ad(filepath)
