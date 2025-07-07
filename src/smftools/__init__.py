"""smftools"""

import logging
import warnings

from . import informatics as inform
from . import machine_learning as ml
from . import plotting as pl
from . import preprocessing as pp
from . import tools as tl

from . import datasets, hmm, readwrite
from .readwrite import adata_to_df, safe_write_h5ad, merge_barcoded_anndatas


from importlib.metadata import version

package_name = "smftools"
__version__ = version(package_name)

__all__ = [
    "adata_to_df",
    "inform",
    "ml",
    "pp",
    "tl",
    "pl",
    "readwrite",
    "datasets",
    "safe_write_h5ad",
    "merge_barcoded_anndatas"
]