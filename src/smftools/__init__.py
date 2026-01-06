"""smftools"""

import logging
import warnings
from importlib.metadata import version

from . import cli, config, datasets, hmm
from . import informatics as inform
from . import machine_learning as ml
from . import plotting as pl
from . import preprocessing as pp
from . import tools as tl
from .readwrite import adata_to_df, merge_barcoded_anndatas_core, safe_read_h5ad, safe_write_h5ad

package_name = "smftools"
__version__ = version(package_name)

__all__ = [
    "adata_to_df",
    "inform",
    "ml",
    "pp",
    "tl",
    "pl",
    "datasets",
    "safe_write_h5ad",
    "safe_read_h5ad",
]
