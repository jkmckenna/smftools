"""smftools"""

import logging
import warnings

from . import informatics as inform
from . import preprocessing as pp
from . import tools as tl
from . import plotting as pl
from . import readwrite, datasets


from importlib.metadata import version

package_name = "smftools"
__version__ = version(package_name)

__all__ = [
    "inform",
    "pp",
    "tl",
    "pl",
    "readwrite",
    "datasets"
]