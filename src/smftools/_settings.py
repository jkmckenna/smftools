from pathlib import Path
from typing import Union

class SMFConfig:
    """\
    Config for smftools.
    """

    def __init__(
        self,
        *,
        datasetdir: Union[Path, str] = "./datasets/"
    ):
        self._datasetdir = Path(datasetdir) if isinstance(datasetdir, str) else datasetdir

    @property
    def datasetdir(self) -> Path:
        return self._datasetdir

settings = SMFConfig()