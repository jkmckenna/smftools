from pathlib import Path

class SMFConfig:
    """\
    Config for smftools.
    """

    def __init__(
        self,
        *,
        datasetdir: Path | str = "./datasets/"
    ):
        self.datasetdir = datasetdir

    @property
    def datasetdir(self) -> Path:
        return self._datasetdir

settings = SMFConfig()