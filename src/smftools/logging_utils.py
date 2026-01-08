"""Logging utilities for smftools."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

DEFAULT_LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: int = logging.INFO,
    fmt: str = DEFAULT_LOG_FORMAT,
    datefmt: str = DEFAULT_DATE_FORMAT,
    log_file: Optional[Union[str, Path]] = None,
) -> None:
    """
    Configure logging for smftools.

    Should be called once by the CLI entrypoint.
    Safe to call multiple times.
    """
    logger = logging.getLogger("smftools")

    if logger.handlers:
        return

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # Console handler (stderr)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Optional file handler
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.setLevel(level)
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)