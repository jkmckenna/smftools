"""Logging utilities for smftools."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

DEFAULT_LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
STAGE_LOGGING_SUBDIR = "logs"


def setup_logging(
    level: int = logging.INFO,
    fmt: str = DEFAULT_LOG_FORMAT,
    datefmt: str = DEFAULT_DATE_FORMAT,
    log_file: Optional[Union[str, Path]] = None,
    reconfigure: bool = False,
) -> None:
    """
    Configure logging for smftools.

    Should be called once by the CLI entrypoint.
    Safe to call multiple times, with optional reconfiguration.
    """
    logger = logging.getLogger("smftools")

    if logger.handlers and not reconfigure:
        if log_file is not None:
            log_path = Path(log_file)
            has_file_handler = any(
                isinstance(handler, logging.FileHandler)
                and Path(getattr(handler, "baseFilename", "")) == log_path
                for handler in logger.handlers
            )
            if not has_file_handler:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                file_handler = logging.FileHandler(log_path)
                file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
                logger.addHandler(file_handler)
        logger.setLevel(level)
        return

    if logger.handlers and reconfigure:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

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


def setup_stage_logging(cfg, stage_directory: Union[str, Path]) -> Optional[Path]:
    """Configure smftools logging once for a CLI stage invocation.

    Must be called at the top of a stage's wrapper function (e.g.
    ``preprocess_adata``), before any branch that might skip the stage's
    ``*_core`` function (early-return-if-done, or dispatch to a partitioned
    executor). Those branches never called ``setup_logging`` themselves, so
    their output silently fell through to whatever log file a previous stage
    in the same process had left attached (or nowhere, if this is the first
    stage) instead of getting their own ``<stage>/logs/`` file.

    Returns the log file path, or ``None`` if ``cfg.emit_log_file`` is falsy.
    """
    stage_directory = Path(stage_directory)
    log_level = getattr(logging, str(getattr(cfg, "log_level", "INFO")).upper(), logging.INFO)

    log_file: Optional[Path] = None
    if getattr(cfg, "emit_log_file", True):
        logging_directory = stage_directory / STAGE_LOGGING_SUBDIR
        logging_directory.mkdir(parents=True, exist_ok=True)
        now = datetime.now()
        log_file = logging_directory / f"{now.strftime('%y%m%d')}_{now.strftime('%H%M%S')}_log.log"
    else:
        stage_directory.mkdir(parents=True, exist_ok=True)

    setup_logging(level=log_level, log_file=log_file, reconfigure=log_file is not None)
    return log_file
