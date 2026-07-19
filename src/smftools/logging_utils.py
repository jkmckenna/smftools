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

    logging_directory = stage_directory / STAGE_LOGGING_SUBDIR
    now = datetime.now()
    stem = f"{now.strftime('%y%m%d')}_{now.strftime('%H%M%S')}"

    log_file: Optional[Path] = None
    if getattr(cfg, "emit_log_file", True):
        logging_directory.mkdir(parents=True, exist_ok=True)
        log_file = logging_directory / f"{stem}_log.log"
    else:
        stage_directory.mkdir(parents=True, exist_ok=True)

    setup_logging(level=log_level, log_file=log_file, reconfigure=log_file is not None)
    _setup_stage_perf_logging(cfg, logging_directory, stem, stage_directory.name)
    return log_file


def _setup_stage_perf_logging(cfg, logging_directory, stem: str, stage_name: str) -> None:
    """Start a per-command perf log (worker counts + memory) beside the stage log.

    Best-effort and observability-only: any failure here (missing psutil, unwritable
    dir) must never block the actual pipeline, so it is swallowed. Closes the
    previous stage's perf logger first, since ``experiment full`` runs raw ->
    preprocess -> spatial -> hmm in one process and each gets its own file.
    """
    from .perf_log import PerfLogger, get_perf_logger, set_perf_logger

    previous = get_perf_logger()
    if previous is not None:
        try:
            previous.close()
        except Exception:
            pass
        set_perf_logger(None)

    if not getattr(cfg, "emit_perf_log", True):
        return
    try:
        logging_directory.mkdir(parents=True, exist_ok=True)
        interval = float(getattr(cfg, "perf_log_sample_interval_seconds", 2.0) or 2.0)
        logger = PerfLogger(
            logging_directory / f"{stem}_perf.jsonl",
            stage_name,
            sample_interval_seconds=interval,
        )
        set_perf_logger(logger)
        global _PERF_ATEXIT_REGISTERED
        if not _PERF_ATEXIT_REGISTERED:
            import atexit

            atexit.register(_close_current_perf_logger)
            _PERF_ATEXIT_REGISTERED = True
    except Exception:
        set_perf_logger(None)


_PERF_ATEXIT_REGISTERED = False


def _close_current_perf_logger() -> None:
    from .perf_log import get_perf_logger, set_perf_logger

    logger = get_perf_logger()
    if logger is not None:
        try:
            logger.close()
        except Exception:
            pass
        set_perf_logger(None)
