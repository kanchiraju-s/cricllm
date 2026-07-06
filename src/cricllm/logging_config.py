"""Sets up logging once — writes to both a rotating log file and the console."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging(log_dir: Path, level: int = logging.INFO) -> logging.Logger:
    """Set up the root ``cricllm`` logger and return it.

    Logs go to ``cricllm.log`` (rotates at 5MB, keeps 5 backups) and to the
    console. You can call this more than once without ending up with
    duplicate log lines — it clears any handlers already attached first.
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("cricllm")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(_LOG_FORMAT)

    file_handler = RotatingFileHandler(
        log_dir / "cricllm.log", maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Grab a child logger, e.g. get_logger("pipeline") -> "cricllm.pipeline"."""
    return logging.getLogger(f"cricllm.{name}")
