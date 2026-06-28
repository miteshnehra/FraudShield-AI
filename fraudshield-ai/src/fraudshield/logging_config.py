"""Central logging configuration for the application."""

from __future__ import annotations

import logging
import os
from pathlib import Path


def configure_logging(log_directory: Path) -> logging.Logger:
    """Configure console and file logging once and return the app logger."""
    log_directory.mkdir(parents=True, exist_ok=True)
    level_name = os.getenv("FRAUDSHIELD_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logger = logging.getLogger("fraudshield")
    logger.setLevel(level)
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_directory / "fraudshield.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

