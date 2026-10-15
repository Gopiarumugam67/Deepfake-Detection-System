"""Shared logger factory so every module logs consistently to console + file."""

import logging
import sys
from pathlib import Path


def get_logger(name: str, log_dir: Path | None = None) -> logging.Logger:
    """
    Create (or fetch) a configured logger.

    Args:
        name: Usually __name__ of the calling module.
        log_dir: Directory to write a rotating log file into. If None,
            logs only to stdout.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        # Already configured (avoid duplicate handlers on repeated imports).
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "project.log")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger
