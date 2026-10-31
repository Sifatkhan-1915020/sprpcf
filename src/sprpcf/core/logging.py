"""Logging and diagnostic infrastructure for SPR-PCF Studio."""

import logging
import sys
from typing import Optional

_LOGGER_NAME = "sprpcf"
_logger: Optional[logging.Logger] = None


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return library logger or a specific sub-logger."""
    global _logger
    if _logger is None:
        _logger = logging.getLogger(_LOGGER_NAME)
        _logger.setLevel(logging.INFO)
        if not _logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            handler.setFormatter(formatter)
            _logger.addHandler(handler)

    if name:
        return logging.getLogger(f"{_LOGGER_NAME}.{name}")
    return _logger


def set_log_level(level: int) -> None:
    """Set global log level (e.g. logging.DEBUG, logging.INFO, logging.WARNING)."""
    get_logger().setLevel(level)

