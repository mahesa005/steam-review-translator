"""Package logging: console (stderr) + rotating file at ``artifacts/logs/run.log``."""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler

from steam_review_translator.core.config import get_path

ROOT_LOGGER = "steam_review_translator"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
LOG_FILE = "run.log"

_configured = False


def _configure() -> None:
    """Attach handlers to the package root logger exactly once."""
    global _configured
    if _configured:
        return
    root = logging.getLogger(ROOT_LOGGER)
    level_name = os.environ.get("SRT_LOG_LEVEL", "INFO").upper()
    level = logging.getLevelName(level_name)
    root.setLevel(level if isinstance(level, int) else logging.INFO)
    root.propagate = False

    formatter = logging.Formatter(LOG_FORMAT)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    try:
        log_dir = get_path("logs", create=True)
        file_handler = RotatingFileHandler(
            log_dir / LOG_FILE, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError as exc:  # read-only checkout etc.: keep console logging only
        root.warning("File logging disabled: %s", exc)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``steam_review_translator`` namespace.

    Safe to call repeatedly: handlers are installed once on the package root logger and
    children propagate to it. Level comes from ``SRT_LOG_LEVEL`` (default ``INFO``).
    """
    _configure()
    if name == ROOT_LOGGER or name.startswith(ROOT_LOGGER + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_LOGGER}.{name}")
