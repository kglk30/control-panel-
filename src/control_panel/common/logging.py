"""Structured logging setup for Control Panel using structlog.

Logs go to stderr. The CLI's Rich table output goes to stdout.
This keeps log lines from polluting Rich tables.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import structlog

__all__ = ["setup_logging", "get_logger"]


def _make_stderr_utf8() -> None:
    """Reconfigure stderr to use UTF-8 if possible (Windows only need)."""
    if hasattr(sys.stderr, 'reconfigure'):
        try:
            sys.stderr.reconfigure(encoding='utf-8')  # type: ignore[union-attr]
        except Exception:
            pass


# Apply at import time so it's done before anything else
_make_stderr_utf8()


def setup_logging(log_dir: Optional[Path] = None, verbose: bool = False) -> None:
    """Configure structlog to write to stderr with optional file sink.

    Call at CLI startup before any business logic.

    Args:
        log_dir: Directory to write daily log files. None = stderr only.
        verbose: If True, use DEBUG level with human-readable dev renderer.
    """
    level = logging.DEBUG if verbose else logging.INFO

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{datetime.now().strftime('%Y-%m-%d')}.log"
        handlers.append(logging.FileHandler(str(log_file), encoding="utf-8"))

    logging.basicConfig(level=level, handlers=handlers, format="%(message)s", force=True)

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if verbose:
        processors: list[structlog.types.Processor] = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(),
        ]
    else:
        processors = [
            *shared_processors,
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        # PrintLoggerFactory with no file arg writes to stdout; explicitly use stderr
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger for the given module name."""
    return structlog.get_logger(name)
