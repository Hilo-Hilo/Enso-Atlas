"""Runtime configuration helpers for API startup."""

from __future__ import annotations

import logging
import os


def env_flag(name: str, default: bool) -> bool:
    """Parse a boolean environment variable with common truthy values."""

    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_int(
    name: str,
    default: int,
    *,
    minimum: int = 1,
    maximum: int | None = None,
    logger: logging.Logger | None = None,
) -> int:
    """Parse and clamp an integer environment variable."""

    raw = os.environ.get(name)
    if raw is None:
        value = int(default)
    else:
        try:
            value = int(raw)
        except (TypeError, ValueError):
            if logger is not None:
                logger.warning(
                    "Invalid integer for %s=%r; using default=%d",
                    name,
                    raw,
                    default,
                )
            value = int(default)

    if value < minimum:
        if logger is not None:
            logger.warning(
                "Value for %s=%r below minimum=%d; clamping.",
                name,
                raw if raw is not None else value,
                minimum,
            )
        value = minimum

    if maximum is not None and value > maximum:
        if logger is not None:
            logger.warning(
                "Value for %s=%r above maximum=%d; clamping.",
                name,
                raw if raw is not None else value,
                maximum,
            )
        value = maximum

    return value
