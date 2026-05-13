"""In-memory audit and analysis-history storage."""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timezone
from typing import Any

MAX_HISTORY_SIZE = 100
analysis_history: deque[dict[str, Any]] = deque(maxlen=MAX_HISTORY_SIZE)
audit_log: deque[dict[str, Any]] = deque(maxlen=500)


def get_timestamp() -> str:
    """Get current ISO timestamp."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def log_audit_event(
    event_type: str,
    slide_id: str | None = None,
    user_id: str = "clinician",
    details: dict[str, Any] | None = None,
    *,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Log an audit event for compliance tracking."""
    entry = {
        "timestamp": get_timestamp(),
        "event_type": event_type,
        "user_id": user_id,
        "slide_id": slide_id,
        "details": details or {},
    }
    audit_log.append(entry)
    if logger is not None:
        logger.info("AUDIT: %s - slide=%s user=%s", event_type, slide_id, user_id)
    return entry


def save_analysis_to_history(
    slide_id: str,
    prediction: str,
    score: float,
    confidence: float,
    patches_analyzed: int,
    top_evidence: list[dict[str, Any]],
    similar_cases: list[dict[str, Any]],
    user_id: str = "clinician",
    *,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Save analysis result to history and return the entry."""
    entry = {
        "id": f"{slide_id}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
        "timestamp": get_timestamp(),
        "slide_id": slide_id,
        "user_id": user_id,
        "prediction": prediction,
        "score": score,
        "confidence": confidence,
        "patches_analyzed": patches_analyzed,
        "top_evidence_count": len(top_evidence),
        "similar_cases_count": len(similar_cases),
    }
    analysis_history.append(entry)
    log_audit_event(
        "analysis_completed",
        slide_id,
        user_id,
        {"prediction": prediction, "confidence": confidence},
        logger=logger,
    )
    return entry
