"""Analysis-history and audit-log API routes."""

from __future__ import annotations

from fastapi import APIRouter

from .audit import analysis_history, audit_log
from .schemas import (
    AnalysisHistoryEntry,
    AnalysisHistoryResponse,
    AuditLogEntry,
    AuditLogResponse,
)

router = APIRouter()


@router.get("/api/history", response_model=AnalysisHistoryResponse)
async def get_analysis_history(
    limit: int = 50,
    offset: int = 0,
    slide_id: str | None = None,
    prediction: str | None = None,
):
    """Get recent analysis history, optionally filtered by slide/prediction."""
    all_entries = list(analysis_history)

    if slide_id:
        all_entries = [e for e in all_entries if e["slide_id"] == slide_id]
    if prediction:
        all_entries = [e for e in all_entries if e["prediction"] == prediction.upper()]

    all_entries.sort(key=lambda x: x["timestamp"], reverse=True)
    limit = min(limit, 100)
    paginated = all_entries[offset : offset + limit]

    return AnalysisHistoryResponse(
        analyses=[AnalysisHistoryEntry(**e) for e in paginated],
        total=len(all_entries),
    )


@router.get("/api/slides/{slide_id}/history", response_model=AnalysisHistoryResponse)
async def get_slide_history(slide_id: str, limit: int = 20):
    """Get analysis history for a specific slide."""
    slide_entries = [e for e in analysis_history if e["slide_id"] == slide_id]
    slide_entries.sort(key=lambda x: x["timestamp"], reverse=True)

    limit = min(limit, 50)
    paginated = slide_entries[:limit]

    return AnalysisHistoryResponse(
        analyses=[AnalysisHistoryEntry(**e) for e in paginated],
        total=len(slide_entries),
    )


@router.get("/api/audit-log", response_model=AuditLogResponse)
async def get_audit_log(
    limit: int = 100,
    offset: int = 0,
    event_type: str | None = None,
    slide_id: str | None = None,
):
    """Get audit log entries for compliance tracking."""
    all_entries = list(audit_log)

    if event_type:
        all_entries = [e for e in all_entries if e["event_type"] == event_type]
    if slide_id:
        all_entries = [e for e in all_entries if e.get("slide_id") == slide_id]

    all_entries.sort(key=lambda x: x["timestamp"], reverse=True)
    limit = min(limit, 500)
    paginated = all_entries[offset : offset + limit]

    return AuditLogResponse(
        entries=[AuditLogEntry(**e) for e in paginated],
        total=len(all_entries),
    )
