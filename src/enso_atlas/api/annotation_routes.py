"""Pathologist annotation route handlers."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, HTTPException

from .schemas import AnnotationCreate, AnnotationUpdate

AuditLogger = Callable[[str, str | None, str, dict[str, Any] | None], None]


def _annotation_payload(row: dict[str, Any]) -> dict[str, Any]:
    created_at = row.get("created_at")
    created_at_text = (
        created_at.isoformat()
        if created_at is not None and hasattr(created_at, "isoformat")
        else str(created_at)
    )
    return {
        "id": row["id"],
        "slide_id": row["slide_id"],
        "type": row["type"],
        "coordinates": row["coordinates"],
        "text": row.get("notes") or row.get("label") or "",
        "label": row.get("label"),
        "notes": row.get("notes"),
        "color": row.get("color", "#3b82f6"),
        "category": row.get("category"),
        "created_at": created_at_text,
        "created_by": None,
    }


def create_annotation_router(
    *,
    db_module: Any,
    logger: logging.Logger,
    log_audit_event: AuditLogger,
) -> APIRouter:
    """Create PostgreSQL-backed annotation routes."""
    router = APIRouter()

    @router.get("/api/slides/{slide_id}/annotations")
    async def get_annotations_endpoint(slide_id: str):
        """Get all annotations for a slide."""
        try:
            rows = await db_module.get_annotations(slide_id)
        except Exception as exc:
            logger.warning("Failed to fetch annotations for %s: %s", slide_id, exc)
            rows = []

        annotations = [_annotation_payload(row) for row in rows]
        return {
            "slide_id": slide_id,
            "annotations": annotations,
            "total": len(annotations),
        }

    @router.post("/api/slides/{slide_id}/annotations")
    async def save_annotation_endpoint(slide_id: str, body: AnnotationCreate):
        """Create a new annotation."""
        annotation_id = f"ann_{uuid.uuid4().hex[:12]}"
        notes = body.notes or body.text or None

        try:
            row = await db_module.create_annotation(
                annotation_id=annotation_id,
                slide_id=slide_id,
                ann_type=body.type,
                coordinates=body.coordinates,
                label=body.label,
                notes=notes,
                color=body.color or "#3b82f6",
                category=body.category,
            )
        except Exception as exc:
            logger.error("Failed to create annotation: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        log_audit_event(
            "annotation_created",
            slide_id,
            "pathologist",
            {"annotation_id": annotation_id, "type": body.type},
        )

        return _annotation_payload(row)

    @router.put("/api/slides/{slide_id}/annotations/{annotation_id}")
    async def update_annotation_endpoint(slide_id: str, annotation_id: str, body: AnnotationUpdate):
        """Update an annotation's label, notes, color, or category."""
        row = await db_module.update_annotation(
            annotation_id=annotation_id,
            label=body.label,
            notes=body.notes,
            color=body.color,
            category=body.category,
        )
        if not row:
            raise HTTPException(status_code=404, detail=f"Annotation {annotation_id} not found")

        return _annotation_payload(row)

    @router.delete("/api/slides/{slide_id}/annotations/{annotation_id}")
    async def delete_annotation_endpoint(slide_id: str, annotation_id: str):
        """Delete an annotation."""
        deleted = await db_module.delete_annotation(annotation_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Annotation {annotation_id} not found")

        log_audit_event(
            "annotation_deleted",
            slide_id,
            "pathologist",
            {"annotation_id": annotation_id},
        )

        return {"success": True, "message": f"Annotation {annotation_id} deleted"}

    @router.get("/api/slides/{slide_id}/annotations/summary")
    async def get_annotations_summary(slide_id: str):
        """Get a summary of annotations for a slide."""
        try:
            rows = await db_module.get_annotations(slide_id)
        except Exception:
            rows = []

        label_counts: dict[str, int] = {}
        for ann in rows:
            label = ann.get("label") or ann.get("category")
            if label:
                label_counts[label] = label_counts.get(label, 0) + 1

        return {
            "slide_id": slide_id,
            "total_annotations": len(rows),
            "by_label": label_counts,
        }

    return router
