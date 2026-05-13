"""Slide display-name, embedding-status, and cached-result routes."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from .schemas import SlideRenameRequest

ProjectValidator = Callable[[str | None], Any]
ProjectSlideIds = Callable[[str | None], Awaitable[set[str] | None]]
ProjectModelIds = Callable[[str | None], Awaitable[set[str] | None]]


def create_slide_status_router(
    *,
    db_module: Any,
    logger: logging.Logger,
    require_project: ProjectValidator,
    project_slide_ids: ProjectSlideIds,
    resolve_project_model_ids: ProjectModelIds,
) -> APIRouter:
    """Create routes for slide metadata/status surfaces that depend on DB state."""
    router = APIRouter()

    @router.patch("/api/slides/{slide_id}")
    async def rename_slide(slide_id: str, body: SlideRenameRequest):
        """Update a slide's display_name (alias). Pass null to clear."""
        updated = await db_module.update_slide_display_name(slide_id, body.display_name)
        if not updated:
            raise HTTPException(status_code=404, detail=f"Slide '{slide_id}' not found")
        return {"slide_id": slide_id, "display_name": body.display_name}

    @router.get("/api/slides/{slide_id}/embedding-status")
    async def get_slide_embedding_status(
        slide_id: str,
        project_id: str | None = Query(
            default=None,
            description="Optional project scope for model cache visibility",
        ),
    ):
        """Get embedding and cached-classification status for a slide."""
        require_project(project_id)

        if project_id is not None:
            allowed_slide_ids = await project_slide_ids(project_id)
            if allowed_slide_ids is not None and slide_id not in allowed_slide_ids:
                raise HTTPException(
                    status_code=404,
                    detail=f"Slide {slide_id} is not available in project '{project_id}'",
                )

        status = await db_module.get_slide_embedding_status(slide_id)
        if "error" in status:
            raise HTTPException(status_code=404, detail=status["error"])

        allowed_model_ids = await resolve_project_model_ids(project_id)
        if allowed_model_ids is not None:
            status["cached_model_ids"] = [
                model_id
                for model_id in status.get("cached_model_ids", [])
                if model_id in allowed_model_ids
            ]

        return status

    @router.get("/api/slides/{slide_id}/cached-results")
    async def get_slide_cached_results(
        slide_id: str,
        project_id: str | None = Query(
            default=None,
            description="Optional project scope for cached result visibility",
        ),
    ):
        """Get all cached analysis results for a slide."""
        require_project(project_id)

        if project_id is not None:
            allowed_slide_ids = await project_slide_ids(project_id)
            if allowed_slide_ids is not None and slide_id not in allowed_slide_ids:
                raise HTTPException(
                    status_code=404,
                    detail=f"Slide {slide_id} is not available in project '{project_id}'",
                )

        allowed_model_ids = await resolve_project_model_ids(project_id)

        try:
            cached = await db_module.get_all_cached_results(slide_id)
        except Exception as exc:
            logger.warning("Failed to fetch cached results for %s: %s", slide_id, exc)
            cached = []

        results = []
        for row in cached:
            if allowed_model_ids is not None and row.get("model_id") not in allowed_model_ids:
                continue
            results.append(
                {
                    "model_id": row["model_id"],
                    "score": row["score"],
                    "label": row["label"],
                    "confidence": row["confidence"],
                    "threshold": row.get("threshold"),
                    "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
                }
            )

        return {
            "slide_id": slide_id,
            "results": results,
            "count": len(results),
            "cached": True,
        }

    return router
