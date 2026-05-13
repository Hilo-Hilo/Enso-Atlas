"""Model registry/listing API routes."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from .schemas import AvailableModelsResponse

ModelIdsResolver = Callable[[str | None], Awaitable[set[str] | None]]
TimingLogger = Callable[[str, float], float]


def create_model_router(
    *,
    multi_model_provider: Callable[[], Any],
    resolve_project_model_ids: ModelIdsResolver,
    log_timing: Callable[..., float],
) -> APIRouter:
    """Create model-listing routes."""
    router = APIRouter()

    @router.get("/api/models", response_model=AvailableModelsResponse)
    async def list_available_models(
        project_id: str | None = Query(None, description="Filter models by project"),
    ):
        """List available TransMIL models, optionally filtered by project."""
        started_at = time.perf_counter()
        multi_model_inference = multi_model_provider()
        if multi_model_inference is None:
            raise HTTPException(status_code=503, detail="Multi-model inference not initialized")

        models = multi_model_inference.get_available_models()
        allowed_ids = await resolve_project_model_ids(project_id)

        if allowed_ids is not None:
            models = [m for m in models if m.get("id", m.get("model_id")) in allowed_ids]

        response = AvailableModelsResponse(models=models)
        log_timing(
            "api.models",
            started_at,
            project_id=project_id,
            total_models=len(response.models),
            scoped=allowed_ids is not None,
        )
        return response

    return router
