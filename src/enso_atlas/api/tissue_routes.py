"""Tissue-region classification routes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter

from .schemas import ClassifyRegionRequest, ClassifyRegionResponse


def create_tissue_router(
    *,
    classify_tissue_type: Callable[[int, int, int | None], dict[str, Any]],
) -> APIRouter:
    """Create tissue-region classification routes."""
    router = APIRouter()

    @router.post("/api/classify-region", response_model=ClassifyRegionResponse)
    async def classify_region(request: ClassifyRegionRequest):
        """Classify tissue type for a region."""
        result = classify_tissue_type(request.x, request.y, request.patch_index)
        return ClassifyRegionResponse(**result)

    return router
