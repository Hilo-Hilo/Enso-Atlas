"""Slide search and pagination routes."""

from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import APIRouter, Query


def create_slide_search_router(
    *,
    list_slides: Callable[..., Awaitable[list[Any]]],
) -> APIRouter:
    router = APIRouter()

    @router.get("/api/slides/search")
    async def search_slides(
        search: str | None = None,
        label: str | None = None,
        has_embeddings: bool | None = None,
        min_patches: int | None = None,
        max_patches: int | None = None,
        tags: str | None = None,
        group_id: str | None = None,
        starred: bool | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        sort_by: str | None = "date",
        sort_order: str | None = "desc",
        page: int = 1,
        per_page: int = 20,
        project_id: str | None = Query(None, description="Filter slides by project"),
    ):
        """Search and paginate slides, optionally scoped by ``project_id``."""
        needs_patch_counts = min_patches is not None or max_patches is not None
        all_slides = await list_slides(project_id=project_id, include_metadata=needs_patch_counts)

        slides_data = [
            {
                "slide_id": s.slide_id,
                "patient_id": s.patient_id,
                "has_wsi": s.has_wsi,
                "has_embeddings": s.has_embeddings,
                "has_level0_embeddings": s.has_level0_embeddings,
                "label": s.label,
                "num_patches": s.num_patches,
                "patient": s.patient.dict() if s.patient else None,
                "dimensions": s.dimensions.dict() if s.dimensions else {"width": 0, "height": 0},
                "mpp": s.mpp,
                "magnification": s.magnification,
            }
            for s in all_slides
        ]

        filtered = slides_data

        if search:
            search_lower = search.lower()
            filtered = [s for s in filtered if search_lower in s.get("slide_id", "").lower()]

        if label:
            label_map = {
                "platinum_sensitive": "1",
                "platinum_resistant": "0",
                "Sensitive": "1",
                "Resistant": "0",
                "sensitive": "1",
                "resistant": "0",
                "responder": "1",
                "non-responder": "0",
            }
            data_label = label_map.get(label, label)
            filtered = [s for s in filtered if s.get("label") == data_label]

        if has_embeddings is not None:
            filtered = [s for s in filtered if s.get("has_embeddings") == has_embeddings]

        if min_patches is not None:
            filtered = [s for s in filtered if (s.get("num_patches") or 0) >= min_patches]

        if max_patches is not None:
            filtered = [s for s in filtered if (s.get("num_patches") or 0) <= max_patches]

        total = len(filtered)
        start = (page - 1) * per_page
        end = start + per_page
        paginated = filtered[start:end]

        return {
            "slides": paginated,
            "total": total,
            "page": page,
            "per_page": per_page,
            "filters": {"label": label, "search": search},
        }

    return router
