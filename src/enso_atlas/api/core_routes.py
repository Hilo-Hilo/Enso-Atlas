"""Core status, perf, and documentation routes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, HTTPException
from starlette.responses import RedirectResponse

from .perf_observability import InMemoryLatencyTracker


def create_core_router(
    *,
    health_provider: Callable[[], dict[str, Any]],
    perf_enabled: bool,
    perf_summary_enabled: bool,
    perf_max_samples: int,
    perf_route_limit: int,
    perf_tracker: InMemoryLatencyTracker,
) -> APIRouter:
    """Create routes that do not own model or slide-processing behavior."""
    router = APIRouter()

    @router.get("/health")
    async def health_check():
        """Health check endpoint."""
        return health_provider()

    @router.get("/api/health")
    async def api_health_check():
        """Health check endpoint under the API prefix."""
        return health_provider()

    @router.get("/api/tags")
    async def get_tags():
        """List all tags."""
        return []

    @router.post("/api/tags")
    async def create_tag(request: dict):
        """Create a tag placeholder until tag persistence is needed."""
        return {
            "name": request.get("name", ""),
            "color": request.get("color", "#888"),
            "count": 0,
        }

    @router.get("/api/perf/latency-summary")
    async def perf_latency_summary():
        """Return rolling p50/p95 latency stats from in-memory request timings."""
        if not perf_enabled:
            return {
                "enabled": False,
                "summary_enabled": False,
                "reason": "Set ENSO_PERF_ENABLED=1 to collect request timings.",
            }

        if not perf_summary_enabled:
            raise HTTPException(
                status_code=404,
                detail="Perf latency summary endpoint disabled. Set ENSO_PERF_SUMMARY_ENABLED=1.",
            )

        return {
            "enabled": True,
            "summary_enabled": True,
            "max_samples": perf_max_samples,
            "route_limit": perf_route_limit,
            "summary": perf_tracker.summary(limit_routes=perf_route_limit),
        }

    @router.get("/")
    async def root():
        """API root endpoint."""
        return {
            "name": "Enso Atlas API",
            "version": "0.1.0",
            "docs": "/api/docs",
        }

    @router.get("/docs", include_in_schema=False)
    async def docs_redirect():
        """Redirect /docs to /api/docs for convenience."""
        return RedirectResponse(url="/api/docs")

    @router.get("/redoc", include_in_schema=False)
    async def redoc_redirect():
        """Redirect /redoc to /api/redoc for convenience."""
        return RedirectResponse(url="/api/redoc")

    return router
