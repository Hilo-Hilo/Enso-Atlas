"""Database administration/status routes."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException


def create_database_router(
    *,
    db_module: Any,
    db_available_provider: Callable[[], bool],
    data_root_provider: Callable[[], Path],
    embeddings_dir_provider: Callable[[], Path],
) -> APIRouter:
    """Create lightweight database operational routes."""
    router = APIRouter()

    @router.post("/api/db/repopulate")
    async def repopulate_database():
        """Force re-population of PostgreSQL from flat files."""
        if not db_available_provider():
            raise HTTPException(status_code=503, detail="Database not available")
        try:
            t0 = time.time()
            await db_module.populate_from_flat_files(
                data_root=data_root_provider(),
                embeddings_dir=embeddings_dir_provider(),
            )
            elapsed = time.time() - t0
            return {
                "status": "ok",
                "message": f"Database repopulated in {elapsed:.1f}s",
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @router.get("/api/db/status")
    async def database_status():
        """Check database connection and population status."""
        if not db_available_provider():
            return {"status": "unavailable", "message": "PostgreSQL not connected"}
        try:
            pool = await db_module.get_pool()
            async with pool.acquire() as conn:
                slide_count = await conn.fetchval("SELECT COUNT(*) FROM slides")
                patient_count = await conn.fetchval("SELECT COUNT(*) FROM patients")
                meta_count = await conn.fetchval("SELECT COUNT(*) FROM slide_metadata")
                dims_count = await conn.fetchval("SELECT COUNT(*) FROM slides WHERE width > 0")
            return {
                "status": "connected",
                "slides": slide_count,
                "patients": patient_count,
                "metadata_entries": meta_count,
                "slides_with_dimensions": dims_count,
            }
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    return router
