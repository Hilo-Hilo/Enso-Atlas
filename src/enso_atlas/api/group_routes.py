"""Persistent slide-group route handlers."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .slide_metadata import GroupResponse, SlideMetadataManager


def _group_payload(group: GroupResponse) -> dict:
    return {
        "id": group.id,
        "name": group.name,
        "description": group.description,
        "slide_ids": group.slide_ids,
        "created_at": group.created_at,
        "updated_at": group.updated_at,
    }


def create_group_router(metadata_manager: SlideMetadataManager) -> APIRouter:
    """Create `/api/groups` and bulk group routes backed by metadata storage."""
    router = APIRouter()

    @router.get("/api/groups")
    async def get_groups():
        """List all slide groups with persistent storage."""
        return [_group_payload(group) for group in metadata_manager.list_groups()]

    @router.post("/api/groups")
    async def create_group_endpoint(request: dict):
        """Create a new slide group with persistent storage."""
        name = request.get("name", "").strip()
        if not name:
            raise HTTPException(status_code=400, detail="Group name is required")
        group = metadata_manager.create_group(
            name,
            request.get("description"),
            request.get("color"),
        )
        return _group_payload(group)

    @router.get("/api/groups/{group_id}")
    async def get_group_endpoint(group_id: str):
        """Get a specific slide group."""
        group = metadata_manager.get_group(group_id)
        if not group:
            raise HTTPException(status_code=404, detail=f"Group {group_id} not found")
        return _group_payload(group)

    @router.patch("/api/groups/{group_id}")
    async def update_group_endpoint(group_id: str, request: dict):
        """Update a slide group."""
        try:
            group = metadata_manager.update_group(
                group_id,
                name=request.get("name"),
                description=request.get("description"),
                color=request.get("color"),
            )
            return _group_payload(group)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.delete("/api/groups/{group_id}")
    async def delete_group_endpoint(group_id: str):
        """Delete a slide group."""
        try:
            metadata_manager.delete_group(group_id)
            return {"success": True}
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/api/groups/{group_id}/slides")
    async def add_slides_to_group_endpoint(group_id: str, request: dict):
        """Add slides to a group."""
        slide_ids = request.get("slide_ids", [])
        if not slide_ids:
            raise HTTPException(status_code=400, detail="slide_ids is required")
        try:
            group = metadata_manager.add_slides_to_group(group_id, slide_ids)
            return {"success": True, "group": _group_payload(group)}
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.delete("/api/groups/{group_id}/slides/{slide_id}")
    async def remove_slide_from_group_endpoint(group_id: str, slide_id: str):
        """Remove a slide from a group."""
        try:
            group = metadata_manager.remove_slide_from_group(group_id, slide_id)
            return {"success": True, "group": _group_payload(group)}
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/api/bulk/group")
    async def bulk_add_to_group(request: dict):
        """Add multiple slides to a group."""
        slide_ids = request.get("slide_ids", [])
        group_id = request.get("group_id", "")
        if not slide_ids:
            raise HTTPException(status_code=400, detail="slide_ids is required")
        if not group_id:
            raise HTTPException(status_code=400, detail="group_id is required")
        try:
            metadata_manager.add_slides_to_group(group_id, slide_ids)
            return {"success": True, "count": len(slide_ids)}
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return router
