"""
Slide Metadata Manager - Persistent storage for slide tags, groups, and custom metadata.

Provides JSON-file based storage for:
- Tags (multiple per slide)
- Groups/collections
- Custom metadata fields
- Notes/comments
- Analysis history references
- Star/favorite status
"""

import json
import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Set
from datetime import datetime
from pydantic import BaseModel, Field
import logging
from threading import Lock

logger = logging.getLogger(__name__)


# ============== Pydantic Models ==============

class SlideMetadata(BaseModel):
    """Metadata for a single slide."""
    slide_id: str
    tags: List[str] = Field(default_factory=list)
    groups: List[str] = Field(default_factory=list)  # Group IDs
    starred: bool = False
    notes: Optional[str] = None
    custom_metadata: Dict[str, Any] = Field(default_factory=dict)
    analysis_history: List[str] = Field(default_factory=list)  # Analysis IDs
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


class SlideGroup(BaseModel):
    """A collection/group of slides."""
    id: str
    name: str
    description: Optional[str] = None
    color: Optional[str] = None  # Hex color for UI display
    slide_ids: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


class MetadataStore(BaseModel):
    """Root storage model for all slide metadata."""
    version: str = "1.0"
    slides: Dict[str, SlideMetadata] = Field(default_factory=dict)
    groups: Dict[str, SlideGroup] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)  # All unique tags


# ============== Request/Response Models ==============

class TagsRequest(BaseModel):
    """Request to add tags to a slide."""
    tags: List[str] = Field(..., min_length=1, description="List of tags to add")


class GroupCreateRequest(BaseModel):
    """Request to create a new group."""
    name: str = Field(..., min_length=1, max_length=256)
    description: Optional[str] = None
    color: Optional[str] = Field(None, pattern="^#[0-9A-Fa-f]{6}$")


class GroupUpdateRequest(BaseModel):
    """Request to update a group."""
    name: Optional[str] = Field(None, min_length=1, max_length=256)
    description: Optional[str] = None
    color: Optional[str] = Field(None, pattern="^#[0-9A-Fa-f]{6}$")


class SlidesAddRequest(BaseModel):
    """Request to add slides to a group."""
    slide_ids: List[str] = Field(..., min_length=1)


class MetadataUpdateRequest(BaseModel):
    """Request to update slide custom metadata."""
    notes: Optional[str] = None
    custom_metadata: Optional[Dict[str, Any]] = None


class SlideMetadataResponse(BaseModel):
    """Response containing slide metadata."""
    slide_id: str
    tags: List[str]
    groups: List[str]
    group_names: List[str] = Field(default_factory=list)  # Resolved group names
    starred: bool
    notes: Optional[str]
    custom_metadata: Dict[str, Any]
    created_at: str
    updated_at: str


class GroupResponse(BaseModel):
    """Response containing group details."""
    id: str
    name: str
    description: Optional[str]
    color: Optional[str]
    slide_count: int
    slide_ids: List[str]
    created_at: str
    updated_at: str


class AllTagsResponse(BaseModel):
    """Response containing all tags."""
    tags: List[str]
    tag_counts: Dict[str, int]


# ============== Metadata Manager ==============

class SlideMetadataManager:
    """
    Thread-safe manager for slide metadata storage.
    
    Uses JSON file for persistence with in-memory caching.
    """
    
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self._store: MetadataStore = MetadataStore()
        self._lock = Lock()
        self._load()
    
    def _load(self) -> None:
        """Load metadata from disk."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                self._store = MetadataStore(**data)
                logger.info(f"Loaded slide metadata: {len(self._store.slides)} slides, {len(self._store.groups)} groups")
            except Exception as e:
                logger.error(f"Failed to load metadata from {self.storage_path}: {e}")
                self._store = MetadataStore()
        else:
            logger.info(f"No existing metadata file, starting fresh at {self.storage_path}")
            self._save()
    
    def _save(self) -> None:
        """Save metadata to disk."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump(self._store.model_dump(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata to {self.storage_path}: {e}")
    
    def _get_timestamp(self) -> str:
        return datetime.utcnow().isoformat() + "Z"
    
    def _ensure_slide(self, slide_id: str) -> SlideMetadata:
        """Ensure a slide entry exists, creating if needed."""
        if slide_id not in self._store.slides:
            self._store.slides[slide_id] = SlideMetadata(slide_id=slide_id)
        return self._store.slides[slide_id]
    
    # ============== Tag Operations ==============
    
    def get_all_tags(self) -> AllTagsResponse:
        """Get all unique tags with counts."""
        with self._lock:
            tag_counts: Dict[str, int] = {}
            for slide in self._store.slides.values():
                for tag in slide.tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            return AllTagsResponse(
                tags=sorted(tag_counts.keys()),
                tag_counts=tag_counts
            )
    
    def add_tags(self, slide_id: str, tags: List[str]) -> SlideMetadata:
        """Add tags to a slide."""
        with self._lock:
            slide = self._ensure_slide(slide_id)
            
            # Add new tags (avoid duplicates)
            for tag in tags:
                tag = tag.strip()
                if tag and tag not in slide.tags:
                    slide.tags.append(tag)
                    # Track in global tags list
                    if tag not in self._store.tags:
                        self._store.tags.append(tag)
            
            slide.updated_at = self._get_timestamp()
            self._save()
            return slide
    
    def remove_tag(self, slide_id: str, tag: str) -> SlideMetadata:
        """Remove a tag from a slide."""
        with self._lock:
            if slide_id not in self._store.slides:
                raise ValueError(f"Slide {slide_id} not found")
            
            slide = self._store.slides[slide_id]
            if tag in slide.tags:
                slide.tags.remove(tag)
                slide.updated_at = self._get_timestamp()
                self._save()
            
            return slide
    
    def remove_tags(self, slide_id: str, tags: List[str]) -> SlideMetadata:
        """Remove multiple tags from a slide."""
        with self._lock:
            if slide_id not in self._store.slides:
                raise ValueError(f"Slide {slide_id} not found")
            
            slide = self._store.slides[slide_id]
            modified = False
            for tag in tags:
                if tag in slide.tags:
                    slide.tags.remove(tag)
                    modified = True
            
            if modified:
                slide.updated_at = self._get_timestamp()
                self._save()
            
            return slide
    
    # ============== Group Operations ==============
    
    def get_all_groups(self) -> List[GroupResponse]:
        """Get all groups."""
        with self._lock:
            return [
                GroupResponse(
                    id=group.id,
                    name=group.name,
                    description=group.description,
                    color=group.color,
                    slide_count=len(group.slide_ids),
                    slide_ids=group.slide_ids,
                    created_at=group.created_at,
                    updated_at=group.updated_at,
                )
                for group in self._store.groups.values()
            ]
    
    def list_groups(self) -> List[GroupResponse]:
        """Alias for get_all_groups() for router compatibility."""
        return self.get_all_groups()
    
    def get_group(self, group_id: str) -> Optional[GroupResponse]:
        """Get a single group by ID."""
        with self._lock:
            if group_id not in self._store.groups:
                return None
            group = self._store.groups[group_id]
            return GroupResponse(
                id=group.id,
                name=group.name,
                description=group.description,
                color=group.color,
                slide_count=len(group.slide_ids),
                slide_ids=group.slide_ids,
                created_at=group.created_at,
                updated_at=group.updated_at,
            )
    
    def create_group(self, name: str, description: Optional[str] = None, 
                     color: Optional[str] = None) -> GroupResponse:
        """Create a new group."""
        with self._lock:
            # Generate unique ID
            group_id = f"grp_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"
            
            group = SlideGroup(
                id=group_id,
                name=name,
                description=description,
                color=color,
            )
            
            self._store.groups[group_id] = group
            self._save()
            
            return GroupResponse(
                id=group.id,
                name=group.name,
                description=group.description,
                color=group.color,
                slide_count=0,
                slide_ids=[],
                created_at=group.created_at,
                updated_at=group.updated_at,
            )
    
    def update_group(self, group_id: str, name: Optional[str] = None,
                     description: Optional[str] = None, 
                     color: Optional[str] = None) -> GroupResponse:
        """Update a group."""
        with self._lock:
            if group_id not in self._store.groups:
                raise ValueError(f"Group {group_id} not found")
            
            group = self._store.groups[group_id]
            
            if name is not None:
                group.name = name
            if description is not None:
                group.description = description
            if color is not None:
                group.color = color
            
            group.updated_at = self._get_timestamp()
            self._save()
            
            return GroupResponse(
                id=group.id,
                name=group.name,
                description=group.description,
                color=group.color,
                slide_count=len(group.slide_ids),
                slide_ids=group.slide_ids,
                created_at=group.created_at,
                updated_at=group.updated_at,
            )
    
    def delete_group(self, group_id: str) -> bool:
        """Delete a group."""
        with self._lock:
            if group_id not in self._store.groups:
                raise ValueError(f"Group {group_id} not found")
            
            group = self._store.groups[group_id]
            
            # Remove group reference from all slides
            for slide_id in group.slide_ids:
                if slide_id in self._store.slides:
                    slide = self._store.slides[slide_id]
                    if group_id in slide.groups:
                        slide.groups.remove(group_id)
            
            del self._store.groups[group_id]
            self._save()
            return True
    
    def add_slides_to_group(self, group_id: str, slide_ids: List[str]) -> GroupResponse:
        """Add slides to a group."""
        with self._lock:
            if group_id not in self._store.groups:
                raise ValueError(f"Group {group_id} not found")
            
            group = self._store.groups[group_id]
            
            for slide_id in slide_ids:
                if slide_id not in group.slide_ids:
                    group.slide_ids.append(slide_id)
                
                # Also update slide's groups list
                slide = self._ensure_slide(slide_id)
                if group_id not in slide.groups:
                    slide.groups.append(group_id)
                    slide.updated_at = self._get_timestamp()
            
            group.updated_at = self._get_timestamp()
            self._save()
            
            return GroupResponse(
                id=group.id,
                name=group.name,
                description=group.description,
                color=group.color,
                slide_count=len(group.slide_ids),
                slide_ids=group.slide_ids,
                created_at=group.created_at,
                updated_at=group.updated_at,
            )
    
    def remove_slide_from_group(self, group_id: str, slide_id: str) -> GroupResponse:
        """Remove a slide from a group."""
        with self._lock:
            if group_id not in self._store.groups:
                raise ValueError(f"Group {group_id} not found")
            
            group = self._store.groups[group_id]
            
            if slide_id in group.slide_ids:
                group.slide_ids.remove(slide_id)
                group.updated_at = self._get_timestamp()
            
            # Also update slide's groups list
            if slide_id in self._store.slides:
                slide = self._store.slides[slide_id]
                if group_id in slide.groups:
                    slide.groups.remove(group_id)
                    slide.updated_at = self._get_timestamp()
            
            self._save()
            
            return GroupResponse(
                id=group.id,
                name=group.name,
                description=group.description,
                color=group.color,
                slide_count=len(group.slide_ids),
                slide_ids=group.slide_ids,
                created_at=group.created_at,
                updated_at=group.updated_at,
            )
    
    def remove_slides_from_group(self, group_id: str, slide_ids: List[str]) -> GroupResponse:
        """Remove multiple slides from a group."""
        with self._lock:
            if group_id not in self._store.groups:
                raise ValueError(f"Group {group_id} not found")
            
            group = self._store.groups[group_id]
            
            for slide_id in slide_ids:
                if slide_id in group.slide_ids:
                    group.slide_ids.remove(slide_id)
                
                # Also update slide's groups list
                if slide_id in self._store.slides:
                    slide = self._store.slides[slide_id]
                    if group_id in slide.groups:
                        slide.groups.remove(group_id)
                        slide.updated_at = self._get_timestamp()
            
            group.updated_at = self._get_timestamp()
            self._save()
            
            return GroupResponse(
                id=group.id,
                name=group.name,
                description=group.description,
                color=group.color,
                slide_count=len(group.slide_ids),
                slide_ids=group.slide_ids,
                created_at=group.created_at,
                updated_at=group.updated_at,
            )
    
    # ============== Metadata Operations ==============
    
    def get_slide_metadata(self, slide_id: str) -> Optional[SlideMetadataResponse]:
        """Get metadata for a slide."""
        with self._lock:
            if slide_id not in self._store.slides:
                return None
            
            slide = self._store.slides[slide_id]
            
            # Resolve group names
            group_names = []
            for gid in slide.groups:
                if gid in self._store.groups:
                    group_names.append(self._store.groups[gid].name)
            
            return SlideMetadataResponse(
                slide_id=slide.slide_id,
                tags=slide.tags,
                groups=slide.groups,
                group_names=group_names,
                starred=slide.starred,
                notes=slide.notes,
                custom_metadata=slide.custom_metadata,
                created_at=slide.created_at,
                updated_at=slide.updated_at,
            )
    
    def update_metadata(self, slide_id: str, notes: Optional[str] = None,
                        custom_metadata: Optional[Dict[str, Any]] = None) -> SlideMetadata:
        """Update slide notes and custom metadata."""
        with self._lock:
            slide = self._ensure_slide(slide_id)
            
            if notes is not None:
                slide.notes = notes
            
            if custom_metadata is not None:
                # Merge with existing metadata
                slide.custom_metadata.update(custom_metadata)
            
            slide.updated_at = self._get_timestamp()
            self._save()
            return slide
    
    def toggle_star(self, slide_id: str) -> bool:
        """Toggle star/favorite status. Returns new status."""
        with self._lock:
            slide = self._ensure_slide(slide_id)
            slide.starred = not slide.starred
            slide.updated_at = self._get_timestamp()
            self._save()
            return slide.starred
    
    # ============== Search Operations ==============
    
    def search_slides(
        self,
        available_slide_ids: List[str],
        tags: Optional[List[str]] = None,
        group_id: Optional[str] = None,
        starred: Optional[bool] = None,
        has_notes: Optional[bool] = None,
    ) -> List[str]:
        """
        Search/filter slides based on metadata criteria.
        Returns list of matching slide IDs.
        """
        with self._lock:
            # Start with all available slides
            result_ids = set(available_slide_ids)
            
            # Filter by group
            if group_id:
                if group_id in self._store.groups:
                    group_slides = set(self._store.groups[group_id].slide_ids)
                    result_ids = result_ids.intersection(group_slides)
                else:
                    return []  # Group doesn't exist
            
            # Filter by tags (any match)
            if tags:
                tag_matches = set()
                for slide_id in result_ids:
                    if slide_id in self._store.slides:
                        slide = self._store.slides[slide_id]
                        if any(t in slide.tags for t in tags):
                            tag_matches.add(slide_id)
                result_ids = tag_matches
            
            # Filter by starred
            if starred is not None:
                starred_matches = set()
                for slide_id in result_ids:
                    if slide_id in self._store.slides:
                        slide = self._store.slides[slide_id]
                        if slide.starred == starred:
                            starred_matches.add(slide_id)
                    elif not starred:
                        # Slides without metadata are not starred
                        starred_matches.add(slide_id)
                result_ids = starred_matches
            
            # Filter by has_notes
            if has_notes is not None:
                notes_matches = set()
                for slide_id in result_ids:
                    if slide_id in self._store.slides:
                        slide = self._store.slides[slide_id]
                        has = bool(slide.notes and slide.notes.strip())
                        if has == has_notes:
                            notes_matches.add(slide_id)
                    elif not has_notes:
                        notes_matches.add(slide_id)
                result_ids = notes_matches
            
            return list(result_ids)
    
    def get_slides_by_group(self, group_id: str) -> List[str]:
        """Get all slide IDs in a group."""
        with self._lock:
            if group_id not in self._store.groups:
                return []
            return list(self._store.groups[group_id].slide_ids)
    
    def get_starred_slides(self) -> List[str]:
        """Get all starred slide IDs."""
        with self._lock:
            return [
                slide.slide_id
                for slide in self._store.slides.values()
                if slide.starred
            ]
    
    # ============== Bulk Operations ==============
    
    def get_metadata_for_slides(self, slide_ids: List[str]) -> Dict[str, SlideMetadataResponse]:
        """Get metadata for multiple slides at once."""
        with self._lock:
            result = {}
            for slide_id in slide_ids:
                if slide_id in self._store.slides:
                    slide = self._store.slides[slide_id]
                    group_names = [
                        self._store.groups[gid].name
                        for gid in slide.groups
                        if gid in self._store.groups
                    ]
                    result[slide_id] = SlideMetadataResponse(
                        slide_id=slide.slide_id,
                        tags=slide.tags,
                        groups=slide.groups,
                        group_names=group_names,
                        starred=slide.starred,
                        notes=slide.notes,
                        custom_metadata=slide.custom_metadata,
                        created_at=slide.created_at,
                        updated_at=slide.updated_at,
                    )
            return result


# ============== FastAPI Router ==============

from fastapi import APIRouter, HTTPException, Query

def create_metadata_router(manager: SlideMetadataManager, get_available_slides: callable) -> APIRouter:
    """Create FastAPI router for slide metadata endpoints."""
    router = APIRouter(prefix="/api/metadata", tags=["Slide Metadata"])
    
    @router.get("/tags")
    async def list_all_tags():
        """Get all unique tags."""
        return manager.get_all_tags()
    
    @router.get("/groups")
    async def list_all_groups():
        """Get all groups."""
        groups = manager.list_groups()
        return {"groups": [g.dict() for g in groups]}
    
    @router.post("/groups")
    async def create_group(request: GroupCreateRequest):
        """Create a new group."""
        group = manager.create_group(request.name, request.description, request.color)
        return {"group": group.dict()}
    
    @router.get("/groups/{group_id}")
    async def get_group(group_id: str):
        """Get a specific group."""
        group = manager.get_group(group_id)
        if not group:
            raise HTTPException(status_code=404, detail=f"Group {group_id} not found")
        return {"group": group.dict()}
    
    @router.patch("/groups/{group_id}")
    async def update_group(group_id: str, request: GroupUpdateRequest):
        """Update a group."""
        group = manager.update_group(group_id, request.name, request.description, request.color)
        if not group:
            raise HTTPException(status_code=404, detail=f"Group {group_id} not found")
        return {"group": group.dict()}
    
    @router.delete("/groups/{group_id}")
    async def delete_group(group_id: str):
        """Delete a group."""
        success = manager.delete_group(group_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Group {group_id} not found")
        return {"status": "deleted"}
    
    @router.post("/groups/{group_id}/slides")
    async def add_slides_to_group(group_id: str, request: SlidesAddRequest):
        """Add slides to a group."""
        group = manager.add_slides_to_group(group_id, request.slide_ids)
        if not group:
            raise HTTPException(status_code=404, detail=f"Group {group_id} not found")
        return {"group": group.dict()}
    
    @router.delete("/groups/{group_id}/slides")
    async def remove_slides_from_group(group_id: str, request: SlidesAddRequest):
        """Remove slides from a group."""
        group = manager.remove_slides_from_group(group_id, request.slide_ids)
        if not group:
            raise HTTPException(status_code=404, detail=f"Group {group_id} not found")
        return {"group": group.dict()}
    
    @router.get("/slides/{slide_id}")
    async def get_slide_metadata(slide_id: str):
        """Get metadata for a specific slide."""
        meta = manager.get_slide_metadata(slide_id)
        if not meta:
            return {"slide_id": slide_id, "tags": [], "groups": [], "starred": False, "notes": None}
        return {"metadata": meta.dict()}
    
    @router.post("/slides/{slide_id}/tags")
    async def add_tags_to_slide(slide_id: str, request: TagsRequest):
        """Add tags to a slide."""
        slide = manager.add_tags(slide_id, request.tags)
        return {"slide_id": slide.slide_id, "tags": slide.tags}
    
    @router.delete("/slides/{slide_id}/tags")
    async def remove_tags_from_slide(slide_id: str, request: TagsRequest):
        """Remove tags from a slide."""
        slide = manager.remove_tags(slide_id, request.tags)
        return {"slide_id": slide.slide_id, "tags": slide.tags}
    
    @router.post("/slides/{slide_id}/star")
    async def toggle_slide_star(slide_id: str):
        """Toggle star/favorite status for a slide."""
        starred = manager.toggle_star(slide_id)
        return {"slide_id": slide_id, "starred": starred}
    
    @router.patch("/slides/{slide_id}")
    async def update_slide_metadata(slide_id: str, notes: Optional[str] = None, 
                                     custom_metadata: Optional[Dict[str, Any]] = None):
        """Update slide notes and custom metadata."""
        slide = manager.update_metadata(slide_id, notes, custom_metadata)
        return {"slide_id": slide.slide_id, "notes": slide.notes, "custom_metadata": slide.custom_metadata}
    
    @router.get("/search")
    async def search_slides(
        tags: Optional[str] = Query(None, description="Comma-separated tags"),
        group_id: Optional[str] = None,
        starred: Optional[bool] = None,
        has_notes: Optional[bool] = None,
    ):
        """Search/filter slides by metadata criteria."""
        tag_list = tags.split(",") if tags else None
        available = get_available_slides()
        results = manager.search_slides(available, tag_list, group_id, starred, has_notes)
        return {"slide_ids": results, "count": len(results)}
    
    @router.post("/bulk/metadata")
    async def get_bulk_metadata(request: SlidesAddRequest):
        """Get metadata for multiple slides at once."""
        metadata = manager.get_metadata_for_slides(request.slide_ids)
        return {"metadata": {k: v.dict() for k, v in metadata.items()}}
    
    return router
