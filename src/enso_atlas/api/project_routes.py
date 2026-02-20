"""
Project-aware API routes for Enso Atlas.

Provides endpoints to list, inspect, create, update, delete, and query
projects. Existing read-only routes continue to work unchanged.

Routes:
    GET    /api/projects                           -- list all projects
    POST   /api/projects                           -- create a new project
    GET    /api/projects/{project_id}              -- get project details
    PUT    /api/projects/{project_id}              -- update project config
    DELETE /api/projects/{project_id}              -- delete project (keeps data)
    GET    /api/projects/{project_id}/slides       -- slides for this project
    GET    /api/projects/{project_id}/status       -- project readiness status
    POST   /api/projects/{project_id}/upload       -- upload a slide file
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from .projects import ProjectConfig, ProjectRegistry, default_dataset_paths

logger = logging.getLogger(__name__)

# The registry is injected at startup via `set_registry()`
_registry: Optional[ProjectRegistry] = None


def set_registry(registry: ProjectRegistry):
    """Called from main.py during startup to inject the loaded registry."""
    global _registry
    _registry = registry


def get_registry() -> ProjectRegistry:
    """Get the project registry, raising 503 if not yet loaded."""
    if _registry is None:
        raise HTTPException(
            status_code=503,
            detail="Project registry not initialized yet. Server is starting up.",
        )
    return _registry


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

ALLOWED_SLIDE_EXTENSIONS = {".svs", ".tiff", ".tif", ".ndpi"}


class CreateProjectRequest(BaseModel):
    """Body for POST /api/projects."""
    id: str = Field(..., min_length=1, max_length=128)
    name: str = Field(..., min_length=1, max_length=256)
    cancer_type: str = Field(..., min_length=1)
    prediction_target: str = Field(..., min_length=1)
    classes: List[str] = Field(default_factory=lambda: ["resistant", "sensitive"])
    positive_class: str = "sensitive"
    description: str = ""
    slide_ids: Optional[List[str]] = None
    model_ids: Optional[List[str]] = None


class UpdateProjectRequest(BaseModel):
    """Body for PUT /api/projects/{project_id}.

    All fields are optional -- only provided fields are updated.
    """
    name: Optional[str] = None
    cancer_type: Optional[str] = None
    prediction_target: Optional[str] = None
    classes: Optional[List[str]] = None
    positive_class: Optional[str] = None
    description: Optional[str] = None
    dataset: Optional[Dict[str, Any]] = None
    models: Optional[Dict[str, Any]] = None
    threshold: Optional[float] = None


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/api/projects", tags=["projects"])


@router.get("")
async def list_projects():
    """
    List all configured projects.

    Returns a list of project summaries (id, name, cancer_type, description, classes).
    """
    reg = get_registry()
    projects = reg.list_projects()
    result = []
    for pid, proj in projects.items():
        result.append({
            "id": proj.id,
            "name": proj.name,
            "cancer_type": proj.cancer_type,
            "prediction_target": proj.prediction_target,
            "classes": proj.classes,
            "positive_class": proj.positive_class,
            "description": proj.description,
            "is_default": pid == reg.default_project_id,
        })
    return {
        "projects": result,
        "default_project": reg.default_project_id,
        "count": len(result),
    }


@router.post("", status_code=201)
async def create_project(body: CreateProjectRequest):
    """
    Create a new project.

    Adds the project to projects.yaml and creates its data directories
    (slides_dir, embeddings_dir). Returns the created project config.
    """
    reg = get_registry()

    # Build config dict from the request body
    dataset_paths = default_dataset_paths(body.id)
    config = {
        "name": body.name,
        "cancer_type": body.cancer_type,
        "prediction_target": body.prediction_target,
        "classes": body.classes,
        "positive_class": body.positive_class,
        "description": body.description,
        "dataset": {
            **dataset_paths,
            "label_column": body.prediction_target,
        },
    }

    try:
        project = reg.add_project(body.id, config)
    except ValueError:
        raise HTTPException(
            status_code=409,
            detail=f"Project '{body.id}' already exists",
        )

    # Create data directories on disk
    os.makedirs(project.dataset.slides_dir, exist_ok=True)
    os.makedirs(project.dataset.embeddings_dir, exist_ok=True)
    logger.info(
        "Created project '%s' with directories: %s, %s",
        body.id,
        project.dataset.slides_dir,
        project.dataset.embeddings_dir,
    )

    # Seed project_slides and project_models if provided
    slides_assigned = 0
    models_assigned = 0
    try:
        from . import database as db

        if body.slide_ids:
            slides_assigned = await db.assign_slides_to_project(body.id, body.slide_ids)
        if body.model_ids:
            models_assigned = await db.assign_models_to_project(body.id, body.model_ids)
    except Exception as e:
        logger.warning("Failed to seed slide/model assignments for project '%s': %s", body.id, e)

    return {
        "project": project.to_dict(),
        "is_default": body.id == reg.default_project_id,
        "slides_assigned": slides_assigned,
        "models_assigned": models_assigned,
    }


@router.get("/{project_id}")
async def get_project(project_id: str):
    """
    Get full details for a specific project.

    Includes dataset paths, model configuration, threshold, class definitions,
    foundation model info, feature toggles, and classification model metadata.
    """
    reg = get_registry()
    proj = reg.get_project(project_id)
    if proj is None:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")

    # Include foundation model info if available
    fm = reg.get_foundation_model(proj.foundation_model)
    foundation_model_info = None
    if fm:
        foundation_model_info = {
            "id": fm.id,
            "name": fm.name,
            "embedding_dim": fm.embedding_dim,
            "description": fm.description,
        }

    # Include classification model metadata
    cls_models = reg.get_project_classification_models(project_id)
    classification_model_details = []
    for cm in cls_models:
        classification_model_details.append({
            "id": cm.id,
            "display_name": cm.display_name,
            "description": cm.description,
            "auc": cm.auc,
            "n_slides": cm.n_slides,
            "category": cm.category,
            "positive_label": cm.positive_label,
            "negative_label": cm.negative_label,
        })

    return {
        "project": proj.to_dict(),
        "is_default": project_id == reg.default_project_id,
        "foundation_model": foundation_model_info,
        "classification_model_details": classification_model_details,
    }


@router.put("/{project_id}")
async def update_project(project_id: str, body: UpdateProjectRequest):
    """
    Update an existing project configuration.

    Only the fields included in the request body are modified; everything else
    is preserved. Writes the updated config back to projects.yaml.
    """
    reg = get_registry()

    updates = body.dict(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields provided for update")

    try:
        project = reg.update_project(project_id, updates)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Project '{project_id}' not found",
        )

    return {
        "project": project.to_dict(),
        "is_default": project_id == reg.default_project_id,
    }


@router.delete("/{project_id}")
async def delete_project(project_id: str):
    """
    Delete a project from the registry.

    Removes the entry from projects.yaml but does NOT delete data files on
    disk. Returns a confirmation message.
    """
    reg = get_registry()

    try:
        reg.remove_project(project_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Project '{project_id}' not found",
        )

    return {
        "detail": f"Project '{project_id}' deleted",
        "remaining_projects": len(reg.list_projects()),
    }


@router.post("/{project_id}/upload", status_code=201)
async def upload_slide(project_id: str, file: UploadFile = File(...)):
    """
    Upload a whole-slide image file to a project.

    Accepts WSI files (.svs, .tiff, .tif, .ndpi). The file is saved into the
    project's slides_dir. If the database is available, the slide is also
    registered in the slides table.
    """
    reg = get_registry()
    proj = reg.get_project(project_id)
    if proj is None:
        raise HTTPException(
            status_code=404,
            detail=f"Project '{project_id}' not found",
        )

    # Validate file extension
    filename = file.filename or ""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_SLIDE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid file extension '{ext}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_SLIDE_EXTENSIONS))}"
            ),
        )

    # Ensure slides directory exists
    slides_dir = Path(proj.dataset.slides_dir)
    os.makedirs(slides_dir, exist_ok=True)

    dest_path = slides_dir / filename
    file_size = 0

    # Stream-write to disk to handle large WSI files
    try:
        with open(dest_path, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)  # 1 MB chunks
                if not chunk:
                    break
                out.write(chunk)
                file_size += len(chunk)
    except Exception as exc:
        # Clean up partial file on failure
        if dest_path.exists():
            dest_path.unlink()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save uploaded file: {exc}",
        )

    slide_id = Path(filename).stem

    # Try to register in PostgreSQL if the database is available
    db_registered = False
    try:
        from . import database as db

        pool = await db.get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO slides (slide_id, filename, file_path, file_size_bytes, project_id)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (slide_id) DO UPDATE
                SET filename = EXCLUDED.filename,
                    file_path = EXCLUDED.file_path,
                    file_size_bytes = EXCLUDED.file_size_bytes,
                    project_id = EXCLUDED.project_id,
                    updated_at = now()
                """,
                slide_id,
                filename,
                str(dest_path),
                file_size,
                project_id,
            )
        db_registered = True
    except Exception as exc:
        logger.warning("Could not register slide in database: %s", exc)

    logger.info(
        "Uploaded slide '%s' (%d bytes) to project '%s' [db_registered=%s]",
        filename,
        file_size,
        project_id,
        db_registered,
    )

    return {
        "slide_id": slide_id,
        "filename": filename,
        "project_id": project_id,
        "file_size_bytes": file_size,
        "path": str(dest_path),
        "db_registered": db_registered,
    }


@router.get("/{project_id}/slides")
async def get_project_slides(project_id: str):
    """
    Get slides associated with a project.

    Uses the project_slides junction table. Falls back to the legacy
    project_id column on slides if the junction table has no rows yet.
    """
    reg = get_registry()
    proj = reg.get_project(project_id)
    if proj is None:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")

    try:
        from . import database as db

        # First try the junction table
        slide_ids = await db.get_project_slides(project_id)

        if slide_ids:
            pool = await db.get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT s.slide_id, s.patient_id, s.filename, s.width, s.height,
                           s.label, s.has_embeddings, s.has_level0_embeddings, s.num_patches
                    FROM slides s
                    WHERE s.slide_id = ANY($1::text[])
                    ORDER BY s.slide_id
                    """,
                    slide_ids,
                )
            return {
                "project_id": project_id,
                "slides": [dict(r) for r in rows],
                "count": len(rows),
            }

        # Fallback: legacy project_id column on slides table
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            col_check = await conn.fetchval(
                """
                SELECT COUNT(*) FROM information_schema.columns
                WHERE table_name = 'slides' AND column_name = 'project_id'
                """
            )
            if col_check > 0:
                rows = await conn.fetch(
                    """
                    SELECT s.slide_id, s.patient_id, s.filename, s.width, s.height,
                           s.label, s.has_embeddings, s.has_level0_embeddings, s.num_patches
                    FROM slides s
                    WHERE s.project_id = $1 OR s.project_id IS NULL
                    ORDER BY s.slide_id
                    """,
                    project_id,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT s.slide_id, s.patient_id, s.filename, s.width, s.height,
                           s.label, s.has_embeddings, s.has_level0_embeddings, s.num_patches
                    FROM slides s
                    ORDER BY s.slide_id
                    """
                )

            return {
                "project_id": project_id,
                "slides": [dict(r) for r in rows],
                "count": len(rows),
            }
    except Exception as e:
        logger.warning(f"Database query failed for project slides: {e}")
        return {
            "project_id": project_id,
            "slides": [],
            "count": 0,
            "error": "Database not available",
        }


class SlideIdsRequest(BaseModel):
    """Body for assigning/unassigning slides."""
    slide_ids: List[str] = Field(..., min_length=1)


class ModelIdsRequest(BaseModel):
    """Body for assigning/unassigning models."""
    model_ids: List[str] = Field(..., min_length=1)


@router.post("/{project_id}/slides", status_code=200)
async def assign_project_slides(project_id: str, body: SlideIdsRequest):
    """Assign slides to a project (idempotent)."""
    reg = get_registry()
    if reg.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")

    from . import database as db
    count = await db.assign_slides_to_project(project_id, body.slide_ids)
    return {
        "project_id": project_id,
        "assigned": count,
        "requested": len(body.slide_ids),
    }


@router.delete("/{project_id}/slides", status_code=200)
async def unassign_project_slides(project_id: str, body: SlideIdsRequest):
    """Remove slide assignments from a project."""
    reg = get_registry()
    if reg.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")

    from . import database as db
    count = await db.unassign_slides_from_project(project_id, body.slide_ids)
    return {
        "project_id": project_id,
        "removed": count,
        "requested": len(body.slide_ids),
    }


@router.get("/{project_id}/models")
async def get_project_models_endpoint(project_id: str):
    """List model_ids assigned to a project."""
    reg = get_registry()
    if reg.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")

    from . import database as db
    model_ids = await db.get_project_models(project_id)
    return {
        "project_id": project_id,
        "model_ids": model_ids,
        "count": len(model_ids),
    }


@router.post("/{project_id}/models", status_code=200)
async def assign_project_models(project_id: str, body: ModelIdsRequest):
    """Assign models to a project (idempotent)."""
    reg = get_registry()
    if reg.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")

    from . import database as db
    count = await db.assign_models_to_project(project_id, body.model_ids)
    return {
        "project_id": project_id,
        "assigned": count,
        "requested": len(body.model_ids),
    }


@router.delete("/{project_id}/models", status_code=200)
async def unassign_project_models(project_id: str, body: ModelIdsRequest):
    """Remove model assignments from a project."""
    reg = get_registry()
    if reg.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")

    from . import database as db
    count = await db.unassign_models_from_project(project_id, body.model_ids)
    return {
        "project_id": project_id,
        "removed": count,
        "requested": len(body.model_ids),
    }


@router.get("/{project_id}/available-models")
async def get_project_available_models(project_id: str):
    """Get classification models available for a project, with full metadata.

    Returns model configs compatible with the project's foundation model.
    This is the source of truth for the frontend ModelPicker -- it should
    NOT rely on hardcoded model lists.
    """
    reg = get_registry()
    proj = reg.get_project(project_id)
    if proj is None:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")

    cls_models = reg.get_project_classification_models(project_id)
    models = []
    for cm in cls_models:
        models.append({
            "id": cm.id,
            "displayName": cm.display_name,
            "description": cm.description,
            "auc": cm.auc,
            "category": cm.category,
            "positiveLabel": cm.positive_label,
            "negativeLabel": cm.negative_label,
        })

    return {
        "project_id": project_id,
        "foundation_model": proj.foundation_model,
        "models": models,
        "count": len(models),
    }


@router.get("/{project_id}/status")
async def get_project_status(project_id: str):
    """
    Get readiness status of a project — are the required models and data available?
    """
    reg = get_registry()
    proj = reg.get_project(project_id)
    if proj is None:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")

    slides_dir = Path(proj.dataset.slides_dir)
    embeddings_dir = Path(proj.dataset.embeddings_dir)
    mil_checkpoint = Path(proj.models.mil_checkpoint)
    labels_file = Path(proj.dataset.labels_file)

    def _count_files(p: Path, glob: str = "*.npy") -> int:
        if not p.exists():
            return 0
        return sum(1 for _ in p.glob(glob))

    exts = ("*.svs", "*.tiff", "*.tif", "*.ndpi")
    slide_count = sum(_count_files(slides_dir, g) for g in exts) if slides_dir.exists() else 0
    embedding_count = _count_files(embeddings_dir) if embeddings_dir.exists() else 0

    return {
        "project_id": project_id,
        "name": proj.name,
        "ready": {
            "slides_dir": slides_dir.exists(),
            "embeddings_dir": embeddings_dir.exists(),
            "mil_checkpoint": mil_checkpoint.exists(),
            "labels_file": labels_file.exists(),
        },
        "counts": {
            "slides": slide_count,
            "embeddings": embedding_count,
        },
        "threshold": proj.threshold,
    }
