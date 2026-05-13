"""Visual patch-search API routes."""

import logging
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException

from .schemas import VisualSearchRequest, VisualSearchResponse, VisualSearchResultPatch

logger = logging.getLogger(__name__)


def create_visual_search_router(
    *,
    require_project: Callable[[str | None], Any],
    project_slide_ids: Callable[[str | None], Awaitable[set[str] | None]],
    resolve_project_embeddings_dir: Callable[..., Path],
    resolve_embedding_path: Callable[..., tuple[Path | None, list[Path]]],
    evidence_generator_provider: Callable[[], Any],
    slide_labels_provider: Callable[[], dict[str, Any]],
    log_audit_event: Callable[..., None],
) -> APIRouter:
    router = APIRouter()

    @router.post("/api/search/visual", response_model=VisualSearchResponse)
    async def visual_search(request: VisualSearchRequest):
        """
        Find visually similar patches across the entire database using FAISS.

        The query can be a direct embedding, a slide patch index, or slide
        coordinates. Results are project-scoped when ``project_id`` is supplied.
        """
        start_time = time.time()

        project_requested = request.project_id is not None
        require_project(request.project_id)
        allowed_slide_ids = await project_slide_ids(request.project_id)
        if allowed_slide_ids is not None and len(allowed_slide_ids) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"Project '{request.project_id}' has no scoped slides for visual search",
            )

        if request.coordinates is not None:
            if len(request.coordinates) != 2:
                raise HTTPException(status_code=400, detail="coordinates must be [x, y]")
            if request.coordinates[0] < 0 or request.coordinates[1] < 0:
                raise HTTPException(status_code=400, detail="coordinates must be non-negative")

        has_embedding = request.patch_embedding is not None
        has_slide_patch = request.slide_id is not None and request.patch_index is not None
        has_slide_coords = request.slide_id is not None and request.coordinates is not None

        if not (has_embedding or has_slide_patch or has_slide_coords):
            raise HTTPException(
                status_code=400,
                detail="Must provide either patch_embedding, (slide_id + patch_index), or (slide_id + coordinates)",
            )

        query_embedding = None
        query_slide_id = request.slide_id
        query_patch_index = request.patch_index
        query_coordinates = request.coordinates

        if (
            query_slide_id
            and allowed_slide_ids is not None
            and query_slide_id not in allowed_slide_ids
        ):
            raise HTTPException(
                status_code=404,
                detail=f"Slide {query_slide_id} is not available in project '{request.project_id}'",
            )

        visual_embeddings_dir = resolve_project_embeddings_dir(
            request.project_id,
            require_exists=project_requested,
        )

        if has_embedding:
            query_embedding = np.asarray(request.patch_embedding, dtype=np.float32)
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            elif query_embedding.ndim != 2 or query_embedding.shape[0] != 1:
                raise HTTPException(
                    status_code=400, detail="patch_embedding must be a single embedding vector"
                )

        elif has_slide_patch:
            slide_id = request.slide_id
            patch_index = request.patch_index
            if slide_id is None or patch_index is None:
                raise HTTPException(
                    status_code=400,
                    detail="Must provide both slide_id and patch_index for patch lookup",
                )
            emb_path, searched_dirs = resolve_embedding_path(
                slide_id,
                level=1,
                project_id=request.project_id,
                base_embeddings_dir=visual_embeddings_dir,
            )
            if emb_path is None:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"Slide {slide_id} not found in embeddings directories: "
                        + ", ".join(str(p) for p in searched_dirs)
                    ),
                )

            embeddings = np.load(emb_path)
            if patch_index >= len(embeddings):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Patch index {patch_index} out of range "
                        f"(slide has {len(embeddings)} patches)"
                    ),
                )

            query_embedding = embeddings[patch_index : patch_index + 1].astype(np.float32)

            coord_path = emb_path.with_name(f"{slide_id}_coords.npy")
            if coord_path.exists():
                coords = np.load(coord_path)
                if patch_index < len(coords):
                    query_coordinates = [
                        int(coords[patch_index][0]),
                        int(coords[patch_index][1]),
                    ]

        elif has_slide_coords:
            slide_id = request.slide_id
            coordinates = request.coordinates
            if slide_id is None or coordinates is None:
                raise HTTPException(
                    status_code=400,
                    detail="Must provide both slide_id and coordinates for coordinate lookup",
                )
            emb_path, searched_dirs = resolve_embedding_path(
                slide_id,
                level=1,
                project_id=request.project_id,
                base_embeddings_dir=visual_embeddings_dir,
            )
            if emb_path is None:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"Slide {slide_id} not found in embeddings directories: "
                        + ", ".join(str(p) for p in searched_dirs)
                    ),
                )

            coord_path = emb_path.with_name(f"{slide_id}_coords.npy")

            if not coord_path.exists():
                raise HTTPException(status_code=404, detail=f"Coordinates not found for slide {slide_id}")

            embeddings = np.load(emb_path)
            coords = np.load(coord_path)

            target_x, target_y = coordinates[0], coordinates[1]
            distances = np.sqrt((coords[:, 0] - target_x) ** 2 + (coords[:, 1] - target_y) ** 2)
            query_patch_index = int(np.argmin(distances))

            query_embedding = embeddings[query_patch_index : query_patch_index + 1].astype(
                np.float32
            )
            query_coordinates = [
                int(coords[query_patch_index][0]),
                int(coords[query_patch_index][1]),
            ]

        evidence_gen = evidence_generator_provider()
        if evidence_gen is None or evidence_gen._faiss_index is None:
            raise HTTPException(status_code=503, detail="FAISS index not initialized")

        faiss_index = evidence_gen._faiss_index
        index_total = int(faiss_index.ntotal)
        if index_total <= 0:
            raise HTTPException(status_code=404, detail="FAISS index is empty")

        if query_embedding is None or query_embedding.ndim != 2:
            raise HTTPException(status_code=400, detail="Could not resolve query embedding")

        if int(query_embedding.shape[1]) != int(faiss_index.d):
            raise HTTPException(
                status_code=400,
                detail=f"Embedding dimension mismatch: expected {faiss_index.d}, got {query_embedding.shape[1]}",
            )

        search_multiplier = 10 if allowed_slide_ids is not None else 3
        if not request.exclude_same_slide and allowed_slide_ids is None:
            search_multiplier = 1
        search_k = min(max(request.top_k * search_multiplier, request.top_k), index_total)

        try:
            distances, indices = faiss_index.search(query_embedding, search_k)
        except Exception as e:
            logger.error("FAISS search failed: %s", e)
            raise HTTPException(status_code=500, detail=f"FAISS search failed: {str(e)}")

        scoped_total_patches = index_total
        if allowed_slide_ids is not None:
            scoped_total_patches = sum(
                1
                for meta in evidence_gen._reference_metadata
                if str(meta.get("slide_id", "")) in allowed_slide_ids
            )

        results = []
        seen_slide_patches = set()
        slide_labels = slide_labels_provider()

        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(evidence_gen._reference_metadata):
                continue

            meta = evidence_gen._reference_metadata[idx]
            result_slide_id = str(meta.get("slide_id", "unknown"))
            result_patch_index = int(meta.get("patch_index", 0))

            if allowed_slide_ids is not None and result_slide_id not in allowed_slide_ids:
                continue

            if request.exclude_same_slide and result_slide_id == query_slide_id:
                continue

            key = (result_slide_id, result_patch_index)
            if key in seen_slide_patches:
                continue
            seen_slide_patches.add(key)

            result_coordinates = None
            emb_path_for_result, _ = resolve_embedding_path(
                result_slide_id,
                level=1,
                project_id=request.project_id,
                base_embeddings_dir=visual_embeddings_dir,
            )
            if emb_path_for_result is not None:
                coord_path = emb_path_for_result.with_name(f"{result_slide_id}_coords.npy")
                if coord_path.exists():
                    try:
                        coords = np.load(coord_path)
                        if result_patch_index < len(coords):
                            result_coordinates = [
                                int(coords[result_patch_index][0]),
                                int(coords[result_patch_index][1]),
                            ]
                    except Exception as e:
                        logger.warning("Failed to load coords for %s: %s", result_slide_id, e)

            result_label = slide_labels.get(result_slide_id)
            similarity = 1.0 / (1.0 + float(dist))

            thumbnail_url = None
            if result_coordinates:
                thumbnail_url = f"/api/slides/{result_slide_id}/patches/{result_patch_index}"
                if request.project_id:
                    thumbnail_url = f"{thumbnail_url}?project_id={request.project_id}"

            results.append(
                VisualSearchResultPatch(
                    slide_id=result_slide_id,
                    patch_index=result_patch_index,
                    coordinates=result_coordinates,
                    distance=float(dist),
                    similarity=similarity,
                    label=result_label,
                    thumbnail_url=thumbnail_url,
                )
            )

            if len(results) >= request.top_k:
                break

        search_time_ms = (time.time() - start_time) * 1000

        log_audit_event(
            "visual_search",
            slide_id=query_slide_id,
            details={
                "project_id": request.project_id,
                "patch_index": query_patch_index,
                "coordinates": query_coordinates,
                "num_results": len(results),
                "search_time_ms": search_time_ms,
            },
        )

        return VisualSearchResponse(
            query_slide_id=query_slide_id,
            query_patch_index=query_patch_index,
            query_coordinates=query_coordinates,
            results=results,
            total_patches_searched=scoped_total_patches,
            search_time_ms=round(search_time_ms, 2),
        )

    return router
