"""Status routes for embedding and search features."""

from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import APIRouter, Query


def create_feature_status_router(
    *,
    embedder_provider: Callable[[], Any],
    medsiglip_embedder_provider: Callable[[], Any],
    slide_siglip_embeddings_provider: Callable[[], dict],
    evidence_generator_provider: Callable[[], Any],
    available_slides_provider: Callable[[], list[str]],
    check_cuda: Callable[[], bool],
    require_project: Callable[[str | None], Any],
    project_slide_ids: Callable[[str | None], Awaitable[set[str] | None]],
) -> APIRouter:
    router = APIRouter()

    @router.get("/api/embed/status")
    async def embedder_status():
        """Check the status of the Path Foundation embedder."""
        embedder = embedder_provider()
        model_loaded = embedder is not None and embedder._model is not None
        device = "unknown"

        if model_loaded:
            device = str(embedder._device)
        else:
            device = "cuda" if check_cuda() else "cpu"

        return {
            "model": "google/path-foundation",
            "model_loaded": model_loaded,
            "device": device,
            "embedding_dim": 384,
            "input_size": 224,
        }

    @router.get("/api/semantic-search/status")
    async def semantic_search_status():
        """Check the status of the MedSigLIP semantic search feature."""
        medsiglip_embedder = medsiglip_embedder_provider()
        slide_siglip_embeddings = slide_siglip_embeddings_provider()
        model_loaded = medsiglip_embedder is not None and medsiglip_embedder._model is not None
        device = "unknown"

        if model_loaded:
            device = str(medsiglip_embedder._device)
        else:
            device = "cuda" if check_cuda() else "cpu"

        return {
            "model": medsiglip_embedder.config.model_id
            if medsiglip_embedder
            else "not initialized",
            "model_loaded": model_loaded,
            "device": device,
            "embedding_dim": medsiglip_embedder.EMBEDDING_DIM if medsiglip_embedder else None,
            "input_size": medsiglip_embedder.INPUT_SIZE if medsiglip_embedder else None,
            "cached_slides": list(slide_siglip_embeddings.keys())
            if slide_siglip_embeddings
            else [],
        }

    @router.get("/api/search/visual/status")
    async def visual_search_status(
        project_id: str | None = Query(
            default=None, description="Optional project scope for visual-search inventory"
        ),
    ):
        """Check the status of the visual search FAISS index."""
        require_project(project_id)
        evidence_gen = evidence_generator_provider()
        index_loaded = evidence_gen is not None and evidence_gen._faiss_index is not None

        total_patches = evidence_gen._faiss_index.ntotal if index_loaded else 0
        total_slides = len(available_slides_provider())

        if index_loaded and project_id is not None:
            allowed_slide_ids = await project_slide_ids(project_id)
            if allowed_slide_ids is None:
                allowed_slide_ids = set()

            scoped_slide_ids = {
                str(meta.get("slide_id", ""))
                for meta in evidence_gen._reference_metadata
                if str(meta.get("slide_id", "")) in allowed_slide_ids
            }
            total_slides = len(scoped_slide_ids)
            total_patches = sum(
                1
                for meta in evidence_gen._reference_metadata
                if str(meta.get("slide_id", "")) in allowed_slide_ids
            )

        return {
            "index_loaded": index_loaded,
            "total_patches": total_patches,
            "total_slides": total_slides,
            "embedding_dim": 384,
        }

    return router
