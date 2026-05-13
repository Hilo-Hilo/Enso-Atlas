"""Similar-case retrieval API routes."""

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from .schemas import SimilarResponse


def create_similar_router(
    *,
    resolve_project_embeddings_dir: Callable[..., Path],
    project_slide_ids: Callable[[str | None], Awaitable[set[str] | None]],
    slide_mean_index_provider: Callable[[], Any],
    slide_mean_ids_provider: Callable[[], list[str]],
    slide_mean_meta_provider: Callable[[], dict[str, Any]],
    slide_labels_provider: Callable[[], dict[str, Any]],
) -> APIRouter:
    router = APIRouter()

    @router.get("/api/similar", response_model=SimilarResponse)
    async def get_similar_cases(
        slide_id: str,
        k: int = 5,
        top_patches: int = 3,
        project_id: str | None = Query(
            default=None, description="Project ID to scope embeddings lookup"
        ),
    ):
        """Find similar slides from the scoped reference cohort."""
        slide_mean_index = slide_mean_index_provider()
        if slide_mean_index is None:
            raise HTTPException(status_code=503, detail="Similarity index not available")

        project_requested = project_id is not None
        similar_embeddings_dir = resolve_project_embeddings_dir(
            project_id,
            require_exists=project_requested,
        )
        allowed_slide_ids = await project_slide_ids(project_id)

        emb_path = similar_embeddings_dir / f"{slide_id}.npy"
        if not emb_path.exists():
            raise HTTPException(status_code=404, detail=f"Slide {slide_id} not found")

        embs = np.load(emb_path)
        if embs is None or len(embs) == 0:
            return SimilarResponse(slide_id=slide_id, similar_cases=[], num_queries=1)

        q = np.asarray(embs, dtype=np.float32).mean(axis=0)
        q = q / (np.linalg.norm(q) + 1e-12)
        q = q.reshape(1, -1).astype(np.float32)

        slide_mean_ids = slide_mean_ids_provider()
        search_k = min(len(slide_mean_ids), max(k + 10, k * 3))
        sims, idxs = slide_mean_index.search(q, search_k)

        similar_cases = []
        seen = set()
        slide_mean_meta = slide_mean_meta_provider()
        slide_labels = slide_labels_provider()

        for sim, idx in zip(sims[0], idxs[0]):
            if idx < 0 or idx >= len(slide_mean_ids):
                continue
            sid = slide_mean_ids[int(idx)]
            if sid == slide_id or sid in seen:
                continue
            if allowed_slide_ids is not None and sid not in allowed_slide_ids:
                continue
            seen.add(sid)

            meta = slide_mean_meta.get(sid, {})
            similar_cases.append(
                {
                    "slide_id": sid,
                    "similarity_score": float(sim),
                    "distance": float(1.0 - float(sim)),
                    "label": meta.get("label") or slide_labels.get(sid),
                    "n_patches": meta.get("n_patches"),
                }
            )

            if len(similar_cases) >= k:
                break

        return SimilarResponse(
            slide_id=slide_id,
            similar_cases=similar_cases,
            num_queries=1,
        )

    return router
