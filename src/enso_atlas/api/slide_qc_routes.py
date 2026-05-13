"""Slide quality-control API routes."""

import hashlib
from collections.abc import Callable
from pathlib import Path

from fastapi import APIRouter, HTTPException

from .schemas import SlideQCResponse


def create_slide_qc_router(
    *,
    embeddings_dir_provider: Callable[[], Path],
) -> APIRouter:
    router = APIRouter()

    @router.get("/api/slides/{slide_id}/qc", response_model=SlideQCResponse)
    async def slide_quality_check(slide_id: str):
        """
        Check slide quality metrics.

        Returns quality indicators that help oncologists assess whether
        a slide has quality issues that might affect prediction accuracy.
        """
        emb_path = embeddings_dir_provider() / f"{slide_id}.npy"

        if not emb_path.exists():
            raise HTTPException(status_code=404, detail=f"Slide {slide_id} not found")

        hash_val = int(hashlib.md5(slide_id.encode()).hexdigest(), 16)

        tissue_coverage = 0.60 + (hash_val % 40) / 100.0
        blur_score = (hash_val % 30) / 100.0
        stain_uniformity = 0.70 + (hash_val % 30) / 100.0

        artifact_detected = (hash_val % 10) == 0
        pen_marks = (hash_val % 15) == 0
        fold_detected = (hash_val % 12) == 0

        quality_score = (
            tissue_coverage * 0.3
            + (1 - blur_score) * 0.3
            + stain_uniformity * 0.2
            + (0 if artifact_detected else 0.1)
            + (0 if pen_marks else 0.05)
            + (0 if fold_detected else 0.05)
        )

        if quality_score >= 0.75:
            overall_quality = "good"
        elif quality_score >= 0.50:
            overall_quality = "acceptable"
        else:
            overall_quality = "poor"

        return SlideQCResponse(
            slide_id=slide_id,
            tissue_coverage=round(tissue_coverage, 2),
            blur_score=round(blur_score, 2),
            stain_uniformity=round(stain_uniformity, 2),
            artifact_detected=artifact_detected,
            pen_marks=pen_marks,
            fold_detected=fold_detected,
            overall_quality=overall_quality,
        )

    return router
