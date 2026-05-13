"""Synchronous batch analysis API route."""

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException

from .schemas import (
    BatchAnalysisResult,
    BatchAnalysisSummary,
    BatchAnalyzeRequest,
    BatchAnalyzeResponse,
)

logger = logging.getLogger(__name__)


def create_batch_analysis_router(
    *,
    classifier_provider: Callable[[], Any],
    resolve_project_embeddings_dir: Callable[..., Path],
    resolve_project_label_pair: Callable[..., tuple[str, str]],
    classifier_threshold: Callable[[], float],
    log_audit_event: Callable[..., None],
) -> APIRouter:
    router = APIRouter()

    @router.post("/api/analyze-batch", response_model=BatchAnalyzeResponse)
    async def analyze_batch(request: BatchAnalyzeRequest):
        """Analyze multiple slides in one request for clinical triage workflows."""
        start_time = time.time()

        classifier = classifier_provider()
        if classifier is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        project_requested = getattr(request, "project_id", None) is not None
        batch_embeddings_dir = resolve_project_embeddings_dir(
            getattr(request, "project_id", None),
            require_exists=project_requested,
        )
        pos_label, _neg_label = resolve_project_label_pair(
            getattr(request, "project_id", None),
            positive_default="RESPONDER",
            negative_default="NON-RESPONDER",
            uppercase=True,
        )

        results = []
        for slide_id in request.slide_ids:
            emb_path = batch_embeddings_dir / f"{slide_id}.npy"

            if not emb_path.exists():
                results.append(
                    BatchAnalysisResult(
                        slide_id=slide_id,
                        prediction="ERROR",
                        score=0.0,
                        confidence=0.0,
                        patches_analyzed=0,
                        requires_review=True,
                        uncertainty_level="unknown",
                        error=f"Slide {slide_id} not found",
                    )
                )
                continue

            try:
                embeddings = np.load(emb_path)

                score, _attention = classifier.predict(embeddings)
                threshold = classifier_threshold()
                label = pos_label if score >= threshold else _neg_label
                confidence = abs(score - threshold) * 2

                if confidence < 0.3:
                    uncertainty_level = "high"
                    requires_review = True
                elif confidence < 0.6:
                    uncertainty_level = "moderate"
                    requires_review = True
                else:
                    uncertainty_level = "low"
                    requires_review = False

                results.append(
                    BatchAnalysisResult(
                        slide_id=slide_id,
                        prediction=label,
                        score=float(score),
                        confidence=float(confidence),
                        patches_analyzed=len(embeddings),
                        requires_review=requires_review,
                        uncertainty_level=uncertainty_level,
                        error=None,
                    )
                )

                log_audit_event(
                    "batch_analysis_slide",
                    slide_id,
                    details={
                        "prediction": label,
                        "confidence": float(confidence),
                        "requires_review": requires_review,
                    },
                )

            except Exception as e:
                logger.error("Batch analysis failed for %s: %s", slide_id, e)
                results.append(
                    BatchAnalysisResult(
                        slide_id=slide_id,
                        prediction="ERROR",
                        score=0.0,
                        confidence=0.0,
                        patches_analyzed=0,
                        requires_review=True,
                        uncertainty_level="unknown",
                        error=str(e),
                    )
                )

        results.sort(
            key=lambda r: (
                0 if r.error else 1,
                r.confidence if not r.error else 999,
            )
        )

        completed = [r for r in results if r.error is None]
        failed = [r for r in results if r.error is not None]
        responders = [r for r in completed if r.prediction == pos_label]
        non_responders = [r for r in completed if r.prediction != pos_label]
        uncertain = [r for r in completed if r.requires_review]
        avg_confidence = sum(r.confidence for r in completed) / len(completed) if completed else 0.0

        summary = BatchAnalysisSummary(
            total=len(results),
            completed=len(completed),
            failed=len(failed),
            responders=len(responders),
            non_responders=len(non_responders),
            uncertain=len(uncertain),
            avg_confidence=round(avg_confidence, 3),
            requires_review_count=sum(1 for r in results if r.requires_review),
        )

        processing_time_ms = (time.time() - start_time) * 1000

        log_audit_event(
            "batch_analysis_completed",
            details={
                "total_slides": len(results),
                "completed": len(completed),
                "failed": len(failed),
                "processing_time_ms": processing_time_ms,
            },
        )

        return BatchAnalyzeResponse(
            results=results,
            summary=summary,
            processing_time_ms=round(processing_time_ms, 2),
        )

    return router
