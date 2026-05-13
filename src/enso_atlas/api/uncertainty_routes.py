"""Uncertainty analysis API routes."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException

from .schemas import UncertaintyRequest, UncertaintyResponse

logger = logging.getLogger(__name__)


def create_uncertainty_router(
    *,
    classifier_provider: Callable[[], Any],
    embeddings_dir_provider: Callable[[], Path],
    log_audit_event: Callable[..., None],
) -> APIRouter:
    router = APIRouter()

    @router.post("/api/analyze-uncertainty", response_model=UncertaintyResponse)
    async def analyze_with_uncertainty(request: UncertaintyRequest):
        """
        Analyze a slide with MC Dropout uncertainty quantification.

        High uncertainty marks the result for human review and returns
        conservative clinical guidance.
        """
        classifier = classifier_provider()
        if classifier is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        slide_id = request.slide_id
        embeddings_dir = embeddings_dir_provider()
        emb_path = embeddings_dir / f"{slide_id}.npy"

        if not emb_path.exists():
            raise HTTPException(status_code=404, detail=f"Slide {slide_id} not found")

        embeddings = np.load(emb_path)

        try:
            result = classifier.predict_with_uncertainty(embeddings, n_samples=request.n_samples)
        except Exception as e:
            logger.error("Uncertainty prediction failed: %s", e)
            raise HTTPException(
                status_code=500,
                detail=f"Uncertainty prediction failed: {str(e)}",
            )

        uncertainty = result["uncertainty"]
        probability = result["probability"]

        if uncertainty < 0.10:
            uncertainty_level = "low"
            requires_review = False
            pred_label = result.get("prediction", "unknown")
            clinical_recommendation = (
                f"Model shows high confidence in {pred_label} prediction. "
                "Consider proceeding with clinical evaluation based on full context."
            )
        elif uncertainty < 0.20:
            uncertainty_level = "moderate"
            requires_review = True
            clinical_recommendation = (
                "Model shows moderate uncertainty. Recommend pathologist review "
                "of high-attention regions and correlation with clinical factors."
            )
        else:
            uncertainty_level = "high"
            requires_review = True
            clinical_recommendation = (
                "Model is uncertain about this case - consider additional testing. "
                "Do not rely solely on this prediction. Recommend molecular profiling "
                "and/or expert pathology consultation."
            )

        coord_path = embeddings_dir / f"{slide_id}_coords.npy"
        coords = None
        if coord_path.exists():
            coords = np.load(coord_path)

        attention = result["attention_weights"]
        attention_std = result["attention_uncertainty"]
        top_k = min(8, len(attention))
        top_indices = np.argsort(attention)[-top_k:][::-1]

        top_evidence = []
        for i, idx in enumerate(top_indices):
            patch_x = int(coords[idx][0]) if coords is not None else 0
            patch_y = int(coords[idx][1]) if coords is not None else 0

            top_evidence.append(
                {
                    "rank": i + 1,
                    "patch_index": int(idx),
                    "attention_weight": float(attention[idx]),
                    "attention_uncertainty": float(attention_std[idx]),
                    "coordinates": [patch_x, patch_y],
                }
            )

        log_audit_event(
            "uncertainty_analysis_completed",
            slide_id,
            details={
                "prediction": result["prediction"],
                "uncertainty": uncertainty,
                "uncertainty_level": uncertainty_level,
                "requires_review": requires_review,
            },
        )

        return UncertaintyResponse(
            slide_id=slide_id,
            prediction=result["prediction"],
            probability=probability,
            uncertainty=uncertainty,
            confidence_interval=result["confidence_interval"],
            is_uncertain=result["is_uncertain"],
            requires_review=requires_review,
            uncertainty_level=uncertainty_level,
            clinical_recommendation=clinical_recommendation,
            patches_analyzed=len(embeddings),
            n_samples=result["n_samples"],
            samples=result["samples"],
            top_evidence=top_evidence,
        )

    return router
