"""Single-slide analysis API route."""

import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException

from .schemas import AnalyzeRequest, AnalyzeResponse

logger = logging.getLogger(__name__)


def create_analysis_router(
    *,
    classifier_provider: Callable[[], Any],
    evidence_generator_provider: Callable[[], Any],
    resolve_project_embeddings_dir: Callable[..., Path],
    classifier_threshold: Callable[[], float],
    resolve_project_label_pair: Callable[..., tuple[str, str]],
    classify_tissue_type: Callable[[int, int, int | None], dict],
    project_slide_ids: Callable[[str | None], Awaitable[set[str] | None]],
    slide_mean_index_provider: Callable[[], Any],
    slide_mean_ids_provider: Callable[[], list[str]],
    slide_mean_meta_provider: Callable[[], dict[str, Any]],
    slide_labels_provider: Callable[[], dict[str, Any]],
    similar_case_slide_id: Callable[[Any], str | None],
    save_analysis_to_history: Callable[..., Any],
) -> APIRouter:
    router = APIRouter()

    @router.post("/api/analyze", response_model=AnalyzeResponse)
    async def analyze_slide(request: AnalyzeRequest):
        """Analyze one slide and return prediction, evidence, and similar cases."""
        classifier = classifier_provider()
        if classifier is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        slide_id = request.slide_id

        project_requested = request.project_id is not None
        analysis_embeddings_dir = resolve_project_embeddings_dir(
            request.project_id,
            require_exists=project_requested,
        )

        emb_path = analysis_embeddings_dir / f"{slide_id}.npy"

        if not emb_path.exists():
            raise HTTPException(status_code=404, detail=f"Slide {slide_id} not found")

        embeddings = np.load(emb_path)

        score, attention = classifier.predict(embeddings)
        threshold = classifier_threshold()
        pos_label, neg_label = resolve_project_label_pair(
            request.project_id,
            positive_default="RESPONDER",
            negative_default="NON-RESPONDER",
            uppercase=True,
        )
        label = pos_label if score >= threshold else neg_label

        if score >= threshold:
            margin = score - threshold
            confidence = min(1.0 - 0.5 * (2.0 ** (-20.0 * margin)), 0.99)
        else:
            margin = threshold - score
            confidence = min(1.0 - 0.5 * (2.0 ** (-20.0 * margin)), 0.99)

        coord_path = analysis_embeddings_dir / f"{slide_id}_coords.npy"
        coords = None
        if coord_path.exists():
            coords = np.load(coord_path)

        top_k = min(8, len(attention))
        top_indices = np.argsort(attention)[-top_k:][::-1]

        top_evidence = []
        for i, idx in enumerate(top_indices):
            patch_x = int(coords[idx][0]) if coords is not None else 0
            patch_y = int(coords[idx][1]) if coords is not None else 0

            tissue_info = classify_tissue_type(patch_x, patch_y, int(idx))

            top_evidence.append(
                {
                    "rank": i + 1,
                    "patch_index": int(idx),
                    "attention_weight": float(attention[idx]),
                    "coordinates": [patch_x, patch_y],
                    "tissue_type": tissue_info["tissue_type"],
                    "tissue_confidence": tissue_info["confidence"],
                }
            )

        similar_cases = []
        allowed_slide_ids = await project_slide_ids(request.project_id)
        slide_mean_index = slide_mean_index_provider()
        slide_mean_ids = slide_mean_ids_provider()
        slide_mean_meta = slide_mean_meta_provider()
        slide_labels = slide_labels_provider()

        if slide_mean_index is not None:
            try:
                q = np.asarray(embeddings, dtype=np.float32).mean(axis=0)
                q = q / (np.linalg.norm(q) + 1e-12)
                q = q.reshape(1, -1).astype(np.float32)

                search_k = min(len(slide_mean_ids), max(15, 5 * 3))
                sims, idxs = slide_mean_index.search(q, search_k)

                seen_slides = set()
                for sim, idx_val in zip(sims[0], idxs[0]):
                    if idx_val < 0 or idx_val >= len(slide_mean_ids):
                        continue
                    sid = slide_mean_ids[int(idx_val)]
                    if sid == slide_id or sid in seen_slides:
                        continue
                    if allowed_slide_ids is not None and sid not in allowed_slide_ids:
                        continue
                    seen_slides.add(sid)
                    meta = slide_mean_meta.get(sid, {})
                    case_label = meta.get("label") or slide_labels.get(sid)
                    similar_cases.append(
                        {
                            "slide_id": sid,
                            "similarity_score": float(sim),
                            "distance": float(1.0 - float(sim)),
                            "label": case_label,
                        }
                    )
                    if len(similar_cases) >= 5:
                        break
            except Exception as e:
                logger.warning("Similar case search (cosine) failed: %s", e)

        evidence_gen = evidence_generator_provider()
        if not similar_cases and evidence_gen is not None:
            try:
                similar_results = evidence_gen.find_similar(
                    embeddings, attention, k=10, top_patches=3
                )
                seen_slides = set()
                for s in similar_results:
                    sid_candidate = similar_case_slide_id(s)
                    if not sid_candidate or sid_candidate == slide_id or sid_candidate in seen_slides:
                        continue
                    sid = sid_candidate
                    if allowed_slide_ids is not None and sid not in allowed_slide_ids:
                        continue
                    seen_slides.add(sid)
                    case_label = slide_labels.get(sid)
                    similar_cases.append(
                        {
                            "slide_id": sid,
                            "similarity_score": 1.0 / (1.0 + s["distance"]),
                            "distance": float(s["distance"]),
                            "label": case_label,
                        }
                    )
                    if len(similar_cases) >= 5:
                        break
            except Exception as e:
                logger.warning("Similar case search (L2 fallback) failed: %s", e)

        save_analysis_to_history(
            slide_id=slide_id,
            prediction=label,
            score=float(score),
            confidence=float(confidence),
            patches_analyzed=len(embeddings),
            top_evidence=top_evidence,
            similar_cases=similar_cases[:5],
        )

        return AnalyzeResponse(
            slide_id=slide_id,
            prediction=label,
            score=float(score),
            confidence=float(confidence),
            patches_analyzed=len(embeddings),
            top_evidence=top_evidence,
            similar_cases=similar_cases[:5],
        )

    return router
