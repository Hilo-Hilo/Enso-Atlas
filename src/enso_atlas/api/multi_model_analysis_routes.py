"""Multi-model slide analysis route."""

import hashlib
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException

from .schemas import ModelPrediction, MultiModelRequest, MultiModelResponse


def create_multi_model_analysis_router(
    *,
    multi_model_provider: Callable[[], Any],
    model_configs_provider: Callable[[], dict[str, Any]],
    resolve_project_model_ids: Callable[[str | None], Any],
    active_batch_embed_info: Callable[[], dict[str, Any] | None],
    resolve_project_embeddings_dir: Callable[..., Path],
    resolve_embedding_path: Callable[..., tuple[Path | None, list[Path]]],
    score_to_prediction: Callable[..., dict[str, Any]],
    db_module: Any,
    log_audit_event: Callable[..., None],
    project_root: Path,
    logger: Any,
) -> APIRouter:
    router = APIRouter()

    @router.post("/api/analyze-multi", response_model=MultiModelResponse)
    async def analyze_slide_multi(request: MultiModelRequest):
        """
        Run multi-model inference for one slide.

        ``request.models`` can select a subset; otherwise all allowed models are
        run. When ``request.project_id`` is supplied, model selection is
        validated against that project's allowlist and embeddings directory.
        Level 0 embeddings are required.
        """
        start_time = time.time()
        multi_model_inference = multi_model_provider()
        model_configs = model_configs_provider()

        if multi_model_inference is None:
            raise HTTPException(status_code=503, detail="Multi-model inference not initialized")

        slide_id = request.slide_id
        level = request.level
        if level != 0:
            raise HTTPException(
                status_code=400, detail="level must be 0 (dense full-resolution policy)"
            )

        active_batch_embedding = active_batch_embed_info()
        if active_batch_embedding:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "SERVER_BUSY",
                    "message": (
                        "Level 0 batch embedding is currently running. "
                        "Multi-model analysis is temporarily unavailable to avoid GPU contention."
                    ),
                    "retry_after_seconds": 30,
                    "active_batch_embedding": active_batch_embedding,
                },
                headers={"Retry-After": "30"},
            )

        allowed_model_ids = await resolve_project_model_ids(request.project_id)
        effective_model_ids: list[str] | None
        if request.models is not None:
            effective_model_ids = list(request.models)
            if allowed_model_ids is not None:
                disallowed = sorted(set(effective_model_ids) - allowed_model_ids)
                if disallowed:
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "MODELS_NOT_ALLOWED_FOR_PROJECT",
                            "project_id": request.project_id,
                            "disallowed_models": disallowed,
                        },
                    )
        else:
            effective_model_ids = (
                sorted(allowed_model_ids) if allowed_model_ids is not None else None
            )

        if not request.force:
            try:
                cached = await db_module.get_all_cached_results(slide_id)
                if cached:
                    cached_predictions = {}
                    cached_by_cat: dict[str, list] = {}
                    requested_models = set(effective_model_ids) if effective_model_ids else None

                    for row in cached:
                        mid = row["model_id"]
                        if requested_models and mid not in requested_models:
                            continue

                        cfg = model_configs.get(mid, {})
                        positive_label = cfg.get("positive_label", "Positive")
                        negative_label = cfg.get("negative_label", "Negative")
                        current_threshold = cfg.get("decision_threshold", cfg.get("threshold", 0.5))

                        cached_eval = score_to_prediction(
                            score=row.get("score", 0.0),
                            decision_threshold=current_threshold,
                            positive_label=positive_label,
                            negative_label=negative_label,
                        )

                        pred_dict = {
                            "model_id": mid,
                            "model_name": cfg.get("display_name", mid),
                            "category": cfg.get("category", "general_pathology"),
                            "score": cached_eval["score"],
                            "decision_threshold": cached_eval["decision_threshold"],
                            "label": cached_eval["label"],
                            "positive_label": positive_label,
                            "negative_label": negative_label,
                            "confidence": min(cached_eval["confidence"], 0.99),
                            "auc": cfg.get("auc", 0.0),
                            "n_training_slides": cfg.get(
                                "n_training_slides", cfg.get("n_slides", 0)
                            ),
                            "description": cfg.get("description", ""),
                        }
                        mp = ModelPrediction(**pred_dict)
                        cached_predictions[mid] = mp
                        cat = cfg.get("category", "general_pathology")
                        cached_by_cat.setdefault(cat, []).append(mp)

                    if cached_predictions:
                        processing_time = (time.time() - start_time) * 1000
                        logger.info(
                            "Returning cached results for %s (%d models)",
                            slide_id,
                            len(cached_predictions),
                        )
                        return MultiModelResponse(
                            slide_id=slide_id,
                            predictions=cached_predictions,
                            by_category=cached_by_cat,
                            n_patches=0,
                            processing_time_ms=processing_time,
                            warnings=["Results loaded from cache"],
                        )
            except Exception as exc:
                logger.warning("Cache lookup failed for %s, running fresh: %s", slide_id, exc)

        project_requested = request.project_id is not None
        analysis_embeddings_dir = resolve_project_embeddings_dir(
            request.project_id,
            require_exists=project_requested,
        )

        emb_path, searched_dirs = resolve_embedding_path(
            slide_id,
            level=level,
            project_id=request.project_id,
            base_embeddings_dir=analysis_embeddings_dir,
        )

        if emb_path is None:
            if level == 0:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "LEVEL0_EMBEDDINGS_REQUIRED",
                        "message": f"Level 0 (full resolution) embeddings do not exist for slide {slide_id}. Generate embeddings first using /api/embed-slide with level=0.",
                        "needs_embedding": True,
                        "slide_id": slide_id,
                        "level": 0,
                        "project_id": request.project_id,
                        "searched_dirs": [str(d) for d in searched_dirs],
                    },
                )
            raise HTTPException(status_code=404, detail=f"Slide {slide_id} not found")

        if not emb_path.exists():
            raise HTTPException(status_code=404, detail=f"Slide {slide_id} not found")

        embeddings = np.load(emb_path)

        try:
            results = multi_model_inference.predict_all(
                embeddings,
                model_ids=effective_model_ids,
                return_attention=True,
            )
        except Exception as exc:
            logger.error("Multi-model inference failed: %s", exc)
            raise HTTPException(status_code=500, detail=f"Inference failed: {str(exc)}")

        processing_time = (time.time() - start_time) * 1000

        try:
            coord_path = emb_path.with_name(f"{slide_id}_coords.npy")
            if coord_path.exists():
                cache_dir = emb_path.parent / "heatmap_cache"
                cache_dir.mkdir(exist_ok=True)
                cache_suffix = request.project_id if request.project_id else "global"

                emb_stat = emb_path.stat()
                coord_stat = coord_path.stat()
                data_sig_raw = f"{emb_stat.st_mtime_ns}:{emb_stat.st_size}:{coord_stat.st_mtime_ns}:{coord_stat.st_size}"
                data_signature = hashlib.sha1(data_sig_raw.encode("utf-8")).hexdigest()[:12]

                def checkpoint_signature_for_model(model_identifier: str) -> str:
                    cfg = model_configs.get(model_identifier, {})
                    model_dir = cfg.get("model_dir")
                    if not model_dir:
                        return "model_dir_missing"
                    checkpoint_path = project_root / "outputs" / str(model_dir) / "best_model.pt"
                    if not checkpoint_path.exists():
                        return "checkpoint_missing"
                    st = checkpoint_path.stat()
                    return f"{int(st.st_mtime_ns)}_{int(st.st_size)}"

                cached_count = 0
                for mid, pred in (results.get("predictions") or {}).items():
                    att = pred.get("attention")
                    if att is None:
                        continue
                    try:
                        att_arr = np.asarray(att, dtype=np.float32)
                        if att_arr.ndim != 1:
                            att_arr = att_arr.reshape(-1)

                        checkpoint_signature = checkpoint_signature_for_model(mid)
                        checkpoint_suffix = hashlib.sha1(
                            checkpoint_signature.encode("utf-8")
                        ).hexdigest()[:10]
                        attention_cache_path = cache_dir / (
                            f"{cache_suffix}_{slide_id}_{mid}_{checkpoint_suffix}_{data_signature}_attn_v1.npy"
                        )
                        np.save(attention_cache_path, att_arr)
                        cached_count += 1
                    except Exception as attn_err:
                        logger.warning(
                            "Failed to cache attention for %s/%s during analyze-multi: %s",
                            slide_id,
                            mid,
                            attn_err,
                        )

                if cached_count:
                    logger.info(
                        "Primed %d attention caches for %s (project=%s)",
                        cached_count,
                        slide_id,
                        request.project_id,
                    )
        except Exception as prewarm_err:
            logger.warning("Attention cache prewarm skipped for %s: %s", slide_id, prewarm_err)

        def normalize_prediction_dict(prediction: dict[str, Any]) -> dict[str, Any]:
            out = {k: v for k, v in prediction.items() if k != "attention"}
            model_id = str(out.get("model_id") or "")
            cfg = model_configs.get(model_id, {})

            if out.get("decision_threshold") is None:
                out["decision_threshold"] = cfg.get("decision_threshold", cfg.get("threshold", 0.5))

            if "confidence" in out and out["confidence"] is not None:
                try:
                    out["confidence"] = min(float(out["confidence"]), 0.99)
                except Exception:
                    pass

            return out

        predictions = {}
        for model_id, pred in results["predictions"].items():
            if "error" not in pred:
                predictions[model_id] = ModelPrediction(**normalize_prediction_dict(pred))

        by_category: dict[str, list[ModelPrediction]] = {}
        for cat_key, cat_preds in results.get("by_category", {}).items():
            by_category[cat_key] = [
                ModelPrediction(**normalize_prediction_dict(pred))
                for pred in cat_preds
                if "error" not in pred
            ]

        warnings: list[str] = list(results.get("warnings") or [])
        try:
            s1 = predictions.get("survival_1y")
            s3 = predictions.get("survival_3y")
            s5 = predictions.get("survival_5y")
            if s1 and s3:
                if s1.label == s1.negative_label and s3.label == s3.positive_label:
                    warnings.append(
                        "Survival predictions inconsistent: 1-year predicts deceased but 3-year predicts survived"
                    )
            if s3 and s5:
                if s3.label == s3.negative_label and s5.label == s5.positive_label:
                    warnings.append(
                        "Survival predictions inconsistent: 3-year predicts deceased but 5-year predicts survived"
                    )
            if s1 and s5:
                if s1.label == s1.negative_label and s5.label == s5.positive_label:
                    warnings.append(
                        "Survival predictions inconsistent: 1-year predicts deceased but 5-year predicts survived"
                    )
        except Exception:
            pass

        log_audit_event(
            "multi_model_analysis",
            slide_id,
            details={
                "models_run": list(predictions.keys()),
                "processing_time_ms": processing_time,
                "project_id": request.project_id,
            },
        )

        try:
            for mid, pred in predictions.items():
                await db_module.save_analysis_result(
                    slide_id=slide_id,
                    model_id=mid,
                    score=pred.score,
                    label=pred.label,
                    confidence=pred.confidence,
                    threshold=pred.decision_threshold,
                )
            logger.info("Saved %d analysis results to cache for %s", len(predictions), slide_id)
        except Exception as exc:
            logger.warning("Failed to cache analysis results for %s: %s", slide_id, exc)

        return MultiModelResponse(
            slide_id=slide_id,
            predictions=predictions,
            by_category=by_category,
            n_patches=results["n_patches"],
            processing_time_ms=processing_time,
            warnings=warnings,
        )

    return router
