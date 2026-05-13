"""Asynchronous batch analysis route and background worker."""

import concurrent.futures
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .batch_tasks import BatchModelResult, BatchSlideResult, BatchTaskStatus


class AsyncBatchRequest(BaseModel):
    """Request for async batch analysis."""

    model_config = ConfigDict(protected_namespaces=())

    slide_ids: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of slide IDs to analyze",
    )
    concurrency: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Number of slides to process in parallel (1-10)",
    )
    model_ids: list[str] | None = Field(
        default=None,
        description="Model IDs to run. If None, uses default classifier.",
    )
    level: int = Field(
        default=0,
        ge=0,
        le=1,
        description="Embedding resolution level (default 0=full, dense; 1=downsampled)",
    )
    force_reembed: bool = Field(
        default=False,
        description="Force re-computation of embeddings even if cached",
    )
    project_id: str | None = Field(
        default=None,
        description="Project ID to scope embeddings and labels",
    )


class AsyncBatchResponse(BaseModel):
    """Response from async batch analysis start."""

    task_id: str
    status: str
    total_slides: int
    message: str


def create_async_batch_analysis_router(
    *,
    classifier_provider: Callable[[], Any],
    multi_model_inference_provider: Callable[[], Any],
    model_configs_provider: Callable[[], dict[str, Any]],
    batch_task_manager: Any,
    resolve_project_embeddings_dir: Callable[..., Path],
    resolve_embedding_path: Callable[..., tuple[Path | None, list[Path]]],
    resolve_project_model_ids: Callable[[str | None], Any],
    resolve_project_label_pair: Callable[..., tuple[str, str]],
    classifier_threshold: Callable[[], float],
    log_audit_event: Callable[..., None],
    logger: Any,
) -> APIRouter:
    router = APIRouter()

    @router.post("/api/analyze-batch/async", response_model=AsyncBatchResponse)
    async def analyze_batch_async(request: AsyncBatchRequest, background_tasks: BackgroundTasks):
        """
        Start asynchronous batch analysis with progress and cancellation support.

        Returns a ``task_id`` immediately. If ``project_id`` is provided, model
        allowlists, embeddings, and labels are enforced for that project scope.
        """
        if classifier_provider() is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        resolve_project_embeddings_dir(
            request.project_id,
            require_exists=request.project_id is not None,
        )
        allowed_model_ids = await resolve_project_model_ids(request.project_id)
        if request.model_ids is not None and allowed_model_ids is not None:
            disallowed = sorted(set(request.model_ids) - allowed_model_ids)
            if disallowed:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "MODELS_NOT_ALLOWED_FOR_PROJECT",
                        "project_id": request.project_id,
                        "disallowed_models": disallowed,
                    },
                )

        task_pos_label, task_neg_label = resolve_project_label_pair(
            request.project_id,
            positive_default="RESPONDER",
            negative_default="NON-RESPONDER",
            uppercase=True,
        )

        task = batch_task_manager.create_task(
            request.slide_ids,
            positive_label=task_pos_label,
            negative_label=task_neg_label,
        )

        def run_batch_analysis():
            run_batch_analysis_background(
                task.task_id,
                request.slide_ids,
                request.concurrency,
                model_ids=request.model_ids,
                level=request.level,
                force_reembed=request.force_reembed,
                project_id=request.project_id,
            )

        background_tasks.add_task(run_batch_analysis)

        return AsyncBatchResponse(
            task_id=task.task_id,
            status="pending",
            total_slides=len(request.slide_ids),
            message=f"Batch analysis started. Poll /api/analyze-batch/status/{task.task_id} for progress.",
        )

    def run_batch_analysis_background(
        task_id: str,
        slide_ids: list[str],
        concurrency: int = 4,
        model_ids: list[str] | None = None,
        level: int = 0,
        force_reembed: bool = False,
        project_id: str | None = None,
    ):
        """Background task to run batch analysis with progress tracking."""
        task = batch_task_manager.get_task(task_id)
        if not task:
            return

        batch_task_manager.update_task(
            task_id,
            status=BatchTaskStatus.RUNNING,
            started_at=time.time(),
            message="Starting batch analysis...",
        )

        multi_model_inference = multi_model_inference_provider()
        use_multi_model = model_ids is not None and multi_model_inference is not None
        project_requested = project_id is not None
        try:
            batch_embeddings_dir = resolve_project_embeddings_dir(
                project_id,
                require_exists=project_requested,
            )
            project_pos_label, project_neg_label = resolve_project_label_pair(
                project_id,
                positive_default="RESPONDER",
                negative_default="NON-RESPONDER",
                uppercase=True,
            )
        except Exception as exc:
            batch_task_manager.update_task(
                task_id,
                status=BatchTaskStatus.FAILED,
                error=str(exc),
                message=f"Batch analysis failed: {str(exc)}",
            )
            return

        effective_model_ids = list(model_ids or [])

        def resolve_emb_path(slide_id: str):
            emb_path, _ = resolve_embedding_path(
                slide_id,
                level=level,
                project_id=project_id,
                base_embeddings_dir=batch_embeddings_dir,
            )
            return emb_path

        def analyze_single_slide(slide_id: str) -> BatchSlideResult:
            emb_path = resolve_emb_path(slide_id)

            if emb_path is None or not emb_path.exists():
                return BatchSlideResult(
                    slide_id=slide_id,
                    prediction="ERROR",
                    error=f"Slide {slide_id} embeddings not found (level {level})",
                )

            try:
                embeddings = np.load(emb_path)

                if use_multi_model:
                    model_results_list = []
                    primary_score = 0.0
                    primary_label = "UNKNOWN"
                    primary_conf = 0.0

                    if not effective_model_ids:
                        return BatchSlideResult(
                            slide_id=slide_id,
                            prediction="ERROR",
                            error="No permitted models available for this request",
                        )

                    model_configs = model_configs_provider()
                    for i, mid in enumerate(effective_model_ids):
                        try:
                            model_obj = multi_model_inference.models.get(mid)
                            if model_obj is None:
                                model_results_list.append(
                                    BatchModelResult(
                                        model_id=mid,
                                        model_name=mid,
                                        error=f"Model {mid} not found",
                                    )
                                )
                                continue
                            cfg = model_configs.get(mid, {})
                            pred_result = multi_model_inference.predict_single(embeddings, mid)
                            if "error" in pred_result:
                                model_results_list.append(
                                    BatchModelResult(
                                        model_id=mid,
                                        model_name=cfg.get("display_name", mid),
                                        error=pred_result["error"],
                                    )
                                )
                                if i == 0:
                                    primary_label = "ERROR"
                                continue
                            score = float(pred_result["score"])
                            confidence = abs(score - 0.5) * 2
                            pos_label = cfg.get("positive_label", "Positive")
                            neg_label = cfg.get("negative_label", "Negative")
                            label = pos_label if score > 0.5 else neg_label
                            model_results_list.append(
                                BatchModelResult(
                                    model_id=mid,
                                    model_name=cfg.get("display_name", mid),
                                    prediction=label,
                                    score=score,
                                    confidence=confidence,
                                    positive_label=pos_label,
                                    negative_label=neg_label,
                                )
                            )
                            if i == 0:
                                primary_score = score
                                primary_label = label
                                primary_conf = confidence
                        except Exception as exc:
                            model_results_list.append(
                                BatchModelResult(
                                    model_id=mid,
                                    model_name=mid,
                                    error=str(exc),
                                )
                            )

                    if primary_conf < 0.3:
                        uncertainty_level = "high"
                        requires_review = True
                    elif primary_conf < 0.6:
                        uncertainty_level = "moderate"
                        requires_review = True
                    else:
                        uncertainty_level = "low"
                        requires_review = False

                    return BatchSlideResult(
                        slide_id=slide_id,
                        prediction=primary_label,
                        score=primary_score,
                        confidence=primary_conf,
                        patches_analyzed=len(embeddings),
                        requires_review=requires_review,
                        uncertainty_level=uncertainty_level,
                        model_results=model_results_list,
                    )

                classifier = classifier_provider()
                score, _attention = classifier.predict(embeddings)
                threshold_val = classifier_threshold()
                label = project_pos_label if score >= threshold_val else project_neg_label
                confidence = abs(score - threshold_val) * 2

                if confidence < 0.3:
                    uncertainty_level = "high"
                    requires_review = True
                elif confidence < 0.6:
                    uncertainty_level = "moderate"
                    requires_review = True
                else:
                    uncertainty_level = "low"
                    requires_review = False

                return BatchSlideResult(
                    slide_id=slide_id,
                    prediction=label,
                    score=float(score),
                    confidence=float(confidence),
                    patches_analyzed=len(embeddings),
                    requires_review=requires_review,
                    uncertainty_level=uncertainty_level,
                )
            except Exception as exc:
                logger.error("Batch analysis failed for %s: %s", slide_id, exc)
                return BatchSlideResult(
                    slide_id=slide_id,
                    prediction="ERROR",
                    error=str(exc),
                )

        try:
            total = len(slide_ids)
            completed = 0

            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                future_to_slide = {
                    executor.submit(analyze_single_slide, slide_id): slide_id
                    for slide_id in slide_ids
                }

                for future in concurrent.futures.as_completed(future_to_slide):
                    if batch_task_manager.is_cancelled(task_id):
                        for future_item in future_to_slide:
                            future_item.cancel()
                        batch_task_manager.update_task(
                            task_id,
                            status=BatchTaskStatus.CANCELLED,
                            message=f"Cancelled after {completed}/{total} slides",
                            completed_at=time.time(),
                        )
                        logger.info(
                            "Batch analysis %s cancelled after %d slides", task_id, completed
                        )
                        return

                    slide_id = future_to_slide[future]
                    try:
                        result = future.result()
                        batch_task_manager.add_result(task_id, result)
                        completed += 1

                        progress = (completed / total) * 100
                        batch_task_manager.update_task(
                            task_id,
                            progress=progress,
                            current_slide_index=completed,
                            current_slide_id=slide_id,
                            message=f"Analyzing slide {completed}/{total}: {slide_id[:20]}...",
                        )

                    except Exception as exc:
                        logger.error("Future failed for %s: %s", slide_id, exc)
                        batch_task_manager.add_result(
                            task_id,
                            BatchSlideResult(
                                slide_id=slide_id,
                                prediction="ERROR",
                                error=str(exc),
                            ),
                        )
                        completed += 1

            batch_task_manager.update_task(
                task_id,
                status=BatchTaskStatus.COMPLETED,
                progress=100,
                message=f"Completed analysis of {total} slides",
                completed_at=time.time(),
            )

            log_audit_event(
                "batch_analysis_async_completed",
                details={
                    "task_id": task_id,
                    "total_slides": total,
                },
            )

            logger.info("Batch analysis %s completed: %d slides", task_id, total)

        except Exception as exc:
            logger.error("Batch analysis %s failed: %s", task_id, exc)
            batch_task_manager.update_task(
                task_id,
                status=BatchTaskStatus.FAILED,
                error=str(exc),
                message=f"Batch analysis failed: {str(exc)}",
            )

    return router
