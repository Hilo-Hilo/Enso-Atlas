"""On-demand and batch slide embedding generation routes."""

import glob
import os
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from .batch_embed_tasks import BatchEmbedSlideResult, BatchEmbedStatus
from .embedding_tasks import TaskStatus
from .schemas import EmbedSlideRequest


class BatchEmbedRequest(BaseModel):
    """Request for batch re-embedding of multiple slides."""

    level: int = Field(
        default=0, ge=0, le=0, description="Resolution level fixed to 0 (dense full-resolution)"
    )
    force: bool = Field(default=True, description="Force re-embedding even if cached")
    slide_ids: list[str] | None = Field(
        default=None, description="Specific slide IDs (omitted = inventory-derived)"
    )
    concurrency: int = Field(
        default=1, ge=1, le=4, description="Concurrent embedding workers (1-4)"
    )
    project_id: str | None = Field(
        default=None, description="Project ID to scope slide + embeddings paths"
    )


def create_embedding_generation_router(
    *,
    task_manager: Any,
    batch_embed_manager: Any,
    resolve_project_embeddings_dir: Callable[..., Path],
    resolve_slide_path: Callable[[str, str | None], Path | None],
    require_project: Callable[[str | None], Any],
    batch_embed_inventory_slide_ids: Callable[[str | None], Awaitable[list[str]]],
    logger: Any,
) -> APIRouter:
    router = APIRouter()

    def resolve_pathfoundation_local() -> str | None:
        """Resolve Path Foundation TF saved-model path from local cache only."""
        hf_home = os.environ.get(
            "HF_HOME",
            os.environ.get(
                "TRANSFORMERS_CACHE",
                os.path.join(os.path.expanduser("~"), ".cache", "huggingface"),
            ),
        )
        model_base = os.path.join(hf_home, "hub", "models--google--path-foundation", "snapshots")
        candidates = [
            model_base,
            "/root/.cache/huggingface/hub/models--google--path-foundation/snapshots",
            "/app/cache/huggingface/hub/models--google--path-foundation/snapshots",
        ]
        for base in candidates:
            if os.path.isdir(base):
                snaps = sorted(glob.glob(os.path.join(base, "*")))
                for snap in reversed(snaps):
                    if os.path.isdir(snap) and os.path.exists(os.path.join(snap, "saved_model.pb")):
                        logger.info("Path Foundation TF model found at: %s", snap)
                        return snap
                    if os.path.isdir(snap):
                        logger.info("Path Foundation snapshot dir found at: %s", snap)
                        return snap
        return None

    def extract_dense_grid_patches(
        *,
        slide_path: Path,
        level: int,
        progress_callback: Callable[[int, int, int], None] | None = None,
    ) -> tuple[list[np.ndarray], list[list[int]]]:
        """Extract all 224x224 dense grid patches for the requested slide level."""
        import openslide

        slide = openslide.OpenSlide(str(slide_path))
        try:
            actual_level = min(level, slide.level_count - 1)
            level_dims = slide.level_dimensions[actual_level]
            downsample = slide.level_downsamples[actual_level]

            width, height = level_dims
            patch_size = 224
            stride = 224
            total_rows = max(0, ((height - patch_size) // stride) + 1)
            total_cols = max(0, ((width - patch_size) // stride) + 1)
            total_potential = max(1, total_rows * total_cols)

            patches: list[np.ndarray] = []
            coords: list[list[int]] = []
            processed = 0

            for y in range(0, height - patch_size + 1, stride):
                for x in range(0, width - patch_size + 1, stride):
                    x0 = int(x * downsample)
                    y0 = int(y * downsample)

                    patch = slide.read_region((x0, y0), actual_level, (patch_size, patch_size))
                    patch = patch.convert("RGB")

                    patches.append(np.array(patch))
                    coords.append([x0, y0])
                    processed += 1

                    if progress_callback is not None:
                        progress_callback(len(patches), processed, total_potential)

            return patches, coords
        finally:
            slide.close()

    def generate_pathfoundation_embeddings(patches: list[np.ndarray]) -> np.ndarray:
        """Generate Path Foundation embeddings from already extracted patches."""
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        import tensorflow as tf

        saved_model_path = resolve_pathfoundation_local()
        if not saved_model_path or not os.path.exists(saved_model_path):
            raise RuntimeError(
                "Path Foundation model not found locally. Pre-download it before running embedding."
            )

        model = tf.saved_model.load(saved_model_path)
        infer = model.signatures["serving_default"]

        batch_size = 64
        all_embeddings = []
        for i in range(0, len(patches), batch_size):
            batch = patches[i : i + batch_size]
            batch_array = np.array(batch, dtype=np.float32) / 255.0
            batch_tensor = tf.constant(batch_array)
            result = infer(inputs=batch_tensor)
            all_embeddings.append(result["output_0"].numpy())

        embeddings: np.ndarray = np.vstack(all_embeddings).astype(np.float32)
        return embeddings

    @router.post("/api/embed-slide")
    async def embed_slide_on_demand(request: EmbedSlideRequest, background_tasks: BackgroundTasks):
        """
        Extract patches and generate embeddings for a slide on-demand.

        Enforces level 0 (full resolution, dense) and starts a background task
        with task_id polling.
        """
        slide_id = request.slide_id
        level = request.level
        force_reembed = request.force
        use_async = request.async_mode

        if level != 0:
            raise HTTPException(
                status_code=400, detail="level must be 0 (dense full-resolution policy)"
            )

        embed_embeddings_dir = resolve_project_embeddings_dir(
            request.project_id,
            require_exists=False,
        )

        level_dir = (
            embed_embeddings_dir
            if (level == 0 and embed_embeddings_dir.name == "level0")
            else embed_embeddings_dir / f"level{level}"
        )
        level_dir.mkdir(parents=True, exist_ok=True)

        emb_path = level_dir / f"{slide_id}.npy"
        coord_path = level_dir / f"{slide_id}_coords.npy"

        if emb_path.exists() and coord_path.exists() and not force_reembed:
            emb = np.load(emb_path)
            return {
                "status": "exists",
                "slide_id": slide_id,
                "level": level,
                "project_id": request.project_id,
                "num_patches": len(emb),
                "message": f"Level {level} embeddings already exist",
            }

        existing_task = task_manager.get_task_by_slide(slide_id, level, request.project_id)
        if existing_task:
            return {
                "status": "in_progress",
                "task_id": existing_task.task_id,
                "slide_id": slide_id,
                "level": level,
                "project_id": request.project_id,
                "progress": existing_task.progress,
                "message": existing_task.message,
            }

        slide_path = resolve_slide_path(slide_id, request.project_id)
        if not slide_path:
            raise HTTPException(
                status_code=404,
                detail=f"Slide file not found for {slide_id}. Level {level} embedding requires the original .svs file.",
            )

        if use_async:
            task = task_manager.create_task(slide_id, level, project_id=request.project_id)

            def run_embedding():
                run_embedding_task(
                    task.task_id,
                    slide_id,
                    level,
                    slide_path,
                    emb_path,
                    coord_path,
                )

            background_tasks.add_task(run_embedding)

            return {
                "status": "started",
                "task_id": task.task_id,
                "slide_id": slide_id,
                "level": level,
                "project_id": request.project_id,
                "message": f"Embedding started in background. Poll /api/embed-slide/status/{task.task_id} for progress.",
                "estimated_time_minutes": 15,
            }

        return run_embedding_inline(
            slide_id,
            level,
            slide_path,
            emb_path,
            coord_path,
            project_id=request.project_id,
        )

    def run_embedding_task(
        task_id: str,
        slide_id: str,
        level: int,
        slide_path: Path,
        emb_path: Path,
        coord_path: Path,
    ):
        """Background task to run embedding."""
        task = task_manager.get_task(task_id)
        if not task:
            return

        task_manager.update_task(
            task_id,
            status=TaskStatus.RUNNING,
            started_at=time.time(),
            message="Starting embedding process...",
        )

        try:
            task_manager.update_task(task_id, progress=5, message="Opening slide file...")
            last_progress_update = time.time()

            def progress_callback(num_patches: int, processed: int, total_potential: int):
                nonlocal last_progress_update
                if time.time() - last_progress_update > 2:
                    extraction_progress = min(processed / total_potential, 1.0) * 40
                    task_manager.update_task(
                        task_id,
                        progress=10 + extraction_progress,
                        message=f"Extracting patches: {num_patches} grid patches extracted ({processed}/{total_potential} checked)",
                    )
                    last_progress_update = time.time()

            task_manager.update_task(
                task_id, progress=10, message="Extracting 224x224 grid patches..."
            )
            patches, coords = extract_dense_grid_patches(
                slide_path=slide_path,
                level=level,
                progress_callback=progress_callback,
            )

            if not patches:
                task_manager.update_task(
                    task_id,
                    status=TaskStatus.FAILED,
                    error="No 224x224 grid patches found in slide",
                )
                return

            task_manager.update_task(
                task_id,
                progress=50,
                message=f"Extracted {len(patches)} patches. Loading Path Foundation model...",
            )

            embeddings = generate_pathfoundation_embeddings(patches)
            coords_array = np.array(coords)

            task_manager.update_task(task_id, progress=95, message="Saving embeddings...")

            np.save(emb_path, embeddings)
            np.save(coord_path, coords_array)

            elapsed = time.time() - task.started_at
            task_manager.update_task(
                task_id,
                status=TaskStatus.COMPLETED,
                progress=100,
                num_patches=len(patches),
                processing_time_seconds=elapsed,
                message=f"Completed: {len(patches)} patches embedded in {elapsed:.1f}s",
                completed_at=time.time(),
            )

            logger.info(
                "Background embedding completed for %s level %s: %d patches in %.1fs",
                slide_id,
                level,
                len(patches),
                elapsed,
            )

        except Exception as exc:
            logger.error("Background embedding failed for %s: %s", slide_id, exc)
            task_manager.update_task(task_id, status=TaskStatus.FAILED, error=str(exc))

    def run_embedding_inline(
        slide_id: str,
        level: int,
        slide_path: Path,
        emb_path: Path,
        coord_path: Path,
        project_id: str | None = None,
    ):
        """Run embedding inline."""
        start_time = time.time()

        try:
            patches, coords = extract_dense_grid_patches(slide_path=slide_path, level=level)
            if not patches:
                raise HTTPException(
                    status_code=400, detail="No 224x224 grid patches found in slide"
                )

            embeddings = generate_pathfoundation_embeddings(patches)
            coords_array = np.array(coords)

            np.save(emb_path, embeddings)
            np.save(coord_path, coords_array)

            elapsed = time.time() - start_time

            return {
                "status": "completed",
                "slide_id": slide_id,
                "level": level,
                "project_id": project_id,
                "num_patches": len(patches),
                "processing_time_seconds": round(elapsed, 1),
                "message": f"Embedded {len(patches)} patches at level {level}",
            }

        except HTTPException:
            raise
        except Exception as exc:
            logger.error("Embedding failed: %s", exc)
            raise HTTPException(status_code=500, detail=f"Embedding failed: {str(exc)}")

    @router.post("/api/embed-slides/batch")
    async def start_batch_embed(request: BatchEmbedRequest, background_tasks: BackgroundTasks):
        """
        Start batch re-embedding of slides.

        If slide_ids is omitted, slide IDs are derived from project-aware inventory.
        """
        active = batch_embed_manager.get_active_task()
        if active:
            return {
                "batch_task_id": active.task_id,
                "status": active.status.value,
                "project_id": active.project_id,
                "message": f"Batch embedding already in progress ({active.completed_slides}/{active.total_slides} done)",
                "total": active.total_slides,
            }

        require_project(request.project_id)

        target_slides = request.slide_ids
        if target_slides is None:
            target_slides = await batch_embed_inventory_slide_ids(request.project_id)

        if not target_slides:
            raise HTTPException(status_code=400, detail="No slides to embed")

        task = batch_embed_manager.create_task(
            slide_ids=target_slides,
            level=request.level,
            force=request.force,
            concurrency=request.concurrency,
            project_id=request.project_id,
        )

        def run_batch_embed():
            run_batch_embed_background(task.task_id)

        background_tasks.add_task(run_batch_embed)

        return {
            "batch_task_id": task.task_id,
            "status": "started",
            "project_id": request.project_id,
            "total": len(target_slides),
            "message": f"Batch embedding started for {len(target_slides)} slides at level {request.level}.",
        }

    def run_batch_embed_background(task_id: str):
        """Background worker: sequentially re-embed each slide."""
        task = batch_embed_manager.get_task(task_id)
        if not task:
            return

        batch_embed_manager.update_task(
            task_id,
            status=BatchEmbedStatus.RUNNING,
            started_at=time.time(),
            message="Starting batch embedding...",
        )

        total = task.total_slides
        project_embeddings_dir = resolve_project_embeddings_dir(
            task.project_id, require_exists=False
        )

        for idx, slide_id in enumerate(task.slide_ids):
            if batch_embed_manager.is_cancelled(task_id):
                batch_embed_manager.update_task(
                    task_id,
                    status=BatchEmbedStatus.CANCELLED,
                    message=f"Cancelled after {idx}/{total} slides",
                    completed_at=time.time(),
                )
                return

            batch_embed_manager.update_task(
                task_id,
                current_slide_index=idx + 1,
                current_slide_id=slide_id,
                progress=(idx / total) * 100,
                message=f"Embedding slide {idx + 1}/{total}: {slide_id[:25]}...",
            )

            level = task.level
            level_dir = (
                project_embeddings_dir
                if (level == 0 and project_embeddings_dir.name == "level0")
                else project_embeddings_dir / f"level{level}"
            )
            level_dir.mkdir(parents=True, exist_ok=True)
            emb_path = level_dir / f"{slide_id}.npy"
            coord_path = level_dir / f"{slide_id}_coords.npy"

            if emb_path.exists() and coord_path.exists() and not task.force:
                try:
                    emb = np.load(emb_path)
                    batch_embed_manager.add_result(
                        task_id,
                        BatchEmbedSlideResult(
                            slide_id=slide_id,
                            status="skipped",
                            num_patches=len(emb),
                        ),
                    )
                except Exception:
                    batch_embed_manager.add_result(
                        task_id,
                        BatchEmbedSlideResult(slide_id=slide_id, status="skipped"),
                    )
                continue

            slide_path = resolve_slide_path(slide_id, task.project_id)
            if not slide_path:
                batch_embed_manager.add_result(
                    task_id,
                    BatchEmbedSlideResult(
                        slide_id=slide_id,
                        status="failed",
                        error=f"Slide file not found for {slide_id} (project_id={task.project_id})",
                    ),
                )
                continue

            slide_start = time.time()
            try:
                patches, coords = extract_dense_grid_patches(slide_path=slide_path, level=level)

                if not patches:
                    batch_embed_manager.add_result(
                        task_id,
                        BatchEmbedSlideResult(
                            slide_id=slide_id,
                            status="failed",
                            error="No 224x224 grid patches found",
                        ),
                    )
                    continue

                embeddings = generate_pathfoundation_embeddings(patches)
                coords_array = np.array(coords)

                if batch_embed_manager.is_cancelled(task_id):
                    batch_embed_manager.update_task(
                        task_id,
                        status=BatchEmbedStatus.CANCELLED,
                        message=f"Cancelled during slide {slide_id}",
                        completed_at=time.time(),
                    )
                    return

                np.save(emb_path, embeddings)
                np.save(coord_path, coords_array)

                elapsed = time.time() - slide_start
                batch_embed_manager.add_result(
                    task_id,
                    BatchEmbedSlideResult(
                        slide_id=slide_id,
                        status="completed",
                        num_patches=len(patches),
                        processing_time_seconds=elapsed,
                    ),
                )

                logger.info(
                    "Batch embed: %s level %s -> %d patches in %.1fs",
                    slide_id,
                    level,
                    len(patches),
                    elapsed,
                )

            except Exception as exc:
                logger.error("Batch embed failed for %s: %s", slide_id, exc)
                batch_embed_manager.add_result(
                    task_id,
                    BatchEmbedSlideResult(
                        slide_id=slide_id,
                        status="failed",
                        error=str(exc),
                        processing_time_seconds=time.time() - slide_start,
                    ),
                )

        batch_embed_manager.update_task(
            task_id,
            status=BatchEmbedStatus.COMPLETED,
            progress=100,
            message=f"Completed batch embedding of {total} slides",
            completed_at=time.time(),
        )
        logger.info("Batch embed %s completed: %d slides", task_id, total)

    return router
