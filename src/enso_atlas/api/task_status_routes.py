"""Task status, listing, and cancellation API routes."""

from collections.abc import Callable
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from .batch_embed_tasks import BatchEmbedStatus, BatchEmbedTaskManager
from .batch_tasks import BatchTaskManager, BatchTaskStatus
from .embedding_tasks import EmbeddingTaskManager, TaskStatus
from .report_tasks import ReportTaskManager


def create_task_status_router(
    *,
    embedding_task_manager: EmbeddingTaskManager,
    batch_task_manager: BatchTaskManager,
    report_task_manager: ReportTaskManager,
    batch_embed_manager: BatchEmbedTaskManager,
    resolve_project_embeddings_dir: Callable[..., Path],
    log_audit_event: Callable[..., None],
) -> APIRouter:
    router = APIRouter()

    @router.get("/api/analyze-batch/status/{task_id}")
    async def get_batch_status(task_id: str):
        """
        Get status of an async batch analysis task.

        Returns:
        - status: pending, running, completed, cancelled, or failed
        - progress: 0-100 percentage
        - current_slide_index: which slide is being processed
        - current_slide_id: ID of current slide
        - message: human-readable status message
        - results: full results (only when completed)
        """
        task = batch_task_manager.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        if task.status in [BatchTaskStatus.COMPLETED, BatchTaskStatus.CANCELLED]:
            return task.to_full_dict()

        return task.to_dict()

    @router.post("/api/analyze-batch/cancel/{task_id}")
    async def cancel_batch_analysis(task_id: str):
        """
        Cancel a running batch analysis task.

        Already completed slides will be retained in the results.
        """
        task = batch_task_manager.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        if task.status != BatchTaskStatus.RUNNING:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel task with status {task.status.value}",
            )

        success = batch_task_manager.request_cancel(task_id)

        if success:
            log_audit_event(
                "batch_analysis_cancelled",
                details={"task_id": task_id, "slides_completed": task.completed_slides},
            )
            return {
                "success": True,
                "message": f"Cancellation requested for task {task_id}",
                "completed_slides": task.completed_slides,
            }

        return {
            "success": False,
            "message": "Failed to request cancellation",
        }

    @router.get("/api/analyze-batch/tasks")
    async def list_batch_tasks(
        status: str | None = Query(None, description="Filter by status"),
    ):
        """
        List all batch analysis tasks.

        Optionally filter by status: pending, running, completed, cancelled, failed
        """
        status_filter = None
        if status:
            try:
                status_filter = BatchTaskStatus(status)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Invalid status: {status}. Valid values: "
                        "pending, running, completed, cancelled, failed"
                    ),
                )

        tasks = batch_task_manager.list_tasks(status_filter)

        return {
            "tasks": tasks,
            "total": len(tasks),
        }

    @router.get("/api/report/status/{task_id}")
    async def get_report_status(task_id: str):
        """
        Get status of an async report generation task.

        Poll this endpoint to track report generation progress.
        When status is 'completed', the result field contains the full report.
        """
        task = report_task_manager.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        return task.to_dict()

    @router.get("/api/report/tasks")
    async def list_report_tasks(
        slide_id: str | None = Query(None),
        status: str | None = Query(None),
    ):
        """List all report generation tasks."""
        tasks = []
        for task in report_task_manager.tasks.values():
            if slide_id and task.slide_id != slide_id:
                continue
            if status and task.status.value != status:
                continue
            tasks.append(task.to_dict())

        return {
            "tasks": sorted(tasks, key=lambda t: t.get("elapsed_seconds", 0), reverse=True),
            "total": len(tasks),
        }

    @router.get("/api/embed-slide/status/{task_id}")
    async def get_embedding_status(task_id: str):
        """Get status of a background embedding task."""
        task = embedding_task_manager.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        result = task.to_dict()

        if task.status == TaskStatus.COMPLETED:
            task_embeddings_dir = resolve_project_embeddings_dir(
                task.project_id, require_exists=False
            )
            level_dir = (
                task_embeddings_dir
                if task.level == 0 and task_embeddings_dir.name == "level0"
                else task_embeddings_dir / f"level{task.level}"
            )
            emb_path = level_dir / f"{task.slide_id}.npy"
            result["embedding_path"] = str(emb_path) if emb_path.exists() else None

        return result

    @router.get("/api/embed-slide/tasks")
    async def list_embedding_tasks(
        slide_id: str | None = Query(None),
        status: str | None = Query(None),
    ):
        """List all embedding tasks."""
        tasks = []
        for task in embedding_task_manager.tasks.values():
            if slide_id and task.slide_id != slide_id:
                continue
            if status and task.status.value != status:
                continue
            tasks.append(task.to_dict())

        return {
            "tasks": sorted(tasks, key=lambda t: t.get("elapsed_seconds", 0), reverse=True),
            "total": len(tasks),
        }

    @router.get("/api/embed-slides/batch/status/{batch_task_id}")
    async def get_batch_embed_status(batch_task_id: str):
        """
        Get progress of a batch embedding task.

        Returns:
        - completed/total/current slide
        - progress percentage
        - Per-slide results (when completed)
        """
        task = batch_embed_manager.get_task(batch_task_id)
        if not task:
            raise HTTPException(
                status_code=404, detail=f"Batch embed task {batch_task_id} not found"
            )

        if task.status in (
            BatchEmbedStatus.COMPLETED,
            BatchEmbedStatus.CANCELLED,
            BatchEmbedStatus.FAILED,
        ):
            return task.to_full_dict()
        return task.to_dict()

    @router.post("/api/embed-slides/batch/cancel/{batch_task_id}")
    async def cancel_batch_embed(batch_task_id: str):
        """Cancel a running batch embedding task."""
        task = batch_embed_manager.get_task(batch_task_id)
        if not task:
            raise HTTPException(
                status_code=404, detail=f"Batch embed task {batch_task_id} not found"
            )
        if task.status != BatchEmbedStatus.RUNNING:
            raise HTTPException(
                status_code=400, detail=f"Cannot cancel task with status {task.status.value}"
            )
        batch_embed_manager.request_cancel(batch_task_id)
        return {"success": True, "message": "Cancellation requested"}

    @router.get("/api/embed-slides/batch/active")
    async def get_active_batch_embed():
        """Get the currently active batch embed task, if any."""
        active = batch_embed_manager.get_active_task()
        if active:
            return active.to_dict()
        return {"status": "idle", "message": "No batch embedding in progress"}

    return router
