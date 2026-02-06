"""
Background batch embedding task management.

Provides async batch re-embedding of all slides with progress tracking.
Used for "Force Re-Embed" across all slides and overnight batch runs.
"""

import threading
import time
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class BatchEmbedStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BatchEmbedSlideResult:
    """Result for a single slide in batch embedding."""
    slide_id: str
    status: str = "pending"  # pending, running, completed, failed, skipped
    num_patches: int = 0
    processing_time_seconds: float = 0.0
    error: Optional[str] = None


@dataclass
class BatchEmbedTask:
    """Represents a background batch embedding task."""
    task_id: str
    slide_ids: List[str]
    level: int = 0
    force: bool = True
    concurrency: int = 1
    status: BatchEmbedStatus = BatchEmbedStatus.PENDING
    progress: float = 0.0  # 0-100
    current_slide_index: int = 0
    current_slide_id: str = ""
    message: str = "Waiting to start..."
    results: List[BatchEmbedSlideResult] = field(default_factory=list)
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    cancel_requested: bool = False

    @property
    def total_slides(self) -> int:
        return len(self.slide_ids)

    @property
    def completed_slides(self) -> int:
        return sum(1 for r in self.results if r.status in ("completed", "failed", "skipped"))

    @property
    def elapsed_seconds(self) -> float:
        if self.started_at:
            end_time = self.completed_at or time.time()
            return end_time - self.started_at
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "level": self.level,
            "force": self.force,
            "concurrency": self.concurrency,
            "progress": round(self.progress, 1),
            "current_slide_index": self.current_slide_index,
            "current_slide_id": self.current_slide_id,
            "total_slides": self.total_slides,
            "completed_slides": self.completed_slides,
            "message": self.message,
            "error": self.error,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "cancel_requested": self.cancel_requested,
        }

    def to_full_dict(self) -> Dict[str, Any]:
        """Include per-slide results."""
        data = self.to_dict()
        data["results"] = [
            {
                "slide_id": r.slide_id,
                "status": r.status,
                "num_patches": r.num_patches,
                "processing_time_seconds": round(r.processing_time_seconds, 1),
                "error": r.error,
            }
            for r in self.results
        ]
        succeeded = sum(1 for r in self.results if r.status == "completed")
        failed = sum(1 for r in self.results if r.status == "failed")
        skipped = sum(1 for r in self.results if r.status == "skipped")
        total_patches = sum(r.num_patches for r in self.results if r.status == "completed")
        data["summary"] = {
            "total": self.total_slides,
            "succeeded": succeeded,
            "failed": failed,
            "skipped": skipped,
            "total_patches": total_patches,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
        }
        return data


class BatchEmbedTaskManager:
    """Manages background batch embedding tasks with thread safety."""

    def __init__(self):
        self.tasks: Dict[str, BatchEmbedTask] = {}
        self.lock = threading.Lock()

    def create_task(
        self,
        slide_ids: List[str],
        level: int = 0,
        force: bool = True,
        concurrency: int = 1,
    ) -> BatchEmbedTask:
        """Create a new batch embedding task."""
        task_id = f"batch_embed_{uuid.uuid4().hex[:12]}"
        task = BatchEmbedTask(
            task_id=task_id,
            slide_ids=slide_ids,
            level=level,
            force=force,
            concurrency=concurrency,
        )
        with self.lock:
            self.tasks[task_id] = task
        return task

    def get_task(self, task_id: str) -> Optional[BatchEmbedTask]:
        with self.lock:
            return self.tasks.get(task_id)

    def update_task(self, task_id: str, **updates) -> Optional[BatchEmbedTask]:
        with self.lock:
            task = self.tasks.get(task_id)
            if task:
                for key, value in updates.items():
                    if hasattr(task, key):
                        setattr(task, key, value)
        return task

    def add_result(self, task_id: str, result: BatchEmbedSlideResult):
        with self.lock:
            task = self.tasks.get(task_id)
            if task:
                task.results.append(result)

    def request_cancel(self, task_id: str) -> bool:
        with self.lock:
            task = self.tasks.get(task_id)
            if task and task.status == BatchEmbedStatus.RUNNING:
                task.cancel_requested = True
                return True
        return False

    def is_cancelled(self, task_id: str) -> bool:
        with self.lock:
            task = self.tasks.get(task_id)
            return task.cancel_requested if task else False

    def get_active_task(self) -> Optional[BatchEmbedTask]:
        """Get the currently running batch embed task, if any."""
        with self.lock:
            for task in self.tasks.values():
                if task.status in (BatchEmbedStatus.PENDING, BatchEmbedStatus.RUNNING):
                    return task
        return None

    def list_tasks(self) -> List[Dict[str, Any]]:
        with self.lock:
            return [
                t.to_dict()
                for t in sorted(self.tasks.values(), key=lambda t: t.created_at, reverse=True)
            ]


# Global instance
batch_embed_manager = BatchEmbedTaskManager()
