"""
Background embedding task management.

Provides async embedding with status polling for long-running level 0 operations.
"""

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class EmbeddingTask:
    """Represents a background embedding task."""

    task_id: str
    slide_id: str
    level: int
    project_id: str | None = None
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0  # 0-100
    message: str = "Waiting to start..."
    num_patches: int = 0
    processing_time_seconds: float = 0.0
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "slide_id": self.slide_id,
            "level": self.level,
            "project_id": self.project_id,
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "num_patches": self.num_patches,
            "processing_time_seconds": self.processing_time_seconds,
            "error": self.error,
            "elapsed_seconds": time.time() - self.created_at,
        }


class EmbeddingTaskManager:
    """Manages background embedding tasks with thread safety."""

    def __init__(self, max_concurrent: int = 1):
        self.tasks: dict[str, EmbeddingTask] = {}
        self.lock = threading.Lock()
        self.max_concurrent = max_concurrent
        self._running_count = 0

    def create_task(
        self, slide_id: str, level: int, project_id: str | None = None
    ) -> EmbeddingTask:
        """Create a new embedding task."""
        project_fragment = project_id or "global"
        task_id = f"emb_{project_fragment}_{slide_id}_{level}_{uuid.uuid4().hex[:8]}"
        task = EmbeddingTask(
            task_id=task_id,
            slide_id=slide_id,
            level=level,
            project_id=project_id,
        )
        with self.lock:
            self.tasks[task_id] = task
        return task

    def get_task(self, task_id: str) -> EmbeddingTask | None:
        """Get task by ID."""
        with self.lock:
            return self.tasks.get(task_id)

    def get_task_by_slide(
        self, slide_id: str, level: int, project_id: str | None = None
    ) -> EmbeddingTask | None:
        """Get active task for a slide/level/project combination."""
        with self.lock:
            for task in self.tasks.values():
                if (
                    task.slide_id == slide_id
                    and task.level == level
                    and task.project_id == project_id
                    and task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]
                ):
                    return task
        return None

    def update_task(self, task_id: str, **updates) -> EmbeddingTask | None:
        """Update task fields."""
        with self.lock:
            task = self.tasks.get(task_id)
            if task:
                for key, value in updates.items():
                    if hasattr(task, key):
                        setattr(task, key, value)
        return task

    def cleanup_old_tasks(self, max_age_seconds: int = 3600):
        """Remove completed tasks older than max_age_seconds."""
        cutoff = time.time() - max_age_seconds
        with self.lock:
            to_remove = [
                tid
                for tid, task in self.tasks.items()
                if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]
                and task.created_at < cutoff
            ]
            for tid in to_remove:
                del self.tasks[tid]


# Global task manager instance
task_manager = EmbeddingTaskManager()
