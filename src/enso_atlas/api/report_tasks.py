"""
Background report generation task management.

Provides async MedGemma report generation with status polling for slow operations.
"""

import threading
import time
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class ReportTaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ReportTask:
    """Represents a background report generation task."""
    task_id: str
    slide_id: str
    status: ReportTaskStatus = ReportTaskStatus.PENDING
    progress: float = 0.0  # 0-100
    message: str = "Waiting to start..."
    stage: str = "initializing"  # initializing, analyzing, generating, formatting, complete
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "task_id": self.task_id,
            "slide_id": self.slide_id,
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "stage": self.stage,
            "error": self.error,
            "elapsed_seconds": round(time.time() - self.created_at, 1),
        }
        if self.result:
            result["result"] = self.result
        return result


class ReportTaskManager:
    """Manages background report generation tasks with thread safety."""
    
    def __init__(self, max_concurrent: int = 2):
        self.tasks: Dict[str, ReportTask] = {}
        self.lock = threading.Lock()
        self.max_concurrent = max_concurrent
        
    def create_task(self, slide_id: str) -> ReportTask:
        """Create a new report task."""
        task_id = f"report_{slide_id}_{uuid.uuid4().hex[:8]}"
        task = ReportTask(
            task_id=task_id,
            slide_id=slide_id,
        )
        with self.lock:
            self.tasks[task_id] = task
        return task
    
    def get_task(self, task_id: str) -> Optional[ReportTask]:
        """Get task by ID."""
        with self.lock:
            return self.tasks.get(task_id)
    
    def get_task_by_slide(self, slide_id: str) -> Optional[ReportTask]:
        """Get active task for a slide."""
        with self.lock:
            for task in self.tasks.values():
                if (task.slide_id == slide_id and 
                    task.status in [ReportTaskStatus.PENDING, ReportTaskStatus.RUNNING]):
                    return task
        return None
    
    def update_task(self, task_id: str, **updates) -> Optional[ReportTask]:
        """Update task fields."""
        with self.lock:
            task = self.tasks.get(task_id)
            if task:
                for key, value in updates.items():
                    if hasattr(task, key):
                        setattr(task, key, value)
        return task
    
    def cleanup_old_tasks(self, max_age_seconds: int = 1800):
        """Remove completed tasks older than max_age_seconds (30 min default)."""
        cutoff = time.time() - max_age_seconds
        with self.lock:
            to_remove = [
                tid for tid, task in self.tasks.items()
                if task.status in [ReportTaskStatus.COMPLETED, ReportTaskStatus.FAILED]
                and task.created_at < cutoff
            ]
            for tid in to_remove:
                del self.tasks[tid]


# Global task manager instance
report_task_manager = ReportTaskManager()
