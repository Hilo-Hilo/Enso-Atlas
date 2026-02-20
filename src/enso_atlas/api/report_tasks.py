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
    project_id: Optional[str] = None
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
            "project_id": self.project_id,
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
        
    def create_task(self, slide_id: str, project_id: Optional[str] = None) -> ReportTask:
        """Create a new report task."""
        task_id = f"report_{slide_id}_{uuid.uuid4().hex[:8]}"
        task = ReportTask(
            task_id=task_id,
            slide_id=slide_id,
            project_id=project_id,
        )
        with self.lock:
            self.tasks[task_id] = task
        return task
    
    def get_task(self, task_id: str) -> Optional[ReportTask]:
        """Get task by ID. Auto-expires stale tasks (>5 min running) with template fallback."""
        stale_cutoff = time.time() - 300  # 5 minutes max
        with self.lock:
            task = self.tasks.get(task_id)
            if task and task.status in [ReportTaskStatus.PENDING, ReportTaskStatus.RUNNING]:
                if task.created_at < stale_cutoff:
                    age = time.time() - task.created_at
                    logger.warning(f"Auto-expiring stale report task {task.task_id} (age={age:.0f}s)")
                    # Complete with a template fallback rather than failing
                    task.status = ReportTaskStatus.COMPLETED
                    task.progress = 100
                    task.stage = "complete"
                    task.message = f"Report generated (template fallback after {age:.0f}s timeout)"
                    task.completed_at = time.time()
                    if task.result is None:
                        # Produce a minimal template result so the frontend can display something
                        task.result = self._create_stale_fallback_result(task.slide_id)
            return task

    def _create_stale_fallback_result(self, slide_id: str) -> Dict[str, Any]:
        """Create a minimal template report for a stale/timed-out task."""
        return {
            "slide_id": slide_id,
            "report_json": {
                "case_id": slide_id,
                "task": "Prediction from H&E histopathology",
                "model_output": {
                    "label": "unknown",
                    "probability": 0.5,
                    "calibration_note": "Report generation timed out. Template report generated as fallback.",
                },
                "evidence": [],
                "limitations": [
                    "Report generation timed out — MedGemma inference was unavailable",
                    "This is a template fallback report with no AI-generated content",
                    "Requires manual pathology review",
                ],
                "suggested_next_steps": [
                    "Retry report generation when GPU resources are available",
                    "Review analysis results and evidence patches manually",
                    "Consult with pathology team",
                ],
                "safety_statement": "This is a research tool. Report generation timed out. All findings must be validated by qualified pathologists.",
            },
            "summary_text": f"CASE ANALYSIS SUMMARY\nCase ID: {slide_id}\n\nReport generation timed out. This is a template fallback.\nPlease retry or review analysis results manually.\n\nDISCLAIMER: This is a research tool.",
        }
    
    def get_task_by_slide(self, slide_id: str, project_id: Optional[str] = None) -> Optional[ReportTask]:
        """Get active task for a slide within the same project scope.

        Returns None if task is stale (>5 min).
        """
        stale_cutoff = time.time() - 300  # 5 minutes max
        with self.lock:
            for task in self.tasks.values():
                if (
                    task.slide_id == slide_id
                    and task.project_id == project_id
                    and task.status in [ReportTaskStatus.PENDING, ReportTaskStatus.RUNNING]
                ):
                    # Mark stale tasks as failed so they don't block retries
                    if task.created_at < stale_cutoff:
                        task.status = ReportTaskStatus.FAILED
                        task.error = "Task timed out (stale)"
                        task.message = "Report generation timed out. Please retry."
                        logger.warning(f"Marked stale report task {task.task_id} as failed")
                        continue
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
