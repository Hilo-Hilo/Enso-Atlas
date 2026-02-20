"""
Background batch analysis task management.

Provides async batch analysis with status polling and cancellation support.
"""

import threading
import time
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class BatchTaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BatchModelResult:
    """Result for a single model prediction on a slide."""
    model_id: str
    model_name: str
    prediction: str = ""
    score: float = 0.0
    confidence: float = 0.0
    positive_label: str = "Positive"
    negative_label: str = "Negative"
    error: Optional[str] = None


@dataclass
class BatchSlideResult:
    """Result for a single slide in batch analysis."""
    slide_id: str
    prediction: str = ""
    score: float = 0.0
    confidence: float = 0.0
    patches_analyzed: int = 0
    requires_review: bool = False
    uncertainty_level: str = "unknown"
    error: Optional[str] = None
    model_results: Optional[List[BatchModelResult]] = None


@dataclass
class BatchTask:
    """Represents a background batch analysis task."""
    task_id: str
    slide_ids: List[str]
    status: BatchTaskStatus = BatchTaskStatus.PENDING
    progress: float = 0.0  # 0-100
    current_slide_index: int = 0
    current_slide_id: str = ""
    message: str = "Waiting to start..."
    results: List[BatchSlideResult] = field(default_factory=list)
    positive_label: str = "RESPONDER"
    negative_label: str = "NON-RESPONDER"
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
        return len(self.results)
    
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
            "progress": self.progress,
            "current_slide_index": self.current_slide_index,
            "current_slide_id": self.current_slide_id,
            "total_slides": self.total_slides,
            "completed_slides": self.completed_slides,
            "message": self.message,
            "error": self.error,
            "elapsed_seconds": self.elapsed_seconds,
            "cancel_requested": self.cancel_requested,
            "results_count": len(self.results),
            "positive_label": self.positive_label,
            "negative_label": self.negative_label,
        }
    
    def to_full_dict(self) -> Dict[str, Any]:
        """Include full results for final response."""
        data = self.to_dict()
        data["results"] = [
            {
                "slide_id": r.slide_id,
                "prediction": r.prediction,
                "score": r.score,
                "confidence": r.confidence,
                "patches_analyzed": r.patches_analyzed,
                "requires_review": r.requires_review,
                "uncertainty_level": r.uncertainty_level,
                "error": r.error,
                "model_results": [
                    {
                        "model_id": mr.model_id,
                        "model_name": mr.model_name,
                        "prediction": mr.prediction,
                        "score": mr.score,
                        "confidence": mr.confidence,
                        "positive_label": mr.positive_label,
                        "negative_label": mr.negative_label,
                        "error": mr.error,
                    }
                    for mr in (r.model_results or [])
                ] if r.model_results else None,
            }
            for r in self.results
        ]
        # Calculate summary
        completed = [r for r in self.results if r.error is None]
        failed = [r for r in self.results if r.error is not None]
        responders = [r for r in completed if r.prediction == self.positive_label]
        non_responders = [r for r in completed if r.prediction == self.negative_label]
        uncertain = [r for r in completed if r.requires_review]
        avg_confidence = (
            sum(r.confidence for r in completed) / len(completed)
            if completed else 0.0
        )
        data["summary"] = {
            "total": len(self.results),
            "completed": len(completed),
            "failed": len(failed),
            "responders": len(responders),
            "non_responders": len(non_responders),
            "uncertain": len(uncertain),
            "avg_confidence": round(avg_confidence, 3),
            "requires_review_count": sum(1 for r in self.results if r.requires_review),
            "positive_label": self.positive_label,
            "negative_label": self.negative_label,
        }
        data["processing_time_ms"] = self.elapsed_seconds * 1000
        return data


class BatchTaskManager:
    """Manages background batch analysis tasks with thread safety."""
    
    def __init__(self, max_concurrent: int = 2):
        self.tasks: Dict[str, BatchTask] = {}
        self.lock = threading.Lock()
        self.max_concurrent = max_concurrent
        
    def create_task(
        self,
        slide_ids: List[str],
        *,
        positive_label: str = "RESPONDER",
        negative_label: str = "NON-RESPONDER",
    ) -> BatchTask:
        """Create a new batch analysis task."""
        task_id = f"batch_{uuid.uuid4().hex[:12]}"
        task = BatchTask(
            task_id=task_id,
            slide_ids=slide_ids,
            positive_label=positive_label or "RESPONDER",
            negative_label=negative_label or "NON-RESPONDER",
        )
        with self.lock:
            self.tasks[task_id] = task
        return task
    
    def get_task(self, task_id: str) -> Optional[BatchTask]:
        """Get task by ID."""
        with self.lock:
            return self.tasks.get(task_id)
    
    def update_task(self, task_id: str, **updates) -> Optional[BatchTask]:
        """Update task fields."""
        with self.lock:
            task = self.tasks.get(task_id)
            if task:
                for key, value in updates.items():
                    if hasattr(task, key):
                        setattr(task, key, value)
        return task
    
    def add_result(self, task_id: str, result: BatchSlideResult) -> Optional[BatchTask]:
        """Add a slide result to the task."""
        with self.lock:
            task = self.tasks.get(task_id)
            if task:
                task.results.append(result)
        return task
    
    def request_cancel(self, task_id: str) -> bool:
        """Request cancellation of a task."""
        with self.lock:
            task = self.tasks.get(task_id)
            if task and task.status == BatchTaskStatus.RUNNING:
                task.cancel_requested = True
                return True
        return False
    
    def is_cancelled(self, task_id: str) -> bool:
        """Check if cancellation was requested."""
        with self.lock:
            task = self.tasks.get(task_id)
            return task.cancel_requested if task else False
    
    def list_tasks(self, status: Optional[BatchTaskStatus] = None) -> List[Dict[str, Any]]:
        """List all tasks, optionally filtered by status."""
        with self.lock:
            tasks = list(self.tasks.values())
            if status:
                tasks = [t for t in tasks if t.status == status]
            return [t.to_dict() for t in sorted(tasks, key=lambda t: t.created_at, reverse=True)]
    
    def cleanup_old_tasks(self, max_age_seconds: int = 3600):
        """Remove completed tasks older than max_age_seconds."""
        cutoff = time.time() - max_age_seconds
        with self.lock:
            to_remove = [
                tid for tid, task in self.tasks.items()
                if task.status in [BatchTaskStatus.COMPLETED, BatchTaskStatus.FAILED, BatchTaskStatus.CANCELLED]
                and task.created_at < cutoff
            ]
            for tid in to_remove:
                del self.tasks[tid]


# Global batch task manager instance
batch_task_manager = BatchTaskManager()
