"""Runtime state containers for the Enso Atlas FastAPI backend.

The app factory historically kept most shared resources in ``create_app``
closure variables. This module provides explicit containers for the same
resource categories so route modules can be split without growing another
implicit monolith.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any


@dataclass
class TimedCache:
    """Small TTL cache used for path/model/metadata lookups.

    It intentionally stores plain values and does not perform background
    eviction; callers decide the cache key and TTL that fit their domain.
    """

    ttl_seconds: float
    values: dict[str, dict[str, Any]] = field(default_factory=dict)

    def get(self, key: str) -> dict[str, Any] | None:
        record = self.values.get(key)
        if not record:
            return None
        if time.time() - float(record.get("ts", 0.0)) >= self.ttl_seconds:
            self.values.pop(key, None)
            return None
        return record

    def set(self, key: str, **fields: Any) -> dict[str, Any]:
        record = {"ts": time.time(), **fields}
        self.values[key] = record
        return record

    def clear(self) -> None:
        self.values.clear()


@dataclass
class ApiRuntimeState:
    """Shared mutable resources owned by one FastAPI application instance."""

    embeddings_dir: Path
    model_path: Path
    data_root: Path = Path("data")
    slides_dir: Path = Path("data/slides")

    classifier: Any = None
    evidence_generator: Any = None
    embedder: Any = None
    medsiglip_embedder: Any = None
    reporter: Any = None
    decision_support: Any = None
    multi_model_inference: Any = None
    project_registry: Any = None

    db_available: bool = False
    available_slides: list[str] = field(default_factory=list)
    slide_labels: dict[str, str] = field(default_factory=dict)
    slide_siglip_embeddings: dict[str, Any] = field(default_factory=dict)
    model_checkpoint_signatures: dict[str, str] = field(default_factory=dict)

    slide_mean_index: Any = None
    slide_mean_ids: list[str] = field(default_factory=list)
    slide_mean_meta: dict[str, dict[str, Any]] = field(default_factory=dict)

    project_model_scope_cache: TimedCache = field(default_factory=lambda: TimedCache(15.0))
    labels_slide_ids_cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    slide_dims_cache: TimedCache = field(default_factory=lambda: TimedCache(300.0))
    cpu_heatmap_model_cache: dict[str, Any] = field(
        default_factory=lambda: {"signature": None, "model": None}
    )
    cpu_heatmap_model_lock: Lock = field(default_factory=Lock)

    def reset_request_caches(self) -> None:
        """Clear short-lived lookup caches after config or assignment changes."""

        self.project_model_scope_cache.clear()
        self.labels_slide_ids_cache.clear()
        self.slide_dims_cache.clear()
