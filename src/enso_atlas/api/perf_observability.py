"""Lightweight request/route observability primitives for dev perf baselines."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
from typing import Any, Iterable


@dataclass(frozen=True)
class RequestLatencyRecord:
    """Single request timing sample."""

    method: str
    path: str
    status: int
    duration_ms: float
    request_id: str
    ts_unix: float


def percentile(values: Iterable[float], p: float) -> float:
    """Compute percentile using linear interpolation (numpy-like)."""
    arr = sorted(float(v) for v in values)
    if not arr:
        return 0.0
    if len(arr) == 1:
        return float(arr[0])

    p = max(0.0, min(100.0, float(p)))
    rank = (len(arr) - 1) * (p / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(arr) - 1)
    if lo == hi:
        return float(arr[lo])

    frac = rank - lo
    return float(arr[lo] + (arr[hi] - arr[lo]) * frac)


class InMemoryLatencyTracker:
    """Thread-safe rolling latency tracker.

    Keeps an in-memory ring buffer of recent request timings and computes
    p50/p95 summaries overall and per route signature ("METHOD path").
    """

    def __init__(self, max_samples: int = 2000):
        self.max_samples = max(100, int(max_samples))
        self._samples: deque[RequestLatencyRecord] = deque(maxlen=self.max_samples)
        self._lock = threading.Lock()

    def add(self, sample: RequestLatencyRecord) -> None:
        with self._lock:
            self._samples.append(sample)

    def size(self) -> int:
        with self._lock:
            return len(self._samples)

    def summary(self, limit_routes: int = 25) -> dict[str, Any]:
        with self._lock:
            items = list(self._samples)

        durations = [s.duration_ms for s in items]
        summary = {
            "window_samples": len(items),
            "p50_ms": round(percentile(durations, 50), 3),
            "p95_ms": round(percentile(durations, 95), 3),
            "max_ms": round(max(durations), 3) if durations else 0.0,
            "routes": [],
        }

        by_route: dict[str, list[float]] = {}
        statuses: dict[str, dict[int, int]] = {}

        for item in items:
            route_key = f"{item.method} {item.path}"
            by_route.setdefault(route_key, []).append(item.duration_ms)
            route_statuses = statuses.setdefault(route_key, {})
            route_statuses[item.status] = route_statuses.get(item.status, 0) + 1

        ranked = sorted(
            by_route.items(),
            key=lambda kv: (percentile(kv[1], 95), len(kv[1])),
            reverse=True,
        )

        for route_key, vals in ranked[: max(1, int(limit_routes))]:
            summary["routes"].append(
                {
                    "route": route_key,
                    "count": len(vals),
                    "p50_ms": round(percentile(vals, 50), 3),
                    "p95_ms": round(percentile(vals, 95), 3),
                    "max_ms": round(max(vals), 3),
                    "status_counts": statuses.get(route_key, {}),
                }
            )

        return summary


def normalize_perf_path(path: str) -> str:
    """Normalize path to reduce high-cardinality in perf summaries."""
    path = (path or "/").strip() or "/"
    if "?" in path:
        path = path.split("?", 1)[0]

    parts = []
    for part in path.split("/"):
        if not part:
            continue
        if len(part) >= 8 and all(c in "0123456789abcdefABCDEF-" for c in part):
            parts.append(":id")
            continue
        if part.isdigit():
            parts.append(":id")
            continue
        parts.append(part)

    return "/" + "/".join(parts)


def should_track_path(path: str) -> bool:
    """Exclude noisy/non-business routes from perf tracking."""
    normalized = normalize_perf_path(path)
    excluded = {
        "/api/perf/latency-summary",
        "/health",
        "/api/health",
        "/api/docs",
        "/api/redoc",
        "/docs",
        "/redoc",
    }
    return normalized not in excluded
