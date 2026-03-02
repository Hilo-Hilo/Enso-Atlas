from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
PERF_PATH = REPO_ROOT / "src" / "enso_atlas" / "api" / "perf_observability.py"

_spec = importlib.util.spec_from_file_location("perf_observability", PERF_PATH)
assert _spec and _spec.loader
_perf = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _perf
_spec.loader.exec_module(_perf)  # type: ignore[attr-defined]


InMemoryLatencyTracker = _perf.InMemoryLatencyTracker
RequestLatencyRecord = _perf.RequestLatencyRecord
normalize_perf_path = _perf.normalize_perf_path
percentile = _perf.percentile
should_track_path = _perf.should_track_path


def test_percentile_linear_interpolation_matches_expected_values():
    vals = [10, 20, 30, 40, 50]
    assert percentile(vals, 50) == 30
    assert percentile(vals, 95) == 48
    assert percentile(vals, 0) == 10
    assert percentile(vals, 100) == 50


def test_latency_tracker_summary_contains_overall_and_route_stats():
    tracker = InMemoryLatencyTracker(max_samples=10)

    for ms in [10, 20, 30]:
        tracker.add(
            RequestLatencyRecord(
                method="GET",
                path="/api/slides",
                status=200,
                duration_ms=ms,
                request_id=f"req-{ms}",
                ts_unix=1.0,
            )
        )

    for ms in [100, 120]:
        tracker.add(
            RequestLatencyRecord(
                method="POST",
                path="/api/analyze",
                status=200,
                duration_ms=ms,
                request_id=f"req-{ms}",
                ts_unix=2.0,
            )
        )

    summary = tracker.summary(limit_routes=5)

    assert summary["window_samples"] == 5
    assert summary["p50_ms"] == 30.0
    assert summary["p95_ms"] == 116.0
    assert len(summary["routes"]) >= 2

    slowest = summary["routes"][0]
    assert slowest["route"] == "POST /api/analyze"
    assert slowest["p95_ms"] >= 100.0


def test_path_normalization_and_exclusions_for_perf_tracking():
    assert normalize_perf_path("/api/slides/12345") == "/api/slides/:id"
    assert normalize_perf_path("/api/slides/abcdefff-aaaa-bbbb-cccc-ddddeeeeffff") == "/api/slides/:id"

    assert should_track_path("/api/slides/12345")
    assert not should_track_path("/api/health")
    assert not should_track_path("/api/perf/latency-summary")


def test_main_api_wires_perf_middleware_and_summary_endpoint():
    main_src = (REPO_ROOT / "src" / "enso_atlas" / "api" / "main.py").read_text(encoding="utf-8")
    assert "@app.middleware(\"http\")" in main_src
    assert "request_timing_middleware" in main_src
    assert "/api/perf/latency-summary" in main_src
    assert "X-Request-ID" in main_src
