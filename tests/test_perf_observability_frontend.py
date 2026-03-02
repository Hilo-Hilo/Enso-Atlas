from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(rel_path: str) -> str:
    return (REPO_ROOT / rel_path).read_text(encoding="utf-8")


def test_root_layout_mounts_perf_observer():
    src = _read("frontend/src/app/layout.tsx")
    assert "PerfObserver" in src
    assert "<PerfObserver />" in src


def test_home_page_logs_panel_switch_timings():
    src = _read("frontend/src/app/page.tsx")
    assert "usePanelSwitchPerf" in src
    assert 'usePanelSwitchPerf("right-sidebar", activeRightPanel)' in src
    assert 'usePanelSwitchPerf("mobile-panel", mobilePanelTab)' in src


def test_perf_logger_supports_toggleable_client_logging():
    src = _read("frontend/src/lib/perfLogger.ts")
    assert "NEXT_PUBLIC_ENABLE_PERF_LOGS" in src
    assert "route_render_ms" not in src  # route metric names belong in hook/component callers
    assert "[enso-perf]" in src


def test_perf_hooks_emit_route_and_panel_metrics():
    src = _read("frontend/src/hooks/usePerfInstrumentation.ts")
    assert '"route_render_ms"' in src
    assert '"panel_render_ms"' in src
