from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_model_heatmap_proxy_forwards_project_id_to_backend():
    src = _read("frontend/src/app/api/heatmap/[slideId]/[modelId]/route.ts")

    assert 'searchParams.get("project_id")' in src
    assert 'backendParams.set("project_id", projectId)' in src


def test_slide_heatmap_proxy_forwards_project_and_alpha_params_to_backend():
    src = _read("frontend/src/app/api/heatmap/[slideId]/route.ts")

    assert 'searchParams.get("project_id")' in src
    assert 'backendParams.set("project_id", projectId)' in src
    assert 'searchParams.get("alpha_power")' in src
    assert 'backendParams.set("alpha_power", alphaPower)' in src


def test_api_client_uses_project_scoped_models_endpoint_and_heatmap_query_param():
    src = _read("frontend/src/lib/api.ts")

    assert "/api/models" in src
    assert "project_id" in src
    assert "getProjectAvailableModels" in src
    assert "getHeatmapUrl" in src
    # lightweight UI uses normalizeProjectId() -> scopedProjectId
    assert "scopedProjectId" in src
    assert (
        "params.set('project_id', scopedProjectId)" in src
        or 'params.set("project_id", scopedProjectId)' in src
    )


def test_api_client_heatmap_url_includes_project_scoping():
    """Verify getHeatmapUrl passes project_id to backend via query params."""
    src = _read("frontend/src/lib/api.ts")

    assert "getHeatmapUrl" in src
    assert "project_id" in src
    # heatmap URL builder must scope by project
    assert "scopedProjectId" in src


def test_model_heatmap_proxy_preserves_backend_error_payload_for_ui_messages():
    src = _read("frontend/src/app/api/heatmap/[slideId]/[modelId]/route.ts")

    assert "const errorBody = await response.text();" in src
    assert "return new NextResponse(" in src
    assert '"Content-Type": response.headers.get("Content-Type") || "application/json"' in src


def test_model_heatmap_proxy_sets_cache_control_on_errors():
    """Error responses must include no-store to prevent stale error caching."""
    src = _read("frontend/src/app/api/heatmap/[slideId]/[modelId]/route.ts")

    assert '"Cache-Control": "no-store"' in src or "'Cache-Control': 'no-store'" in src


def test_model_heatmap_proxy_forwards_coverage_headers():
    """Proxy must expose heatmap alignment headers to the frontend."""
    src = _read("frontend/src/app/api/heatmap/[slideId]/[modelId]/route.ts")

    assert "X-Coverage-Width" in src
    assert "X-Coverage-Height" in src
    assert "Access-Control-Expose-Headers" in src


def test_slide_heatmap_proxy_preserves_backend_error_payload_for_ui_messages():
    src = _read("frontend/src/app/api/heatmap/[slideId]/route.ts")

    assert "const errorBody = await response.text();" in src
    assert "return new NextResponse(" in src
    assert '"Content-Type": response.headers.get("Content-Type") || "application/json"' in src


def test_page_heatmap_model_options_are_guarded_by_current_project_scope():
    src = _read("frontend/src/app/page.tsx")

    assert "projectAvailableModelsScopeId" in src
    assert "projectAvailableModelsScopeId === currentProject.id" in src
    assert "normalizedHeatmapModel" in src
    assert "!scopedProjectModelIds.has(heatmapModel)" in src


def test_page_uses_project_context_for_analysis():
    """Main page must read current project from ProjectContext for analysis calls."""
    src = _read("frontend/src/app/page.tsx")

    assert "currentProject" in src
    assert "useProject" in src
