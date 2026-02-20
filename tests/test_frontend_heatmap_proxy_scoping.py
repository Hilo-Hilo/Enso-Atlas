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
    assert "params.set('project_id', projectId)" in src or 'params.set("project_id", projectId)' in src


def test_page_heatmap_model_options_are_guarded_by_current_project_scope():
    src = _read("frontend/src/app/page.tsx")

    assert "projectAvailableModelsScopeId" in src
    assert "projectAvailableModelsScopeId === currentProject.id" in src
    assert "normalizedHeatmapModel" in src
    assert "!scopedProjectModelIds.has(heatmapModel)" in src
