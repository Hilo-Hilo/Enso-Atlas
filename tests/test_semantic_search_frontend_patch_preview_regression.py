from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_semantic_panel_uses_result_coordinates_for_preview_and_click():
    src = _read("frontend/src/components/panels/SemanticSearchPanel.tsx")

    assert "projectId?: string;" in src
    assert "coordinates: result.coordinates" in src
    assert "patchSize: result.patch_size" in src
    assert "if (!result.coordinates) return;" in src
    assert "x: result.coordinates[0]" in src
    assert "y: result.coordinates[1]" in src
    assert "result.coordinates?.[0] ?? 0" not in src


def test_patch_proxy_forwards_query_params_to_backend():
    src = _read("frontend/src/app/api/slides/[slideId]/patches/[patchId]/route.ts")

    assert "const requestUrl = new URL(request.url);" in src
    assert "requestUrl.searchParams.forEach" in src
    assert "backendUrl.searchParams.set(key, value);" in src


def test_get_patch_url_supports_project_and_explicit_coordinates():
    src = _read("frontend/src/lib/api.ts")

    assert "projectId?: string;" in src
    assert "coordinates?: [number, number];" in src
    assert 'params.set("project_id", options.projectId);' in src
    assert 'params.set("x", String(options.coordinates[0]));' in src
    assert 'params.set("patch_size", String(options.patchSize));' in src
