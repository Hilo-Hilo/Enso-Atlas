from __future__ import annotations

import ast
from pathlib import Path
from typing import Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = REPO_ROOT / "src" / "enso_atlas" / "api" / "main.py"


def _read_main() -> str:
    return MAIN_PATH.read_text(encoding="utf-8")


def _load_slide_dims_from_coords_helper():
    source = _read_main()
    tree = ast.parse(source)

    create_app = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "create_app"
    )
    helper = next(
        node
        for node in create_app.body
        if isinstance(node, ast.FunctionDef) and node.name == "_slide_dims_from_coords"
    )

    module = ast.Module(body=[helper], type_ignores=[])
    namespace = {
        "np": np,
        "Optional": Optional,
    }
    exec(compile(module, str(MAIN_PATH), "exec"), namespace)
    return namespace["_slide_dims_from_coords"]


def test_project_model_scope_path_uses_short_ttl_cache():
    src = _read_main()

    assert "_PROJECT_SCOPE_CACHE_TTL_S = 15.0" in src
    assert "_project_model_scope_cache" in src
    assert "async def _resolve_project_model_scope_cached(" in src
    assert "scope = await _resolve_project_model_scope_cached(project_id)" in src


def test_project_slide_scope_memoizes_legacy_column_check_and_labels_loads():
    src = _read_main()

    assert "_legacy_project_column_exists: Optional[bool] = None" in src
    assert "nonlocal _legacy_project_column_exists" in src
    assert "_load_slide_ids_from_labels_file_cached(_project_labels_path(project_id))" in src


def test_heatmap_endpoints_emit_structured_timing_logs():
    src = _read_main()

    assert '"api.models"' in src
    assert '"api.heatmap.slide"' in src
    assert '"api.heatmap.model"' in src


def test_slide_dims_from_coords_preserves_existing_geometry_semantics():
    helper = _load_slide_dims_from_coords_helper()

    coords = np.array([[0, 0], [224, 448]], dtype=np.int64)
    assert helper(coords, patch_size=224) == (448, 672)
    assert helper(None, patch_size=224) == (224, 224)
    assert helper(np.empty((0, 2), dtype=np.int64), patch_size=224) == (224, 224)
