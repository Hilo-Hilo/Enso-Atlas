from pathlib import Path


def _main_source() -> str:
    main_py = Path(__file__).resolve().parents[1] / "src" / "enso_atlas" / "api" / "main.py"
    return main_py.read_text()


def test_models_and_multi_analysis_share_project_model_resolution():
    src = _main_source()
    assert "allowed_ids = await _resolve_project_model_ids(project_id)" in src
    assert "allowed_model_ids = await _resolve_project_model_ids(request.project_id)" in src
    assert "scope = await resolve_project_model_scope(" in src


def test_async_batch_propagates_project_id_to_background_worker():
    src = _main_source()
    assert "project_id: Optional[str] = Field(" in src
    assert "project_id=request.project_id" in src
    assert "def _run_batch_analysis_background(" in src
    assert "project_id: Optional[str] = None," in src


def test_heatmap_and_dzi_paths_use_project_scoped_slide_resolution():
    src = _main_source()
    assert "slide_path = resolve_slide_path(slide_id, project_id=project_id)" in src
    assert "_require_project(project_id)\n        result = get_slide_and_dz(slide_id, project_id=project_id)" in src


def test_project_scoped_routes_fail_fast_on_unknown_project():
    src = _main_source()
    assert "proj_cfg = _require_project(project_id)" in src
    assert "raise HTTPException(status_code=404, detail=f\"Project '{project_id}' not found\")" in src


def test_report_paths_resolve_project_specific_embeddings_before_processing():
    src = _main_source()
    assert "report_embeddings_dir = _resolve_project_embeddings_dir(" in src
    assert "_report_embeddings_dir = _resolve_project_embeddings_dir(" in src
