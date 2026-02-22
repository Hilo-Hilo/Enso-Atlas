from pathlib import Path


def _main_source() -> str:
    main_py = Path(__file__).resolve().parents[1] / "src" / "enso_atlas" / "api" / "main.py"
    return main_py.read_text()


def _batch_tasks_source() -> str:
    path = Path(__file__).resolve().parents[1] / "src" / "enso_atlas" / "api" / "batch_tasks.py"
    return path.read_text()


def _report_tasks_source() -> str:
    report_tasks_py = Path(__file__).resolve().parents[1] / "src" / "enso_atlas" / "api" / "report_tasks.py"
    return report_tasks_py.read_text()


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


def test_flat_file_listing_uses_project_labels_without_global_fallback():
    src = _main_source()
    assert "if proj_cfg:\n            labels_path = _project_labels_path(project_id)\n        else:\n            labels_path = _data_root / \"labels.csv\"" in src


def test_patient_context_global_label_fallback_is_non_project_only():
    src = _main_source()
    assert "labels_path = _project_labels_path(project_id)" in src
    assert "if project_id:\n            if labels_path is None:\n                return None\n        else:\n            if labels_path is None:\n                labels_path = _data_root / \"labels.csv\"" in src


def test_async_report_uses_classifier_threshold_consistently_with_sync_path():
    src = _main_source()
    assert "threshold_val = _classifier_threshold()" in src
    assert "threshold = _classifier_threshold()" in src
    assert "getattr(classifier.config, \"threshold\", 0.5)" not in src


def test_async_batch_task_summary_uses_task_label_pair_not_hardcoded_response_labels():
    src = _batch_tasks_source()
    assert 'responders = [r for r in completed if r.prediction == self.positive_label]' in src
    assert 'non_responders = [r for r in completed if r.prediction == self.negative_label]' in src
    assert 'r.prediction == "RESPONDER"' not in src
    assert 'r.prediction == "NON-RESPONDER"' not in src


def test_async_report_preflight_is_project_scoped():
    main_src = _main_source()
    task_src = _report_tasks_source()

    assert "existing_task = report_task_manager.get_task_by_slide(slide_id, request.project_id)" in main_src
    assert "task = report_task_manager.create_task(slide_id, request.project_id)" in main_src

    assert "project_id: Optional[str] = None" in task_src
    assert "def get_task_by_slide(self, slide_id: str, project_id: Optional[str] = None)" in task_src
    assert "task.project_id == project_id" in task_src


def test_project_scoped_slides_fail_closed_when_embeddings_are_missing_or_empty():
    src = _main_source()
    assert "returning 0 slides (no global fallback)" in src
    assert "list_slides._cache[_cache_key] = {\"data\": [], \"ts\": time.time()}" in src
    assert "_candidate_emb_dirs = [_fallback_embeddings_dir]" in src


def test_project_scoped_listing_filters_contaminated_slide_ids_by_project_bases():
    src = _main_source()
    assert "_allowed_base_ids: set[str] = {sid.split(\".\", 1)[0] for sid in slide_data.keys()}" in src
    assert "sid.split(\".\", 1)[0] in _allowed_base_ids" in src
    assert "Filtered %d contaminated slide IDs from project '%s' listing" in src


def test_resolve_slide_path_supports_unambiguous_uuid_suffix_fallback():
    src = _main_source()
    assert "slide_base = slide_id.split(\".\", 1)[0]" in src
    assert "matches.extend(sorted(d.glob(f\"{slide_base}.*{ext}\")))" in src
    assert "Ambiguous WSI fallback match" in src
