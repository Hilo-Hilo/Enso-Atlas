from pathlib import Path


def _main_source() -> str:
    main_py = Path(__file__).resolve().parents[1] / "src" / "enso_atlas" / "api" / "main.py"
    return main_py.read_text()


def _batch_tasks_source() -> str:
    path = Path(__file__).resolve().parents[1] / "src" / "enso_atlas" / "api" / "batch_tasks.py"
    return path.read_text()


def _batch_embed_tasks_source() -> str:
    path = Path(__file__).resolve().parents[1] / "src" / "enso_atlas" / "api" / "batch_embed_tasks.py"
    return path.read_text()


def _batch_reembed_script_source() -> str:
    path = Path(__file__).resolve().parents[1] / "scripts" / "batch_reembed_level0.py"
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


def test_project_scoped_listing_filters_contaminated_slide_ids_by_authoritative_membership():
    src = _main_source()
    assert "include_embedding_fallback=False" in src
    assert "_filter_project_candidate_slide_ids(" in src
    assert "slide scoping guard filtered" in src


def test_resolve_slide_path_supports_unambiguous_uuid_suffix_fallback():
    src = _main_source()
    assert "def _resolve_slide_path_in_dirs(" in src
    assert "pattern = f\"{base}.*{ext}\"" in src
    assert "Ambiguous WSI fallback for slide_id=%s" in src


def test_issue81_embed_slide_endpoint_is_typed_and_project_scoped():
    src = _main_source()

    assert "class EmbedSlideRequest(BaseModel):" in src
    assert "async_mode: bool = Field(default=True, alias=\"async\"" in src
    assert "async def embed_slide_on_demand(request: EmbedSlideRequest" in src
    assert "_embed_embeddings_dir = _resolve_project_embeddings_dir(" in src
    assert "slide_path = resolve_slide_path(slide_id, project_id=request.project_id)" in src
    assert '"project_id": request.project_id' in src


def test_issue81_report_paths_use_embedding_resolver_for_level0_lookup():
    src = _main_source()

    assert "emb_path, searched_dirs = _resolve_embedding_path(" in src
    assert "report_emb_path, searched_dirs = _resolve_embedding_path(" in src
    assert "Level 0 embeddings not found for slide" in src


def test_issue81_batch_embed_defaults_use_project_inventory_and_project_threading():
    src = _main_source()
    task_src = _batch_embed_tasks_source()

    assert "target_slides = await _batch_embed_inventory_slide_ids(request.project_id)" in src
    assert "target_slides = list(available_slides)" not in src
    assert "project_embeddings_dir = _resolve_project_embeddings_dir(task.project_id, require_exists=False)" in src
    assert "slide_path = resolve_slide_path(slide_id, project_id=task.project_id)" in src

    assert "project_id: Optional[str] = None" in task_src
    assert "project_id=project_id" in task_src


def test_issue81_batch_reembed_script_sends_explicit_slide_ids_and_project_scope():
    src = _batch_reembed_script_source()

    assert "--project-id" in src
    assert '"slide_ids": slide_ids' in src
    assert '"project_id": args.project_id' in src
    assert '"slide_ids": slide_ids if args.slide_ids else None' not in src
