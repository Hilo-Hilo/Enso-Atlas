from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(rel_path: str) -> str:
    return (REPO_ROOT / rel_path).read_text()


def test_project_upload_route_has_filename_path_and_size_guardrails():
    src = _read("src/enso_atlas/api/project_routes.py")

    assert "SAFE_UPLOAD_FILENAME_RE" in src
    assert "def _safe_upload_filename" in src
    assert "MAX_UPLOAD_SIZE_BYTES" in src
    assert "if file_size > MAX_UPLOAD_SIZE_BYTES" in src
    assert "status_code=413" in src
    assert "os.replace(tmp_path, dest_path)" in src


def test_project_slides_use_explicit_assignments_only():
    src = _read("src/enso_atlas/api/project_routes.py")

    assert "slide_ids = await db.get_project_slides(project_id)" in src
    assert "WHERE s.slide_id = ANY($1::text[])" in src
    assert "WHERE s.project_id = $1" not in src
    assert "WHERE s.project_id = $1 OR s.project_id IS NULL" not in src


def test_project_model_assignment_validates_project_compatible_model_ids():
    src = _read("src/enso_atlas/api/project_routes.py")

    assert "def _validate_project_model_ids" in src
    assert "validated_model_ids = _validate_project_model_ids(" in src


def test_visual_search_is_project_scoped_and_filters_candidates():
    schema_src = _read("src/enso_atlas/api/schemas.py")
    src = _read("src/enso_atlas/api/visual_search_routes.py")

    assert "project_id: str | None = Field(" in schema_src or "project_id: Optional[str] = Field(" in schema_src
    assert 'description="Project ID to scope visual search candidates"' in schema_src
    assert "allowed_slide_ids = await project_slide_ids(request.project_id)" in src
    assert "if allowed_slide_ids is not None and result_slide_id not in allowed_slide_ids:" in src
    assert '"project_id": request.project_id' in src


def test_cached_result_and_embedding_status_endpoints_accept_project_scope():
    src = _read("src/enso_atlas/api/slide_status_routes.py")

    assert '@router.get("/api/slides/{slide_id}/embedding-status")' in src
    assert "project_id: str | None = Query(" in src or "project_id: Optional[str] = Query(" in src
    assert 'description="Optional project scope for model cache visibility"' in src
    assert "allowed_model_ids = await resolve_project_model_ids(project_id)" in src
    assert '@router.get("/api/slides/{slide_id}/cached-results")' in src
    assert "detail=f\"Slide {slide_id} is not available in project '{project_id}'\"" in src


def test_frontend_scopes_slide_search_visual_search_and_embedding_status_calls():
    slides_page = _read("frontend/src/app/slides/page.tsx")
    app_page = _read("frontend/src/app/page.tsx")
    api_ts = _read("frontend/src/lib/api.ts")
    model_picker = _read("frontend/src/components/panels/ModelPicker.tsx")

    assert "searchSlides(filters, currentProject.id)" in slides_page
    assert "projectId: currentProject.id" in app_page
    assert "if (scopedProjectId) body.project_id = scopedProjectId;" in api_ts
    assert "getSlideEmbeddingStatus(selectedSlideId, currentProject.id)" in model_picker


def test_frontend_upload_modal_has_client_side_size_guardrail():
    src = _read("frontend/src/app/projects/page.tsx")

    assert "MAX_UPLOAD_BYTES = 10 * 1024 * 1024 * 1024" in src
    assert "exceed 10 GiB upload limit" in src
    assert "Max 10 GiB each" in src
