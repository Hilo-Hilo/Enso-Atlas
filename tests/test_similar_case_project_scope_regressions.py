from pathlib import Path


def _api_source(filename: str) -> str:
    path = Path(__file__).resolve().parents[1] / "src" / "enso_atlas" / "api" / filename
    return path.read_text()


def test_project_slide_scope_prefers_db_assignments_before_filesystem_fallback():
    src = _api_source("main.py")
    assert "assigned = [sid for sid in (await db.get_project_slides(project_id)) if sid]" in src
    assert 'source="project_slides"' in src
    assert "SELECT slide_id FROM slides" not in src
    assert "WHERE project_id = $1" not in src


def test_similar_case_endpoints_use_project_slide_scope_resolution():
    similar_src = _api_source("similar_routes.py")
    report_src = _api_source("report_routes.py")
    assert "await project_slide_ids(" in similar_src
    assert "await project_slide_ids(request.project_id)" in report_src
    assert "asyncio.run(project_slide_ids(project_id))" in report_src


def test_report_similarity_paths_extract_slide_id_from_metadata_payloads():
    main_src = _api_source("main.py")
    report_src = _api_source("report_routes.py")
    assert (
        "def _similar_case_slide_id(candidate: Any) -> str | None:" in main_src
        or "def _similar_case_slide_id(candidate: Any) -> Optional[str]:" in main_src
    )
    assert 'meta = candidate.get("metadata")' in main_src
    assert "sid = similar_case_slide_id(similar)" in report_src
