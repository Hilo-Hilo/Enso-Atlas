from pathlib import Path


def _main_source() -> str:
    main_py = Path(__file__).resolve().parents[1] / "src" / "enso_atlas" / "api" / "main.py"
    return main_py.read_text()


def test_project_slide_scope_prefers_db_assignments_before_filesystem_fallback():
    src = _main_source()
    assert "assigned = [sid for sid in (await db.get_project_slides(project_id)) if sid]" in src
    assert "SELECT slide_id FROM slides" in src
    assert "WHERE project_id = $1" in src


def test_similar_case_endpoints_use_project_slide_scope_resolution():
    src = _main_source()
    assert src.count("await _project_slide_ids(") >= 3
    assert "asyncio.run(_project_slide_ids(project_id))" in src


def test_report_similarity_paths_extract_slide_id_from_metadata_payloads():
    src = _main_source()
    assert "def _similar_case_slide_id(candidate: Any) -> Optional[str]:" in src
    assert "meta = candidate.get(\"metadata\")" in src
    assert "sid = _similar_case_slide_id(s)" in src
