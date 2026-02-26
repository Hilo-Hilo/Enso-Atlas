"""Regression tests for disabled project creation (issue: disable-project-create).

Project creation via the API and UI is disabled. Projects must be configured
directly in config/projects.yaml on the backend.

These tests verify:
1. POST /api/projects returns 403 Forbidden with clear message
2. Frontend projects page does not have "New Project" or "Create First Project" buttons
3. Frontend does not render the create project modal
4. Edit and delete functionality remains intact
"""

from pathlib import Path


def _project_routes_source() -> str:
    """Read the project_routes.py source file."""
    path = Path(__file__).resolve().parents[1] / "src" / "enso_atlas" / "api" / "project_routes.py"
    return path.read_text()


def _projects_page_source() -> str:
    """Read the frontend projects page source file."""
    path = Path(__file__).resolve().parents[1] / "frontend" / "src" / "app" / "projects" / "page.tsx"
    return path.read_text()


# ---------------------------------------------------------------------------
# Backend API Tests
# ---------------------------------------------------------------------------


def test_post_projects_endpoint_returns_403_forbidden():
    """POST /api/projects must return 403 Forbidden."""
    src = _project_routes_source()
    # The endpoint should raise HTTPException with status_code=403
    assert "status_code=403" in src
    assert "Project creation via API is disabled" in src


def test_post_projects_endpoint_mentions_backend_config():
    """The 403 response must mention config/projects.yaml."""
    src = _project_routes_source()
    assert "config/projects.yaml" in src


def test_create_project_endpoint_docstring_indicates_disabled():
    """The create_project endpoint docstring should indicate it's disabled."""
    src = _project_routes_source()
    assert "**DISABLED**" in src or "DISABLED" in src


# ---------------------------------------------------------------------------
# Frontend UI Tests
# ---------------------------------------------------------------------------


def test_frontend_no_new_project_button():
    """Frontend must not have a 'New Project' button."""
    src = _projects_page_source()
    # The "New Project" button should be removed/commented out
    assert 'onClick={() => setShowCreate(true)}' not in src
    assert '>New Project<' not in src
    # Should have a comment indicating it's disabled
    assert "Project creation disabled" in src or "projects must be configured" in src.lower()


def test_frontend_no_create_first_project_button():
    """Frontend empty state must not have 'Create First Project' button."""
    src = _projects_page_source()
    assert "Create First Project" not in src


def test_frontend_no_create_project_modal():
    """Frontend must not render the create project modal."""
    src = _projects_page_source()
    # The showCreate modal rendering should be removed
    assert "{showCreate && (" not in src
    assert 'mode="create"' not in src or 'ProjectFormModal' not in src.split('mode="create"')[0][-200:]


def test_frontend_empty_state_mentions_backend_config():
    """Frontend empty state should mention config/projects.yaml."""
    src = _projects_page_source()
    assert "config/projects.yaml" in src


def test_frontend_no_create_project_api_helper():
    """Frontend should not have the createProject API helper function."""
    src = _projects_page_source()
    # The createProject function should be removed
    assert "async function createProject(" not in src


def test_frontend_no_handle_create_handler():
    """Frontend should not have the handleCreate handler."""
    src = _projects_page_source()
    # The handleCreate function should be removed
    assert "const handleCreate = async" not in src


def test_frontend_no_show_create_state():
    """Frontend should not have showCreate state."""
    src = _projects_page_source()
    # The showCreate state should be removed
    assert "const [showCreate, setShowCreate]" not in src


# ---------------------------------------------------------------------------
# Preserved Functionality Tests
# ---------------------------------------------------------------------------


def test_frontend_edit_project_still_works():
    """Edit project functionality must be preserved."""
    src = _projects_page_source()
    # Edit modal should still exist
    assert "editProject && (" in src or "{editProject &&" in src
    assert "setEditProject" in src
    assert 'mode="edit"' in src


def test_frontend_delete_project_still_works():
    """Delete project functionality must be preserved."""
    src = _projects_page_source()
    # Delete modal should still exist
    assert "deleteTarget && (" in src or "{deleteTarget &&" in src
    assert "setDeleteTarget" in src
    assert "DeleteConfirmModal" in src


def test_backend_delete_endpoint_still_works():
    """DELETE /api/projects/{project_id} must still work."""
    src = _project_routes_source()
    assert '@router.delete("/{project_id}")' in src
    assert "async def delete_project" in src


def test_backend_update_endpoint_still_works():
    """PUT /api/projects/{project_id} must still work."""
    src = _project_routes_source()
    assert '@router.put("/{project_id}")' in src
    assert "async def update_project" in src


def test_backend_get_endpoints_still_work():
    """GET endpoints must still work."""
    src = _project_routes_source()
    assert '@router.get("")' in src
    assert "async def list_projects" in src
    assert '@router.get("/{project_id}")' in src
    assert "async def get_project" in src
