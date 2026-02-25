from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_issue86_removes_legacy_right_panel_controls_and_labeling():
    src = _read("frontend/src/app/page.tsx")

    assert "Right panel" not in src
    assert "rightPanelRef" not in src
    assert "rightSidebarOpen" not in src
    assert "Collapse right sidebar" not in src
    assert "Expand right sidebar" not in src
    assert src.count("<PanelResizeHandle") == 1


def test_issue86_workspace_tabs_keep_core_tools_accessible_in_main_workspace():
    src = _read("frontend/src/app/page.tsx")

    assert "WorkspacePanelTabs" in src
    assert 'value: "medgemma", label: "MedGemma"' in src
    assert 'value: "prediction"' in src
    assert 'value: "multi-model"' in src
    assert 'value: "semantic-search", label: "Semantic Search"' in src
    assert 'value: "similar-cases", label: "Similar Cases"' in src
    assert 'value: "patch-classifier", label: "Patch Classifier"' in src


def test_issue86_workspace_panel_toggle_uses_inline_header_controls():
    src = _read("frontend/src/app/page.tsx")

    assert "const [workspacePanelCollapsed, setWorkspacePanelCollapsed] = useState(false);" in src
    assert "setWorkspacePanelCollapsed(false);" in src
    assert '{workspacePanelCollapsed ? "Show tools" : "Hide tools"}' in src
    assert "activeWorkspacePanelLabel" in src
    assert "Hide case panel" in src
    assert "Show case panel" in src
