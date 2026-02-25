from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_issue93_wsi_viewer_has_overlay_menu_collapse_state():
    src = _read("frontend/src/components/viewer/WSIViewer.tsx")

    assert "const [showOverlayMenu, setShowOverlayMenu] = useState(true);" in src
    assert 'aria-expanded={showOverlayMenu}' in src
    assert 'aria-controls="wsi-overlay-controls-menu"' in src
    assert 'id="wsi-overlay-controls-menu"' in src


def test_issue93_overlay_menu_button_has_expand_collapse_copy():
    src = _read("frontend/src/components/viewer/WSIViewer.tsx")

    assert '"Collapse overlay controls"' in src
    assert '"Expand overlay controls"' in src
    assert '{showOverlayMenu ? "Hide controls" : "Show controls"}' in src
