from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_issue94_semantic_panel_supports_patch_deselect_callback():
    src = _read("frontend/src/components/panels/SemanticSearchPanel.tsx")

    assert "onPatchDeselect?: () => void;" in src
    assert "if (isSelected) {" in src
    assert "onPatchDeselect?.();" in src
    assert "Click again to deselect" in src


def test_issue94_main_page_exposes_handle_patch_deselect():
    src = _read("frontend/src/app/page.tsx")

    assert "const handlePatchDeselect = useCallback(() => {" in src
    assert "setSelectedPatchId(undefined);" in src
    assert "setTargetCoordinates(null);" in src


def test_issue94_semantic_panel_wiring_uses_deselect_callback():
    src = _read("frontend/src/app/page.tsx")

    assert "onPatchDeselect={handlePatchDeselect}" in src
    assert "handlePatchDeselect();" in src
