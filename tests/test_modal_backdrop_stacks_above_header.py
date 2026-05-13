from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_header_z_index_reference_is_explicit():
    src = _read("frontend/src/components/layout/Header.tsx")
    assert "z-[120]" in src, "Header should keep explicit z-index reference for stacking order"


def test_core_modals_stack_above_header_with_high_z_index():
    modal_files = [
        "frontend/src/components/modals/SettingsModal.tsx",
        "frontend/src/components/modals/SystemStatusModal.tsx",
        "frontend/src/components/modals/KeyboardShortcutsModal.tsx",
        "frontend/src/components/modals/PatchZoomModal.tsx",
        "frontend/src/components/modals/CompareSelectModal.tsx",
        "frontend/src/components/slides/SlideModals.tsx",
    ]

    for file_path in modal_files:
        src = _read(file_path)
        assert "fixed inset-0 z-[300] flex items-center justify-center" in src, (
            f"{file_path} should use z-[300] so modal backdrop covers header"
        )


def test_projects_page_popups_stack_above_header():
    src = _read("frontend/src/app/projects/page.tsx")
    count = src.count(
        "fixed inset-0 z-[300] flex items-center justify-center bg-black/50 backdrop-blur-sm"
    )
    assert count >= 3, "Projects popups should all use z-[300] backdrop wrappers"
