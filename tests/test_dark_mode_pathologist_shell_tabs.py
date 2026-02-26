"""
Regression tests for pathologist workspace shell dark-mode polish.

Scope:
- frontend/src/app/page.tsx only (shell/tab surfaces around Pathologist mode)
- Ensure dark-mode tab states, WSI header strip, and pathologist fallback card
  are all dark-coherent while preserving existing light-mode classes.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PAGE_TSX = REPO_ROOT / "frontend" / "src" / "app" / "page.tsx"


def _read_page() -> str:
    return PAGE_TSX.read_text(encoding="utf-8")


class TestPathologistRightSidebarTabsDarkMode:
    def test_tablist_shell_has_dark_surface(self):
        src = _read_page()
        assert "bg-white dark:bg-navy-800" in src, (
            "Right tablist shell must keep light bg-white and add dark:bg-navy-800"
        )
        assert "border-gray-200 dark:border-navy-700" in src, (
            "Right tablist shell must keep light border and add dark border"
        )

    def test_active_tab_has_dark_text_and_bg(self):
        src = _read_page()
        assert "dark:text-clinical-300" in src, (
            "Active right tab must have dark:text-clinical-300"
        )
        assert "dark:bg-clinical-900/40" in src, (
            "Active right tab must have dark:bg-clinical-900/40"
        )

    def test_inactive_tab_has_dark_idle_and_hover_states(self):
        src = _read_page()
        assert "dark:text-gray-500" in src, (
            "Inactive right tab must have dark:text-gray-500"
        )
        assert "dark:hover:text-gray-200" in src, (
            "Inactive right tab hover text must have dark override"
        )
        assert "dark:hover:bg-navy-700/80" in src, (
            "Inactive right tab hover bg must have dark override"
        )


class TestPathologistShellDarkMode:
    def test_pathologist_wsi_header_strip_dark_classes(self):
        src = _read_page()
        assert "bg-violet-50 dark:bg-violet-950/40" in src, (
            "Pathologist WSI header strip must have dark background"
        )
        assert "border-violet-200 dark:border-violet-800/70" in src, (
            "Pathologist WSI header strip must have dark border"
        )
        assert "text-violet-700 dark:text-violet-200" in src, (
            "Pathologist WSI header title must have dark text"
        )
        assert "text-violet-500 dark:text-violet-300" in src, (
            "Pathologist WSI header subtitle must have dark text"
        )

    def test_pathologist_fallback_card_dark_classes(self):
        src = _read_page()
        # Scope near fallback copy to avoid false positives.
        marker = "Select a slide to open Pathologist Workspace."
        idx = src.find(marker)
        assert idx >= 0, "Pathologist fallback message text must exist"
        nearby = src[max(0, idx - 220): idx + 80]

        assert "dark:border-violet-800/70" in nearby, (
            "Pathologist fallback card must have dark border"
        )
        assert "dark:bg-violet-950/40" in nearby, (
            "Pathologist fallback card must have dark background"
        )
        assert "dark:text-violet-200" in nearby, (
            "Pathologist fallback card must have dark text"
        )


class TestPathologistShellLightModeUnchanged:
    def test_active_tab_keeps_light_mode_classes(self):
        src = _read_page()
        assert "text-clinical-700" in src, "Active tab light text must remain"
        assert "bg-clinical-50/40" in src, "Active tab light bg must remain"

    def test_pathologist_header_keeps_light_mode_classes(self):
        src = _read_page()
        assert "bg-violet-50" in src, "Pathologist header light bg must remain"
        assert "border-violet-200" in src, "Pathologist header light border must remain"
        assert "text-violet-700" in src, "Pathologist header title light text must remain"
        assert "text-violet-500" in src, "Pathologist header subtitle light text must remain"

    def test_pathologist_fallback_keeps_light_mode_classes(self):
        src = _read_page()
        marker = "Select a slide to open Pathologist Workspace."
        idx = src.find(marker)
        assert idx >= 0, "Pathologist fallback message text must exist"
        nearby = src[max(0, idx - 220): idx + 80]

        assert "border-violet-200" in nearby, "Fallback light border must remain"
        assert "bg-violet-50" in nearby, "Fallback light bg must remain"
        assert "text-violet-800" in nearby, "Fallback light text must remain"
