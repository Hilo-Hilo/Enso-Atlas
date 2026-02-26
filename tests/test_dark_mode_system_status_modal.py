"""
Regression tests for dark-mode styling of SystemStatusModal.

Issue: System status card/modal was fully light-themed even in dark mode.
Fix: Added dark: Tailwind variant classes to modal shell, header, footer,
     stat tiles, service rows, status colors, text, buttons, and borders.

These tests verify:
1. SystemStatusModal has dark: variant classes throughout
2. Modal shell / header / footer use navy dark backgrounds and borders
3. Status color helper returns dark-mode-aware classes for online/offline/degraded
4. Stat tiles (Uptime, Version) have dark backgrounds and text
5. Service section heading and timestamp have dark text variants
6. Close button has dark hover/text styles
7. Status icon classes remain unchanged (both themes share the same icon colors)
8. Light-mode classes are preserved (no regressions)
"""

from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_SRC = REPO_ROOT / "frontend" / "src"


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _modal_src() -> str:
    return _read("frontend/src/components/modals/SystemStatusModal.tsx")


# ---------------------------------------------------------------------------
# 1. Basic dark-mode presence
# ---------------------------------------------------------------------------

class TestSystemStatusModalDarkModePresence:
    """SystemStatusModal must contain dark: variant classes."""

    def test_has_dark_variants(self):
        src = _modal_src()
        count = src.count("dark:")
        assert count >= 15, (
            f"SystemStatusModal should have ≥15 dark: variants (found {count})"
        )

    def test_no_light_only_modal_shell(self):
        """Modal shell bg should not be light-only."""
        src = _modal_src()
        assert "bg-white dark:bg-navy-800" in src, (
            "Modal shell must have dark:bg-navy-800 alongside bg-white"
        )


# ---------------------------------------------------------------------------
# 2. Modal shell / header / footer
# ---------------------------------------------------------------------------

class TestModalStructureDarkMode:
    """Header, footer, and modal container have proper dark styles."""

    def test_header_dark_background(self):
        src = _modal_src()
        assert "dark:bg-navy-900" in src, (
            "Header should have dark:bg-navy-900 background"
        )

    def test_header_dark_border(self):
        src = _modal_src()
        assert "dark:border-navy-600" in src, (
            "Header border should be dark:border-navy-600"
        )

    def test_footer_dark_background(self):
        src = _modal_src()
        # Footer uses same pattern as header
        # Both header and footer have bg-gray-50 dark:bg-navy-900
        occurrences = src.count("bg-gray-50 dark:bg-navy-900")
        assert occurrences >= 2, (
            f"Both header and footer should have bg-gray-50 dark:bg-navy-900 (found {occurrences})"
        )

    def test_footer_dark_border(self):
        src = _modal_src()
        # Both header and footer use border-gray-200 dark:border-navy-600
        occurrences = src.count("border-gray-200 dark:border-navy-600")
        assert occurrences >= 2, (
            f"Both header and footer borders should have dark variant (found {occurrences})"
        )


# ---------------------------------------------------------------------------
# 3. Status color semantics preserved in both themes
# ---------------------------------------------------------------------------

class TestStatusColorsDarkMode:
    """getStatusColor must return dark-aware classes while keeping light ones."""

    def test_online_has_dark_green(self):
        src = _modal_src()
        assert "dark:text-green-400" in src, "Online status should have dark:text-green-400"
        assert "dark:bg-green-900/30" in src, "Online status should have dark:bg-green-900/30"
        assert "dark:border-green-800" in src, "Online status should have dark:border-green-800"

    def test_offline_has_dark_red(self):
        src = _modal_src()
        assert "dark:text-red-400" in src, "Offline status should have dark:text-red-400"
        assert "dark:bg-red-900/30" in src, "Offline status should have dark:bg-red-900/30"
        assert "dark:border-red-800" in src, "Offline status should have dark:border-red-800"

    def test_degraded_has_dark_amber(self):
        src = _modal_src()
        assert "dark:text-amber-400" in src, "Degraded status should have dark:text-amber-400"
        assert "dark:bg-amber-900/30" in src, "Degraded status should have dark:bg-amber-900/30"
        assert "dark:border-amber-800" in src, "Degraded status should have dark:border-amber-800"

    def test_light_mode_status_colors_preserved(self):
        """Light-mode status classes must still exist."""
        src = _modal_src()
        assert "text-green-600 bg-green-50 border-green-200" in src
        assert "text-red-600 bg-red-50 border-red-200" in src
        assert "text-amber-600 bg-amber-50 border-amber-200" in src

    def test_overall_status_subtitle_dark_colors(self):
        """The overall status subtitle (All Systems Operational, etc.) should have dark variants."""
        src = _modal_src()
        # These appear in the header subtitle
        assert 'text-green-600 dark:text-green-400' in src
        assert 'text-amber-600 dark:text-amber-400' in src
        assert 'text-red-600 dark:text-red-400' in src


# ---------------------------------------------------------------------------
# 4. Stat tiles (Uptime / Version)
# ---------------------------------------------------------------------------

class TestStatTilesDarkMode:
    """Quick stat tiles should have dark backgrounds and text."""

    def test_stat_tile_dark_background(self):
        src = _modal_src()
        assert "dark:bg-navy-700/50" in src, (
            "Stat tiles should have dark:bg-navy-700/50 background"
        )

    def test_stat_tile_label_dark_text(self):
        src = _modal_src()
        # Labels like "Uptime", "Version" use text-gray-500 dark:text-gray-400
        assert "text-xs text-gray-500 dark:text-gray-400" in src, (
            "Stat tile labels should have dark:text-gray-400"
        )

    def test_stat_tile_value_dark_text(self):
        src = _modal_src()
        assert "text-sm font-semibold text-gray-900 dark:text-gray-100" in src, (
            "Stat tile values should have dark:text-gray-100"
        )

    def test_stat_icon_dark_color(self):
        src = _modal_src()
        assert "text-gray-400 dark:text-gray-500" in src, (
            "Stat tile icons should have dark:text-gray-500"
        )


# ---------------------------------------------------------------------------
# 5. Services heading, timestamp, close button
# ---------------------------------------------------------------------------

class TestMiscDarkElements:
    """Various text and interactive elements have dark styles."""

    def test_services_heading_dark_text(self):
        src = _modal_src()
        assert 'text-sm font-semibold text-gray-900 dark:text-gray-100 mb-3' in src, (
            "Services heading should have dark:text-gray-100"
        )

    def test_last_checked_dark_text(self):
        src = _modal_src()
        assert 'text-xs text-gray-500 dark:text-gray-400 text-center' in src, (
            "Last-checked timestamp should have dark:text-gray-400"
        )

    def test_close_button_dark_styles(self):
        src = _modal_src()
        assert "dark:text-gray-300" in src, "Close button should have dark:text-gray-300"
        assert "dark:hover:bg-navy-700" in src, "Close button should have dark:hover:bg-navy-700"

    def test_title_dark_text(self):
        src = _modal_src()
        assert 'text-lg font-semibold text-gray-900 dark:text-gray-100' in src, (
            "Modal title 'System Status' should have dark:text-gray-100"
        )


# ---------------------------------------------------------------------------
# 6. Status icons unchanged (shared between themes)
# ---------------------------------------------------------------------------

class TestStatusIconsUnchanged:
    """Status icons use the same colors in both themes (green-500, red-500, amber-500)."""

    def test_online_icon_color(self):
        src = _modal_src()
        assert 'text-green-500' in src, "Online icon should remain text-green-500"

    def test_offline_icon_color(self):
        src = _modal_src()
        assert 'text-red-500' in src, "Offline icon should remain text-red-500"

    def test_degraded_icon_color(self):
        src = _modal_src()
        assert 'text-amber-500' in src, "Degraded icon should remain text-amber-500"


# ---------------------------------------------------------------------------
# 7. Light-mode regression guard
# ---------------------------------------------------------------------------

class TestLightModePreserved:
    """All original light-mode classes must still be present."""

    def test_modal_bg_white(self):
        src = _modal_src()
        assert "bg-white" in src

    def test_header_bg_gray_50(self):
        src = _modal_src()
        assert "bg-gray-50" in src

    def test_border_gray_200(self):
        src = _modal_src()
        assert "border-gray-200" in src

    def test_text_gray_900(self):
        src = _modal_src()
        assert "text-gray-900" in src

    def test_text_gray_500(self):
        src = _modal_src()
        assert "text-gray-500" in src

    def test_status_bg_green_50(self):
        src = _modal_src()
        assert "bg-green-50" in src

    def test_status_bg_red_50(self):
        src = _modal_src()
        assert "bg-red-50" in src

    def test_status_bg_amber_50(self):
        src = _modal_src()
        assert "bg-amber-50" in src


# ---------------------------------------------------------------------------
# 8. Header status trigger in Header.tsx
# ---------------------------------------------------------------------------

class TestHeaderStatusTriggerDarkMode:
    """Header.tsx status dot button should have dark hover styles."""

    def test_status_dot_button_has_dark_hover(self):
        src = _read("frontend/src/components/layout/Header.tsx")
        assert "dark:hover:bg-navy-700" in src, (
            "Status dot button in Header should have dark:hover:bg-navy-700"
        )
