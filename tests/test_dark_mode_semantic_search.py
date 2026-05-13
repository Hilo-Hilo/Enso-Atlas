"""
Regression tests for dark-mode styling on the Semantic Search panel.

Covers:
1. Search input dark background, border, text, placeholder, focus ring.
2. Select (topK) dropdown dark styles.
3. Example-query chips dark background, border, text, hover.
4. Result cards dark background, border, text, selected state.
5. Empty / error / unavailable states dark styles.
6. Miscellaneous icon and label contrast.
7. Light-mode classes are preserved (no regressions).
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


SRC_PATH = "frontend/src/components/panels/SemanticSearchPanel.tsx"


# ─── 1. Search Input Dark Mode ──────────────────────────────────────────


class TestSearchInputDarkMode:
    """The main text input must be readable on navy backgrounds."""

    def test_input_has_dark_bg(self):
        src = _read(SRC_PATH)
        assert "dark:bg-navy-700" in src, "Search input must have dark:bg-navy-700 background"

    def test_input_has_dark_border(self):
        src = _read(SRC_PATH)
        assert "dark:border-navy-600" in src, "Search input must have dark:border-navy-600"

    def test_input_has_dark_text(self):
        src = _read(SRC_PATH)
        assert "dark:text-gray-100" in src, (
            "Search input must have dark:text-gray-100 for readable text"
        )

    def test_input_has_dark_placeholder(self):
        src = _read(SRC_PATH)
        assert "dark:placeholder:text-gray-400" in src, (
            "Search input placeholder must have dark variant"
        )

    def test_input_has_dark_focus_border(self):
        src = _read(SRC_PATH)
        assert "dark:focus:border-clinical-400" in src, (
            "Search input focus border must use clinical-400 in dark mode"
        )

    def test_input_has_dark_focus_ring(self):
        src = _read(SRC_PATH)
        assert "dark:focus:ring-clinical-800" in src, (
            "Search input focus ring must use clinical-800 in dark mode"
        )


# ─── 2. Select (topK) Dropdown Dark Mode ────────────────────────────────


class TestSelectDropdownDarkMode:
    """The results-count <select> must be styled for dark mode."""

    def test_select_has_dark_bg(self):
        src = _read(SRC_PATH)
        # The select element has the same dark:bg-navy-700
        count = src.count("dark:bg-navy-700")
        assert count >= 2, f"Both input and select must have dark:bg-navy-700 (found {count})"

    def test_select_label_has_dark_text(self):
        src = _read(SRC_PATH)
        assert "dark:text-gray-400" in src, "Results label must have dark:text-gray-400"


# ─── 3. Example-Query Chips Dark Mode ───────────────────────────────────


class TestExampleChipsDarkMode:
    """Example query chips must be visible on dark backgrounds."""

    def test_chips_have_dark_bg(self):
        src = _read(SRC_PATH)
        assert "dark:bg-navy-700" in src, "Example chips must have dark background"

    def test_chips_have_dark_border(self):
        src = _read(SRC_PATH)
        assert "dark:border-navy-600" in src, "Example chips must have dark border"

    def test_chips_have_dark_text(self):
        src = _read(SRC_PATH)
        assert "dark:text-gray-300" in src, "Example chips must have dark:text-gray-300"

    def test_chips_have_dark_hover_border(self):
        src = _read(SRC_PATH)
        assert "dark:hover:border-clinical-500" in src, "Example chips must have dark hover border"

    def test_chips_have_dark_hover_bg(self):
        src = _read(SRC_PATH)
        assert "dark:hover:bg-navy-600" in src, "Example chips must have dark hover background"

    def test_chips_have_dark_hover_text(self):
        src = _read(SRC_PATH)
        assert "dark:hover:text-clinical-300" in src, (
            "Example chips must have dark hover text color"
        )

    def test_try_label_has_dark_text(self):
        src = _read(SRC_PATH)
        # The "Try:" label next to the lightbulb icon
        assert "text-gray-500 dark:text-gray-400" in src, (
            "'Try:' label must have dark:text-gray-400"
        )


# ─── 4. Result Card Dark Mode ───────────────────────────────────────────


class TestResultCardDarkMode:
    """Search result items must be styled for dark mode."""

    def test_card_default_has_dark_bg(self):
        src = _read(SRC_PATH)
        assert "dark:bg-navy-800" in src, "Result card default state must have dark:bg-navy-800"

    def test_card_default_has_dark_border(self):
        src = _read(SRC_PATH)
        # border-gray-200 ... dark:border-navy-600 in the default branch
        assert "bg-white dark:border-navy-600 dark:bg-navy-800" in src, (
            "Result card default must have dark border and bg"
        )

    def test_card_selected_has_dark_bg(self):
        src = _read(SRC_PATH)
        assert "dark:bg-clinical-900/40" in src, (
            "Selected result card must have dark:bg-clinical-900/40"
        )

    def test_card_selected_has_dark_border(self):
        src = _read(SRC_PATH)
        assert "dark:border-clinical-500" in src, (
            "Selected result card must have dark:border-clinical-500"
        )

    def test_card_selected_has_dark_ring(self):
        src = _read(SRC_PATH)
        assert "dark:ring-clinical-700" in src, (
            "Selected result card must have dark:ring-clinical-700"
        )

    def test_card_hover_has_dark_border(self):
        src = _read(SRC_PATH)
        assert "dark:hover:border-clinical-400" in src, (
            "Result card hover must have dark:hover:border-clinical-400"
        )

    def test_card_hover_has_dark_bg(self):
        src = _read(SRC_PATH)
        assert "dark:hover:bg-clinical-900/30" in src, (
            "Result card hover must have dark:hover:bg-clinical-900/30"
        )

    def test_card_focus_ring_dark(self):
        src = _read(SRC_PATH)
        assert "dark:focus:ring-clinical-400" in src, (
            "Result card focus ring must have dark variant"
        )

    def test_card_disabled_has_dark_hover_overrides(self):
        src = _read(SRC_PATH)
        assert "dark:hover:border-navy-600 dark:hover:bg-navy-800" in src, (
            "Disabled result card must override hover in dark mode"
        )

    def test_patch_name_has_dark_text(self):
        src = _read(SRC_PATH)
        assert "text-gray-900 dark:text-gray-100" in src, "Patch name must have dark:text-gray-100"

    def test_coordinates_text_has_dark_variant(self):
        src = _read(SRC_PATH)
        assert "text-gray-500 dark:text-gray-400" in src, (
            "Coordinates text must have dark:text-gray-400"
        )

    def test_thumbnail_border_has_dark_variant(self):
        src = _read(SRC_PATH)
        assert "dark:border-navy-600 dark:group-hover:border-clinical-500" in src, (
            "Thumbnail border must have dark mode variants"
        )

    def test_zoom_icon_has_dark_color(self):
        src = _read(SRC_PATH)
        assert "text-clinical-600 dark:text-clinical-400" in src, (
            "ZoomIn icon must have dark:text-clinical-400"
        )

    def test_coordinates_unavailable_has_dark_text(self):
        src = _read(SRC_PATH)
        assert "dark:text-amber-400" in src, (
            "Coordinates-unavailable warning must have dark:text-amber-400"
        )


# ─── 5. Results Divider and Header ──────────────────────────────────────


class TestResultsHeaderDarkMode:
    """The results section header/divider must be dark-styled."""

    def test_results_border_has_dark_variant(self):
        src = _read(SRC_PATH)
        assert "border-gray-100 dark:border-navy-700" in src, (
            "Results section divider must have dark:border-navy-700"
        )

    def test_results_header_text_has_dark_variant(self):
        src = _read(SRC_PATH)
        # "Matching Patches" and "Similarity Score" are in text-gray-500 dark:text-gray-400
        assert "text-gray-500 dark:text-gray-400" in src, (
            "Results header text must have dark variant"
        )

    def test_selected_tip_has_dark_variant(self):
        src = _read(SRC_PATH)
        assert "dark:text-clinical-300" in src, (
            "Selected patch tip must have dark:text-clinical-300"
        )


# ─── 6. Error State Dark Mode ───────────────────────────────────────────


class TestErrorStateDarkMode:
    """Error banner must be readable in dark mode."""

    def test_error_bg_has_dark_variant(self):
        src = _read(SRC_PATH)
        assert "dark:bg-red-900/30" in src, "Error banner must have dark:bg-red-900/30"

    def test_error_border_has_dark_variant(self):
        src = _read(SRC_PATH)
        assert "dark:border-red-800" in src, "Error banner must have dark:border-red-800"

    def test_error_text_has_dark_variant(self):
        src = _read(SRC_PATH)
        assert "dark:text-red-300" in src, "Error text must have dark:text-red-300"


# ─── 7. Unavailable / Empty States Dark Mode ────────────────────────────


class TestUnavailableStateDarkMode:
    """The 'search unavailable' state must not clash in dark mode."""

    def test_unavailable_icon_bg_has_dark_variant(self):
        src = _read(SRC_PATH)
        assert "dark:bg-navy-700" in src, "Unavailable icon background must have dark variant"

    def test_unavailable_title_has_dark_text(self):
        src = _read(SRC_PATH)
        assert "text-gray-600 dark:text-gray-300" in src, (
            "Unavailable title must have dark:text-gray-300"
        )

    def test_no_results_state_has_dark_text(self):
        src = _read(SRC_PATH)
        assert "text-gray-500 dark:text-gray-400" in src, (
            "No-results state must have dark text variant"
        )


# ─── 8. Clear Button Dark Mode ──────────────────────────────────────────


class TestClearButtonDarkMode:
    """Clear (X) button must be visible in dark mode."""

    def test_clear_button_has_dark_text(self):
        src = _read(SRC_PATH)
        assert "dark:text-gray-500 dark:hover:text-gray-300" in src, (
            "Clear button must have dark text and hover variants"
        )


# ─── 9. Light Mode Unchanged ────────────────────────────────────────────


class TestLightModePreserved:
    """Verify light-mode classes are still present (no regressions)."""

    def test_input_still_has_light_border(self):
        src = _read(SRC_PATH)
        assert "border-gray-300" in src, "Input must keep light border-gray-300"

    def test_input_still_has_light_focus(self):
        src = _read(SRC_PATH)
        assert "focus:border-clinical-500" in src, "Input must keep light focus:border-clinical-500"

    def test_chips_still_have_light_bg(self):
        src = _read(SRC_PATH)
        assert "bg-gray-50" in src, "Chips must keep light bg-gray-50"

    def test_chips_still_have_light_hover(self):
        src = _read(SRC_PATH)
        assert "hover:bg-clinical-50" in src, "Chips must keep light hover:bg-clinical-50"

    def test_result_card_still_has_bg_white(self):
        src = _read(SRC_PATH)
        assert "bg-white" in src, "Result cards must keep light bg-white"

    def test_error_still_has_light_bg(self):
        src = _read(SRC_PATH)
        assert "bg-red-50" in src, "Error banner must keep light bg-red-50"

    def test_unavailable_still_has_light_bg(self):
        src = _read(SRC_PATH)
        assert "bg-gray-100" in src, "Unavailable state must keep light bg-gray-100"

    def test_result_selected_still_has_light_bg(self):
        src = _read(SRC_PATH)
        assert "bg-clinical-50" in src, "Selected result card must keep light bg-clinical-50"


# ─── 10. Title Icon Dark Mode ───────────────────────────────────────────


class TestTitleIconDarkMode:
    """Panel title icon must have dark variant."""

    def test_search_icon_in_header_has_dark_variant(self):
        src = _read(SRC_PATH)
        assert "text-clinical-600 dark:text-clinical-400" in src, (
            "Header search icon must have dark:text-clinical-400"
        )
