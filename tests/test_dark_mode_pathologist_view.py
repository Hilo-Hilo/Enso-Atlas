"""
Regression tests for PathologistView dark-mode polish.

These are source-level checks to ensure key PathologistView regions include
explicit dark-mode classes without removing light-mode classes.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PATHOLOGIST_VIEW = REPO_ROOT / "frontend/src/components/panels/PathologistView.tsx"


def _src() -> str:
    return PATHOLOGIST_VIEW.read_text(encoding="utf-8")


class TestPathologistHeaderDarkMode:
    def test_header_strip_dark_surface(self):
        src = _src()
        assert "dark:border-violet-800 dark:bg-violet-900/20" in src

    def test_header_text_dark_contrast(self):
        src = _src()
        assert "text-violet-900 dark:text-violet-100" in src
        assert "text-violet-600 dark:text-violet-300" in src

    def test_switch_button_dark_variant(self):
        src = _src()
        assert "dark:bg-violet-900/40" in src
        assert "dark:text-violet-200" in src


class TestPathologistAnnotationAreaDarkMode:
    def test_tool_button_dark_inactive_style(self):
        src = _src()
        assert "dark:bg-navy-700 dark:text-gray-200 dark:hover:bg-navy-600" in src

    def test_active_tool_helper_dark_style(self):
        src = _src()
        assert "dark:border-violet-800 dark:bg-violet-900/20 dark:text-violet-300" in src

    def test_note_input_dark_style(self):
        src = _src()
        assert "dark:border-navy-600 dark:bg-navy-900 dark:text-gray-100 dark:placeholder:text-gray-400" in src

    def test_annotation_list_dark_surface(self):
        src = _src()
        assert "dark:border dark:border-navy-600 dark:bg-navy-900/30" in src

    def test_annotation_row_dark_states(self):
        src = _src()
        assert "dark:border-violet-700 dark:bg-violet-900/30" in src
        assert "dark:bg-navy-700/70" in src

    def test_annotation_controls_dark_text(self):
        src = _src()
        assert "dark:text-gray-300" in src
        assert "dark:text-gray-400 dark:hover:text-gray-200" in src
        assert "dark:text-red-400 dark:hover:text-red-300" in src


class TestPathologistGradingMitoticMorphologyDarkMode:
    def test_grading_cards_dark_states(self):
        src = _src()
        assert "dark:border-violet-700 dark:bg-violet-900/20" in src
        assert "dark:border-navy-600 dark:bg-navy-900/20 dark:hover:border-navy-500" in src

    def test_mitotic_counter_box_dark_surface(self):
        src = _src()
        assert "dark:border dark:border-navy-600 dark:bg-navy-900/40" in src

    def test_mitotic_helper_text_dark_style(self):
        src = _src()
        assert "dark:text-gray-400" in src
        assert "text-amber-500 dark:text-amber-400" in src

    def test_morphology_cards_dark_states(self):
        src = _src()
        assert "dark:border-navy-600 dark:bg-navy-900/20 dark:hover:border-navy-500" in src
        assert "text-gray-600 dark:text-gray-300" in src
        assert "text-gray-400 dark:text-gray-500" in src


class TestPathologistExportDarkMode:
    def test_export_area_dark_container(self):
        src = _src()
        assert "dark:border dark:border-navy-700 dark:bg-navy-900/40 dark:p-3" in src

    def test_secondary_action_buttons_dark_overrides(self):
        src = _src()
        assert "const secondaryButtonDarkClass = \"dark:bg-navy-700 dark:text-gray-100 dark:border-navy-500 dark:hover:bg-navy-600" in src

    def test_saved_state_icon_dark_green(self):
        src = _src()
        assert "text-green-600 dark:text-green-400" in src


class TestPathologistLightModePreserved:
    def test_header_light_classes_remain(self):
        src = _src()
        assert "border-violet-200 bg-violet-50" in src

    def test_note_input_light_classes_remain(self):
        src = _src()
        assert "border border-gray-200 bg-white" in src

    def test_tool_button_light_classes_remain(self):
        src = _src()
        assert "bg-gray-100 text-gray-700 hover:bg-gray-200" in src
