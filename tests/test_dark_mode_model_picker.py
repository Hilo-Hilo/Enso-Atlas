"""
Regression tests for dark-mode polish on Model Picker, Analysis Controls,
and Badge components.

Covers:
1. ModelPicker outer container dark background/border.
2. ModelPicker header text, icons, and hover states in dark mode.
3. Resolution level buttons: active and inactive dark variants.
4. Quick-action chip buttons dark variants.
5. Model section headers dark text.
6. ModelCheckbox row dark hover, selected, text, checkbox border.
7. Badge component dark variants for every variant type.
8. AnalysisControls info boxes (violet embedding, amber warning, clinical
   progress, sky run-button) dark variants.
9. Light-mode classes are preserved alongside dark: overrides.
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

def _src_model_picker() -> str:
    return _read("frontend/src/components/panels/ModelPicker.tsx")


def _src_analysis_controls() -> str:
    return _read("frontend/src/components/panels/AnalysisControls.tsx")


def _src_badge() -> str:
    return _read("frontend/src/components/ui/Badge.tsx")


# ─── 1. ModelPicker Container ────────────────────────────────────────────

class TestModelPickerContainerDarkMode:
    """Outer wrapper should have dark bg/border."""

    def test_container_has_dark_bg(self):
        src = _src_model_picker()
        assert "dark:bg-navy-800" in src, (
            "ModelPicker container must include dark:bg-navy-800"
        )

    def test_container_has_dark_border(self):
        src = _src_model_picker()
        assert "dark:border-navy-600" in src, (
            "ModelPicker container must include dark:border-navy-600"
        )

    def test_container_keeps_light_bg(self):
        src = _src_model_picker()
        assert "bg-white" in src, (
            "ModelPicker container must retain bg-white for light mode"
        )


# ─── 2. ModelPicker Header ───────────────────────────────────────────────

class TestModelPickerHeaderDarkMode:
    """Header button text and icons need dark overrides."""

    def test_header_hover_dark(self):
        src = _src_model_picker()
        assert "dark:hover:bg-navy-700" in src, (
            "Header button must include dark:hover:bg-navy-700"
        )

    def test_header_title_dark_text(self):
        src = _src_model_picker()
        # "Model Selection" label
        assert 'dark:text-gray-100' in src, (
            "Header title must include dark:text-gray-100"
        )

    def test_header_icon_dark_color(self):
        src = _src_model_picker()
        assert "dark:text-clinical-400" in src, (
            "FlaskConical icon must include dark:text-clinical-400"
        )

    def test_chevron_dark_color(self):
        src = _src_model_picker()
        assert "dark:text-gray-500" in src, (
            "Chevron icons must include dark:text-gray-500"
        )


# ─── 3. Resolution Level Buttons ────────────────────────────────────────

class TestResolutionLevelDarkMode:
    """Active and inactive resolution buttons need dark variants."""

    def test_active_resolution_dark_bg(self):
        src = _src_model_picker()
        assert "dark:bg-purple-900/30" in src, (
            "Active resolution button must include dark:bg-purple-900/30"
        )

    def test_active_resolution_dark_text(self):
        src = _src_model_picker()
        assert "dark:text-purple-300" in src, (
            "Active resolution button must include dark:text-purple-300"
        )

    def test_active_resolution_dark_border(self):
        src = _src_model_picker()
        assert "dark:border-purple-600" in src, (
            "Active resolution button must include dark:border-purple-600"
        )

    def test_inactive_resolution_dark_bg(self):
        src = _src_model_picker()
        assert "dark:bg-navy-700" in src, (
            "Inactive resolution button must include dark:bg-navy-700"
        )

    def test_inactive_resolution_dark_text(self):
        src = _src_model_picker()
        assert "dark:text-gray-300" in src, (
            "Inactive resolution button must include dark:text-gray-300"
        )

    def test_inactive_resolution_dark_hover(self):
        src = _src_model_picker()
        assert "dark:hover:bg-navy-600" in src, (
            "Inactive resolution button must include dark:hover:bg-navy-600"
        )

    def test_resolution_section_border_dark(self):
        src = _src_model_picker()
        assert "dark:border-navy-700" in src, (
            "Resolution section borders must include dark:border-navy-700"
        )

    def test_resolution_label_dark_text(self):
        src = _src_model_picker()
        assert "dark:text-gray-400" in src, (
            "Resolution Level label must include dark:text-gray-400"
        )


# ─── 4. Embedding Status Indicators ─────────────────────────────────────

class TestEmbeddingStatusDarkMode:
    """CheckCircle/Circle and status text need dark variants."""

    def test_ready_icon_dark_color(self):
        src = _src_model_picker()
        assert "dark:text-green-400" in src, (
            "Green check icon must include dark:text-green-400"
        )

    def test_amber_warning_dark(self):
        src = _src_model_picker()
        assert "dark:text-amber-400" in src, (
            "Amber embedding warning must include dark:text-amber-400"
        )


# ─── 5. Force Re-embed Checkbox ─────────────────────────────────────────

class TestForceReembedDarkMode:
    """Checkbox and labels need dark overrides."""

    def test_checkbox_dark_border(self):
        src = _src_model_picker()
        assert "dark:border-navy-500" in src, (
            "Checkboxes must include dark:border-navy-500"
        )

    def test_checkbox_dark_bg(self):
        src = _src_model_picker()
        assert "dark:bg-navy-700" in src, (
            "Checkboxes must include dark:bg-navy-700 for background"
        )

    def test_force_reembed_label_dark(self):
        src = _src_model_picker()
        assert "dark:text-gray-300" in src, (
            "Force Re-embed label must include dark:text-gray-300"
        )


# ─── 6. Quick Action Chips ──────────────────────────────────────────────

class TestQuickActionChipsDarkMode:
    """All/None/Cancer/General chips need dark backgrounds."""

    def test_all_none_chips_dark_bg(self):
        src = _src_model_picker()
        # These share the same dark:bg-navy-700 class
        count = src.count("dark:bg-navy-700")
        assert count >= 3, (
            f"Quick action 'All'/'None' chips should use dark:bg-navy-700 (found {count})"
        )

    def test_cancer_chip_dark_bg(self):
        src = _src_model_picker()
        assert "dark:bg-pink-900/30" in src, (
            "Cancer-specific chip must include dark:bg-pink-900/30"
        )

    def test_cancer_chip_dark_text(self):
        src = _src_model_picker()
        assert "dark:text-pink-300" in src, (
            "Cancer-specific chip must include dark:text-pink-300"
        )

    def test_general_chip_dark_bg(self):
        src = _src_model_picker()
        assert "dark:bg-blue-900/30" in src, (
            "General chip must include dark:bg-blue-900/30"
        )

    def test_general_chip_dark_text(self):
        src = _src_model_picker()
        assert "dark:text-blue-300" in src, (
            "General chip must include dark:text-blue-300"
        )


# ─── 7. Model Checkbox Row ──────────────────────────────────────────────

class TestModelCheckboxRowDarkMode:
    """Each model row needs dark hover, selected, text overrides."""

    def test_row_hover_dark(self):
        src = _src_model_picker()
        assert "dark:hover:bg-navy-700/50" in src, (
            "Model row must include dark:hover:bg-navy-700/50"
        )

    def test_row_selected_dark(self):
        src = _src_model_picker()
        assert "dark:bg-clinical-900/30" in src, (
            "Selected model row must include dark:bg-clinical-900/30"
        )

    def test_model_name_dark_text(self):
        src = _src_model_picker()
        assert "dark:text-gray-100" in src, (
            "Model display name must include dark:text-gray-100"
        )

    def test_model_description_dark_text(self):
        src = _src_model_picker()
        # text-gray-500 dark:text-gray-400 for description
        assert "dark:text-gray-400" in src, (
            "Model description must include dark:text-gray-400"
        )

    def test_cached_badge_dark(self):
        src = _src_model_picker()
        assert "dark:bg-green-900/30" in src, (
            "Cached badge must include dark:bg-green-900/30"
        )
        assert "dark:text-green-300" in src, (
            "Cached badge must include dark:text-green-300"
        )


# ─── 8. Badge Component Dark Variants ───────────────────────────────────

class TestBadgeDarkMode:
    """Badge.tsx must have dark: overrides for every variant."""

    def test_badge_default_dark(self):
        src = _src_badge()
        assert "dark:bg-navy-700" in src
        assert "dark:text-gray-300" in src
        assert "dark:border-navy-600" in src

    def test_badge_success_dark(self):
        src = _src_badge()
        assert "dark:from-green-900/30" in src
        assert "dark:text-green-300" in src
        assert "dark:border-green-700" in src

    def test_badge_warning_dark(self):
        src = _src_badge()
        assert "dark:from-amber-900/30" in src
        assert "dark:text-amber-300" in src
        assert "dark:border-amber-700" in src

    def test_badge_danger_dark(self):
        src = _src_badge()
        assert "dark:from-red-900/30" in src
        assert "dark:text-red-300" in src
        assert "dark:border-red-700" in src

    def test_badge_info_dark(self):
        src = _src_badge()
        assert "dark:from-blue-900/30" in src
        assert "dark:text-blue-300" in src
        assert "dark:border-blue-700" in src

    def test_badge_clinical_dark(self):
        src = _src_badge()
        assert "dark:from-clinical-900/30" in src
        assert "dark:text-clinical-300" in src
        assert "dark:border-clinical-700" in src

    def test_badge_light_variants_preserved(self):
        """Ensure light-mode classes remain."""
        src = _src_badge()
        assert "bg-gray-100" in src
        assert "text-gray-700" in src
        assert "from-green-50" in src
        assert "from-blue-50" in src
        assert "from-clinical-50" in src


# ─── 9. AnalysisControls Dark Mode ──────────────────────────────────────

class TestAnalysisControlsDarkMode:
    """AnalysisControls info boxes and buttons need dark variants."""

    def test_no_slide_msg_dark(self):
        src = _src_analysis_controls()
        assert "dark:text-gray-400" in src, (
            "'Select a case' text must include dark:text-gray-400"
        )

    def test_embedding_progress_box_dark_bg(self):
        src = _src_analysis_controls()
        assert "dark:bg-violet-900/30" in src, (
            "Embedding progress box must include dark:bg-violet-900/30"
        )

    def test_embedding_progress_box_dark_border(self):
        src = _src_analysis_controls()
        assert "dark:border-violet-700" in src, (
            "Embedding progress box must include dark:border-violet-700"
        )

    def test_embedding_progress_text_dark(self):
        src = _src_analysis_controls()
        assert "dark:text-violet-300" in src, (
            "Embedding title must include dark:text-violet-300"
        )
        assert "dark:text-violet-400" in src, (
            "Embedding message must include dark:text-violet-400"
        )

    def test_embedding_progress_bar_dark(self):
        src = _src_analysis_controls()
        assert "dark:bg-violet-800" in src, (
            "Embedding progress bar track must include dark:bg-violet-800"
        )

    def test_analysis_progress_box_dark(self):
        src = _src_analysis_controls()
        assert "dark:bg-clinical-900/30" in src, (
            "Analysis progress box must include dark:bg-clinical-900/30"
        )
        assert "dark:border-clinical-700" in src, (
            "Analysis progress box must include dark:border-clinical-700"
        )

    def test_amber_warning_box_dark(self):
        src = _src_analysis_controls()
        assert "dark:bg-amber-900/30" in src, (
            "Amber warning box must include dark:bg-amber-900/30"
        )
        assert "dark:border-amber-700" in src, (
            "Amber warning box must include dark:border-amber-700"
        )
        assert "dark:text-amber-400" in src, (
            "Amber warning text must include dark:text-amber-400"
        )

    def test_run_analysis_button_dark(self):
        src = _src_analysis_controls()
        assert "dark:bg-sky-900/30" in src, (
            "Run Analysis button must include dark:bg-sky-900/30"
        )
        assert "dark:text-sky-200" in src, (
            "Run Analysis button must include dark:text-sky-200"
        )
        assert "dark:border-sky-600" in src, (
            "Run Analysis button must include dark:border-sky-600"
        )

    def test_generate_embeddings_button_dark(self):
        src = _src_analysis_controls()
        assert "dark:border-violet-600" in src, (
            "Generate Embeddings button must include dark:border-violet-600"
        )
        assert "dark:text-violet-300" in src, (
            "Generate Embeddings button must include dark:text-violet-300"
        )

    def test_no_model_warning_dark(self):
        src = _src_analysis_controls()
        # "Select one or more models" amber message
        count = src.count("dark:text-amber-400")
        assert count >= 2, (
            f"Amber warning texts should have dark:text-amber-400 (found {count})"
        )


# ─── 10. Light Mode Preservation ────────────────────────────────────────

class TestLightModePreserved:
    """Ensure light mode classes are not removed."""

    def test_model_picker_keeps_bg_white(self):
        src = _src_model_picker()
        assert "bg-white" in src

    def test_model_picker_keeps_border_gray_200(self):
        src = _src_model_picker()
        assert "border-gray-200" in src

    def test_model_picker_keeps_text_gray_900(self):
        src = _src_model_picker()
        assert "text-gray-900" in src

    def test_model_picker_keeps_hover_bg_gray_50(self):
        src = _src_model_picker()
        assert "hover:bg-gray-50" in src

    def test_analysis_controls_keeps_bg_violet_50(self):
        src = _src_analysis_controls()
        assert "bg-violet-50" in src

    def test_analysis_controls_keeps_bg_amber_50(self):
        src = _src_analysis_controls()
        assert "bg-amber-50" in src

    def test_analysis_controls_keeps_bg_sky_100(self):
        src = _src_analysis_controls()
        assert "bg-sky-100" in src

    def test_badge_keeps_light_gradients(self):
        src = _src_badge()
        for color in ["green", "amber", "red", "blue", "clinical"]:
            assert f"from-{color}-50" in src, (
                f"Badge must retain from-{color}-50 for light mode"
            )
