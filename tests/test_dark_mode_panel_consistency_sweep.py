"""Regression coverage for the UI consistency sweep.

Focus:
- Right-sidebar tab panels in oncologist/pathologist workflows
- Empty/error/helper states typography + dark-mode contrast
- Core modals and batch page shell dark-mode parity
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


EVIDENCE = "frontend/src/components/panels/EvidencePanel.tsx"
OUTLIER = "frontend/src/components/panels/OutlierDetectorPanel.tsx"
PREDICTION = "frontend/src/components/panels/PredictionPanel.tsx"
MULTI_MODEL = "frontend/src/components/panels/MultiModelPredictionPanel.tsx"
REPORT = "frontend/src/components/panels/ReportPanel.tsx"
COMPARE_MODAL = "frontend/src/components/modals/CompareSelectModal.tsx"
SHORTCUTS_MODAL = "frontend/src/components/modals/KeyboardShortcutsModal.tsx"
PATCH_ZOOM_MODAL = "frontend/src/components/modals/PatchZoomModal.tsx"
BATCH_PAGE = "frontend/src/app/batch/page.tsx"


class TestPathologistPanelConsistency:
    def test_evidence_panel_has_dark_empty_and_shell_tokens(self):
        src = _read(EVIDENCE)
        assert "dark:bg-navy-700" in src
        assert "dark:border-navy-700" in src
        assert "text-gray-600 dark:text-gray-300" in src

    def test_evidence_panel_has_dark_interactive_states(self):
        src = _read(EVIDENCE)
        assert "dark:hover:bg-clinical-900/30" in src
        assert "dark:border-clinical-500" in src
        assert "dark:text-clinical-400" in src

    def test_outlier_panel_dark_error_cards_and_empty_states(self):
        src = _read(OUTLIER)
        assert "dark:bg-red-900/30" in src
        assert "dark:border-red-800" in src
        assert "text-gray-400 dark:text-gray-500" in src


class TestOncologistPanelConsistency:
    def test_prediction_panel_dark_stat_cards_and_disclaimer(self):
        src = _read(PREDICTION)
        assert "dark:bg-navy-900" in src
        assert "dark:border-navy-700" in src
        assert "dark:text-gray-400" in src

    def test_prediction_panel_dark_clinical_warning_palette(self):
        src = _read(PREDICTION)
        assert "dark:bg-red-900/30" in src
        assert "dark:bg-yellow-900/30" in src
        assert "dark:text-red-300" in src

    def test_multi_model_panel_dark_prediction_rows(self):
        src = _read(MULTI_MODEL)
        assert "dark:bg-sky-900/20" in src
        assert "dark:bg-orange-900/20" in src
        assert "dark:text-sky-300" in src

    def test_report_panel_dark_sections_and_guideline_cards(self):
        src = _read(REPORT)
        assert "dark:bg-navy-800" in src
        assert "dark:border-blue-800" in src
        assert "dark:text-blue-200" in src


class TestModalAndPageShellConsistency:
    def test_compare_select_modal_dark_shell(self):
        src = _read(COMPARE_MODAL)
        assert "dark:bg-navy-800" in src
        assert "dark:border-navy-700" in src
        assert "dark:text-gray-100" in src

    def test_keyboard_shortcuts_modal_dark_shell(self):
        src = _read(SHORTCUTS_MODAL)
        assert "dark:bg-navy-800" in src
        assert "dark:bg-navy-900" in src
        assert "dark:text-gray-300" in src

    def test_patch_zoom_modal_dark_shell_and_info_panel(self):
        src = _read(PATCH_ZOOM_MODAL)
        assert "dark:bg-navy-800" in src
        assert "dark:border-navy-700" in src
        assert "dark:text-gray-100" in src

    def test_batch_page_header_dark_typography(self):
        src = _read(BATCH_PAGE)
        assert "text-gray-900 dark:text-gray-100" in src
        assert "text-gray-500 dark:text-gray-400" in src
