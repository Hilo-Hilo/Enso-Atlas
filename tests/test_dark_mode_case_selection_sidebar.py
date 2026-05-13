"""
Regression tests for dark-mode polish on the Case Selection panel (SlideSelector),
AnalysisControls panel, and left-sidebar error display in page.tsx.

Covers:
1. SlideSelector slide-list rows have dark-mode bg/text/border.
2. Search input (via CSS class) has dark-mode overrides.
3. Sort panel has dark-mode bg, border, and text.
4. Selected-case summary has dark-mode clinical color overrides.
5. Loading / empty / error states have dark text and backgrounds.
6. SortButton inactive state has dark-mode classes.
7. SlideThumbnail has dark-mode fallback bg and border.
8. QCBadge variants carry dark-mode color overrides.
9. SlideRenameInline has dark-mode input and button styles.
10. AnalysisControls embedding progress, amber warning, and analysis
    progress bars have dark-mode overrides.
11. page.tsx left-sidebar error display has dark-mode overrides.
12. Left sidebar card stack has coherent dark backgrounds/borders.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


# ─── Helpers ──────────────────────────────────────────────────────────────

SLIDE_SELECTOR = "frontend/src/components/panels/SlideSelector.tsx"
ANALYSIS_CONTROLS = "frontend/src/components/panels/AnalysisControls.tsx"
PAGE_TSX = "frontend/src/app/page.tsx"
GLOBALS_CSS = "frontend/src/app/globals.css"


# ─── 1. SlideItem dark mode ──────────────────────────────────────────────


class TestSlideItemDarkMode:
    """SlideItem rows must be readable in dark mode."""

    def test_slide_item_default_bg(self):
        src = _read(SLIDE_SELECTOR)
        assert "dark:bg-navy-800" in src, (
            "SlideItem default (unselected) must have dark:bg-navy-800"
        )

    def test_slide_item_hover(self):
        src = _read(SLIDE_SELECTOR)
        assert "dark:hover:bg-navy-700" in src, (
            "SlideItem hover state must have dark:hover:bg-navy-700"
        )

    def test_slide_item_selected(self):
        src = _read(SLIDE_SELECTOR)
        assert "dark:bg-clinical-900/20" in src, (
            "SlideItem selected state must have dark:bg-clinical-900/20"
        )

    def test_slide_item_text(self):
        src = _read(SLIDE_SELECTOR)
        assert "dark:text-gray-100" in src, "SlideItem text must have dark:text-gray-100"

    def test_slide_item_border(self):
        src = _read(SLIDE_SELECTOR)
        assert "dark:border-navy-600" in src, "SlideItem border must include dark:border-navy-600"


# ─── 2. Search input dark mode (CSS) ─────────────────────────────────────


class TestSearchInputDarkMode:
    """Search input via .search-input CSS class must have dark overrides."""

    def test_search_input_dark_bg(self):
        css = _read(GLOBALS_CSS)
        assert "dark .search-input" in css, (
            "globals.css must contain a dark mode rule for .search-input"
        )
        assert "bg-navy-800" in css, "search-input dark rule must set bg-navy-800"

    def test_search_input_dark_border(self):
        css = _read(GLOBALS_CSS)
        assert "border-navy-600" in css, "search-input dark rule must set border-navy-600"

    def test_search_input_dark_text(self):
        css = _read(GLOBALS_CSS)
        assert "text-gray-100" in css, "search-input dark rule must set text-gray-100"


# ─── 3. Sort panel dark mode ─────────────────────────────────────────────


class TestSortPanelDarkMode:
    """Sort options panel must have dark bg, border, and text."""

    def test_sort_panel_bg(self):
        src = _read(SLIDE_SELECTOR)
        assert "dark:bg-navy-900" in src, "Sort panel must have dark:bg-navy-900 background"

    def test_sort_panel_border(self):
        src = _read(SLIDE_SELECTOR)
        # The sort panel wrapping div should have dark border
        assert "dark:border-navy-600" in src, "Sort panel must have dark:border-navy-600"

    def test_sort_by_label_dark_text(self):
        src = _read(SLIDE_SELECTOR)
        assert "dark:text-gray-400" in src, "Sort By label must have dark:text-gray-400"


# ─── 4. Selected-case summary dark mode ──────────────────────────────────


class TestSelectedCaseSummaryDarkMode:
    """The selected-case summary box must be readable in dark mode."""

    def test_summary_bg(self):
        src = _read(SLIDE_SELECTOR)
        # Look for the specific selected-case container pattern
        assert "dark:bg-clinical-900/20" in src, (
            "Selected case summary bg must have dark:bg-clinical-900/20"
        )

    def test_summary_border(self):
        src = _read(SLIDE_SELECTOR)
        assert "dark:border-clinical-800" in src, (
            "Selected case summary border must have dark:border-clinical-800"
        )

    def test_summary_title_text(self):
        src = _read(SLIDE_SELECTOR)
        assert "dark:text-clinical-300" in src, (
            "Selected case 'Selected case' label must have dark:text-clinical-300"
        )

    def test_summary_name_text(self):
        src = _read(SLIDE_SELECTOR)
        assert "dark:text-clinical-100" in src, (
            "Selected case name must have dark:text-clinical-100"
        )

    def test_summary_id_text(self):
        src = _read(SLIDE_SELECTOR)
        assert "dark:text-clinical-400" in src, "Selected case ID must have dark:text-clinical-400"


# ─── 5. Loading / empty / error states ───────────────────────────────────


class TestSlideListStatesDarkMode:
    """Loading, empty, and error states in SlideSelector must be dark-ready."""

    def test_loading_text(self):
        src = _read(SLIDE_SELECTOR)
        # "Loading cases..." text
        idx = src.find("Loading cases...")
        nearby = src[max(0, idx - 200) : idx]
        assert "dark:text-gray-400" in nearby, "Loading state text must have dark:text-gray-400"

    def test_empty_icon_bg(self):
        src = _read(SLIDE_SELECTOR)
        assert "dark:bg-navy-700" in src, "Empty state icon circle must have dark:bg-navy-700"

    def test_empty_heading_text(self):
        src = _read(SLIDE_SELECTOR)
        assert "dark:text-gray-300" in src, "Empty state heading must have dark:text-gray-300"

    def test_error_bg(self):
        src = _read(SLIDE_SELECTOR)
        assert "dark:bg-red-900/20" in src, "Error state must have dark:bg-red-900/20 background"

    def test_error_border(self):
        src = _read(SLIDE_SELECTOR)
        assert "dark:border-red-800" in src, "Error state must have dark:border-red-800"

    def test_error_text(self):
        src = _read(SLIDE_SELECTOR)
        assert "dark:text-red-300" in src, "Error state text must have dark:text-red-300"


# ─── 6. SortButton inactive dark mode ────────────────────────────────────


class TestSortButtonDarkMode:
    """Inactive SortButton must have dark-mode bg, text, border."""

    def test_inactive_bg(self):
        src = _read(SLIDE_SELECTOR)
        assert "dark:bg-navy-700" in src, "Inactive SortButton must have dark:bg-navy-700"

    def test_inactive_text(self):
        src = _read(SLIDE_SELECTOR)
        assert "dark:text-gray-300" in src, "Inactive SortButton text must have dark:text-gray-300"

    def test_inactive_hover(self):
        src = _read(SLIDE_SELECTOR)
        assert "dark:hover:bg-navy-600" in src, (
            "Inactive SortButton hover must have dark:hover:bg-navy-600"
        )


# ─── 7. SlideThumbnail dark mode ─────────────────────────────────────────


class TestSlideThumbnailDarkMode:
    """Thumbnail placeholder and skeleton must have dark backgrounds."""

    def test_thumbnail_container_bg(self):
        src = _read(SLIDE_SELECTOR)
        # The outer container has bg-gray-100 in light; needs dark
        assert "dark:bg-navy-700" in src, "Thumbnail container must have dark:bg-navy-700"

    def test_thumbnail_skeleton_bg(self):
        src = _read(SLIDE_SELECTOR)
        assert "dark:bg-navy-600" in src, "Thumbnail loading skeleton must have dark:bg-navy-600"

    def test_thumbnail_error_gradient(self):
        src = _read(SLIDE_SELECTOR)
        assert "dark:from-navy-700" in src, (
            "Thumbnail error gradient must include dark:from-navy-700"
        )


# ─── 8. QCBadge dark mode ────────────────────────────────────────────────


class TestQCBadgeDarkMode:
    """QCBadge for good/acceptable/poor must have dark-mode color variants."""

    def test_good_qc_dark_bg(self):
        src = _read(SLIDE_SELECTOR)
        assert "dark:bg-green-900/30" in src, "Good QC badge must have dark:bg-green-900/30"

    def test_good_qc_dark_text(self):
        src = _read(SLIDE_SELECTOR)
        assert "dark:text-green-400" in src, "Good QC badge must have dark:text-green-400"

    def test_acceptable_qc_dark_bg(self):
        src = _read(SLIDE_SELECTOR)
        assert "dark:bg-yellow-900/30" in src, "Acceptable QC badge must have dark:bg-yellow-900/30"

    def test_poor_qc_dark_bg(self):
        src = _read(SLIDE_SELECTOR)
        assert "dark:bg-red-900/30" in src, "Poor QC badge must have dark:bg-red-900/30"

    def test_poor_qc_dark_border(self):
        src = _read(SLIDE_SELECTOR)
        assert "dark:border-red-800" in src, "Poor QC badge must have dark:border-red-800"


# ─── 9. SlideRenameInline dark mode ──────────────────────────────────────


class TestSlideRenameInlineDarkMode:
    """Rename input and buttons must be dark-ready."""

    def test_rename_trigger_dark_text(self):
        src = _read(SLIDE_SELECTOR)
        assert "dark:text-clinical-400" in src, "Rename trigger must have dark:text-clinical-400"

    def test_rename_input_dark_bg(self):
        src = _read(SLIDE_SELECTOR)
        # Rename input should have dark bg
        assert "dark:bg-navy-800" in src, "Rename inline input must have dark:bg-navy-800"

    def test_rename_cancel_dark_text(self):
        src = _read(SLIDE_SELECTOR)
        assert "dark:hover:text-gray-200" in src, "Cancel button must have dark:hover:text-gray-200"


# ─── 10. AnalysisControls dark mode ──────────────────────────────────────


class TestAnalysisControlsDarkMode:
    """AnalysisControls embedding progress, amber warning, etc. must be dark-ready."""

    def test_no_slide_hint_dark_text(self):
        src = _read(ANALYSIS_CONTROLS)
        assert "dark:text-gray-400" in src, "'Select a case' hint must have dark:text-gray-400"

    def test_embedding_progress_dark_bg(self):
        src = _read(ANALYSIS_CONTROLS)
        assert "dark:bg-violet-900/20" in src, (
            "Embedding progress box must have dark:bg-violet-900/20"
        )

    def test_embedding_progress_dark_border(self):
        src = _read(ANALYSIS_CONTROLS)
        assert "dark:border-violet-800" in src, (
            "Embedding progress box must have dark:border-violet-800"
        )

    def test_embedding_progress_dark_text(self):
        src = _read(ANALYSIS_CONTROLS)
        assert "dark:text-violet-200" in src, (
            "Embedding progress heading must have dark:text-violet-200"
        )

    def test_embedding_progress_bar_dark(self):
        src = _read(ANALYSIS_CONTROLS)
        assert "dark:bg-violet-800" in src, (
            "Embedding progress bar track must have dark:bg-violet-800"
        )

    def test_analysis_progress_dark_bg(self):
        src = _read(ANALYSIS_CONTROLS)
        assert "dark:bg-clinical-900/20" in src, (
            "Analysis progress box must have dark:bg-clinical-900/20"
        )

    def test_amber_warning_dark_bg(self):
        src = _read(ANALYSIS_CONTROLS)
        assert "dark:bg-amber-900/20" in src, "Amber L0 warning must have dark:bg-amber-900/20"

    def test_amber_warning_dark_text(self):
        src = _read(ANALYSIS_CONTROLS)
        assert "dark:text-amber-300" in src, "Amber L0 warning text must have dark:text-amber-300"

    def test_generate_embeddings_btn_dark_border(self):
        src = _read(ANALYSIS_CONTROLS)
        assert "dark:border-violet-700" in src, (
            "Generate Embeddings button must have dark:border-violet-700"
        )

    def test_model_hint_dark_text(self):
        src = _read(ANALYSIS_CONTROLS)
        # "Select at least one model" amber hint
        idx = src.find("Select one or more models")
        assert idx >= 0, "Model selection hint text must exist"
        nearby = src[max(0, idx - 200) : idx + 100]
        assert "dark:text-amber-300" in nearby, "Model selection hint must have dark:text-amber-300"


# ─── 11. page.tsx left-sidebar error display ─────────────────────────────


class TestPageErrorDisplayDarkMode:
    """Error display in page.tsx left sidebar must be dark-mode ready."""

    def test_error_bg(self):
        src = _read(PAGE_TSX)
        assert "dark:bg-red-900/20" in src, "page.tsx error display must have dark:bg-red-900/20"

    def test_error_border(self):
        src = _read(PAGE_TSX)
        assert "dark:border-red-800" in src, "page.tsx error display must have dark:border-red-800"

    def test_error_title_text(self):
        src = _read(PAGE_TSX)
        assert "dark:text-red-200" in src, "page.tsx error title must have dark:text-red-200"

    def test_error_body_text(self):
        src = _read(PAGE_TSX)
        assert "dark:text-red-300" in src, "page.tsx error body text must have dark:text-red-300"

    def test_error_retry_link(self):
        src = _read(PAGE_TSX)
        # The retry button text
        assert "dark:hover:text-red-100" in src, (
            "page.tsx retry link must have dark:hover:text-red-100"
        )

    def test_error_dismiss_link(self):
        src = _read(PAGE_TSX)
        assert "dark:text-red-400" in src, "page.tsx dismiss link must have dark:text-red-400"


# ─── 12. Left sidebar coherent dark backgrounds ──────────────────────────


class TestLeftSidebarDarkBackground:
    """Left sidebar <aside> wrappers must have dark bg and border."""

    def test_mobile_sidebar_dark_bg(self):
        """Mobile left sidebar should have dark bg (inherited from bg-white dark:bg-navy-900)."""
        src = _read(PAGE_TSX)
        # The mobile aside for slides has bg-white and dark:bg-navy-900
        assert "dark:bg-navy-900" in src, "Left sidebar must have dark:bg-navy-900"

    def test_sidebar_dark_border(self):
        src = _read(PAGE_TSX)
        assert "dark:border-navy-700" in src, "Left sidebar must have dark:border-navy-700"

    def test_card_component_dark_bg(self):
        """Card component (used by SlideSelector and AnalysisControls) must have dark bg."""
        src = _read("frontend/src/components/ui/Card.tsx")
        assert "dark:bg-navy-800" in src, "Card component must have dark:bg-navy-800"

    def test_card_header_dark_gradient(self):
        src = _read("frontend/src/components/ui/Card.tsx")
        assert "dark:from-navy-900" in src, "CardHeader must have dark gradient from-navy-900"


# ─── 13. "N cases found" and "Show all" dark mode ────────────────────────


class TestCaseCountDarkMode:
    """The case-count row and Show-all toggle must be dark-ready."""

    def test_case_count_text(self):
        src = _read(SLIDE_SELECTOR)
        # The "N cases found" paragraph — search for the JSX fragment
        marker = 'case{filteredSlides.length !== 1 ? "s" : ""} found'
        found_idx = src.find(marker)
        assert found_idx >= 0, "Case count JSX fragment must exist"
        nearby = src[max(0, found_idx - 200) : found_idx + 50]
        assert "dark:text-gray-400" in nearby, "Case count text must have dark:text-gray-400"

    def test_show_all_toggle(self):
        src = _read(SLIDE_SELECTOR)
        assert "dark:text-clinical-400" in src, "Show all toggle must have dark:text-clinical-400"

    def test_show_all_hover(self):
        src = _read(SLIDE_SELECTOR)
        assert "dark:hover:text-clinical-300" in src, (
            "Show all toggle hover must have dark:hover:text-clinical-300"
        )


# ─── 14. Slide list container dark mode ──────────────────────────────────


class TestSlideListContainerDarkMode:
    """The bg-white wrapper around slide rows must have a dark counterpart."""

    def test_list_container_dark_bg(self):
        src = _read(SLIDE_SELECTOR)
        # The <div className="bg-white dark:bg-navy-800 ..."> wrapping slide items
        # Check that bg-white is paired with dark:bg-navy-800
        idx = src.find('className="bg-white dark:bg-navy-800')
        assert idx >= 0, "Slide list container must have bg-white paired with dark:bg-navy-800"
