"""
Regression tests for dark-mode polish on the Similar Cases panel.

Covers:
1. SimilarCasesPanel.tsx — all hardcoded light-mode colours have dark: overrides
   (cards, outcome summary, toggle pills, bars, labels, metadata, expand details).
2. Badge.tsx — all variant rows carry dark: overrides for bg, text, border.
3. No bare `bg-white` without a dark: companion in the panel.
4. Group-header borders & text have dark: variants (green, red, gray groups).
5. Comparison-note callout has dark: bg/border/text.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


# ─── Helper ───────────────────────────────────────────────────────────────

PANEL_PATH = "frontend/src/components/panels/SimilarCasesPanel.tsx"
BADGE_PATH = "frontend/src/components/ui/Badge.tsx"


# ─── 1. SimilarCaseItem card backgrounds ────────────────────────────────


class TestSimilarCaseCardDarkMode:
    """Each case card must not render as a white rectangle in dark mode."""

    def test_card_default_state_has_dark_bg(self):
        src = _read(PANEL_PATH)
        assert "dark:bg-navy-800" in src, "Default (collapsed) case card must have dark:bg-navy-800"

    def test_card_default_state_has_dark_border(self):
        src = _read(PANEL_PATH)
        assert "dark:border-navy-600" in src, (
            "Default case card border must include dark:border-navy-600"
        )

    def test_card_expanded_state_has_dark_bg(self):
        src = _read(PANEL_PATH)
        assert "dark:bg-clinical-900/30" in src, (
            "Expanded case card must have dark:bg-clinical-900/30"
        )

    def test_card_expanded_state_has_dark_border(self):
        src = _read(PANEL_PATH)
        assert "dark:border-clinical-600" in src, (
            "Expanded case card border must include dark:border-clinical-600"
        )

    def test_card_hover_has_dark_variant(self):
        src = _read(PANEL_PATH)
        assert "dark:hover:bg-navy-700/50" in src, (
            "Card hover state must have dark:hover:bg-navy-700/50"
        )

    def test_no_bare_bg_white(self):
        """Every bg-white in the panel must be accompanied by a dark: variant."""
        src = _read(PANEL_PATH)
        # Find all lines with bg-white
        for i, line in enumerate(src.splitlines(), 1):
            if "bg-white" in line and "dark:" not in line:
                assert False, f"Line {i} has bg-white without a dark: companion:\n  {line.strip()}"


# ─── 2. Outcome Summary Box ─────────────────────────────────────────────


class TestOutcomeSummaryDarkMode:
    """Outcome distribution box must be styled for dark mode."""

    def test_summary_box_dark_bg(self):
        src = _read(PANEL_PATH)
        assert "dark:bg-navy-900" in src, "Outcome summary box must have dark:bg-navy-900"

    def test_summary_box_dark_border(self):
        src = _read(PANEL_PATH)
        # The border on the outcome summary
        assert "dark:border-navy-600" in src, (
            "Outcome summary border must include dark:border-navy-600"
        )

    def test_summary_bar_track_dark(self):
        src = _read(PANEL_PATH)
        assert "dark:bg-navy-700" in src, "Outcome bar track must have dark:bg-navy-700"

    def test_summary_label_dark_text(self):
        src = _read(PANEL_PATH)
        assert "dark:text-gray-300" in src, "Outcome legend labels must have dark:text-gray-300"


# ─── 3. View Mode Toggle Pills ──────────────────────────────────────────


class TestViewModeToggleDarkMode:
    """Grouped/All toggle pills must be readable in dark mode."""

    def test_active_pill_dark_bg(self):
        src = _read(PANEL_PATH)
        assert "dark:bg-navy-700" in src, "Active toggle pill must have dark:bg-navy-700"

    def test_active_pill_dark_text(self):
        src = _read(PANEL_PATH)
        assert "dark:text-clinical-300" in src, (
            "Active toggle pill text must have dark:text-clinical-300"
        )

    def test_inactive_pill_dark_text(self):
        src = _read(PANEL_PATH)
        assert "dark:text-gray-400" in src, "Inactive toggle pill must have dark:text-gray-400"

    def test_inactive_pill_dark_hover(self):
        src = _read(PANEL_PATH)
        assert "dark:hover:text-gray-200" in src, (
            "Inactive toggle pill hover must have dark:hover:text-gray-200"
        )


# ─── 4. Group Headers ───────────────────────────────────────────────────


class TestGroupHeadersDarkMode:
    """Positive/Negative/Unknown group headers need dark borders & text."""

    def test_positive_group_dark_border(self):
        src = _read(PANEL_PATH)
        assert "dark:border-green-800" in src, (
            "Positive group header border must include dark:border-green-800"
        )

    def test_positive_group_dark_text(self):
        src = _read(PANEL_PATH)
        assert "dark:text-green-300" in src, (
            "Positive group header text must include dark:text-green-300"
        )

    def test_positive_icon_dark(self):
        src = _read(PANEL_PATH)
        assert "dark:text-green-400" in src, (
            "Positive group CheckCircle icon must include dark:text-green-400"
        )

    def test_negative_group_dark_border(self):
        src = _read(PANEL_PATH)
        assert "dark:border-red-800" in src, (
            "Negative group header border must include dark:border-red-800"
        )

    def test_negative_group_dark_text(self):
        src = _read(PANEL_PATH)
        assert "dark:text-red-300" in src, (
            "Negative group header text must include dark:text-red-300"
        )

    def test_negative_icon_dark(self):
        src = _read(PANEL_PATH)
        assert "dark:text-red-400" in src, (
            "Negative group XCircle icon must include dark:text-red-400"
        )

    def test_unknown_group_dark_border(self):
        src = _read(PANEL_PATH)
        # Unknown group uses gray/navy border
        assert "dark:border-navy-600" in src, (
            "Unknown group header border must include dark:border-navy-600"
        )


# ─── 5. Expanded Details Metadata ───────────────────────────────────────


class TestExpandedDetailsDarkMode:
    """Expanded case details (Slide ID, Distance, etc.) must be readable."""

    def test_detail_border_dark(self):
        src = _read(PANEL_PATH)
        assert "dark:border-navy-700" in src, (
            "Expanded details border-t must include dark:border-navy-700"
        )

    def test_detail_labels_dark(self):
        src = _read(PANEL_PATH)
        assert "dark:text-gray-400" in src, (
            "Detail labels (Slide ID, Distance) must have dark:text-gray-400"
        )

    def test_detail_values_dark(self):
        src = _read(PANEL_PATH)
        assert "dark:text-gray-300" in src, "Detail values must have dark:text-gray-300"


# ─── 6. Similarity Bar ──────────────────────────────────────────────────


class TestSimilarityBarDarkMode:
    """Per-case similarity bar must be visible in dark mode."""

    def test_bar_track_dark(self):
        src = _read(PANEL_PATH)
        assert "dark:bg-navy-700" in src, (
            "Similarity bar track bg-gray-200 must have dark:bg-navy-700"
        )

    def test_bar_fill_dark(self):
        src = _read(PANEL_PATH)
        assert "dark:bg-clinical-400" in src, "Similarity bar fill must have dark:bg-clinical-400"

    def test_score_text_dark(self):
        src = _read(PANEL_PATH)
        # The % label
        assert "dark:text-gray-300" in src, "Similarity score text must have dark:text-gray-300"


# ─── 7. Case Title Text ─────────────────────────────────────────────────


class TestCaseTitleDarkMode:
    """Case title 'Case XXXX' must be readable in dark mode."""

    def test_case_title_dark_text(self):
        src = _read(PANEL_PATH)
        assert "dark:text-gray-100" in src, "Case title must include dark:text-gray-100"


# ─── 8. Comparison Note Callout ─────────────────────────────────────────


class TestComparisonNoteDarkMode:
    """Blue callout box comparing responders vs non-responders."""

    def test_note_dark_bg(self):
        src = _read(PANEL_PATH)
        assert "dark:bg-blue-900/30" in src, "Comparison note must have dark:bg-blue-900/30"

    def test_note_dark_border(self):
        src = _read(PANEL_PATH)
        assert "dark:border-blue-800" in src, (
            "Comparison note border must include dark:border-blue-800"
        )

    def test_note_dark_text(self):
        src = _read(PANEL_PATH)
        assert "dark:text-blue-300" in src, "Comparison note text must include dark:text-blue-300"


# ─── 9. Error State ─────────────────────────────────────────────────────


class TestErrorStateDarkMode:
    """Error state must not show white circles / unreadable text in dark mode."""

    def test_error_circle_dark_bg(self):
        src = _read(PANEL_PATH)
        assert "dark:bg-red-900/30" in src, "Error circle must have dark:bg-red-900/30"

    def test_error_heading_dark(self):
        src = _read(PANEL_PATH)
        assert "dark:text-red-300" in src, "Error heading must have dark:text-red-300"

    def test_error_detail_dark(self):
        src = _read(PANEL_PATH)
        assert "dark:text-red-400" in src, "Error detail text must have dark:text-red-400"


# ─── 10. Empty & Loading States ─────────────────────────────────────────


class TestEmptyLoadingDarkMode:
    """Empty and loading states must not show white circles in dark mode."""

    def test_empty_circle_dark_bg(self):
        src = _read(PANEL_PATH)
        assert "dark:bg-navy-700" in src, "Empty-state circle must have dark:bg-navy-700"

    def test_empty_heading_dark(self):
        src = _read(PANEL_PATH)
        assert "dark:text-gray-300" in src, "Empty-state heading must have dark:text-gray-300"

    def test_loading_icon_dark(self):
        src = _read(PANEL_PATH)
        assert "dark:text-clinical-400" in src, (
            "Loading GitCompare icon must have dark:text-clinical-400"
        )

    def test_loading_text_dark(self):
        src = _read(PANEL_PATH)
        assert "dark:text-gray-400" in src, "Loading hint text must have dark:text-gray-400"


# ─── 11. Thumbnail Placeholder ──────────────────────────────────────────


class TestThumbnailDarkMode:
    """Thumbnail borders and placeholders must not clash in dark mode."""

    def test_thumbnail_border_dark(self):
        src = _read(PANEL_PATH)
        # thumbnail border
        lines = [line for line in src.splitlines() if "w-12 h-12" in line and "border" in line]
        assert any("dark:border-navy-600" in line for line in lines), (
            "Thumbnail border must include dark:border-navy-600"
        )

    def test_thumbnail_placeholder_neutral_dark(self):
        src = _read(PANEL_PATH)
        assert "dark:bg-navy-700" in src, "Neutral thumbnail placeholder must have dark:bg-navy-700"

    def test_thumbnail_placeholder_responder_dark(self):
        src = _read(PANEL_PATH)
        assert "dark:bg-green-900/30" in src, (
            "Responder thumbnail placeholder must have dark:bg-green-900/30"
        )

    def test_thumbnail_placeholder_nonresponder_dark(self):
        src = _read(PANEL_PATH)
        assert "dark:bg-red-900/30" in src, (
            "Non-responder thumbnail placeholder must have dark:bg-red-900/30"
        )


# ─── 12. Badge Component Dark Mode ──────────────────────────────────────


class TestBadgeDarkMode:
    """Badge variants must carry dark: overrides for bg, text, and border."""

    def test_badge_default_dark_bg(self):
        src = _read(BADGE_PATH)
        assert "dark:bg-navy-700" in src, "Badge default variant must include dark:bg-navy-700"

    def test_badge_default_dark_text(self):
        src = _read(BADGE_PATH)
        assert "dark:text-gray-300" in src, "Badge default variant must include dark:text-gray-300"

    def test_badge_success_dark_text(self):
        src = _read(BADGE_PATH)
        assert "dark:text-green-300" in src, (
            "Badge success variant must include dark:text-green-300"
        )

    def test_badge_danger_dark_text(self):
        src = _read(BADGE_PATH)
        assert "dark:text-red-300" in src, "Badge danger variant must include dark:text-red-300"

    def test_badge_info_dark_text(self):
        src = _read(BADGE_PATH)
        assert "dark:text-blue-300" in src, "Badge info variant must include dark:text-blue-300"

    def test_badge_clinical_dark_text(self):
        src = _read(BADGE_PATH)
        assert "dark:text-clinical-300" in src, (
            "Badge clinical variant must include dark:text-clinical-300"
        )

    def test_badge_success_dark_border(self):
        src = _read(BADGE_PATH)
        assert "dark:border-green-800" in src, (
            "Badge success variant must include dark:border-green-800"
        )

    def test_badge_danger_dark_border(self):
        src = _read(BADGE_PATH)
        assert "dark:border-red-800" in src, "Badge danger variant must include dark:border-red-800"

    def test_badge_all_variants_have_dark(self):
        """Every variant line in Badge must contain at least one dark: class."""
        src = _read(BADGE_PATH)
        in_variants = False
        for line in src.splitlines():
            if "const variants" in line:
                in_variants = True
                continue
            if in_variants:
                if "};" in line:
                    break
                # Each variant line should have dark:
                if '":' in line or "':":
                    assert "dark:" in line, (
                        f"Badge variant line missing dark: override:\n  {line.strip()}"
                    )


# ─── 13. Footer Info ────────────────────────────────────────────────────


class TestFooterInfoDarkMode:
    """Footer separator and FAISS info text must be dark-aware."""

    def test_footer_border_dark(self):
        src = _read(PANEL_PATH)
        assert "dark:border-navy-700" in src, "Footer border-t must include dark:border-navy-700"

    def test_footer_text_dark(self):
        src = _read(PANEL_PATH)
        # Info text about FAISS
        assert "dark:text-gray-400" in src, "Footer info text must include dark:text-gray-400"


# ─── 14. Show More / Less Button ────────────────────────────────────────


class TestShowMoreButtonDarkMode:
    """Show More/Less button must be readable in dark mode."""

    def test_show_more_dark_text(self):
        src = _read(PANEL_PATH)
        assert "dark:text-gray-400" in src, "Show More button must have dark:text-gray-400"

    def test_show_more_dark_hover(self):
        src = _read(PANEL_PATH)
        assert "dark:hover:text-gray-100" in src, (
            "Show More button hover must have dark:hover:text-gray-100"
        )
