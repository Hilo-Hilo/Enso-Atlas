"""
Regression tests for dark-mode polish across full pages.

Covers:
1. Logo wordmark readability in dark mode (dark: text overrides).
2. Slide Manager page (/slides) dark-mode classes on container, header,
   filter panel, cards/table, pagination.
3. Project Manager page (/projects) dark-mode classes on container,
   cards, modals, status badges.
4. Card UI component dark variants.
5. FilterPanel, SlideGrid, SlideTable dark mode classes.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_SRC = REPO_ROOT / "frontend" / "src"


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


# ─── 1. Logo Wordmark Contrast ───────────────────────────────────────────


class TestLogoWordmarkDarkMode:
    """Logo wordmark must be readable on dark backgrounds."""

    def test_logo_enso_text_has_dark_override(self):
        src = _read("frontend/src/components/ui/Logo.tsx")
        # The "Enso" text (light-bg branch) must have dark:text-white
        assert "dark:text-white" in src, (
            "Logo 'Enso' wordmark must include dark:text-white for contrast"
        )

    def test_logo_atlas_text_has_dark_override(self):
        src = _read("frontend/src/components/ui/Logo.tsx")
        assert "dark:text-clinical-400" in src, (
            "Logo 'Atlas' text must include dark:text-clinical-400"
        )

    def test_logo_subtitle_has_dark_override(self):
        src = _read("frontend/src/components/ui/Logo.tsx")
        assert "dark:text-gray-400" in src, "Logo subtitle must include dark:text-gray-400"


# ─── 2. Card Component Dark Mode ─────────────────────────────────────────


class TestCardComponentDarkMode:
    """Card UI primitive must carry dark-mode styles."""

    def test_card_default_has_dark_bg(self):
        src = _read("frontend/src/components/ui/Card.tsx")
        assert "dark:bg-navy-800" in src, "Card default variant must include dark:bg-navy-800"

    def test_card_default_has_dark_border(self):
        src = _read("frontend/src/components/ui/Card.tsx")
        assert "dark:border-navy-600" in src, (
            "Card default variant must include dark:border-navy-600"
        )

    def test_card_title_has_dark_text(self):
        src = _read("frontend/src/components/ui/Card.tsx")
        assert "dark:text-gray-100" in src, "CardTitle must include dark:text-gray-100"

    def test_card_header_has_dark_bg(self):
        src = _read("frontend/src/components/ui/Card.tsx")
        assert "dark:from-navy-900" in src or "dark:bg-navy" in src, (
            "CardHeader must include dark mode background"
        )

    def test_card_footer_has_dark_bg(self):
        src = _read("frontend/src/components/ui/Card.tsx")
        assert "dark:bg-navy-900" in src, "CardFooter must include dark:bg-navy-900"


# ─── 3. Slides Page Dark Mode ────────────────────────────────────────────


class TestSlidesPageDarkMode:
    """Slide Manager page must not have white-on-dark mismatch."""

    def test_slides_page_container_has_dark_bg(self):
        src = _read("frontend/src/app/slides/page.tsx")
        assert "dark:bg-navy" in src, "Slides page container must have dark:bg-navy-* class"

    def test_slides_page_header_bar_dark(self):
        src = _read("frontend/src/app/slides/page.tsx")
        assert "dark:bg-navy-900" in src, "Slides page header bar must have dark:bg-navy-900"

    def test_slides_page_heading_dark_text(self):
        src = _read("frontend/src/app/slides/page.tsx")
        assert 'dark:text-gray-100">Slide Manager' in src, (
            "Slides page h1 must include dark:text-gray-100"
        )

    def test_slides_page_view_toggle_dark(self):
        src = _read("frontend/src/app/slides/page.tsx")
        assert "dark:bg-navy-800" in src, "Slides page view mode toggle wrapper must have dark bg"

    def test_slides_page_filter_sidebar_dark_border(self):
        src = _read("frontend/src/app/slides/page.tsx")
        assert "dark:border-navy-700" in src, "Slides page filter sidebar must have dark border"

    def test_slides_page_pagination_dark(self):
        src = _read("frontend/src/app/slides/page.tsx")
        # Pagination wrapper
        assert "dark:bg-navy-800" in src, "Slides page pagination must have dark background"

    def test_slides_page_table_wrapper_dark(self):
        src = _read("frontend/src/app/slides/page.tsx")
        # The table wrapper div around <SlideTable>
        count = src.count("dark:bg-navy-800")
        assert count >= 2, (
            f"Slides page needs dark:bg-navy-800 on multiple elements (found {count})"
        )


# ─── 4. Filter Panel Dark Mode ───────────────────────────────────────────


class TestFilterPanelDarkMode:
    """Filter panel must be styled for dark mode."""

    def test_filter_panel_root_dark_bg(self):
        src = _read("frontend/src/components/slides/FilterPanel.tsx")
        assert "dark:bg-navy-900" in src, "FilterPanel root must include dark:bg-navy-900"

    def test_filter_panel_header_dark(self):
        src = _read("frontend/src/components/slides/FilterPanel.tsx")
        assert "dark:from-navy-800" in src, "FilterPanel header gradient must have dark variant"

    def test_filter_section_border_dark(self):
        src = _read("frontend/src/components/slides/FilterPanel.tsx")
        assert "dark:border-navy-700" in src, (
            "FilterSection dividers must have dark:border-navy-700"
        )

    def test_filter_section_title_dark_text(self):
        src = _read("frontend/src/components/slides/FilterPanel.tsx")
        assert "dark:text-gray-200" in src, "Filter section title text must have dark:text-gray-200"

    def test_filter_search_input_dark(self):
        src = _read("frontend/src/components/slides/FilterPanel.tsx")
        assert "dark:bg-navy-800" in src, "Filter search input must have dark background"


# ─── 5. SlideGrid Dark Mode ──────────────────────────────────────────────


class TestSlideGridDarkMode:
    """Slide cards in grid must carry dark mode classes."""

    def test_slide_card_name_dark_text(self):
        src = _read("frontend/src/components/slides/SlideGrid.tsx")
        assert "dark:text-gray-100" in src, "SlideCard name must include dark:text-gray-100"

    def test_slide_card_menu_dark_bg(self):
        src = _read("frontend/src/components/slides/SlideGrid.tsx")
        assert "dark:bg-navy-800" in src, "SlideCard context menu must have dark:bg-navy-800"

    def test_slide_card_menu_dark_border(self):
        src = _read("frontend/src/components/slides/SlideGrid.tsx")
        assert "dark:border-navy-600" in src, (
            "SlideCard context menu must have dark:border-navy-600"
        )

    def test_slide_grid_empty_state_dark(self):
        src = _read("frontend/src/components/slides/SlideGrid.tsx")
        assert "dark:text-gray-600" in src, "SlideGrid empty state icon must have dark text variant"


# ─── 6. SlideTable Dark Mode ─────────────────────────────────────────────


class TestSlideTableDarkMode:
    """Table view must carry dark mode classes."""

    def test_table_header_row_dark(self):
        src = _read("frontend/src/components/slides/SlideTable.tsx")
        assert "dark:bg-navy-900" in src, "SlideTable header row must have dark:bg-navy-900"

    def test_table_row_dark_border(self):
        src = _read("frontend/src/components/slides/SlideTable.tsx")
        assert "dark:border-navy-700" in src, "SlideTable rows must have dark:border-navy-700"

    def test_table_row_hover_dark(self):
        src = _read("frontend/src/components/slides/SlideTable.tsx")
        assert "dark:hover:bg-navy" in src, "SlideTable rows must have dark hover background"

    def test_table_name_dark_text(self):
        src = _read("frontend/src/components/slides/SlideTable.tsx")
        assert "dark:text-gray-100" in src, "SlideTable name column must have dark text"

    def test_table_sort_header_dark(self):
        src = _read("frontend/src/components/slides/SlideTable.tsx")
        assert "dark:text-clinical-400" in src, (
            "SlideTable sortable header must have dark active color"
        )

    def test_table_context_menu_dark(self):
        src = _read("frontend/src/components/slides/SlideTable.tsx")
        assert "dark:bg-navy-800" in src, "SlideTable context menu must have dark:bg-navy-800"


# ─── 7. Projects Page Dark Mode ──────────────────────────────────────────


class TestProjectsPageDarkMode:
    """Project Manager page must have dark mode support."""

    def test_projects_page_container_dark(self):
        src = _read("frontend/src/app/projects/page.tsx")
        assert "dark:from-navy-950" in src or "dark:from-navy-900" in src, (
            "Projects page container must have dark gradient"
        )

    def test_projects_page_heading_dark(self):
        src = _read("frontend/src/app/projects/page.tsx")
        assert 'dark:text-gray-100">Project Management' in src, (
            "Projects page h1 must include dark:text-gray-100"
        )

    def test_project_card_dark_bg(self):
        src = _read("frontend/src/app/projects/page.tsx")
        # ProjectCard root
        dark_bg_count = src.count("dark:bg-navy-800")
        assert dark_bg_count >= 2, (
            f"Projects page must have dark:bg-navy-800 on cards and modals (found {dark_bg_count})"
        )

    def test_project_card_dark_border(self):
        src = _read("frontend/src/app/projects/page.tsx")
        assert "dark:border-navy-700" in src, "ProjectCard must have dark:border-navy-700"

    def test_project_card_title_dark(self):
        src = _read("frontend/src/app/projects/page.tsx")
        # h3 in ProjectCard
        assert "dark:text-gray-100 truncate" in src, (
            "ProjectCard title must include dark:text-gray-100"
        )

    def test_project_card_description_dark(self):
        src = _read("frontend/src/app/projects/page.tsx")
        assert "dark:text-gray-300" in src, (
            "ProjectCard description must include dark:text-gray-300"
        )

    def test_project_card_footer_dark(self):
        src = _read("frontend/src/app/projects/page.tsx")
        assert "dark:bg-navy-900/50" in src, "ProjectCard footer must have dark:bg-navy-900/50"

    def test_project_modal_dark(self):
        src = _read("frontend/src/app/projects/page.tsx")
        # Edit/create modal shell
        assert "dark:bg-navy-800" in src, "Project modals must have dark:bg-navy-800"

    def test_project_form_inputs_dark(self):
        src = _read("frontend/src/app/projects/page.tsx")
        assert "dark:bg-navy-700" in src, (
            "Project form inputs must have dark:bg-navy-700 background"
        )

    def test_project_form_labels_dark(self):
        src = _read("frontend/src/app/projects/page.tsx")
        assert "dark:text-gray-200 mb-1" in src, "Project form labels must have dark:text-gray-200"

    def test_project_empty_state_dark(self):
        src = _read("frontend/src/app/projects/page.tsx")
        assert 'dark:text-gray-100">No projects configured' in src, (
            "Projects empty state must have dark text"
        )


# ─── 8. Light Mode Unchanged ─────────────────────────────────────────────


class TestLightModeUnchanged:
    """Verify light mode base classes are still present (not replaced)."""

    def test_slides_page_still_has_light_bg(self):
        src = _read("frontend/src/app/slides/page.tsx")
        assert "bg-gray-50" in src, "Slides page must keep light bg-gray-50"

    def test_projects_page_still_has_light_gradient(self):
        src = _read("frontend/src/app/projects/page.tsx")
        assert "from-gray-50" in src, "Projects page must keep light gradient"

    def test_card_still_has_bg_white(self):
        src = _read("frontend/src/components/ui/Card.tsx")
        assert "bg-white" in src, "Card component must keep light bg-white"

    def test_filter_panel_still_has_bg_white(self):
        src = _read("frontend/src/components/slides/FilterPanel.tsx")
        assert "bg-white" in src, "FilterPanel must keep light bg-white"

    def test_logo_still_has_light_bg_styles(self):
        src = _read("frontend/src/components/ui/Logo.tsx")
        assert "text-slate-800" in src, "Logo must keep text-slate-800 for light-bg contrast"
