"""
Regression tests for issue #86: Remove awkward right-panel collapse UX.

These tests verify the frontend source code meets the acceptance criteria:
1. No floating/overlapping collapse button on the right panel divider
2. No "Right panel" label in UI
3. Right sidebar uses tab-based navigation (not dropdown selector)
4. No right-side collapsible panel state coupling
5. Left sidebar floating button also removed for consistency
"""

import pathlib
import re

FRONTEND_SRC = pathlib.Path(__file__).resolve().parent.parent / "frontend" / "src"
PAGE_TSX = FRONTEND_SRC / "app" / "page.tsx"


def _read_page() -> str:
    """Read the main page.tsx file."""
    assert PAGE_TSX.exists(), f"page.tsx not found at {PAGE_TSX}"
    return PAGE_TSX.read_text()


class TestNoFloatingCollapseButtons:
    """Verify that floating collapse buttons on panel dividers are removed."""

    def test_no_right_panel_collapse_button(self):
        """Right panel divider should NOT have a floating collapse button."""
        source = _read_page()
        # The old code had: title={rightSidebarOpen ? "Collapse right sidebar" : "Expand right sidebar"}
        assert "Collapse right sidebar" not in source, (
            "Found 'Collapse right sidebar' text — floating button still present"
        )
        assert "Expand right sidebar" not in source, (
            "Found 'Expand right sidebar' text — floating button still present"
        )

    def test_no_left_panel_collapse_button(self):
        """Left panel divider should NOT have a floating collapse button."""
        source = _read_page()
        assert "Collapse left sidebar" not in source, (
            "Found 'Collapse left sidebar' text — floating button still present"
        )
        assert "Expand left sidebar" not in source, (
            "Found 'Expand left sidebar' text — floating button still present"
        )

    def test_no_right_sidebar_open_state(self):
        """rightSidebarOpen state should be removed (no collapse state coupling)."""
        source = _read_page()
        assert "rightSidebarOpen" not in source, (
            "Found 'rightSidebarOpen' — right panel collapse state still exists"
        )

    def test_no_right_panel_ref(self):
        """rightPanelRef should be removed (no imperative collapse handle)."""
        source = _read_page()
        assert "rightPanelRef" not in source, (
            "Found 'rightPanelRef' — right panel imperative collapse handle still exists"
        )


class TestNoRightPanelLabel:
    """Verify the 'Right panel' label and selector framing are removed."""

    def test_no_right_panel_label_element(self):
        """No 'Right panel' label rendered in the UI (comments are OK)."""
        source = _read_page()
        # Filter out comment lines — only check rendered strings
        non_comment_lines = [
            line for line in source.split("\n")
            if "Right panel" in line and not line.strip().startswith("//") and not line.strip().startswith("*")
        ]
        assert len(non_comment_lines) == 0, (
            f"Found 'Right panel' in non-comment code: {non_comment_lines}"
        )

    def test_no_right_panel_selector_id(self):
        """The old 'right-panel-selector' select element should be gone."""
        source = _read_page()
        assert "right-panel-selector" not in source, (
            "Found 'right-panel-selector' ID — old dropdown still present"
        )


class TestTabBasedNavigation:
    """Verify the new tab-based right sidebar navigation exists."""

    def test_right_sidebar_tabs_component_exists(self):
        """RightSidebarTabs component should be defined."""
        source = _read_page()
        assert "function RightSidebarTabs" in source, (
            "RightSidebarTabs component not found — tab navigation missing"
        )

    def test_right_sidebar_tabs_uses_role_tablist(self):
        """Tab bar should use proper ARIA role='tablist'."""
        source = _read_page()
        assert 'role="tablist"' in source, (
            "Tab bar missing role='tablist' — accessibility issue"
        )

    def test_right_sidebar_tabs_uses_role_tab(self):
        """Individual tabs should use role='tab'."""
        source = _read_page()
        assert 'role="tab"' in source, (
            "Individual tabs missing role='tab' — accessibility issue"
        )

    def test_icon_mapping_exists(self):
        """RIGHT_PANEL_ICONS mapping should exist for all panel keys."""
        source = _read_page()
        assert "RIGHT_PANEL_ICONS" in source, (
            "RIGHT_PANEL_ICONS mapping not found"
        )
        # Check that all expected panel types have icons
        expected_panels = [
            "pathologist-workspace",
            "medgemma",
            "evidence",
            "prediction",
            "multi-model",
            "semantic-search",
            "similar-cases",
            "outlier-detector",
        ]
        for panel in expected_panels:
            assert f'"{panel}"' in source, (
                f"Panel '{panel}' not found in icon mapping or panel options"
            )

    def test_tabs_rendered_in_right_sidebar(self):
        """RightSidebarTabs should be rendered in the right sidebar content."""
        source = _read_page()
        assert "<RightSidebarTabs" in source, (
            "RightSidebarTabs component not rendered in the right sidebar"
        )


class TestRightPanelAlwaysVisible:
    """Verify the right panel is no longer collapsible."""

    def test_right_panel_not_collapsible(self):
        """Right panel should NOT have collapsible prop."""
        source = _read_page()
        # Find the right panel section — it should not have collapsible
        # The old code had: collapsible collapsedSize="0%"
        # We need to check that the right panel specifically doesn't have these
        # Look for the pattern near "Right Sidebar - Desktop"
        right_sidebar_section = source[source.find("Right Sidebar - Desktop"):]
        if right_sidebar_section:
            # Get the next ~500 chars which should contain the Panel props
            panel_section = right_sidebar_section[:500]
            assert "collapsible" not in panel_section, (
                "Right panel still has 'collapsible' prop"
            )
            assert 'collapsedSize' not in panel_section, (
                "Right panel still has 'collapsedSize' prop"
            )

    def test_right_panel_has_minimum_size(self):
        """Right panel should have a reasonable minSize to stay visible."""
        source = _read_page()
        right_sidebar_section = source[source.find("Right Sidebar - Desktop"):]
        if right_sidebar_section:
            panel_section = right_sidebar_section[:500]
            assert 'minSize=' in panel_section, (
                "Right panel missing minSize constraint"
            )


class TestFunctionalParity:
    """Verify all original panel content types are still available."""

    def test_all_panel_types_in_render(self):
        """All panel rendering branches should still exist."""
        source = _read_page()
        expected_panels = [
            "pathologist-workspace",
            "medgemma",
            "evidence",
            "prediction",
            "multi-model",
            "semantic-search",
            "similar-cases",
            "outlier-detector",
        ]
        for panel in expected_panels:
            assert f'activeRightPanel === "{panel}"' in source, (
                f"Panel rendering for '{panel}' missing — functional parity broken"
            )

    def test_mobile_panel_tabs_still_exist(self):
        """Mobile panel tabs should still be present for responsive layout."""
        source = _read_page()
        assert "MobilePanelTabs" in source, (
            "MobilePanelTabs component missing — mobile layout broken"
        )

    def test_left_panel_still_collapsible(self):
        """Left panel should remain collapsible (it's useful to hide slide list)."""
        source = _read_page()
        left_section = source[source.find("Left Sidebar - Desktop"):source.find("Center - WSI")]
        assert "collapsible" in left_section, (
            "Left panel lost collapsible behavior"
        )


class TestNoChevronImports:
    """Verify removed icons are cleaned up."""

    def test_no_chevron_imports_in_page(self):
        """ChevronLeft/ChevronRight should not be imported (no longer needed)."""
        source = _read_page()
        # Check the import line specifically
        import_lines = [line for line in source.split("\n") if "from \"lucide-react\"" in line]
        for line in import_lines:
            assert "ChevronLeft" not in line, (
                "ChevronLeft still imported — unused icon"
            )
            assert "ChevronRight" not in line, (
                "ChevronRight still imported — unused icon"
            )
