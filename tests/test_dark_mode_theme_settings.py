"""
Regression tests for dark mode theme settings functionality.
Issue: Dark mode did not work from Settings theme selector.

These tests verify:
1. Tailwind is configured with darkMode: 'class' strategy
2. ThemeScript component exists for FOUC prevention
3. SettingsModal properly applies theme changes
4. Key components have dark: variant classes
5. localStorage key 'atlas-theme' is used correctly
6. System preference detection is implemented
"""

from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_SRC = REPO_ROOT / "frontend" / "src"


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


class TestTailwindDarkModeConfiguration:
    """Verify Tailwind is configured for class-based dark mode."""

    def test_tailwind_config_has_darkmode_class(self):
        """darkMode: 'class' must be set for dark: variants to work."""
        config = _read("frontend/tailwind.config.ts")
        # Check for darkMode: "class" or darkMode: 'class'
        assert re.search(r'darkMode:\s*["\']class["\']', config), (
            "tailwind.config.ts must have darkMode: 'class' for dark mode to work"
        )


class TestThemeInitialization:
    """Verify theme is initialized before React hydration to prevent FOUC."""

    def test_theme_script_component_exists(self):
        """ThemeScript component should exist for inline theme initialization."""
        theme_script_path = FRONTEND_SRC / "components" / "ThemeScript.tsx"
        assert theme_script_path.exists(), (
            "ThemeScript.tsx should exist to prevent flash of wrong theme"
        )

    def test_theme_script_reads_localstorage(self):
        """ThemeScript should read atlas-theme from localStorage."""
        src = _read("frontend/src/components/ThemeScript.tsx")
        assert "atlas-theme" in src, (
            "ThemeScript should read 'atlas-theme' from localStorage"
        )
        assert "localStorage.getItem" in src, (
            "ThemeScript should use localStorage.getItem"
        )

    def test_theme_script_checks_system_preference(self):
        """ThemeScript should check prefers-color-scheme media query."""
        src = _read("frontend/src/components/ThemeScript.tsx")
        assert "prefers-color-scheme" in src, (
            "ThemeScript should check system color scheme preference"
        )

    def test_theme_script_applies_dark_class(self):
        """ThemeScript should add/remove 'dark' class on documentElement."""
        src = _read("frontend/src/components/ThemeScript.tsx")
        assert "classList.add" in src or "classList.remove" in src, (
            "ThemeScript should manipulate classList for dark mode"
        )
        assert "'dark'" in src or '"dark"' in src, (
            "ThemeScript should reference the 'dark' class"
        )

    def test_layout_includes_theme_script(self):
        """Layout should include ThemeScript for FOUC prevention."""
        layout = _read("frontend/src/app/layout.tsx")
        assert "ThemeScript" in layout, (
            "layout.tsx should import and use ThemeScript component"
        )
        assert "<ThemeScript" in layout, (
            "layout.tsx should render ThemeScript in head"
        )


class TestSettingsModalThemeBehavior:
    """Verify SettingsModal properly handles theme selection."""

    def test_settings_modal_has_theme_options(self):
        """SettingsModal should offer light, dark, and system options."""
        src = _read("frontend/src/components/modals/SettingsModal.tsx")
        assert '"light"' in src, "SettingsModal should have light theme option"
        assert '"dark"' in src, "SettingsModal should have dark theme option"
        assert '"system"' in src, "SettingsModal should have system theme option"

    def test_settings_modal_persists_to_localstorage(self):
        """Theme selection should be saved to localStorage."""
        src = _read("frontend/src/components/modals/SettingsModal.tsx")
        assert "localStorage.setItem" in src, (
            "SettingsModal should persist theme to localStorage"
        )
        assert "atlas-theme" in src, (
            "SettingsModal should use 'atlas-theme' localStorage key"
        )

    def test_settings_modal_applies_dark_class(self):
        """Selecting dark theme should add 'dark' class to document."""
        src = _read("frontend/src/components/modals/SettingsModal.tsx")
        assert "classList.add" in src, (
            "SettingsModal should add 'dark' class when dark theme selected"
        )
        assert "classList.remove" in src, (
            "SettingsModal should remove 'dark' class when light theme selected"
        )

    def test_settings_modal_listens_for_system_changes(self):
        """SettingsModal should listen for system preference changes."""
        src = _read("frontend/src/components/modals/SettingsModal.tsx")
        assert "addEventListener" in src and "change" in src, (
            "SettingsModal should listen for media query changes"
        )

    def test_settings_modal_has_dark_mode_styles(self):
        """SettingsModal should have dark: variant classes for its own UI."""
        src = _read("frontend/src/components/modals/SettingsModal.tsx")
        assert "dark:" in src, (
            "SettingsModal should have dark: Tailwind variants for dark mode styling"
        )


class TestAppShellDarkModeStyles:
    """Verify key app shell components have dark mode styling."""

    def test_globals_css_has_dark_body_styles(self):
        """globals.css should style body differently in dark mode."""
        css = _read("frontend/src/app/globals.css")
        assert ".dark body" in css or ".dark ::-webkit-scrollbar" in css, (
            "globals.css should have .dark body or .dark descendant styles"
        )

    def test_globals_css_dark_mode_variables(self):
        """globals.css should define CSS variables for dark mode."""
        css = _read("frontend/src/app/globals.css")
        assert ".dark {" in css, (
            "globals.css should have .dark selector with CSS variables"
        )
        assert "--surface-primary" in css and "--surface-secondary" in css, (
            "globals.css should define surface color variables for theming"
        )

    def test_header_has_dark_mode_styles(self):
        """Header component should have dark: variant classes."""
        src = _read("frontend/src/components/layout/Header.tsx")
        assert "dark:" in src, (
            "Header.tsx should have dark: Tailwind variants"
        )
        assert "dark:bg-navy" in src or "dark:border-navy" in src, (
            "Header should use navy colors for dark mode background/borders"
        )

    def test_footer_has_dark_mode_styles(self):
        """Footer component should have dark: variant classes."""
        src = _read("frontend/src/components/layout/Footer.tsx")
        assert "dark:" in src, (
            "Footer.tsx should have dark: Tailwind variants"
        )

    def test_clinical_card_has_dark_mode_in_css(self):
        """Clinical card component class should have dark mode styles."""
        css = _read("frontend/src/app/globals.css")
        assert ".dark .clinical-card" in css or ":is(.dark .clinical-card)" in css, (
            "globals.css should have dark mode styles for .clinical-card"
        )


class TestThemePersistence:
    """Verify theme persistence behavior is correctly implemented."""

    def test_localstorage_key_is_atlas_theme(self):
        """The localStorage key should be exactly 'atlas-theme'."""
        # Check in both ThemeScript and SettingsModal
        theme_script = _read("frontend/src/components/ThemeScript.tsx")
        settings = _read("frontend/src/components/modals/SettingsModal.tsx")

        assert "'atlas-theme'" in theme_script or '"atlas-theme"' in theme_script, (
            "ThemeScript should use 'atlas-theme' key"
        )
        assert "'atlas-theme'" in settings or '"atlas-theme"' in settings, (
            "SettingsModal should use 'atlas-theme' key"
        )

    def test_theme_values_are_valid(self):
        """Theme values should be 'light', 'dark', or 'system'."""
        src = _read("frontend/src/components/modals/SettingsModal.tsx")
        # Check ThemeMode type definition
        assert 'ThemeMode = "light" | "dark" | "system"' in src or \
               "ThemeMode = 'light' | 'dark' | 'system'" in src, (
            "ThemeMode type should be defined as light | dark | system"
        )


class TestDarkModeAppShellVisibility:
    """
    Verify dark mode visibly affects the app shell.
    Issue: Clicking Dark mode in Settings appeared to do nothing because
    key layout elements lacked dark: variant classes.
    """

    def test_layout_body_has_dark_mode_classes(self):
        """Body in layout.tsx should have dark: variant classes for visible theme change."""
        layout = _read("frontend/src/app/layout.tsx")
        assert "dark:bg-navy" in layout, (
            "layout.tsx body should have dark:bg-navy-* for dark mode background"
        )
        assert "dark:text-gray" in layout, (
            "layout.tsx body should have dark:text-gray-* for dark mode text color"
        )

    def test_main_page_container_has_dark_mode(self):
        """Main page container should have dark mode background."""
        page = _read("frontend/src/app/page.tsx")
        # The main flex container should have dark mode
        assert "dark:bg-navy" in page, (
            "page.tsx main container should have dark:bg-navy-* for dark mode"
        )

    def test_sidebars_have_dark_mode_styles(self):
        """Both left and right sidebars should have dark mode backgrounds."""
        page = _read("frontend/src/app/page.tsx")
        # Count dark:bg-navy occurrences for sidebars (should have multiple)
        dark_bg_count = page.count("dark:bg-navy")
        assert dark_bg_count >= 3, (
            f"page.tsx should have at least 3 dark:bg-navy-* classes for sidebars and containers (found {dark_bg_count})"
        )

    def test_sidebar_borders_have_dark_mode(self):
        """Sidebar borders should have dark mode styling."""
        page = _read("frontend/src/app/page.tsx")
        assert "dark:border-navy" in page, (
            "page.tsx sidebars should have dark:border-navy-* for dark mode borders"
        )

    def test_mobile_panel_tabs_have_dark_mode(self):
        """Mobile panel tabs should have dark mode styling."""
        page = _read("frontend/src/app/page.tsx")
        # MobilePanelTabs function should have dark mode
        assert "dark:bg-navy-800" in page, (
            "Mobile panel tabs should have dark:bg-navy-800 for dark mode"
        )


class TestSettingsOpenHeaderFade:
    """
    Verify header/banner visibly fades when Settings modal is open.
    Issue: While Settings modal is open, header/demo banner still looked too bright.
    """

    def test_header_fades_when_settings_open(self):
        """Header should have opacity fade when settingsOpen is true."""
        src = _read("frontend/src/components/layout/Header.tsx")
        # Header should conditionally apply opacity based on settingsOpen
        assert "settingsOpen && " in src and "opacity" in src, (
            "Header should apply opacity class when settingsOpen is true"
        )

    def test_header_disables_interaction_when_settings_open(self):
        """Header should disable pointer events when settingsOpen is true."""
        src = _read("frontend/src/components/layout/Header.tsx")
        assert "pointer-events-none" in src, (
            "Header should have pointer-events-none when settings modal is open"
        )

    def test_disconnection_banner_fades_when_settings_open(self):
        """Disconnection banner should also fade when settings is open."""
        src = _read("frontend/src/components/layout/Header.tsx")
        # The DisconnectionBanner usage section (in return JSX) should have conditional fade
        # Find the JSX return section which includes the banner wrapper
        banner_usage_start = src.find("{/* Disconnection Banner */}")
        assert banner_usage_start != -1, "Header should have Disconnection Banner comment marker"
        banner_section = src[banner_usage_start:banner_usage_start + 500]
        assert "settingsOpen" in banner_section, (
            "DisconnectionBanner wrapper should reference settingsOpen for fade effect"
        )

    def test_fade_uses_transition_for_smooth_animation(self):
        """Fade effect should use CSS transition for smooth animation."""
        src = _read("frontend/src/components/layout/Header.tsx")
        # Should have transition-all or transition-opacity for smooth fade
        assert "transition-all" in src or "transition-opacity" in src, (
            "Header fade should use CSS transition for smooth animation"
        )
