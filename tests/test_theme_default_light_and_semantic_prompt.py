"""
Regression tests for UX polish updates:
1) Light mode should be default on first visit.
2) Header should expose a top-level dark/light toggle.
3) Semantic search preconditions should use clearer copy.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


class TestLightModeDefaultBehavior:
    def test_theme_script_defaults_to_light_on_first_visit(self):
        src = _read("frontend/src/components/ThemeScript.tsx")
        assert "(!theme && prefersDark)" not in src, (
            "ThemeScript should not auto-enable dark mode when no preference is saved"
        )
        assert "theme === 'dark' || (theme === 'system' && prefersDark)" in src, (
            "ThemeScript should only enable dark mode for explicit dark/system preferences"
        )

    def test_settings_modal_defaults_to_light(self):
        src = _read("frontend/src/components/modals/SettingsModal.tsx")
        assert 'useState<ThemeMode>("light")' in src, (
            "Settings modal should default local state to light"
        )
        assert 'savedTheme || "light"' in src, (
            "Settings modal should fall back to light when no saved theme exists"
        )
        assert 'setTheme("light")' in src and 'applyTheme("light")' in src, (
            "Reset to defaults should reset to light"
        )


class TestHeaderThemeToggle:
    def test_header_has_quick_theme_toggle_handler(self):
        src = _read("frontend/src/components/layout/Header.tsx")
        assert "const handleThemeToggle = () =>" in src, (
            "Header should include a quick theme toggle handler"
        )
        assert 'localStorage.setItem("atlas-theme"' in src, (
            "Header theme toggle should persist atlas-theme preference"
        )

    def test_header_renders_theme_toggle_button(self):
        src = _read("frontend/src/components/layout/Header.tsx")
        assert "Switch to light mode" in src and "Switch to dark mode" in src, (
            "Theme toggle button should expose accessible mode-switch labels"
        )
        assert '<Sun className="h-4 w-4" />' in src and '<Moon className="h-4 w-4" />' in src, (
            "Theme toggle button should render sun/moon icons"
        )


class TestSemanticSearchPreconditionCopy:
    def test_semantic_search_prompts_for_patient_selection_first(self):
        src = _read("frontend/src/components/panels/SemanticSearchPanel.tsx")
        assert "Select a patient to start" in src, (
            "Semantic search should prompt to select a patient before analysis"
        )
        assert "Choose a case from the left panel to start MedSigLIP semantic search." in src, (
            "Semantic search should show patient-selection guidance"
        )

    def test_semantic_search_uses_analysis_prompt_after_selection(self):
        src = _read("frontend/src/components/panels/SemanticSearchPanel.tsx")
        assert "Run analysis to enable search" in src, (
            "Semantic search should explicitly ask user to run analysis once a slide is selected"
        )
        assert (
            "Run analysis on this slide to enable MedSigLIP semantic search by description." in src
        ), "Semantic search should include clear post-selection guidance"
