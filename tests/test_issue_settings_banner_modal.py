"""
Regression tests for GitHub issue: Settings Banner Modal
https://github.com/Hilo-Hilo/Enso-Atlas/issues/settings-banner-modal

When the Settings modal is open, the demo toggle in the header should appear
disabled/greyed with a visual strike treatment and proper accessibility semantics.
"""
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_demo_toggle_accepts_disabled_prop():
    """DemoToggle component must accept a disabled prop for modal-open state."""
    src = _read("frontend/src/components/demo/DemoMode.tsx")

    # Interface should include disabled prop
    assert "disabled?: boolean" in src, "DemoToggle should accept optional disabled prop"
    # Component should destructure disabled with default false
    assert "disabled = false" in src, "DemoToggle should default disabled to false"


def test_demo_toggle_disabled_applies_accessibility_attributes():
    """Disabled DemoToggle must have aria-disabled and disabled attributes."""
    src = _read("frontend/src/components/demo/DemoMode.tsx")

    assert "aria-disabled={disabled}" in src, "DemoToggle should set aria-disabled"
    assert "disabled={disabled}" in src, "DemoToggle should set native disabled attribute"


def test_demo_toggle_disabled_has_visual_treatment():
    """Disabled DemoToggle must have greyed/muted styling and strike-through."""
    src = _read("frontend/src/components/demo/DemoMode.tsx")

    # Check for disabled-specific styling classes (grey, cursor-not-allowed, opacity)
    assert "cursor-not-allowed" in src, "Disabled state should show not-allowed cursor"
    assert "opacity-60" in src or "opacity-50" in src, "Disabled state should reduce opacity"
    # Check for strike-through visual element
    assert "-rotate-12" in src or "line-through" in src, "Disabled state should have strike visual treatment"


def test_demo_toggle_disabled_prevents_interaction():
    """Disabled DemoToggle must not trigger onToggle."""
    src = _read("frontend/src/components/demo/DemoMode.tsx")

    # onClick should be conditionally disabled
    assert "disabled ? undefined : onToggle" in src, "onClick should be blocked when disabled"


def test_header_passes_settings_open_to_demo_toggle():
    """Header component must pass settingsOpen state to DemoToggle as disabled."""
    src = _read("frontend/src/components/layout/Header.tsx")

    # Desktop DemoToggle
    assert "disabled={settingsOpen}" in src, "Header should pass settingsOpen to DemoToggle"
    # Should appear in both desktop and mobile menu contexts
    occurrences = src.count("disabled={settingsOpen}")
    assert occurrences >= 2, f"settingsOpen should be passed to both desktop and mobile DemoToggle (found {occurrences})"
