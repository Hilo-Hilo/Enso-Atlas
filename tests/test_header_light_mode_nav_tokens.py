from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


def test_header_uses_theme_aware_nav_tokens_for_light_and_dark_modes():
    src = _read("frontend/src/components/layout/Header.tsx")
    assert "bg-white/90 text-sky-900" in src, (
        "Header nav controls should use readable light-mode surfaces instead of white text on light backgrounds"
    )
    assert "dark:border-navy-600 dark:bg-navy-800/90 dark:text-gray-100" in src, (
        "Header nav controls should switch to dark surfaces in dark mode"
    )


def test_demo_toggle_inactive_state_is_theme_aware():
    src = _read("frontend/src/components/demo/DemoMode.tsx")
    assert "border border-sky-200 bg-white/90 text-sky-900" in src, (
        "Inactive demo toggle should be readable in light mode"
    )
    assert "dark:border-navy-600 dark:bg-navy-800/90 dark:text-gray-100" in src, (
        "Inactive demo toggle should switch to dark surfaces in dark mode"
    )
