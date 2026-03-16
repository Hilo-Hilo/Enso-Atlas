from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


def test_header_uses_theme_aware_light_mode_nav_tokens():
    src = _read("frontend/src/components/layout/Header.tsx")
    assert "bg-white/90 text-sky-900" in src, (
        "Header nav controls should use readable light-mode surfaces instead of white text on light backgrounds"
    )
    assert "dark:bg-gradient-to-r dark:from-clinical-500 dark:to-clinical-600 dark:text-white" in src, (
        "Header nav controls should preserve the stronger branded dark-mode treatment"
    )


def test_demo_toggle_inactive_state_is_readable_in_light_mode():
    src = _read("frontend/src/components/demo/DemoMode.tsx")
    assert "border border-sky-200 bg-white/90 text-sky-900" in src, (
        "Inactive demo toggle should be readable in light mode"
    )
    assert "dark:bg-gradient-to-r dark:from-clinical-500 dark:to-clinical-600 dark:text-white" in src, (
        "Inactive demo toggle should retain the dark-mode branded styling"
    )
