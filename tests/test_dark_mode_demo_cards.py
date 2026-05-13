from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _src() -> str:
    return (REPO_ROOT / "frontend/src/components/demo/DemoMode.tsx").read_text(encoding="utf-8")


def test_custom_tooltip_supports_dark_shell_and_text():
    src = _src()
    assert "bg-white dark:bg-navy-800" in src
    assert "dark:border-navy-600" in src
    assert "text-gray-900 dark:text-gray-100" in src
    assert "text-gray-600 dark:text-gray-300" in src


def test_custom_tooltip_supports_dark_feature_block_and_back_button():
    src = _src()
    assert "bg-gray-50 dark:bg-navy-700/60" in src
    assert "dark:bg-clinical-900/40" in src
    assert "dark:text-clinical-300" in src
    assert "dark:bg-navy-700" in src
    assert "dark:hover:bg-navy-600" in src


def test_joyride_arrow_and_background_switch_in_dark_mode():
    src = _src()
    assert "const isDarkTheme" in src
    assert 'arrowColor: isDarkTheme ? "#1e293b" : "#fff"' in src
    assert 'backgroundColor: isDarkTheme ? "#1e293b" : "#fff"' in src


def test_welcome_modal_cards_and_text_have_dark_variants():
    src = _src()
    assert "relative bg-white dark:bg-navy-800" in src
    assert "bg-gray-50 dark:bg-navy-700/60" in src
    assert "text-gray-500 dark:text-gray-400" in src


def test_demo_step_copy_has_dark_secondary_text_variants():
    src = _src()
    assert "text-sm text-gray-500 dark:text-gray-400" in src
    assert "text-amber-600 dark:text-amber-300 text-sm" in src
    assert "text-green-600 dark:text-green-300 font-medium" in src
