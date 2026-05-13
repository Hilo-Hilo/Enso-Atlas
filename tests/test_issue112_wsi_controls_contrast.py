from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_issue112_wsi_overlay_titles_and_zoom_use_high_contrast_text():
    src = _read("frontend/src/components/viewer/WSIViewer.tsx")

    # Zoom factor in top toolbar should remain high contrast in dark mode.
    assert "text-sm font-mono font-semibold tracking-tight text-gray-900 dark:text-gray-50" in src
    assert "text-xs font-mono text-gray-700" not in src

    # Overlay toggle/title row should stay readable on dark translucent toolbars.
    assert 'className="flex items-center gap-2 text-gray-900 dark:text-gray-100"' in src

    # Section titles for the right-side hotbars should stay bright in dark mode.
    assert src.count("text-sm font-semibold text-gray-900 dark:text-gray-100") >= 2


def test_issue112_slider_label_and_value_support_dark_mode_contrast():
    src = _read("frontend/src/components/ui/Slider.tsx")

    assert "text-sm font-medium text-gray-800 dark:text-gray-100" in src
    assert "text-sm text-gray-600 dark:text-gray-200 font-mono" in src
