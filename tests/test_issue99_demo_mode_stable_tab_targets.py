from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_issue99_demo_steps_target_stable_right_sidebar_tabs():
    src = _read("frontend/src/components/demo/DemoMode.tsx")

    assert '[data-demo="right-tab-prediction"]' in src
    assert '[data-demo="right-tab-evidence"]' in src
    assert '[data-demo="right-tab-similar-cases"]' in src
    assert '[data-demo="right-tab-medgemma"]' in src

    # Tour should no longer anchor to panel bodies that mount/unmount during tab switches.
    assert '[data-demo="prediction-panel"]' not in src
    assert '[data-demo="evidence-panel"]' not in src
    assert '[data-demo="similar-cases"]' not in src
    assert '[data-demo="report-panel"]' not in src


def test_issue99_right_sidebar_tabs_expose_demo_hooks_for_each_panel_button():
    src = _read("frontend/src/app/page.tsx")

    assert 'data-demo={`right-tab-${opt.value}`}' in src
