from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_issue96_demo_mode_starts_after_first_target_not_all_targets():
    src = _read("frontend/src/components/demo/DemoMode.tsx")

    assert "const firstStepSelector = getStepSelector(0);" in src
    assert "tourSteps.every" not in src


def test_issue96_demo_mode_uses_bounded_single_lane_target_retry():
    src = _read("frontend/src/components/demo/DemoMode.tsx")

    assert "const MAX_TARGET_RETRY_ATTEMPTS = 8;" in src
    assert "if (activeRetry.step === missingStep && activeRetry.timer)" in src
    assert "if (retryState.attempts >= MAX_TARGET_RETRY_ATTEMPTS)" in src
    assert "if (type === EVENTS.STEP_BEFORE)" in src


def test_issue96_page_wires_demo_step_preconditions_and_panel_mapping():
    src = _read("frontend/src/app/page.tsx")

    assert "const DEMO_RIGHT_PANEL_BY_STEP" in src
    assert '3: "prediction"' in src
    assert '4: "evidence"' in src
    assert '5: "similar-cases"' in src
    assert '6: "medgemma"' in src
    assert "leftPanelRef.current?.expand?.();" in src
    assert "onStepChange={handleDemoStepChange}" in src


def test_issue96_analysis_button_has_demo_target_selector():
    src = _read("frontend/src/components/panels/AnalysisControls.tsx")

    assert 'data-demo="analyze-button"' in src
