from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_issue98_demo_mode_defers_step_transition_after_step_after_event():
    src = _read("frontend/src/components/demo/DemoMode.tsx")

    assert "requestAnimationFrame(() => setTourStep(currentIndex + 1));" in src
    assert "requestAnimationFrame(() => setTourStep(currentIndex - 1));" in src


def test_issue98_demo_mode_disables_joyride_scrolling_side_effects():
    src = _read("frontend/src/components/demo/DemoMode.tsx")

    assert "disableScrolling" in src
    assert "disableScrollParentFix" in src
    assert "disableAnimation: true" in src


def test_issue98_set_tour_step_updates_index_ref_without_direct_onstepchange_call():
    src = _read("frontend/src/components/demo/DemoMode.tsx")

    assert "stepIndexRef.current = bounded;" in src
    assert "const setTourStep = useCallback((nextStep: number) => {" in src
