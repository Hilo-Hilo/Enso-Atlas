from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_issue103_last_step_next_closes_demo_in_controlled_mode():
    src = _read("frontend/src/components/demo/DemoMode.tsx")

    assert "const nextStep = currentIndex + 1;" in src
    assert "if (nextStep >= tourSteps.length)" in src
    assert "setRun(false);" in src
    assert "onClose();" in src


def test_issue103_primes_next_step_state_before_advancing_index():
    src = _read("frontend/src/components/demo/DemoMode.tsx")

    assert "onStepChange?.(nextStep);" in src
    assert "requestAnimationFrame(() => setTourStep(currentIndex + 1));" in src


def test_issue103_primes_prev_step_state_before_rewinding_index():
    src = _read("frontend/src/components/demo/DemoMode.tsx")

    assert "const prevStep = currentIndex - 1;" in src
    assert "onStepChange?.(prevStep);" in src
    assert "requestAnimationFrame(() => setTourStep(currentIndex - 1));" in src
