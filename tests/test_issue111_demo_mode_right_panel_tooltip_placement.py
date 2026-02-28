from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_issue111_right_sidebar_tour_steps_use_left_start_placement_to_avoid_flip_overlap():
    src = _read("frontend/src/components/demo/DemoMode.tsx")

    assert 'target: \'[data-demo="right-tab-prediction"]\'' in src
    assert 'target: \'[data-demo="right-tab-semantic-search"]\'' in src
    assert 'target: \'[data-demo="right-tab-similar-cases"]\'' in src
    assert 'target: \'[data-demo="right-tab-medgemma"]\'' in src

    # Use left-start (not plain left) so floater keeps the tooltip aligned near
    # the top tab row and avoids auto-flipping to top/center over panel content.
    assert src.count('placement: "left-start" as const,') >= 4
    assert 'placement: "left" as const,' not in src
    assert 'placement: "auto" as const,' not in src
