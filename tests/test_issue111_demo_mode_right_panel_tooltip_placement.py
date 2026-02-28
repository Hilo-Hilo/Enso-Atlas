from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_issue111_right_sidebar_tour_steps_use_left_placement_to_avoid_covering_panels():
    src = _read("frontend/src/components/demo/DemoMode.tsx")

    assert 'target: \'[data-demo="right-tab-prediction"]\'' in src
    assert 'target: \'[data-demo="right-tab-semantic-search"]\'' in src
    assert 'target: \'[data-demo="right-tab-similar-cases"]\'' in src
    assert 'target: \'[data-demo="right-tab-medgemma"]\'' in src

    # Keep tooltips left of right-sidebar tabs to avoid covering the content region.
    assert src.count('placement: "left" as const,') >= 4
    assert 'placement: "auto" as const,' not in src
