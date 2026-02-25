from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_issue91_oncologist_sidebar_includes_semantic_search_tab():
    src = _read("frontend/src/app/page.tsx")

    assert '{ value: "semantic-search", label: "Semantic Search" }' in src


def test_issue91_main_page_connection_recovers_after_idle_events():
    src = _read("frontend/src/app/page.tsx")

    assert 'document.addEventListener("visibilitychange", handleVisibilityChange);' in src
    assert 'window.addEventListener("focus", handleWindowFocus);' in src
    assert 'window.addEventListener("online", handleWindowFocus);' in src
    assert "let failureStreak = 0;" in src
    assert "failureStreak >= 2" in src


def test_issue91_projects_page_connection_recovers_after_idle_events():
    src = _read("frontend/src/app/projects/page.tsx")

    assert 'document.addEventListener("visibilitychange", handleVisibilityChange);' in src
    assert 'window.addEventListener("focus", handleWindowFocus);' in src
    assert 'window.addEventListener("online", handleWindowFocus);' in src
    assert "let failureStreak = 0;" in src
    assert "failureStreak >= 2" in src
