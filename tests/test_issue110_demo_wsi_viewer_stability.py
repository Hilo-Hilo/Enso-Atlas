from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_issue110_demo_mode_preserves_real_wsi_slide_when_available():
    src = _read("frontend/src/app/page.tsx")

    assert "const fallbackSlide =" in src
    assert "slideListRef.current.find((slide) => slide.hasWsi !== false)" in src
    assert "selectedSlide && selectedSlide.hasWsi !== false ? selectedSlide : null" in src
    assert "if (selectedSlide?.id !== demoViewerSlide.id)" in src


def test_issue110_demo_mode_no_longer_forces_synthetic_slide_id_for_viewer_step():
    src = _read("frontend/src/app/page.tsx")

    assert "setSelectedSlide(DEMO_SLIDE);" not in src
    assert "slideInfo: demoAnalysisSlideInfo" in src


def test_issue110_slide_list_ref_kept_current_for_demo_fallback_selection():
    src = _read("frontend/src/app/page.tsx")

    assert "const slideListRef = useRef<SlideInfo[]>([]);" in src
    assert "slideListRef.current = slideList;" in src
