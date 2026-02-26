from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _src() -> str:
    return (REPO_ROOT / "frontend/src/components/layout/Header.tsx").read_text(encoding="utf-8")


def test_top_floating_banners_defined():
    src = _src()
    assert "const topFloatingBanners: FloatingBannerItem[]" in src
    assert 'id: "research-preview"' in src
    assert 'id: "dgx-capacity"' in src


def test_research_preview_uses_project_disclaimer_with_fallback():
    src = _src()
    assert "currentProject?.disclaimer?.trim()" in src
    assert "research-only" in src


def test_dgx_capacity_notice_message_present():
    src = _src()
    assert "NVIDIA DGX Spark" in src
    assert "avoid stress testing" in src
    assert "strict permissions" in src


def test_floating_container_is_fixed_and_high_z_index():
    src = _src()
    assert "fixed top-2 left-1/2 z-[260]" in src


def test_cross_out_behavior_is_implemented_per_banner():
    src = _src()
    assert "handleCrossOutBanner" in src
    assert "Cross out banner" in src
    assert "before:bg-red-500/80" in src


def test_dismissing_one_banner_reflows_remaining_banner_stack():
    src = _src()
    assert "visibleTopBanners = topFloatingBanners.filter(" in src
    assert "!dismissedTopBanners.includes(banner.id)" in src
