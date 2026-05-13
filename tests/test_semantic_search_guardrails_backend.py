from __future__ import annotations

from pathlib import Path

from src.enso_atlas.api.semantic_guardrails import resolve_semantic_siglip_plan

REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = REPO_ROOT / "src" / "enso_atlas" / "api" / "main.py"


def _read_main() -> str:
    return MAIN_PATH.read_text(encoding="utf-8")


def _read_api(filename: str) -> str:
    return (REPO_ROOT / "src" / "enso_atlas" / "api" / filename).read_text(encoding="utf-8")


def test_semantic_siglip_guardrails_default_to_disabled_and_bounded_patch_limit():
    src = _read_main()

    assert "semantic_allow_on_the_fly_siglip = _env_flag(" in src
    allow_block = src.split("semantic_allow_on_the_fly_siglip = _env_flag(", 1)[1].split(")", 1)[0]
    assert '"ENSO_SEMANTIC_ALLOW_ON_THE_FLY_SIGLIP"' in allow_block
    assert "default=False" in allow_block

    assert "semantic_on_the_fly_max_patches = _env_int(" in src
    patch_limit_block = src.split("semantic_on_the_fly_max_patches = _env_int(", 1)[1].split(
        ")", 1
    )[0]
    assert '"ENSO_SEMANTIC_ON_THE_FLY_MAX_PATCHES"' in patch_limit_block
    assert "default=1024" in patch_limit_block
    assert "minimum=256" in patch_limit_block
    assert "maximum=4096" in patch_limit_block


def test_resolve_semantic_siglip_plan_prefers_fallback_when_guardrails_trigger():
    assert resolve_semantic_siglip_plan(
        has_cached_siglip=True,
        allow_on_the_fly=False,
        patch_count=None,
        max_patches=1024,
    ) == ("cache", "cache-hit")

    assert resolve_semantic_siglip_plan(
        has_cached_siglip=False,
        allow_on_the_fly=False,
        patch_count=512,
        max_patches=1024,
    ) == ("fallback", "on-the-fly-disabled")

    assert resolve_semantic_siglip_plan(
        has_cached_siglip=False,
        allow_on_the_fly=True,
        patch_count=None,
        max_patches=1024,
    ) == ("fallback", "missing-coordinates")

    assert resolve_semantic_siglip_plan(
        has_cached_siglip=False,
        allow_on_the_fly=True,
        patch_count=0,
        max_patches=1024,
    ) == ("fallback", "empty-coordinates")

    assert resolve_semantic_siglip_plan(
        has_cached_siglip=False,
        allow_on_the_fly=True,
        patch_count=5000,
        max_patches=1024,
    ) == ("fallback", "too-many-patches")

    assert resolve_semantic_siglip_plan(
        has_cached_siglip=False,
        allow_on_the_fly=True,
        patch_count=512,
        max_patches=1024,
    ) == ("on-the-fly", "eligible")


def test_semantic_search_endpoint_uses_guardrail_plan_and_reasoned_fallback_logging():
    src = _read_api("semantic_search_routes.py")

    assert "plan_mode, plan_reason = resolve_semantic_siglip_plan(" in src
    assert "Failed to load MedSigLIP cache for %s (%s); using fallback." in src
    assert "No MedSigLIP embeddings available for %s (reason=%s), using fallback" in src
