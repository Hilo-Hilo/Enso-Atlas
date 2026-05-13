"""Semantic-search execution planning helpers."""

from __future__ import annotations


def resolve_semantic_siglip_plan(
    *,
    has_cached_siglip: bool,
    allow_on_the_fly: bool,
    patch_count: int | None,
    max_patches: int,
) -> tuple[str, str]:
    """Return cache/on-the-fly/fallback strategy for SigLIP retrieval."""
    if has_cached_siglip:
        return "cache", "cache-hit"

    if not allow_on_the_fly:
        return "fallback", "on-the-fly-disabled"

    safe_limit = max(1, int(max_patches))

    if patch_count is None:
        return "fallback", "missing-coordinates"

    try:
        patch_count_int = int(patch_count)
    except (TypeError, ValueError):
        return "fallback", "invalid-patch-count"

    if patch_count_int <= 0:
        return "fallback", "empty-coordinates"

    if patch_count_int > safe_limit:
        return "fallback", "too-many-patches"

    return "on-the-fly", "eligible"
