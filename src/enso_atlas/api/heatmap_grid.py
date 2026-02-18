"""Deterministic helpers for heatmap/grid alignment math.

Issue #43 context:
The model heatmap is generated on a patch grid (224px patch size). The grid
coverage can exceed real slide bounds when dimensions are not divisible by 224.
These helpers centralize the ceil-based coverage math to keep backend headers
and frontend overlay scaling coherent.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class HeatmapGridCoverage:
    patch_size: int
    slide_width: int
    slide_height: int
    grid_width: int
    grid_height: int
    coverage_width: int
    coverage_height: int


def compute_heatmap_grid_coverage(
    slide_width: int,
    slide_height: int,
    *,
    patch_size: int = 224,
) -> HeatmapGridCoverage:
    """Compute patch-grid size and pixel coverage for a slide.

    Returns:
      - grid_width/grid_height in patch cells
      - coverage_width/coverage_height in pixels
    """

    if slide_width <= 0 or slide_height <= 0:
        raise ValueError("slide dimensions must be positive")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")

    grid_w = int(math.ceil(slide_width / patch_size))
    grid_h = int(math.ceil(slide_height / patch_size))

    return HeatmapGridCoverage(
        patch_size=patch_size,
        slide_width=int(slide_width),
        slide_height=int(slide_height),
        grid_width=grid_w,
        grid_height=grid_h,
        coverage_width=grid_w * patch_size,
        coverage_height=grid_h * patch_size,
    )


def compute_osd_heatmap_overlay_width(
    *,
    slide_bounds_width: float,
    slide_pixel_width: int,
    coverage_pixel_width: int,
) -> float:
    """Compute overlay width in OSD viewport space.

    Frontend uses: bounds.width * (coverageW / contentSize.x)
    """

    if slide_bounds_width <= 0:
        raise ValueError("slide_bounds_width must be positive")
    if slide_pixel_width <= 0:
        raise ValueError("slide_pixel_width must be positive")
    if coverage_pixel_width <= 0:
        raise ValueError("coverage_pixel_width must be positive")

    return float(slide_bounds_width) * (float(coverage_pixel_width) / float(slide_pixel_width))
