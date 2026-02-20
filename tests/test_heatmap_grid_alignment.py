"""Regression checks for issue #43 (heatmap-grid alignment)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HEATMAP_GRID_PATH = REPO_ROOT / "src" / "enso_atlas" / "api" / "heatmap_grid.py"
_spec = importlib.util.spec_from_file_location("heatmap_grid", HEATMAP_GRID_PATH)
assert _spec and _spec.loader, "Failed to load heatmap_grid module"
_heatmap_grid = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _heatmap_grid
_spec.loader.exec_module(_heatmap_grid)

compute_heatmap_grid_coverage = _heatmap_grid.compute_heatmap_grid_coverage
compute_osd_heatmap_overlay_width = _heatmap_grid.compute_osd_heatmap_overlay_width


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_grid_coverage_uses_ceil_patch_math_for_non_divisible_slide_dims():
    coverage = compute_heatmap_grid_coverage(1000, 500, patch_size=224)

    assert coverage.grid_width == 5
    assert coverage.grid_height == 3
    assert coverage.coverage_width == 1120
    assert coverage.coverage_height == 672


def test_grid_coverage_exact_multiple_does_not_overshoot():
    coverage = compute_heatmap_grid_coverage(448, 672, patch_size=224)

    assert coverage.grid_width == 2
    assert coverage.grid_height == 3
    assert coverage.coverage_width == 448
    assert coverage.coverage_height == 672


def test_osd_overlay_width_scales_to_grid_coverage_ratio():
    # Mirrors WSIViewer formula: bounds.width * (coverageW / contentSize.x)
    overlay_width = compute_osd_heatmap_overlay_width(
        slide_bounds_width=1.0,
        slide_pixel_width=1000,
        coverage_pixel_width=1120,
    )

    assert abs(overlay_width - 1.12) < 1e-9


def test_frontend_alignment_formula_present_in_viewer_regression_contract():
    src = _read("frontend/src/components/viewer/WSIViewer.tsx")

    assert "const fallbackCoverageW = Math.ceil(contentSize.x / DEFAULT_PATCH_SIZE_PX) * DEFAULT_PATCH_SIZE_PX;" in src
    assert "const widthScale = bounds.width / contentSize.x;" in src
    assert "const heatmapWorldWidth = coverageW * widthScale;" in src


def test_backend_model_heatmap_sets_coverage_headers_from_shared_helper():
    src = _read("src/enso_atlas/api/main.py")

    assert "from .heatmap_grid import compute_heatmap_grid_coverage" in src
    assert '"X-Coverage-Width": str(_coverage.coverage_width)' in src
    assert '"X-Coverage-Height": str(_coverage.coverage_height)' in src


def test_pdf_heatmap_path_does_not_fabricate_synthetic_coords_when_missing():
    src = _read("src/enso_atlas/api/main.py")

    assert "cannot generate truthful PDF heatmap" in src
