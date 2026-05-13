from pathlib import Path

import numpy as np

from src.enso_atlas.api.patch_coordinates import (
    infer_patch_size_from_coords,
    normalize_coords_to_level0,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_infer_patch_size_from_coords_handles_dense_and_low_mag_grids():
    dense = np.array([[0, 0], [224, 0], [448, 0], [0, 224]], dtype=np.int64)
    low_mag = np.array([[0, 0], [448, 0], [896, 0], [0, 448]], dtype=np.int64)

    assert infer_patch_size_from_coords(dense) == 224
    assert infer_patch_size_from_coords(low_mag) == 448


def test_normalize_coords_to_level0_upscales_when_slide_coverage_matches():
    # Simulates historical level-1 coords for a slide that should cover 1344x1344 at level-0.
    coords = np.array([[0, 0], [224, 224], [448, 448]], dtype=np.int64)

    normalized, scale = normalize_coords_to_level0(coords, slide_dims=(1344, 1344), patch_size=224)

    assert scale == 2
    assert normalized is not None
    assert normalized.tolist() == [[0, 0], [448, 448], [896, 896]]


def test_normalize_coords_to_level0_keeps_localized_level0_coords_unchanged():
    coords = np.array([[1000, 1000], [1224, 1224], [1448, 1448]], dtype=np.int64)

    normalized, scale = normalize_coords_to_level0(
        coords, slide_dims=(10000, 10000), patch_size=224
    )

    assert scale == 1
    assert normalized is not None
    assert np.array_equal(normalized, coords)
