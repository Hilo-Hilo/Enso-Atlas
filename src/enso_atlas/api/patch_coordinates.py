"""Patch coordinate inference and normalization helpers."""

from __future__ import annotations

import numpy as np


def infer_patch_size_from_coords(
    coords: np.ndarray | None,
    *,
    default_patch_size: int = 224,
    max_patch_size: int = 2048,
) -> int:
    """Infer patch span in level-0 pixels from coordinate spacing."""
    if coords is None:
        return default_patch_size

    arr = np.asarray(coords)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
        return default_patch_size

    min_deltas: list[int] = []
    for axis in (0, 1):
        unique_vals = np.unique(arr[:, axis].astype(np.int64, copy=False))
        if unique_vals.size < 2:
            continue
        deltas = np.diff(unique_vals)
        deltas = deltas[deltas > 0]
        if deltas.size > 0:
            min_deltas.append(int(deltas.min()))

    if not min_deltas:
        return default_patch_size

    inferred = max(default_patch_size, min(min_deltas))
    return max(default_patch_size, min(inferred, max_patch_size))


def normalize_coords_to_level0(
    coords: np.ndarray | None,
    *,
    slide_dims: tuple[int, int] | None,
    patch_size: int = 224,
) -> tuple[np.ndarray | None, int]:
    """Normalize cached coordinates to level-0 pixel space when needed."""
    if coords is None:
        return None, 1

    arr = np.asarray(coords)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
        return arr, 1

    if not slide_dims or slide_dims[0] <= 0 or slide_dims[1] <= 0:
        return arr, 1

    work = arr[:, :2].astype(np.int64, copy=False)
    max_x = int(work[:, 0].max()) if work.size else 0
    max_y = int(work[:, 1].max()) if work.size else 0
    slide_w, slide_h = int(slide_dims[0]), int(slide_dims[1])

    if max_x <= 0 or max_y <= 0:
        return arr, 1

    def coverage_error(scale: int) -> float:
        cov_x = (max_x + patch_size) * scale
        cov_y = (max_y + patch_size) * scale
        err_x = abs(cov_x - slide_w) / max(slide_w, 1)
        err_y = abs(cov_y - slide_h) / max(slide_h, 1)
        return (err_x + err_y) / 2.0

    base_error = coverage_error(1)
    raw_cov_x = (max_x + patch_size) / max(slide_w, 1)
    raw_cov_y = (max_y + patch_size) / max(slide_h, 1)

    best_scale = 1
    best_error = base_error
    for scale in (2, 4, 8, 16):
        err = coverage_error(scale)
        if err < best_error:
            best_scale = scale
            best_error = err

    if (
        best_scale > 1
        and (base_error - best_error) >= 0.2
        and best_error <= 0.35
        and max(raw_cov_x, raw_cov_y) >= 0.2
    ):
        scaled = arr.astype(np.int64, copy=True)
        scaled[:, 0] *= best_scale
        scaled[:, 1] *= best_scale
        return scaled, best_scale

    return arr, 1
