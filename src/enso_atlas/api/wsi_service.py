"""Cached WSI and DeepZoom loading service."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any


class WsiService:
    """Load and cache OpenSlide/DeepZoom objects by project-scoped slide ID."""

    def __init__(
        self,
        *,
        resolve_slide_path: Callable[[str, str | None], Path | None],
        logger: logging.Logger,
    ):
        self._resolve_slide_path = resolve_slide_path
        self._logger = logger
        self._cache: dict[str, Any] = {}

    def get_slide_and_dz(self, slide_id: str, project_id: str | None = None):
        """Get or create cached OpenSlide/DeepZoom objects for a slide."""
        cache_key = f"{project_id or '__any__'}::{slide_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        slide_path = self._resolve_slide_path(slide_id, project_id)

        if slide_path is None:
            self._logger.warning("WSI file not found for slide_id=%s", slide_id)
            return None

        try:
            import openslide
            from openslide.deepzoom import DeepZoomGenerator

            slide = openslide.OpenSlide(str(slide_path))
            dz = DeepZoomGenerator(slide, tile_size=254, overlap=1, limit_bounds=True)
            self._cache[cache_key] = (slide, dz)
            self._logger.info("Loaded WSI: %s", slide_path)
            return slide, dz
        except Exception as e:
            self._logger.error("Failed to load WSI %s: %s", slide_path, e)
            return None

    def clear(self) -> None:
        self._cache.clear()
