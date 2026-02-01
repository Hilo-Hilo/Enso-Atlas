"""
WSI Processing - Tissue detection and patch extraction.

This module handles:
- Loading whole-slide images (SVS, NDPI, MRXS, TIFF)
- Tissue detection using Otsu thresholding
- Patch sampling (grid-based + adaptive refinement)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Optional
import logging

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class WSIConfig:
    """WSI processing configuration."""
    patch_size: int = 224
    magnification: int = 20
    tissue_threshold: float = 0.5
    min_tissue_area: float = 0.1
    max_patches_coarse: int = 2000
    max_patches_refine: int = 8000


class WSIProcessor:
    """
    Processor for whole-slide images.

    Handles loading, tissue detection, and patch extraction from WSIs.
    Uses OpenSlide for broad format support.
    """

    def __init__(self, config: WSIConfig):
        self.config = config
        self._slide = None
        self._current_path = None

        # Lazy import OpenSlide
        try:
            import openslide
            self._openslide = openslide
        except ImportError:
            logger.warning("OpenSlide not installed. WSI support limited.")
            self._openslide = None

    def load_slide(self, path: str | Path) -> None:
        """Load a whole-slide image."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Slide not found: {path}")

        if self._openslide is None:
            raise ImportError("OpenSlide required for WSI loading")

        self._slide = self._openslide.OpenSlide(str(path))
        self._current_path = path
        logger.info(f"Loaded slide: {path.name} ({self._slide.dimensions})")

    def get_slide_dimensions(self, path: Optional[str | Path] = None) -> Tuple[int, int]:
        """Get slide dimensions at level 0."""
        if path is not None:
            self.load_slide(path)
        if self._slide is None:
            raise RuntimeError("No slide loaded")
        return self._slide.dimensions

    def get_thumbnail(self, size: Tuple[int, int] = (1024, 1024)) -> np.ndarray:
        """Get a thumbnail of the slide."""
        if self._slide is None:
            raise RuntimeError("No slide loaded")
        thumb = self._slide.get_thumbnail(size)
        return np.array(thumb)

    def detect_tissue(self, thumbnail: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Detect tissue regions using Otsu thresholding.

        Returns:
            Binary mask where 1 = tissue, 0 = background
        """
        import cv2

        if thumbnail is None:
            thumbnail = self.get_thumbnail()

        # Convert to grayscale
        if len(thumbnail.shape) == 3:
            gray = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2GRAY)
        else:
            gray = thumbnail

        # Otsu thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Invert (tissue is typically darker)
        binary = 255 - binary

        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Convert to 0-1 mask
        mask = (binary > 0).astype(np.uint8)

        tissue_ratio = np.sum(mask) / mask.size
        logger.info(f"Tissue detection: {tissue_ratio:.1%} coverage")

        return mask

    def sample_patches(
        self,
        tissue_mask: np.ndarray,
        max_patches: int,
        attention_weights: Optional[np.ndarray] = None,
    ) -> List[Tuple[int, int]]:
        """
        Sample patch coordinates from tissue regions.

        Args:
            tissue_mask: Binary mask of tissue regions
            max_patches: Maximum number of patches to sample
            attention_weights: Optional weights for adaptive sampling

        Returns:
            List of (x, y) coordinates at level 0
        """
        if self._slide is None:
            raise RuntimeError("No slide loaded")

        slide_w, slide_h = self._slide.dimensions
        mask_h, mask_w = tissue_mask.shape

        # Calculate scaling factor
        scale_x = slide_w / mask_w
        scale_y = slide_h / mask_h

        # Calculate patch grid
        patch_size = self.config.patch_size
        step = patch_size  # Non-overlapping for now

        grid_x = np.arange(0, mask_w, step // scale_x)
        grid_y = np.arange(0, mask_h, step // scale_y)

        # Find valid patch locations (tissue present)
        valid_coords = []
        for y in grid_y:
            for x in grid_x:
                y_int, x_int = int(y), int(x)
                if y_int < mask_h and x_int < mask_w:
                    # Check tissue coverage in this region
                    region_size = max(1, int(step // scale_x))
                    region = tissue_mask[
                        y_int:min(y_int + region_size, mask_h),
                        x_int:min(x_int + region_size, mask_w)
                    ]
                    if region.size > 0 and np.mean(region) > self.config.tissue_threshold:
                        # Convert to slide coordinates
                        slide_x = int(x * scale_x)
                        slide_y = int(y * scale_y)
                        valid_coords.append((slide_x, slide_y))

        logger.info(f"Found {len(valid_coords)} valid patch locations")

        # Subsample if too many
        if len(valid_coords) > max_patches:
            if attention_weights is not None:
                # Weighted sampling based on attention
                probs = attention_weights / attention_weights.sum()
                indices = np.random.choice(
                    len(valid_coords), size=max_patches, replace=False, p=probs
                )
            else:
                # Random sampling
                indices = np.random.choice(
                    len(valid_coords), size=max_patches, replace=False
                )
            valid_coords = [valid_coords[i] for i in indices]

        return valid_coords

    def extract_patch(self, x: int, y: int, level: int = 0) -> np.ndarray:
        """Extract a single patch from the slide."""
        if self._slide is None:
            raise RuntimeError("No slide loaded")

        patch_size = self.config.patch_size
        region = self._slide.read_region((x, y), level, (patch_size, patch_size))

        # Convert RGBA to RGB
        patch = np.array(region.convert("RGB"))
        return patch

    def extract_patches(
        self,
        path: str | Path,
        refine_with_attention: bool = False,
        attention_weights: Optional[np.ndarray] = None,
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """
        Extract all patches from a slide.

        Args:
            path: Path to the WSI file
            refine_with_attention: Whether to do adaptive refinement
            attention_weights: Attention weights for refinement

        Returns:
            Tuple of (patches, coordinates)
        """
        self.load_slide(path)

        # Get thumbnail and detect tissue
        thumbnail = self.get_thumbnail()
        tissue_mask = self.detect_tissue(thumbnail)

        # Sample patch coordinates
        max_patches = self.config.max_patches_coarse
        if refine_with_attention and attention_weights is not None:
            max_patches = self.config.max_patches_refine

        coords = self.sample_patches(tissue_mask, max_patches, attention_weights)

        # Extract patches
        patches = []
        for x, y in coords:
            try:
                patch = self.extract_patch(x, y)
                patches.append(patch)
            except Exception as e:
                logger.warning(f"Failed to extract patch at ({x}, {y}): {e}")

        logger.info(f"Extracted {len(patches)} patches")
        return patches, coords

    def close(self) -> None:
        """Close the current slide."""
        if self._slide is not None:
            self._slide.close()
            self._slide = None
            self._current_path = None
