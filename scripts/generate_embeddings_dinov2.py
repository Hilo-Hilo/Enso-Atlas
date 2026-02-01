#!/usr/bin/env python3
"""Generate embeddings for WSIs using DINOv2."""

import os
import sys
from pathlib import Path
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from enso_atlas.embedding.embedder_dinov2 import DINOv2Embedder


def extract_patches_simple(slide_path: Path, patch_size: int = 224, max_patches: int = 500):
    """Simple patch extraction from WSI."""
    import openslide
    import cv2
    from PIL import Image
    
    try:
        slide = openslide.OpenSlide(str(slide_path))
    except Exception as e:
        logger.error(f"Could not open slide {slide_path}: {e}")
        return [], []
    
    # Get dimensions at level 0
    width, height = slide.dimensions
    
    # Use a lower resolution level if available
    level = min(1, slide.level_count - 1)
    level_dims = slide.level_dimensions[level]
    downsample = slide.level_downsamples[level]
    
    logger.info(f"Slide dimensions: {width}x{height}, using level {level} ({level_dims})")
    
    patches = []
    coords = []
    
    # Calculate step size
    step = int(patch_size * downsample)
    
    # Sample patches
    for y in range(0, height - step, step * 4):  # Skip every 4 patches for speed
        for x in range(0, width - step, step * 4):
            if len(patches) >= max_patches:
                break
            
            try:
                # Read region
                region = slide.read_region((x, y), level, (patch_size, patch_size))
                region = region.convert("RGB")
                patch = np.array(region)
                
                # Simple tissue detection (not too white, not too dark)
                gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
                mean_val = np.mean(gray)
                std_val = np.std(gray)
                
                if 50 < mean_val < 220 and std_val > 20:
                    patches.append(patch)
                    coords.append((x, y))
            except Exception as e:
                continue
        
        if len(patches) >= max_patches:
            break
    
    slide.close()
    logger.info(f"Extracted {len(patches)} patches from {slide_path.name}")
    return patches, coords


def main():
    slides_dir = Path("data/slides")
    output_dir = Path("data/embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize embedder
    embedder = DINOv2Embedder(cache_dir=str(output_dir), batch_size=32)
    
    # Find all slides
    slides = list(slides_dir.glob("*.svs"))
    logger.info(f"Found {len(slides)} slides")
    
    for slide_path in slides:
        output_path = output_dir / f"{slide_path.stem}.npy"
        coords_path = output_dir / f"{slide_path.stem}_coords.npy"
        
        if output_path.exists():
            logger.info(f"Skipping {slide_path.name} (already processed)")
            continue
        
        logger.info(f"Processing {slide_path.name}...")
        
        # Extract patches
        patches, coords = extract_patches_simple(slide_path)
        
        if len(patches) == 0:
            logger.warning(f"No patches extracted from {slide_path.name}")
            continue
        
        # Generate embeddings
        embeddings = embedder.embed(patches, show_progress=True)
        
        # Save
        np.save(output_path, embeddings)
        np.save(coords_path, np.array(coords))
        
        logger.info(f"Saved embeddings {embeddings.shape} to {output_path}")
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
