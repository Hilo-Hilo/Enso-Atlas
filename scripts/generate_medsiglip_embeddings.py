#!/usr/bin/env python3
"""
Generate MedSigLIP embeddings for all slides.

Pre-computes MedSigLIP patch embeddings to enable fast text-to-patch 
semantic search at runtime.

Usage:
    python scripts/generate_medsiglip_embeddings.py [--slides-dir PATH] [--force]
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.enso_atlas.embedding.medsiglip import MedSigLIPEmbedder, MedSigLIPConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_patches_from_slide(slide_path: Path, coords_path: Path, max_patches: int = 5000) -> list:
    """Load patches from a slide file using coordinates."""
    import openslide
    from PIL import Image
    
    if not slide_path.exists():
        logger.warning(f"Slide not found: {slide_path}")
        return []
    
    if not coords_path.exists():
        logger.warning(f"Coordinates not found: {coords_path}")
        return []
    
    coords = np.load(coords_path)
    patches = []
    
    try:
        slide = openslide.OpenSlide(str(slide_path))
        
        # Subsample if too many patches
        if len(coords) > max_patches:
            indices = np.random.choice(len(coords), max_patches, replace=False)
            coords = coords[indices]
        
        for x, y in coords:
            try:
                patch = slide.read_region((int(x), int(y)), 0, (224, 224))
                patch = patch.convert('RGB')
                patches.append(np.array(patch))
            except Exception as e:
                logger.debug(f"Failed to read patch at ({x}, {y}): {e}")
        
        slide.close()
        
    except Exception as e:
        logger.error(f"Failed to open slide {slide_path}: {e}")
    
    return patches


def generate_embeddings_from_pf(
    pf_embeddings_dir: Path,
    slides_dir: Path,
    output_dir: Path,
    embedder: MedSigLIPEmbedder,
    force: bool = False,
    max_patches: int = 5000,
):
    """
    Generate MedSigLIP embeddings for slides that have Path Foundation embeddings.
    
    If slide file exists, extracts patches and embeds them.
    If only PF embeddings exist (no slide), cannot generate MedSigLIP embeddings.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all slides with PF embeddings
    pf_files = list(pf_embeddings_dir.glob('*.npy'))
    pf_files = [f for f in pf_files if not f.name.endswith('_coords.npy')]
    
    logger.info(f"Found {len(pf_files)} slides with Path Foundation embeddings")
    
    processed = 0
    skipped = 0
    failed = 0
    
    for pf_path in tqdm(pf_files, desc="Processing slides"):
        slide_id = pf_path.stem
        output_path = output_dir / f"{slide_id}_siglip.npy"
        
        # Skip if already exists
        if output_path.exists() and not force:
            skipped += 1
            continue
        
        # Find slide file
        slide_path = None
        for ext in ['.svs', '.tiff', '.tif', '.ndpi']:
            candidate = slides_dir / f"{slide_id}{ext}"
            if candidate.exists():
                slide_path = candidate
                break
        
        # Also check alternative slide directories
        if slide_path is None:
            alt_dirs = [
                pf_embeddings_dir.parent / 'tcga_full' / 'slides',
                pf_embeddings_dir.parent / 'ovarian_bev' / 'slides',
                pf_embeddings_dir.parent / 'demo' / 'slides',
            ]
            for alt_dir in alt_dirs:
                for ext in ['.svs', '.tiff', '.tif', '.ndpi']:
                    candidate = alt_dir / f"{slide_id}{ext}"
                    if candidate.exists():
                        slide_path = candidate
                        break
                if slide_path:
                    break
        
        if slide_path is None:
            logger.debug(f"No slide file for {slide_id}, skipping MedSigLIP")
            skipped += 1
            continue
        
        # Load coordinates
        coords_path = pf_embeddings_dir / f"{slide_id}_coords.npy"
        
        try:
            # Load patches from slide
            patches = load_patches_from_slide(slide_path, coords_path, max_patches)
            
            if len(patches) == 0:
                logger.warning(f"No patches loaded for {slide_id}")
                failed += 1
                continue
            
            # Generate MedSigLIP embeddings
            embeddings = embedder.embed_patches(
                patches,
                show_progress=False,
            )
            
            # Save
            np.save(output_path, embeddings)
            processed += 1
            
            logger.debug(f"Generated {len(embeddings)} embeddings for {slide_id}")
            
        except Exception as e:
            logger.error(f"Failed to process {slide_id}: {e}")
            failed += 1
    
    logger.info(f"Done! Processed: {processed}, Skipped: {skipped}, Failed: {failed}")
    return processed, skipped, failed


def main():
    parser = argparse.ArgumentParser(description="Generate MedSigLIP embeddings")
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=Path("data/embeddings"),
        help="Directory with Path Foundation embeddings",
    )
    parser.add_argument(
        "--slides-dir",
        type=Path,
        default=Path("data/slides"),
        help="Directory with slide files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: embeddings-dir/medsiglip_cache)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if cache exists",
    )
    parser.add_argument(
        "--max-patches",
        type=int,
        default=5000,
        help="Max patches per slide (default: 5000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for embedding (default: 16)",
    )
    
    args = parser.parse_args()
    
    # Setup output directory
    if args.output_dir is None:
        args.output_dir = args.embeddings_dir / "medsiglip_cache"
    
    logger.info(f"Embeddings dir: {args.embeddings_dir}")
    logger.info(f"Slides dir: {args.slides_dir}")
    logger.info(f"Output dir: {args.output_dir}")
    
    # Initialize embedder
    config = MedSigLIPConfig(
        batch_size=args.batch_size,
        cache_dir=str(args.output_dir),
    )
    embedder = MedSigLIPEmbedder(config)
    
    # Generate embeddings
    start = time.time()
    processed, skipped, failed = generate_embeddings_from_pf(
        pf_embeddings_dir=args.embeddings_dir,
        slides_dir=args.slides_dir,
        output_dir=args.output_dir,
        embedder=embedder,
        force=args.force,
        max_patches=args.max_patches,
    )
    
    elapsed = time.time() - start
    logger.info(f"Total time: {elapsed:.1f}s")
    
    if processed > 0:
        logger.info(f"Average: {elapsed/processed:.1f}s per slide")


if __name__ == "__main__":
    main()
