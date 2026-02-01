#!/usr/bin/env python3
"""
Batch Embedding Generation using Path Foundation.

GPU-accelerated embedding generation with:
- Path Foundation (Google's histopathology foundation model)
- Batched GPU inference
- Incremental saving (crash recovery)
- Progress tracking

Usage:
    python scripts/generate_embeddings_pathfound.py \
        --input data/slides \
        --output data/embeddings \
        --batch-size 64
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class SlideResult:
    """Result of processing a single slide."""
    slide_id: str
    status: str
    num_patches: int = 0
    embedding_shape: Tuple[int, int] = (0, 0)
    duration_seconds: float = 0.0
    error: Optional[str] = None


class PathFoundationBatchEmbedder:
    """
    GPU-accelerated batch embedding generator using Path Foundation.
    
    Uses google/path-foundation (384-dim embeddings) with batched inference.
    """
    
    EMBEDDING_DIM = 384
    INPUT_SIZE = 224
    
    def __init__(
        self,
        device: str = "auto",
        batch_size: int = 64,
        patch_size: int = 224,
        max_patches_per_slide: int = 2000,
        tissue_threshold: float = 0.3,
    ):
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.max_patches = max_patches_per_slide
        self.tissue_threshold = tissue_threshold
        self._model = None
        self._infer = None
        
    def _load_model(self):
        if self._model is not None:
            return
            
        import tensorflow as tf
        from huggingface_hub import snapshot_download
        
        # Configure GPU
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Using {len(gpus)} GPU(s)")
        
        logger.info("Loading Path Foundation model...")
        
        # Download the model files from HuggingFace Hub
        model_dir = snapshot_download(
            repo_id="google/path-foundation",
            allow_patterns=["*.pb", "variables/*", "keras_metadata.pb"]
        )
        logger.info(f"Model downloaded to: {model_dir}")
        
        # Load as TensorFlow SavedModel
        self._model = tf.saved_model.load(model_dir)
        self._infer = self._model.signatures["serving_default"]
        logger.info("Path Foundation loaded successfully")
    
    def extract_patches(self, slide_path: Path) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """Extract tissue patches from a WSI."""
        import openslide
        import cv2
        
        try:
            slide = openslide.OpenSlide(str(slide_path))
        except Exception as e:
            logger.error(f"Failed to open {slide_path}: {e}")
            return [], []
        
        width, height = slide.dimensions
        
        # Choose appropriate level
        level = 0
        if slide.level_count > 1:
            # Use level 1 if available for efficiency
            level = 1
        
        level_dims = slide.level_dimensions[level]
        downsample = slide.level_downsamples[level]
        
        logger.info(f"Slide: {width}x{height}, level {level}")
        
        patches = []
        coords = []
        step = self.patch_size
        
        for y in range(0, int(level_dims[1]) - step, step):
            if len(patches) >= self.max_patches:
                break
            for x in range(0, int(level_dims[0]) - step, step):
                if len(patches) >= self.max_patches:
                    break
                
                x0 = int(x * downsample)
                y0 = int(y * downsample)
                
                try:
                    region = slide.read_region((x0, y0), level, (self.patch_size, self.patch_size))
                    region = region.convert("RGB")
                    patch = np.array(region)
                    
                    if self._is_tissue(patch):
                        patches.append(patch)
                        coords.append((x0, y0))
                except Exception:
                    continue
        
        slide.close()
        logger.info(f"Extracted {len(patches)} patches")
        return patches, coords
    
    def _is_tissue(self, patch: np.ndarray) -> bool:
        """Simple tissue detection."""
        import cv2
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        
        if mean_val < 30 or mean_val > 230:
            return False
        if std_val < 15:
            return False
        
        non_white = np.sum(gray < 220) / gray.size
        return non_white >= self.tissue_threshold
    
    def embed_patches(self, patches: List[np.ndarray]) -> np.ndarray:
        """Generate embeddings for patches."""
        self._load_model()
        
        import tensorflow as tf
        
        if not patches:
            return np.zeros((0, self.EMBEDDING_DIM), dtype=np.float16)
        
        all_embeddings = []
        
        for i in range(0, len(patches), self.batch_size):
            batch = patches[i:i + self.batch_size]
            
            # Convert to tensor: [N, 224, 224, 3], float32, [0,1]
            batch_array = np.stack(batch, axis=0).astype(np.float32) / 255.0
            batch_tensor = tf.constant(batch_array)
            
            # Inference
            outputs = self._infer(batch_tensor)
            embeddings = outputs['output_0'].numpy()
            all_embeddings.append(embeddings)
        
        return np.concatenate(all_embeddings, axis=0).astype(np.float16)
    
    def embed_slide(self, slide_path: Path, output_dir: Path, force: bool = False) -> SlideResult:
        """Process a single slide."""
        slide_id = slide_path.stem
        output_path = output_dir / f"{slide_id}.npy"
        coords_path = output_dir / f"{slide_id}_coords.npy"
        
        start_time = time.time()
        
        if output_path.exists() and coords_path.exists() and not force:
            return SlideResult(slide_id=slide_id, status="skipped")
        
        try:
            patches, coords = self.extract_patches(slide_path)
            
            if not patches:
                return SlideResult(
                    slide_id=slide_id,
                    status="failed",
                    error="No tissue patches",
                    duration_seconds=time.time() - start_time
                )
            
            embeddings = self.embed_patches(patches)
            
            # Save atomically
            temp_emb = output_path.with_suffix('.tmp.npy')
            temp_coords = coords_path.with_suffix('.tmp.npy')
            
            np.save(temp_emb, embeddings)
            np.save(temp_coords, np.array(coords))
            
            temp_emb.rename(output_path)
            temp_coords.rename(coords_path)
            
            return SlideResult(
                slide_id=slide_id,
                status="success",
                num_patches=len(patches),
                embedding_shape=embeddings.shape,
                duration_seconds=time.time() - start_time
            )
            
        except Exception as e:
            return SlideResult(
                slide_id=slide_id,
                status="failed",
                error=str(e),
                duration_seconds=time.time() - start_time
            )
    
    def process_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        pattern: str = "*.svs",
        force: bool = False
    ) -> List[SlideResult]:
        """Process all slides in directory."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        slides = list(input_dir.glob(pattern))
        slides.extend(input_dir.glob("*.tif"))
        slides.extend(input_dir.glob("*.tiff"))
        slides.extend(input_dir.glob("*.ndpi"))
        slides = sorted(set(slides))
        
        if not slides:
            logger.warning(f"No slides found in {input_dir}")
            return []
        
        logger.info(f"Found {len(slides)} slides")
        
        # Pre-load model
        self._load_model()
        
        results = []
        total_patches = 0
        start_time = time.time()
        
        for i, slide_path in enumerate(slides, 1):
            logger.info(f"\n[{i}/{len(slides)}] Processing {slide_path.name}")
            result = self.embed_slide(slide_path, output_dir, force)
            results.append(result)
            
            if result.status == "success":
                total_patches += result.num_patches
                logger.info(f"SUCCESS: {result.num_patches} patches in {result.duration_seconds:.1f}s")
            elif result.status == "skipped":
                logger.info("SKIPPED (already exists)")
            else:
                logger.error(f"Failed {slide_path.stem}: {result.error}")
        
        # Summary
        elapsed = time.time() - start_time
        success = sum(1 for r in results if r.status == "success")
        skipped = sum(1 for r in results if r.status == "skipped")
        failed = sum(1 for r in results if r.status == "failed")
        
        logger.info("\n" + "="*60)
        logger.info("Processing Complete - Path Foundation")
        logger.info("="*60)
        logger.info(f"Total: {len(slides)}, Success: {success}, Skipped: {skipped}, Failed: {failed}")
        logger.info(f"Total patches: {total_patches}")
        logger.info(f"Total time: {timedelta(seconds=int(elapsed))}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Generate Path Foundation embeddings")
    parser.add_argument("--input", "-i", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path, default=Path("data/embeddings"))
    parser.add_argument("--batch-size", "-b", type=int, default=64)
    parser.add_argument("--max-patches", type=int, default=2000)
    parser.add_argument("--pattern", type=str, default="*.svs")
    parser.add_argument("--force", "-f", action="store_true")
    
    args = parser.parse_args()
    
    if not args.input.exists():
        logger.error(f"Input not found: {args.input}")
        sys.exit(1)
    
    embedder = PathFoundationBatchEmbedder(
        batch_size=args.batch_size,
        max_patches_per_slide=args.max_patches
    )
    
    results = embedder.process_directory(
        input_dir=args.input,
        output_dir=args.output,
        pattern=args.pattern,
        force=args.force
    )
    
    failed = sum(1 for r in results if r.status == "failed")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
