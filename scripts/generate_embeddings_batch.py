#!/usr/bin/env python3
"""
Batch Embedding Generation Pipeline for Histopathology Slides.

GPU-accelerated DINOv2 embedding generation with:
- Parallel slide processing (I/O bound: reading slides)
- Batched GPU inference
- Incremental saving (crash recovery)
- Progress tracking and time estimation

Usage:
    python scripts/generate_embeddings_batch.py \
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

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
    status: str  # success, skipped, failed
    num_patches: int = 0
    embedding_shape: Tuple[int, int] = (0, 0)
    duration_seconds: float = 0.0
    error: Optional[str] = None


@dataclass
class ProcessingStats:
    """Overall processing statistics."""
    total_slides: int = 0
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    total_patches: int = 0
    start_time: float = 0.0
    
    def elapsed(self) -> float:
        return time.time() - self.start_time
    
    def eta_seconds(self) -> float:
        if self.processed == 0:
            return 0
        remaining = self.total_slides - self.processed - self.skipped - self.failed
        avg_time = self.elapsed() / self.processed
        return remaining * avg_time
    
    def eta_str(self) -> str:
        eta = self.eta_seconds()
        if eta <= 0:
            return "N/A"
        return str(timedelta(seconds=int(eta)))


class BatchEmbedder:
    """
    GPU-accelerated batch embedding generator for histopathology slides.
    
    Uses DINOv2-small (384-dim embeddings) with batched inference.
    Processes slides in parallel for I/O, sequential GPU for compute.
    """
    
    EMBEDDING_DIM = 384
    INPUT_SIZE = 224
    
    def __init__(
        self,
        model_name: str = "facebook/dinov2-small",
        device: str = "auto",
        batch_size: int = 64,
        precision: str = "fp16",
        patch_size: int = 224,
        max_patches_per_slide: int = 2000,
        tissue_threshold: float = 0.15,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.precision = precision
        self.patch_size = patch_size
        self.max_patches = max_patches_per_slide
        self.tissue_threshold = tissue_threshold
        
        self._model = None
        self._processor = None
        self._device = None
        self._device_str = device
        
    def _load_model(self) -> None:
        """Lazy load model on first use."""
        if self._model is not None:
            return
            
        import torch
        from transformers import AutoModel, AutoImageProcessor
        
        # Determine device
        if self._device_str == "auto":
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self._device = torch.device("mps")
            else:
                self._device = torch.device("cpu")
        else:
            self._device = torch.device(self._device_str)
            
        logger.info(f"Loading {self.model_name} on {self._device}")
        
        self._processor = AutoImageProcessor.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model = self._model.to(self._device)
        
        # Use FP16 for CUDA
        if self.precision == "fp16" and self._device.type == "cuda":
            self._model = self._model.half()
            
        self._model.eval()
        logger.info(f"Model loaded successfully (device={self._device}, precision={self.precision})")
        
    def extract_patches(
        self,
        slide_path: Path,
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """
        Extract tissue patches from a whole slide image.
        
        Returns:
            patches: List of 224x224 RGB patches
            coords: List of (x, y) coordinates at level 0
        """
        import cv2
        
        try:
            import openslide
        except ImportError:
            logger.error("openslide-python not installed. Run: pip install openslide-python")
            raise
            
        try:
            slide = openslide.OpenSlide(str(slide_path))
        except Exception as e:
            logger.error(f"Failed to open slide {slide_path}: {e}")
            return [], []
            
        # Get slide dimensions
        width, height = slide.dimensions
        
        # Use level 0 or 1 depending on slide size
        level = 0
        if slide.level_count > 1 and width * height > 100000 * 100000:
            level = 1
        
        level_dims = slide.level_dimensions[level]
        downsample = slide.level_downsamples[level]
        
        logger.debug(f"Slide {slide_path.name}: {width}x{height}, using level {level}")
        
        patches = []
        coords = []
        
        # Calculate step (with some overlap reduction for efficiency)
        step_at_level = self.patch_size
        step_at_level0 = int(step_at_level * downsample)
        
        # Grid sampling
        y_steps = list(range(0, int(level_dims[1]) - step_at_level, step_at_level))
        x_steps = list(range(0, int(level_dims[0]) - step_at_level, step_at_level))
        
        for y in y_steps:
            for x in x_steps:
                if len(patches) >= self.max_patches:
                    break
                    
                # Convert to level 0 coordinates
                x0 = int(x * downsample)
                y0 = int(y * downsample)
                
                try:
                    # Read region at level
                    region = slide.read_region((x0, y0), level, (self.patch_size, self.patch_size))
                    region = region.convert("RGB")
                    patch = np.array(region)
                    
                    # Tissue detection
                    if self._is_tissue(patch):
                        patches.append(patch)
                        coords.append((x0, y0))
                        
                except Exception:
                    continue
                    
            if len(patches) >= self.max_patches:
                break
                
        slide.close()
        return patches, coords
        
    def _is_tissue(self, patch: np.ndarray) -> bool:
        """
        Detect if patch contains tissue (not background).
        
        Uses simple RGB-based detection:
        - Not too white (background)
        - Not too dark (artifacts)
        - Has sufficient color variation
        """
        import cv2
        
        # Convert to grayscale
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        
        # Check mean intensity (tissue is typically between 50-220)
        mean_val = np.mean(gray)
        if mean_val < 30 or mean_val > 230:
            return False
            
        # Check standard deviation (tissue has texture)
        std_val = np.std(gray)
        if std_val < 15:
            return False
            
        # Check for sufficient non-white pixels
        non_white = np.sum(gray < 220) / gray.size
        if non_white < self.tissue_threshold:
            return False
            
        return True
        
    def embed_patches(self, patches: List[np.ndarray]) -> np.ndarray:
        """
        Generate embeddings for a list of patches.
        
        Args:
            patches: List of 224x224 RGB numpy arrays
            
        Returns:
            embeddings: (N, 384) array of embeddings
        """
        self._load_model()
        
        import torch
        from PIL import Image
        
        if len(patches) == 0:
            return np.zeros((0, self.EMBEDDING_DIM), dtype=np.float32)
            
        all_embeddings = []
        
        for i in range(0, len(patches), self.batch_size):
            batch = patches[i:i + self.batch_size]
            
            # Convert to PIL images
            pil_images = [
                Image.fromarray(p) if isinstance(p, np.ndarray) else p
                for p in batch
            ]
            
            # Process batch
            inputs = self._processor(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            
            # FP16 conversion for CUDA
            if self.precision == "fp16" and self._device.type == "cuda":
                inputs = {
                    k: v.half() if v.dtype == torch.float32 else v
                    for k, v in inputs.items()
                }
                
            # Forward pass
            with torch.no_grad():
                outputs = self._model(**inputs)
                # Use CLS token embedding
                embeddings = outputs.last_hidden_state[:, 0, :]
                
            all_embeddings.append(embeddings.cpu().numpy())
            
        # Concatenate all batches
        embeddings = np.concatenate(all_embeddings, axis=0)
        
        # Keep float16 for storage efficiency (matches existing embeddings)
        if self.precision == "fp16":
            embeddings = embeddings.astype(np.float16)
        else:
            embeddings = embeddings.astype(np.float32)
        
        return embeddings
        
    def embed_slide(
        self,
        slide_path: Path,
        output_dir: Path,
        force: bool = False,
    ) -> SlideResult:
        """
        Process a single slide: extract patches, generate embeddings, save.
        
        Args:
            slide_path: Path to .svs or other WSI file
            output_dir: Directory to save embeddings
            force: Re-process even if output exists
            
        Returns:
            SlideResult with processing outcome
        """
        slide_id = slide_path.stem
        output_path = output_dir / f"{slide_id}.npy"
        coords_path = output_dir / f"{slide_id}_coords.npy"
        
        start_time = time.time()
        
        # Check if already processed
        if output_path.exists() and coords_path.exists() and not force:
            return SlideResult(
                slide_id=slide_id,
                status="skipped",
            )
            
        try:
            # Extract patches
            patches, coords = self.extract_patches(slide_path)
            
            if len(patches) == 0:
                return SlideResult(
                    slide_id=slide_id,
                    status="failed",
                    error="No tissue patches extracted",
                    duration_seconds=time.time() - start_time,
                )
                
            # Generate embeddings
            embeddings = self.embed_patches(patches)
            
            # Save atomically (write to temp, then rename)
            temp_emb = output_path.with_name(output_path.stem + '.tmp.npy')
            temp_coords = coords_path.with_name(coords_path.stem + '.tmp.npy')
            
            np.save(temp_emb, embeddings)
            np.save(temp_coords, np.array(coords))
            
            # Atomic rename
            temp_emb.rename(output_path)
            temp_coords.rename(coords_path)
            
            duration = time.time() - start_time
            
            return SlideResult(
                slide_id=slide_id,
                status="success",
                num_patches=len(patches),
                embedding_shape=embeddings.shape,
                duration_seconds=duration,
            )
            
        except Exception as e:
            return SlideResult(
                slide_id=slide_id,
                status="failed",
                error=str(e),
                duration_seconds=time.time() - start_time,
            )
            
    def process_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        pattern: str = "*.svs",
        num_workers: int = 4,
        force: bool = False,
    ) -> List[SlideResult]:
        """
        Process all slides in a directory.
        
        Uses thread pool for I/O parallelism (reading slides).
        GPU inference is sequential (model is shared).
        
        Args:
            input_dir: Directory containing slide files
            output_dir: Directory for output embeddings
            pattern: Glob pattern for slide files
            num_workers: Number of parallel workers for I/O
            force: Re-process existing files
            
        Returns:
            List of SlideResult for each slide
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all slides
        slides = list(input_dir.glob(pattern))
        
        # Also check for other common formats
        if pattern == "*.svs":
            slides.extend(input_dir.glob("*.tif"))
            slides.extend(input_dir.glob("*.tiff"))
            slides.extend(input_dir.glob("*.ndpi"))
            
        slides = sorted(set(slides))
        
        if not slides:
            logger.warning(f"No slides found in {input_dir} with pattern {pattern}")
            return []
            
        logger.info(f"Found {len(slides)} slides to process")
        
        # Initialize stats
        stats = ProcessingStats(
            total_slides=len(slides),
            start_time=time.time(),
        )
        
        results = []
        
        # Pre-load model before starting
        self._load_model()
        
        # Process slides
        # Note: We process sequentially because GPU is the bottleneck
        # Parallel I/O doesn't help much when GPU is saturated
        for i, slide_path in enumerate(slides, 1):
            result = self.embed_slide(slide_path, output_dir, force=force)
            results.append(result)
            
            # Update stats
            if result.status == "success":
                stats.processed += 1
                stats.total_patches += result.num_patches
            elif result.status == "skipped":
                stats.skipped += 1
            else:
                stats.failed += 1
                
            # Log progress
            self._log_progress(result, i, stats)
            
        # Final summary
        self._log_summary(stats, output_dir)
        
        return results
        
    def _log_progress(self, result: SlideResult, current: int, stats: ProcessingStats):
        """Log progress for a completed slide."""
        total = stats.total_slides
        pct = (current / total) * 100
        
        if result.status == "success":
            patches_str = f"{result.num_patches} patches"
            time_str = f"{result.duration_seconds:.1f}s"
            logger.info(
                f"[{current}/{total}] {result.slide_id}: {patches_str} in {time_str} "
                f"(ETA: {stats.eta_str()})"
            )
        elif result.status == "skipped":
            logger.info(f"[{current}/{total}] {result.slide_id}: skipped (already exists)")
        else:
            logger.warning(f"[{current}/{total}] {result.slide_id}: FAILED - {result.error}")
            
    def _log_summary(self, stats: ProcessingStats, output_dir: Path):
        """Log final processing summary."""
        elapsed = timedelta(seconds=int(stats.elapsed()))
        
        logger.info("=" * 60)
        logger.info("Processing Complete")
        logger.info("=" * 60)
        logger.info(f"Total slides: {stats.total_slides}")
        logger.info(f"  Processed: {stats.processed}")
        logger.info(f"  Skipped:   {stats.skipped}")
        logger.info(f"  Failed:    {stats.failed}")
        logger.info(f"Total patches: {stats.total_patches}")
        logger.info(f"Total time: {elapsed}")
        logger.info(f"Output: {output_dir}")
        
        if stats.processed > 0:
            avg_time = stats.elapsed() / stats.processed
            avg_patches = stats.total_patches / stats.processed
            logger.info(f"Avg time/slide: {avg_time:.1f}s")
            logger.info(f"Avg patches/slide: {avg_patches:.0f}")


def save_processing_log(results: List[SlideResult], output_dir: Path):
    """Save processing results to a JSON log file."""
    log_path = output_dir / "processing_log.json"
    
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "total": len(results),
        "success": sum(1 for r in results if r.status == "success"),
        "skipped": sum(1 for r in results if r.status == "skipped"),
        "failed": sum(1 for r in results if r.status == "failed"),
        "results": [asdict(r) for r in results],
    }
    
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)
        
    logger.info(f"Processing log saved to {log_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate DINOv2 embeddings for histopathology slides",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input directory containing slide files",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/embeddings"),
        help="Output directory for embeddings",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=64,
        help="Batch size for GPU inference",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device for inference",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=["fp16", "fp32"],
        help="Model precision (fp16 for faster inference)",
    )
    parser.add_argument(
        "--max-patches",
        type=int,
        default=2000,
        help="Maximum patches per slide",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.svs",
        help="Glob pattern for slide files",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Re-process existing files",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (for future I/O optimization)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Validate input directory
    if not args.input.exists():
        logger.error(f"Input directory does not exist: {args.input}")
        sys.exit(1)
        
    # Create embedder
    embedder = BatchEmbedder(
        batch_size=args.batch_size,
        device=args.device,
        precision=args.precision,
        max_patches_per_slide=args.max_patches,
    )
    
    # Process directory
    results = embedder.process_directory(
        input_dir=args.input,
        output_dir=args.output,
        pattern=args.pattern,
        num_workers=args.workers,
        force=args.force,
    )
    
    # Save processing log
    if results:
        save_processing_log(results, args.output)
        
    # Exit with error code if any failures
    failed = sum(1 for r in results if r.status == "failed")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
