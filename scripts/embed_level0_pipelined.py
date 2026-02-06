#!/usr/bin/env python3
"""
Pipelined Path Foundation embedding script for level 0 (full resolution) patches.
Uses producer/consumer pattern to keep GPU busy during CPU-bound extraction.

Key optimization: GPU embeds batches while CPU extracts more patches in parallel.
"""

import os
import sys
import glob
import logging
import queue
import threading
import time
from pathlib import Path

import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.environ.get('EMBED_LOG', '/tmp/embed_pipelined.log'))
    ]
)
logger = logging.getLogger(__name__)

# Lazy imports for heavy libraries
openslide = None
tf = None
path_foundation = None

def import_libs():
    """Lazy import heavy libraries."""
    global openslide, tf, path_foundation
    if openslide is None:
        import openslide as _openslide
        openslide = _openslide
    if tf is None:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        import tensorflow as _tf
        tf = _tf
    if path_foundation is None:
        sys.path.insert(0, '/workspace/path-foundation')
        import path_foundation as _pf
        path_foundation = _pf


class PatchExtractor:
    """Extracts patches from a slide in a background thread."""
    
    def __init__(self, slide_path, patch_size=224, step=None, max_patches=50000,
                 batch_size=512, tissue_threshold=0.05):
        self.slide_path = slide_path
        self.patch_size = patch_size
        self.step = step or patch_size  # Dense by default
        self.max_patches = max_patches
        self.batch_size = batch_size
        self.tissue_threshold = tissue_threshold
        
        self.slide = None
        self.dims = None
        self.total_positions = 0
        
    def open_slide(self):
        """Open the slide and get dimensions."""
        import_libs()
        self.slide = openslide.OpenSlide(self.slide_path)
        self.dims = self.slide.level_dimensions[0]
        
        cols = (self.dims[0] - self.patch_size) // self.step + 1
        rows = (self.dims[1] - self.patch_size) // self.step + 1
        self.total_positions = cols * rows
        
        logger.info(f"Slide: {self.dims[0]}x{self.dims[1]} at level 0")
        logger.info(f"Grid: {cols}x{rows} = {self.total_positions} possible positions")
        
    def is_tissue(self, patch_array):
        """Check if patch contains tissue (not background)."""
        # Convert to grayscale
        gray = np.mean(patch_array[:, :, :3], axis=2)
        # Background is typically white (>200)
        tissue_pixels = np.sum(gray < 200)
        tissue_ratio = tissue_pixels / (self.patch_size * self.patch_size)
        return tissue_ratio > self.tissue_threshold
    
    def extract_to_queue(self, patch_queue, coord_queue, stop_event):
        """Extract patches and put batches into queue. Runs in background thread."""
        try:
            self.open_slide()
            
            batch = []
            coords = []
            extracted = 0
            skipped = 0
            
            cols = (self.dims[0] - self.patch_size) // self.step + 1
            rows = (self.dims[1] - self.patch_size) // self.step + 1
            
            for row in range(rows):
                if stop_event.is_set():
                    break
                if extracted >= self.max_patches:
                    break
                    
                for col in range(cols):
                    if stop_event.is_set():
                        break
                    if extracted >= self.max_patches:
                        break
                    
                    x = col * self.step
                    y = row * self.step
                    
                    # Read patch
                    patch = self.slide.read_region((x, y), 0, (self.patch_size, self.patch_size))
                    patch_array = np.array(patch.convert('RGB'))
                    
                    # Filter non-tissue
                    if not self.is_tissue(patch_array):
                        skipped += 1
                        continue
                    
                    batch.append(patch_array)
                    coords.append((x, y))
                    extracted += 1
                    
                    # Send batch when full
                    if len(batch) >= self.batch_size:
                        patch_queue.put(np.array(batch))
                        coord_queue.put(coords.copy())
                        batch = []
                        coords = []
                        
                        # Log progress
                        if extracted % 2000 == 0:
                            logger.info(f"Extracted {extracted}/{self.max_patches} patches (skipped {skipped} background)")
            
            # Send remaining patches
            if batch:
                patch_queue.put(np.array(batch))
                coord_queue.put(coords)
            
            logger.info(f"Extraction complete: {extracted} tissue patches, {skipped} background skipped")
            
        except Exception as e:
            logger.error(f"Extraction error: {e}")
            raise
        finally:
            # Signal end of extraction
            patch_queue.put(None)
            coord_queue.put(None)
            if self.slide:
                self.slide.close()


class PathFoundationEmbedder:
    """Embeds patches using Path Foundation model."""
    
    def __init__(self, model_path='/workspace/path-foundation'):
        self.model_path = model_path
        self.model = None
        
    def load_model(self):
        """Load Path Foundation model."""
        import_libs()
        logger.info("Loading Path Foundation model...")
        self.model = path_foundation.load_model(self.model_path)
        logger.info("Model loaded")
        
    def embed_batch(self, patches):
        """Embed a batch of patches."""
        # Path Foundation expects float32 in [0, 1]
        patches_normalized = patches.astype(np.float32) / 255.0
        embeddings = self.model(patches_normalized)
        return embeddings.numpy()


def process_slide_pipelined(slide_path, output_dir, patch_size=224, step=None,
                            max_patches=50000, batch_size=512):
    """
    Process a single slide with pipelined extraction and embedding.
    GPU stays busy while CPU extracts patches in parallel.
    """
    slide_name = Path(slide_path).stem
    output_path = Path(output_dir) / f"{slide_name}.npy"
    coord_path = Path(output_dir) / f"{slide_name}_coords.npy"
    
    # Skip if already processed
    if output_path.exists():
        logger.info(f"SKIP: {output_path} already exists")
        return
    
    logger.info(f"Processing: {slide_path}")
    
    # Initialize components
    extractor = PatchExtractor(
        slide_path, 
        patch_size=patch_size,
        step=step,
        max_patches=max_patches,
        batch_size=batch_size
    )
    embedder = PathFoundationEmbedder()
    embedder.load_model()
    
    # Queues for producer/consumer pattern
    # maxsize limits memory usage - extractor waits if queue is full
    patch_queue = queue.Queue(maxsize=4)  # 4 batches buffered
    coord_queue = queue.Queue(maxsize=4)
    stop_event = threading.Event()
    
    # Start extraction in background thread
    extract_thread = threading.Thread(
        target=extractor.extract_to_queue,
        args=(patch_queue, coord_queue, stop_event)
    )
    extract_thread.start()
    
    # Embed batches as they arrive
    all_embeddings = []
    all_coords = []
    batch_count = 0
    start_time = time.time()
    
    try:
        while True:
            # Get next batch (blocks until available)
            patches = patch_queue.get(timeout=300)  # 5 min timeout
            coords = coord_queue.get(timeout=300)
            
            if patches is None:  # End signal
                break
            
            # Embed batch
            embeddings = embedder.embed_batch(patches)
            all_embeddings.append(embeddings)
            all_coords.extend(coords)
            
            batch_count += 1
            if batch_count % 10 == 0:
                elapsed = time.time() - start_time
                patches_done = len(all_coords)
                rate = patches_done / elapsed
                logger.info(f"Embedded {patches_done} patches ({rate:.1f} patches/sec)")
        
        # Wait for extractor to finish
        extract_thread.join(timeout=10)
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        stop_event.set()
        extract_thread.join(timeout=5)
        raise
    
    # Concatenate and save
    if all_embeddings:
        final_embeddings = np.concatenate(all_embeddings, axis=0)
        final_coords = np.array(all_coords)
        
        np.save(output_path, final_embeddings)
        np.save(coord_path, final_coords)
        
        elapsed = time.time() - start_time
        logger.info(f"SUCCESS: {len(all_coords)} patches -> {output_path}")
        logger.info(f"Total time: {elapsed:.1f}s ({len(all_coords)/elapsed:.1f} patches/sec)")
    else:
        logger.warning(f"No patches extracted from {slide_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Pipelined Path Foundation embedding')
    parser.add_argument('--slide-dir', default='/workspace/slides', help='Directory with slides')
    parser.add_argument('--output-dir', default='/workspace/embeddings', help='Output directory')
    parser.add_argument('--patch-size', type=int, default=224, help='Patch size')
    parser.add_argument('--step', type=int, default=None, help='Step size (default: patch_size for dense)')
    parser.add_argument('--max-patches', type=int, default=20000, help='Max patches per slide')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size for embedding')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find slides
    slide_patterns = ['*.svs', '*.ndpi', '*.tiff', '*.tif', '*.mrxs']
    slides = []
    for pattern in slide_patterns:
        slides.extend(glob.glob(os.path.join(args.slide_dir, pattern)))
    
    if not slides:
        logger.error(f"No slides found in {args.slide_dir}")
        return
    
    logger.info(f"Found {len(slides)} slides")
    
    # Process each slide
    for i, slide_path in enumerate(slides, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Slide {i}/{len(slides)}: {Path(slide_path).name}")
        logger.info(f"{'='*60}")
        
        try:
            process_slide_pipelined(
                slide_path,
                args.output_dir,
                patch_size=args.patch_size,
                step=args.step,
                max_patches=args.max_patches,
                batch_size=args.batch_size
            )
        except Exception as e:
            logger.error(f"Failed to process {slide_path}: {e}")
            continue
    
    logger.info("\nAll slides processed!")


if __name__ == '__main__':
    main()
