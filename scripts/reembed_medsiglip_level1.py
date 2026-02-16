#!/usr/bin/env python3
"""
Re-embed all slides with MedSigLIP at level 1 (lower magnification).

Level 1 gives ~4x wider field of view per patch so SigLIP can see tissue
architecture instead of just cellular detail. Also filters out whitespace
patches before embedding.

Usage:
    python scripts/reembed_medsiglip_level1.py
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from typing import List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def is_tissue_patch(patch_rgb: np.ndarray, white_thresh: int = 220, tissue_frac: float = 0.3) -> bool:
    """Check if a patch contains enough tissue (not whitespace)."""
    # Convert to grayscale
    gray = np.mean(patch_rgb, axis=2)
    # Count non-white pixels
    tissue_pixels = np.sum(gray < white_thresh)
    total_pixels = gray.size
    return (tissue_pixels / total_pixels) >= tissue_frac


def extract_patches_level1(
    slide_path: str,
    patch_size: int = 224,
    stride: int = 224,
    white_thresh: int = 220,
    tissue_frac: float = 0.3,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Extract non-whitespace patches from a slide at level 1.
    
    Returns:
        patches: List of RGB numpy arrays
        coords: (N, 2) array of (x, y) coordinates at level 1
    """
    import openslide
    
    slide = openslide.OpenSlide(slide_path)
    
    # Use level 1 if available, otherwise level 0
    if slide.level_count > 1:
        level = 1
    else:
        level = 0
        print(f"  Warning: only 1 level available, using level 0")
    
    w, h = slide.level_dimensions[level]
    downsample = slide.level_downsamples[level]
    
    patches = []
    coords = []
    skipped_white = 0
    
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            # Read at the chosen level
            # read_region takes level-0 coordinates, so scale up
            x0 = int(x * downsample)
            y0 = int(y * downsample)
            
            region = slide.read_region((x0, y0), level, (patch_size, patch_size))
            
            # Convert RGBA to RGB
            if region.mode == 'RGBA':
                rgb = np.array(region)[:, :, :3]
            else:
                rgb = np.array(region.convert('RGB'))
            
            # Filter whitespace
            if not is_tissue_patch(rgb, white_thresh, tissue_frac):
                skipped_white += 1
                continue
            
            patches.append(rgb)
            coords.append([x0, y0])  # Store level-0 coords for viewer navigation
    
    slide.close()
    
    coords_arr = np.array(coords, dtype=np.int64) if coords else np.zeros((0, 2), dtype=np.int64)
    return patches, coords_arr, skipped_white, level, downsample


def main():
    import torch
    from transformers import SiglipImageProcessor, SiglipModel
    from PIL import Image
    from tqdm import tqdm
    
    embeddings_dir = Path("/app/data/embeddings")
    slides_dir = Path("/app/data/tcga_full/slides")
    cache_dir = embeddings_dir / "medsiglip_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Find MedSigLIP model
    model_path = "/app/models/medsiglip"
    if not Path(model_path).exists():
        model_path = "google/siglip-so400m-patch14-384"
    
    print(f"Loading MedSigLIP from {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    processor = SiglipImageProcessor.from_pretrained(model_path)
    model = SiglipModel.from_pretrained(model_path).to(device)
    if device.type == "cuda":
        model = model.half()
    model.eval()
    
    print(f"Model loaded. Embedding dim: {model.config.vision_config.hidden_size}")
    
    # Find all slides with existing PF embeddings
    pf_files = sorted(embeddings_dir.glob("*.npy"))
    slide_ids = set()
    for f in pf_files:
        name = f.stem
        if not name.endswith("_coords") and not name.endswith("_siglip"):
            slide_ids.add(name)
    
    print(f"\nFound {len(slide_ids)} slides with PF embeddings")
    
    # Find corresponding SVS files
    svs_files = {}
    for svs in slides_dir.glob("*.svs"):
        svs_files[svs.stem] = svs
    
    batch_size = 64
    total_start = time.time()
    
    for i, slide_id in enumerate(sorted(slide_ids)):
        cache_path = cache_dir / f"{slide_id}_siglip.npy"
        coords_cache = cache_dir / f"{slide_id}_siglip_coords.npy"
        
        # Check if SVS exists
        if slide_id not in svs_files and slide_id != "slide_000":
            print(f"[{i+1}/{len(slide_ids)}] {slide_id[:40]}... - SVS not found, skipping")
            continue
        
        # For slide_000 (demo), skip if no SVS
        if slide_id == "slide_000":
            print(f"[{i+1}/{len(slide_ids)}] slide_000 - demo slide, skipping")
            continue
        
        svs_path = str(svs_files[slide_id])
        
        print(f"\n[{i+1}/{len(slide_ids)}] {slide_id[:50]}...")
        start = time.time()
        
        # Extract patches at level 1 with whitespace filtering
        patches, coords, n_skipped, level, downsample = extract_patches_level1(svs_path)
        
        if len(patches) == 0:
            print(f"  No tissue patches found, skipping")
            continue
        
        print(f"  Level {level} (downsample {downsample:.1f}x): {len(patches)} tissue patches, {n_skipped} whitespace skipped")
        
        # Embed in batches
        all_embeddings = []
        
        for batch_start in tqdm(range(0, len(patches), batch_size), desc="  Embedding"):
            batch = patches[batch_start:batch_start + batch_size]
            pil_images = [Image.fromarray(p) for p in batch]
            
            inputs = processor(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            if device.type == "cuda":
                inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
            
            with torch.no_grad():
                features = model.get_image_features(**inputs)
                if hasattr(features, "pooler_output"):
                    features = features.pooler_output
                features = features / features.norm(dim=-1, keepdim=True)
            
            all_embeddings.append(features.cpu().float().numpy())
        
        embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
        
        # Save
        np.save(cache_path, embeddings)
        np.save(coords_cache, coords)
        
        elapsed = time.time() - start
        print(f"  Saved: {embeddings.shape} in {elapsed:.1f}s")
    
    total_elapsed = time.time() - total_start
    print(f"\nDone! Total time: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
