#!/usr/bin/env python3
"""Generate DINOv2 embeddings for Ovarian Bevacizumab Response slides.

Uses patch-based embedding extraction:
1. Extract patches at 20x magnification
2. Encode with DINOv2-G
3. Aggregate to slide-level embedding (mean pooling)
"""
import sys
sys.path.insert(0, '/home/hansonwen/med-gemma-hackathon/venv/lib/python3.12/site-packages')

import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import openslide
from PIL import Image
import torchvision.transforms as transforms

# Config
PATCH_SIZE = 256
STRIDE = 256  # Non-overlapping patches
MAG_LEVEL = 0  # Usually 20x or 40x
MAX_PATCHES = 2000  # Max patches per slide to limit memory
TISSUE_THRESHOLD = 0.5  # Min tissue content

def get_tissue_mask(slide, level, threshold=0.9):
    """Get a rough tissue mask using grayscale threshold."""
    dims = slide.level_dimensions[level]
    thumb = slide.get_thumbnail((dims[0]//32, dims[1]//32))
    gray = np.array(thumb.convert('L'))
    # Tissue is darker than background
    mask = gray < (255 * threshold)
    return mask

def extract_patches(slide_path, patch_size=256, stride=256, max_patches=2000):
    """Extract tissue patches from a WSI."""
    slide = openslide.OpenSlide(str(slide_path))
    
    # Get dimensions at level 0
    width, height = slide.level_dimensions[0]
    
    # Get rough tissue locations
    tissue_mask = get_tissue_mask(slide, 0)
    mask_h, mask_w = tissue_mask.shape
    scale_x = width / mask_w / 32
    scale_y = height / mask_h / 32
    
    patches = []
    coords = []
    
    # Sample patches
    for y in range(0, height - patch_size, stride):
        for x in range(0, width - patch_size, stride):
            # Check tissue mask
            mask_x = int(x / scale_x / 32)
            mask_y = int(y / scale_y / 32)
            if mask_x < mask_w and mask_y < mask_h:
                if not tissue_mask[mask_y, mask_x]:
                    continue
            
            # Extract patch
            patch = slide.read_region((x, y), 0, (patch_size, patch_size))
            patch = patch.convert('RGB')
            
            # Check if patch has enough tissue (not too white)
            gray = np.array(patch.convert('L'))
            if np.mean(gray) > 230:  # Too white, likely background
                continue
            
            patches.append(patch)
            coords.append((x, y))
            
            if len(patches) >= max_patches:
                break
        if len(patches) >= max_patches:
            break
    
    slide.close()
    return patches, coords

def load_dinov2():
    """Load DINOv2-G model."""
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model

def get_transforms():
    """Get image transforms for DINOv2."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

def embed_patches(model, patches, transform, batch_size=32):
    """Embed patches using DINOv2."""
    embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            batch = patches[i:i+batch_size]
            tensors = torch.stack([transform(p) for p in batch])
            if torch.cuda.is_available():
                tensors = tensors.cuda()
            emb = model(tensors)
            embeddings.append(emb.cpu().numpy())
    
    if embeddings:
        return np.vstack(embeddings)
    return np.array([])

def main():
    data_dir = Path('/home/hansonwen/med-gemma-hackathon/data/ovarian_bev')
    slides_dir = data_dir / 'slides'
    embed_dir = data_dir / 'embeddings'
    embed_dir.mkdir(exist_ok=True)
    
    # Load clinical data
    clinical = pd.read_csv(data_dir / 'clinical.csv')
    slide_files = list(slides_dir.glob('*.svs'))
    
    print(f'Found {len(slide_files)} slides')
    print(f'Clinical data has {len(clinical)} entries')
    
    # Load model
    print('Loading DINOv2-G...')
    model = load_dinov2()
    transform = get_transforms()
    
    # Process each slide
    for slide_path in tqdm(slide_files, desc='Processing slides'):
        slide_id = slide_path.stem
        embed_path = embed_dir / f'{slide_id}.npy'
        
        if embed_path.exists():
            continue
        
        try:
            # Extract patches
            patches, coords = extract_patches(
                slide_path, 
                patch_size=PATCH_SIZE, 
                stride=STRIDE,
                max_patches=MAX_PATCHES
            )
            
            if len(patches) == 0:
                print(f'Warning: No patches extracted from {slide_id}')
                continue
            
            # Embed patches
            embeddings = embed_patches(model, patches, transform)
            
            # Save patch embeddings (for MIL) and aggregated embedding
            np.save(embed_path, {
                'patch_embeddings': embeddings,
                'coords': np.array(coords),
                'aggregated': embeddings.mean(axis=0)  # Mean pooling
            })
            
            print(f'{slide_id}: {len(patches)} patches, embedding shape: {embeddings.shape}')
            
        except Exception as e:
            print(f'Error processing {slide_id}: {e}')
            continue
    
    print('Done!')

if __name__ == '__main__':
    main()
