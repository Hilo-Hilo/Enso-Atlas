"""Re-embed slide_000 with lower tissue threshold (0.01 instead of 0.05)."""
import sys, os, time, glob
import numpy as np

# Path Foundation encoder
sys.path.insert(0, "src")

SLIDE_PATH = "data/slides/slide_000.svs"
OUTPUT_DIR = "data/embeddings/level0"
PATCH_SIZE = 224
TISSUE_THRESHOLD = 0.01  # Was 0.05

import openslide
slide = openslide.OpenSlide(SLIDE_PATH)
dims = slide.dimensions
print(f"Slide: {dims[0]}x{dims[1]}")

cols = (dims[0] - PATCH_SIZE) // PATCH_SIZE + 1
rows = (dims[1] - PATCH_SIZE) // PATCH_SIZE + 1
print(f"Grid: {cols}x{rows} = {cols*rows} positions")
print(f"Tissue threshold: {TISSUE_THRESHOLD}")

# Extract patches
t0 = time.time()
patches = []
coords = []
for row in range(rows):
    for col in range(cols):
        x = col * PATCH_SIZE
        y = row * PATCH_SIZE
        patch = np.array(slide.read_region((x, y), 0, (PATCH_SIZE, PATCH_SIZE)).convert("RGB"))
        gray = np.mean(patch[:,:,:3], axis=2)
        tissue_ratio = np.sum(gray < 200) / (PATCH_SIZE * PATCH_SIZE)
        if tissue_ratio > TISSUE_THRESHOLD:
            patches.append(patch)
            coords.append((x, y))

slide.close()
print(f"Extracted: {len(patches)} patches in {time.time()-t0:.1f}s (was 6680)")

# Save coords
coords_arr = np.array(coords)
np.save(os.path.join(OUTPUT_DIR, "slide_000_coords.npy"), coords_arr)
print(f"Saved coords: {coords_arr.shape}")

# Embed with Path Foundation
from enso_atlas.embeddings.path_foundation import PathFoundationEncoder
encoder = PathFoundationEncoder()

batch_size = 64
all_embeddings = []
total = len(patches)
t1 = time.time()
for i in range(0, total, batch_size):
    batch = np.array(patches[i:i+batch_size])
    embs = encoder.encode_batch(batch)
    all_embeddings.append(embs)
    done = min(i + batch_size, total)
    if done % (batch_size * 10) == 0 or done == total:
        elapsed = time.time() - t1
        rate = done / elapsed if elapsed > 0 else 0
        print(f"  Embedded {done}/{total} ({rate:.0f} patches/s)")

embeddings = np.concatenate(all_embeddings, axis=0)
np.save(os.path.join(OUTPUT_DIR, "slide_000.npy"), embeddings)
print(f"Saved embeddings: {embeddings.shape}")

# Clear heatmap cache
for f in glob.glob(os.path.join(OUTPUT_DIR, "heatmap_cache", "slide_000_*.png")):
    os.remove(f)
    print(f"Cleared cache: {os.path.basename(f)}")

print(f"Total time: {time.time()-t0:.1f}s")
print("Done! Restart the backend or it will use cached predictions.")
