"""Re-embed slide_000 with tissue_threshold=0.01.
Avoids cv2 import issue by importing TF before keras."""

import glob
import os
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Import TF first to avoid the cv2/keras conflict
import numpy as np
import tensorflow as tf

SLIDE_PATH = "data/slides/slide_000.svs"
OUTPUT_DIR = "data/embeddings/level0"
PATCH_SIZE = 224
TISSUE_THRESHOLD = 0.01
BATCH_SIZE = 64

# Step 1: Extract patches
import openslide

slide = openslide.OpenSlide(SLIDE_PATH)
dims = slide.dimensions
cols = len(range(0, dims[0], PATCH_SIZE))  # 206 for 46000/224
rows = len(range(0, dims[1], PATCH_SIZE))  # 147 for 32914/224
print(f"Slide: {dims[0]}x{dims[1]}, Grid: {cols}x{rows}")

t0 = time.time()
patches = []
coords = []
for y in range(0, dims[1], PATCH_SIZE):
    for x in range(0, dims[0], PATCH_SIZE):
        patch = np.array(slide.read_region((x, y), 0, (PATCH_SIZE, PATCH_SIZE)).convert("RGB"))
        gray = np.mean(patch[:, :, :3], axis=2)
        ratio = np.sum(gray < 200) / (PATCH_SIZE * PATCH_SIZE)
        if ratio > TISSUE_THRESHOLD:
            patches.append(patch)
            coords.append((x, y))
slide.close()
print(
    f"Extracted: {len(patches)} patches in {time.time() - t0:.1f}s (was 6680, gained {len(patches) - 6680})"
)

# Save coords
coords_arr = np.array(coords)
np.save(os.path.join(OUTPUT_DIR, "slide_000_coords.npy"), coords_arr)

# Step 2: Load Path Foundation model directly
MODEL_NAME = "google/path-foundation"
from huggingface_hub import snapshot_download

model_dir = snapshot_download(MODEL_NAME, local_dir=None)
model = tf.saved_model.load(model_dir)
infer = model.signatures["serving_default"]
print(f"Path Foundation loaded from {model_dir}")

# Step 3: Embed
all_embeddings = []
total = len(patches)
t1 = time.time()
for i in range(0, total, BATCH_SIZE):
    batch = np.array(patches[i : i + BATCH_SIZE], dtype=np.float32) / 255.0
    tensor = tf.constant(batch)
    result = infer(tensor)
    key = list(result.keys())[0]
    embs = result[key].numpy()
    all_embeddings.append(embs)
    done = min(i + BATCH_SIZE, total)
    if done % (BATCH_SIZE * 20) == 0 or done == total:
        elapsed = time.time() - t1
        rate = done / elapsed if elapsed > 0 else 0
        print(f"  Embedded {done}/{total} ({rate:.0f}/s)")

embeddings = np.concatenate(all_embeddings, axis=0)
np.save(os.path.join(OUTPUT_DIR, "slide_000.npy"), embeddings)
print(f"Saved: {embeddings.shape}")

# Clear heatmap cache
for f in glob.glob(os.path.join(OUTPUT_DIR, "heatmap_cache", "slide_000_*.png")):
    os.remove(f)
    print(f"Cleared: {os.path.basename(f)}")

print(f"Total: {time.time() - t0:.1f}s. Done!")
