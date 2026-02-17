#!/usr/bin/env python3
"""
BRCA RunPod Pipeline - Download slides from GDC + embed with Path Foundation.
Run this ON RunPod (GPU instance).

Usage:
    # On RunPod:
    pip install openslide-python requests torch torchvision tensorflow tensorflow-hub numpy
    apt-get install -y openslide-tools
    python brca_runpod_pipeline.py --manifest /workspace/download_manifest.json --output /workspace/brca_embeddings
"""

import argparse
import json
import os
import sys
import time
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess

import numpy as np
import requests


GDC_DATA_URL = "https://api.gdc.cancer.gov/data"


def download_slide(file_id, filename, output_dir):
    """Download a single slide from GDC."""
    out_path = output_dir / filename
    if out_path.exists() and out_path.stat().st_size > 1000:
        return out_path, "cached"

    url = f"{GDC_DATA_URL}/{file_id}"
    r = requests.get(url, stream=True, timeout=300)
    r.raise_for_status()

    # GDC returns tar for single files
    content_type = r.headers.get("Content-Type", "")
    if "application/x-tar" in content_type:
        # Save tar then extract
        tar_path = output_dir / f"{file_id}.tar"
        with open(tar_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        # Extract
        import tarfile
        with tarfile.open(tar_path) as tf:
            for member in tf.getmembers():
                if member.name.endswith(".svs"):
                    member.name = filename
                    tf.extract(member, output_dir)
                    break
        tar_path.unlink()
    else:
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    return out_path, "downloaded"


def download_all_slides(manifest, output_dir, max_workers=4):
    """Download slides in parallel."""
    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(manifest)
    completed = 0
    failed = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for item in manifest:
            fut = pool.submit(download_slide, item["id"], item["filename"], output_dir)
            futures[fut] = item

        for fut in as_completed(futures):
            item = futures[fut]
            completed += 1
            try:
                path, status = fut.result()
                if completed % 10 == 0 or completed == total:
                    print(f"  [{completed}/{total}] {item['filename']} - {status}")
            except Exception as e:
                failed.append(item["filename"])
                print(f"  [{completed}/{total}] FAILED: {item['filename']} - {e}")

    print(f"\nDownloaded: {total - len(failed)}/{total}, Failed: {len(failed)}")
    if failed:
        print(f"Failed files: {failed[:10]}")
    return failed


def embed_slides(slides_dir, output_dir, batch_size=256, max_patches=5000):
    """Embed slides using Path Foundation (TF SavedModel)."""
    import openslide
    import tensorflow as tf
    import tensorflow_hub as hub

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Path Foundation
    print("Loading Path Foundation model...")
    model_url = "https://www.kaggle.com/models/google/path-foundation/TensorFlow2/path-foundation/1"
    model = hub.KerasLayer(model_url)
    print("Model loaded")

    slide_files = sorted(slides_dir.glob("*.svs"))
    print(f"Found {len(slide_files)} slides to embed")

    for idx, slide_path in enumerate(slide_files):
        slide_id = slide_path.stem
        emb_path = output_dir / f"{slide_id}.npy"
        if emb_path.exists():
            print(f"[{idx+1}/{len(slide_files)}] {slide_id} - cached")
            continue

        print(f"[{idx+1}/{len(slide_files)}] {slide_id}...", end=" ", flush=True)
        t0 = time.time()

        try:
            slide = openslide.OpenSlide(str(slide_path))
            W, H = slide.dimensions

            # Extract patches at level 0, 224x224
            patch_size = 224
            stride = patch_size  # non-overlapping

            coords = []
            for y in range(0, H - patch_size, stride):
                for x in range(0, W - patch_size, stride):
                    coords.append((x, y))

            # Subsample if too many
            if len(coords) > max_patches:
                rng = np.random.RandomState(42)
                indices = rng.choice(len(coords), max_patches, replace=False)
                coords = [coords[i] for i in sorted(indices)]

            # Extract and embed in batches
            all_embeddings = []
            for batch_start in range(0, len(coords), batch_size):
                batch_coords = coords[batch_start:batch_start + batch_size]
                patches = []
                valid_coords = []
                for x, y in batch_coords:
                    patch = slide.read_region((x, y), 0, (patch_size, patch_size))
                    patch = np.array(patch.convert("RGB"))
                    # Skip mostly white patches
                    if patch.mean() > 230:
                        continue
                    patches.append(patch)
                    valid_coords.append((x, y))

                if not patches:
                    continue

                batch_array = np.stack(patches).astype(np.float32) / 255.0
                batch_tensor = tf.constant(batch_array)
                embeddings = model(batch_tensor)
                all_embeddings.append(embeddings.numpy())

            if all_embeddings:
                all_embeddings = np.concatenate(all_embeddings, axis=0)
                np.save(emb_path, all_embeddings)
                elapsed = time.time() - t0
                print(f"{all_embeddings.shape[0]} patches, {elapsed:.0f}s")
            else:
                print("no valid patches!")

            slide.close()

        except Exception as e:
            print(f"ERROR: {e}")


def main():
    parser = argparse.ArgumentParser(description="BRCA RunPod Pipeline")
    parser.add_argument("--manifest", type=str, required=True, help="Path to download_manifest.json")
    parser.add_argument("--slides-dir", type=str, default="/workspace/brca_slides")
    parser.add_argument("--output", type=str, default="/workspace/brca_embeddings")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-embed", action="store_true")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-patches", type=int, default=5000)
    args = parser.parse_args()

    slides_dir = Path(args.slides_dir)
    output_dir = Path(args.output)

    with open(args.manifest) as f:
        manifest = json.load(f)

    print(f"=== BRCA RunPod Pipeline ===")
    print(f"Slides: {len(manifest)}")
    print(f"Slides dir: {slides_dir}")
    print(f"Embeddings dir: {output_dir}")
    print()

    if not args.skip_download:
        print("=== Phase 1: Download from GDC ===")
        failed = download_all_slides(manifest, slides_dir, max_workers=args.max_workers)
        print()

    if not args.skip_embed:
        print("=== Phase 2: Embed with Path Foundation ===")
        embed_slides(slides_dir, output_dir, batch_size=args.batch_size, max_patches=args.max_patches)
        print()

    # Summary
    n_slides = len(list(slides_dir.glob("*.svs"))) if slides_dir.exists() else 0
    n_emb = len(list(output_dir.glob("*.npy"))) if output_dir.exists() else 0
    print(f"=== Done ===")
    print(f"Slides downloaded: {n_slides}")
    print(f"Embeddings generated: {n_emb}")


if __name__ == "__main__":
    main()
