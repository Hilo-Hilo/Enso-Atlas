#!/usr/bin/env python3
"""Recover missing *_coords.npy files for existing embedding files.

This reconstructs coordinates using the same extraction logic used by the
LUAD/BRCA runpod embedding scripts:
- level 0
- patch_size=stride=224 (default)
- optional deterministic max_patches subsampling (RandomState(42))
- skip mostly white patches (mean > white_threshold)

Why this exists:
If embeddings were saved without coords, attention heatmaps cannot be mapped
truthfully. Synthetic fallback grids are misleading.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import openslide


def reconstruct_coords_for_slide(
    slide_path: Path,
    emb_path: Path,
    out_coords_path: Path,
    *,
    patch_size: int,
    max_patches: int,
    white_threshold: float,
) -> tuple[bool, str]:
    emb = np.load(emb_path, mmap_mode="r")
    n_embeddings = int(emb.shape[0]) if emb.ndim >= 1 else 0

    slide = openslide.OpenSlide(str(slide_path))
    w, h = slide.dimensions

    coords: list[tuple[int, int]] = []
    for y in range(0, h - patch_size, patch_size):
        for x in range(0, w - patch_size, patch_size):
            coords.append((x, y))

    if len(coords) > max_patches:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(coords), max_patches, replace=False)
        coords = [coords[i] for i in sorted(indices)]

    kept: list[tuple[int, int]] = []
    for x, y in coords:
        patch = slide.read_region((x, y), 0, (patch_size, patch_size))
        arr = np.array(patch.convert("RGB"))
        if float(arr.mean()) > white_threshold:
            continue
        kept.append((x, y))

    slide.close()

    if len(kept) != n_embeddings:
        return (
            False,
            f"count mismatch: embeddings={n_embeddings}, reconstructed_coords={len(kept)}",
        )

    out_coords_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_coords_path, np.asarray(kept, dtype=np.int32))
    return True, f"ok ({len(kept)} coords)"


def main() -> int:
    ap = argparse.ArgumentParser(description="Recover missing *_coords.npy files")
    ap.add_argument("--slides-dir", required=True, help="Directory containing .svs files")
    ap.add_argument("--embeddings-dir", required=True, help="Directory containing embeddings .npy")
    ap.add_argument("--match", default=None, help="Optional substring filter on slide stem")
    ap.add_argument("--patch-size", type=int, default=224)
    ap.add_argument("--max-patches", type=int, default=5000)
    ap.add_argument("--white-threshold", type=float, default=230.0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    slides_dir = Path(args.slides_dir)
    emb_dir = Path(args.embeddings_dir)

    if not slides_dir.exists():
        print(f"slides-dir does not exist: {slides_dir}", file=sys.stderr)
        return 2
    if not emb_dir.exists():
        print(f"embeddings-dir does not exist: {emb_dir}", file=sys.stderr)
        return 2

    emb_files = sorted([p for p in emb_dir.glob("*.npy") if not p.name.endswith("_coords.npy")])
    if args.match:
        emb_files = [p for p in emb_files if args.match in p.stem]

    total = len(emb_files)
    print(f"Found {total} embedding files to inspect")

    recovered = 0
    skipped = 0
    failed = 0

    for i, emb_path in enumerate(emb_files, start=1):
        stem = emb_path.stem
        coords_path = emb_dir / f"{stem}_coords.npy"

        if coords_path.exists() and not args.overwrite:
            print(f"[{i}/{total}] {stem}: skip (coords exists)")
            skipped += 1
            continue

        slide_path = slides_dir / f"{stem}.svs"
        if not slide_path.exists():
            print(f"[{i}/{total}] {stem}: FAIL (slide not found: {slide_path.name})")
            failed += 1
            continue

        ok, msg = reconstruct_coords_for_slide(
            slide_path,
            emb_path,
            coords_path,
            patch_size=args.patch_size,
            max_patches=args.max_patches,
            white_threshold=args.white_threshold,
        )
        if ok:
            print(f"[{i}/{total}] {stem}: recovered - {msg}")
            recovered += 1
        else:
            print(f"[{i}/{total}] {stem}: FAIL - {msg}")
            failed += 1

    print(
        f"Done: recovered={recovered}, skipped={skipped}, failed={failed}, total={total}"
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
