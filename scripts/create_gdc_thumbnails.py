#!/usr/bin/env python3
import argparse
import csv
import json
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import httpx
import numpy as np
import openslide
from google.cloud import storage

SOURCE_BUCKET_NAME = "gdc-tcga-phs000178-open"
GCS_CHUNK_SIZE = 8 * 1024 * 1024


def load_full_path_map(scan_csv: Optional[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not scan_csv:
        return mapping
    with open(scan_csv, "r", newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            fid = str(row.get("file_id", "")).strip()
            full_path = str(row.get("full_path", "")).strip()
            if fid and full_path:
                mapping[fid] = full_path
    return mapping


def barcode_from_full_path(full_path: str) -> str:
    fname = Path(full_path).name
    return fname.split(".")[0]


@lru_cache(maxsize=50000)
def resolve_current_file_id_from_barcode(barcode: str) -> Optional[str]:
    """
    Same selection logic as embedder_32-path.py:
    query GDC by filename barcode prefix and pick most recent open/released SVS.
    """
    url = "https://api.gdc.cancer.gov/files"
    filters = {
        "op": "=",
        "content": {
            "field": "files.file_name",
            "value": [barcode + "*"],
        },
    }
    params = {
        "filters": json.dumps(filters),
        "fields": "file_id,file_name,access,state,data_format,file_size,created_datetime",
        "format": "JSON",
        "size": 100,
        "sort": "created_datetime:desc",
    }
    r = httpx.get(url, params=params, timeout=60.0)
    r.raise_for_status()
    hits = r.json().get("data", {}).get("hits", [])

    svs = [
        h
        for h in hits
        if (h.get("data_format") == "SVS") or str(h.get("file_name", "")).lower().endswith(".svs")
    ]
    if not svs:
        return None

    cand = [h for h in svs if h.get("access") == "open"] or svs
    cand2 = [h for h in cand if str(h.get("state", "")).lower() == "released"] or cand
    cand2.sort(
        key=lambda h: (str(h.get("created_datetime", "")), int(h.get("file_size") or 0)),
        reverse=True,
    )
    return cand2[0].get("file_id")


def download_blob_to_file(
    bucket: storage.Bucket,
    blob_path: str,
    local_path: str,
    *,
    file_id: Optional[str] = None,
) -> Tuple[float, float]:
    """
    Same logic as embedder_32-path.py:
    1) try GCS object first
    2) if missing/fails, fallback to GDC /data/<file_id> streaming download
    """
    t0 = time.time()
    blob = bucket.blob(blob_path)
    blob.chunk_size = GCS_CHUNK_SIZE

    try:
        if not blob.exists():
            raise FileNotFoundError(f"GCS missing: gs://{bucket.name}/{blob_path}")
        blob.download_to_filename(local_path)
    except Exception:
        if not file_id:
            raise
        url = f"https://api.gdc.cancer.gov/data/{file_id}"
        timeout = httpx.Timeout(60.0, read=None)
        with httpx.stream("GET", url, follow_redirects=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_bytes(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

    dt = time.time() - t0
    size_mb = os.path.getsize(local_path) / (1024 * 1024)
    return size_mb, dt


def rgba_to_rgb_composite_white(rgba: np.ndarray) -> np.ndarray:
    if rgba.ndim != 3 or rgba.shape[2] not in (3, 4):
        raise ValueError(f"Expected HxWx3/4, got {rgba.shape}")
    if rgba.shape[2] == 3:
        return rgba

    rgb = rgba[..., :3].astype(np.float32)
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    out = rgb * a + 255.0 * (1.0 - a)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def choose_level_no_upsample(slide: openslide.OpenSlide, downsample_req: float) -> int:
    ds = [float(d) for d in slide.level_downsamples]
    candidates = [i for i, d in enumerate(ds) if d <= downsample_req]
    return max(candidates) if candidates else 0


def make_thumbnail_inter_area_stream(
    slide: openslide.OpenSlide,
    max_dim: int = 6000,
    max_level_mpx: float = 1e12,
    stripe_max_mpx: float = 60.0,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Same thumbnail algorithm as embedder_32-path.py.
    max_level_mpx defaults high here so explicit thumbnail export is not skipped.
    """
    w0, h0 = slide.dimensions
    scale = min(float(max_dim) / float(max(w0, h0)), 1.0)
    tw = max(1, int(round(w0 * scale)))
    th = max(1, int(round(h0 * scale)))

    downsample_req = max(w0 / tw, h0 / th)
    lvl = choose_level_no_upsample(slide, downsample_req)
    lvl_w, lvl_h = slide.level_dimensions[lvl]
    lvl_down = float(slide.level_downsamples[lvl])
    lvl_mpx = (float(lvl_w) * float(lvl_h)) / 1e6

    info: Dict[str, Any] = {
        "thumb_out_w": int(tw),
        "thumb_out_h": int(th),
        "thumb_level": int(lvl),
        "thumb_level_mpx": float(lvl_mpx),
        "thumb_skipped": False,
        "thumb_skip_reason": "",
    }

    if lvl_mpx > float(max_level_mpx):
        info["thumb_skipped"] = True
        info["thumb_skip_reason"] = f"level_mpx>{max_level_mpx}"
        return None, info

    stripe_max_px = max(1.0, float(stripe_max_mpx) * 1e6)
    stripe_h = int(max(1, min(lvl_h, int(stripe_max_px / max(lvl_w, 1)))))
    stripe_h = max(64, stripe_h) if lvl_h >= 64 else stripe_h
    stripe_h = min(stripe_h, lvl_h)

    out = np.empty((th, tw, 3), dtype=np.uint8)
    y_lvl = 0
    while y_lvl < lvl_h:
        h = min(stripe_h, lvl_h - y_lvl)
        y0_l0 = int(round(y_lvl * lvl_down))
        pil_rgba = slide.read_region((0, y0_l0), lvl, (lvl_w, h))
        rgba = np.asarray(pil_rgba, dtype=np.uint8)
        rgb = rgba_to_rgb_composite_white(rgba)

        y0_out = (y_lvl * th) // lvl_h
        y1_out = ((y_lvl + h) * th) // lvl_h
        if y_lvl + h >= lvl_h:
            y1_out = th
        out_h = int(max(1, y1_out - y0_out))

        interp = cv2.INTER_AREA
        if out_h > h or tw > lvl_w:
            interp = cv2.INTER_LINEAR
        strip_resized = cv2.resize(rgb, (tw, out_h), interpolation=interp)
        out[y0_out:y0_out + out_h, :, :] = strip_resized
        y_lvl += h

    return out, info


def get_file_name_from_gdc(file_id: str) -> str:
    r = httpx.get(f"https://api.gdc.cancer.gov/files/{file_id}", timeout=60.0)
    r.raise_for_status()
    obj = r.json()
    data = obj.get("data", {})
    file_name = data.get("file_name")
    if not file_name:
        raise RuntimeError(f"GDC API did not return file_name for {file_id}")
    return str(file_name)


def download_slide_with_fallback(
    source_bucket: storage.Bucket,
    file_id: str,
    full_path: str,
    local_slide: Path,
) -> Tuple[float, float, Optional[str]]:
    """
    Same as embedder_32-path.py:
    - try file_id/full_path
    - if stale/missing, resolve by barcode and retry with new file_id.
    """
    try:
        size_mb, dt = download_blob_to_file(
            source_bucket,
            full_path,
            str(local_slide),
            file_id=file_id,
        )
        return size_mb, dt, None
    except Exception as e:
        try:
            if local_slide.exists():
                local_slide.unlink()
        except Exception:
            pass

        barcode = barcode_from_full_path(full_path)
        new_id = resolve_current_file_id_from_barcode(barcode)
        if not new_id or new_id == file_id:
            raise RuntimeError(
                f"Download failed and barcode fallback unavailable for {file_id} ({barcode})"
            ) from e

        fname = Path(full_path).name
        new_full_path = f"{new_id}/{fname}"
        size_mb, dt = download_blob_to_file(
            source_bucket,
            new_full_path,
            str(local_slide),
            file_id=new_id,
        )
        return size_mb, dt, new_id


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket_out", required=True, help="Destination bucket name")
    ap.add_argument("--out_prefix", default="thumbnails", help="Destination object prefix")
    ap.add_argument("--scan_csv", default="", help="Optional local bucket_physical_scan.csv for file_id->full_path mapping")
    ap.add_argument("--max_dim", type=int, default=6000)
    ap.add_argument("--stripe_max_mpx", type=float, default=60.0)
    ap.add_argument("file_ids", nargs="+")
    args = ap.parse_args()

    work_dir = Path("/tmp/thumb_jobs")
    work_dir.mkdir(parents=True, exist_ok=True)

    client = storage.Client()
    source_bucket = client.bucket(SOURCE_BUCKET_NAME)
    out_bucket = client.bucket(args.bucket_out)
    full_path_map = load_full_path_map(args.scan_csv or None)

    for file_id in args.file_ids:
        full_path = full_path_map.get(file_id)
        if not full_path:
            file_name = get_file_name_from_gdc(file_id)
            full_path = f"{file_id}/{file_name}"
        local_slide = work_dir / f"{file_id}.svs"
        local_thumb = work_dir / f"{file_id}.jpg"

        print(f"[{file_id}] downloading {full_path}")
        size_mb, dt, resolved_id = download_slide_with_fallback(
            source_bucket,
            file_id,
            full_path,
            local_slide,
        )
        print(f"[{file_id}] downloaded {size_mb:.2f} MB in {dt:.1f}s")
        if resolved_id:
            print(f"[{file_id}] barcode fallback resolved current file_id={resolved_id}")

        slide = openslide.OpenSlide(str(local_slide))
        thumb_rgb, info = make_thumbnail_inter_area_stream(
            slide,
            max_dim=args.max_dim,
            max_level_mpx=1e12,
            stripe_max_mpx=args.stripe_max_mpx,
        )
        slide.close()

        if thumb_rgb is None:
            raise RuntimeError(f"[{file_id}] thumbnail skipped: {info}")

        ok = cv2.imwrite(
            str(local_thumb),
            cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), 92],
        )
        if not ok:
            raise RuntimeError(f"[{file_id}] failed to write thumbnail jpeg")

        remote_key = f"{args.out_prefix.rstrip('/')}/{file_id}.jpg"
        blob = out_bucket.blob(remote_key)
        blob.chunk_size = GCS_CHUNK_SIZE
        blob.upload_from_filename(str(local_thumb))
        print(
            f"[{file_id}] uploaded gs://{args.bucket_out}/{remote_key} "
            f"({info['thumb_out_w']}x{info['thumb_out_h']}, level={info['thumb_level']})"
        )

        try:
            local_slide.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    main()
