#!/usr/bin/env python3
"""
Batch re-embed all slides at Level 0 (full resolution).

Safe overnight runner for DGX. Sequential by default (concurrency=1)
to avoid GPU OOM. Uses the locally-cached Path Foundation TF model
and never downloads from HuggingFace.

Usage:
    # Run against the local API server:
    python scripts/batch_reembed_level0.py --api-url http://localhost:8000

    # Run on DGX (default port 8003 from docker-compose):
    python scripts/batch_reembed_level0.py --api-url http://100.111.126.23:8003

    # Dry run (just list slides):
    python scripts/batch_reembed_level0.py --api-url http://localhost:8000 --dry-run

    # Custom concurrency (careful - GPU memory):
    python scripts/batch_reembed_level0.py --api-url http://localhost:8000 --concurrency 2

    # Only specific slides:
    python scripts/batch_reembed_level0.py --api-url http://localhost:8000 --slide-ids TCGA-01 TCGA-02
"""

import argparse
import json
import sys
import time
import requests
from typing import Optional


def main():
    parser = argparse.ArgumentParser(
        description="Batch re-embed all slides at Level 0 (full resolution)"
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="Backend API base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=0,
        choices=[0, 1],
        help="Embedding resolution level (default: 0 = full res)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent embedding workers (default: 1, max: 4)",
    )
    parser.add_argument(
        "--no-force",
        action="store_true",
        help="Skip slides that already have embeddings",
    )
    parser.add_argument(
        "--slide-ids",
        nargs="+",
        default=None,
        help="Specific slide IDs to re-embed (default: all slides)",
    )
    parser.add_argument(
        "--project-id",
        default=None,
        help="Optional project ID to scope slide listing and batch embedding",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Just list slides and exit without embedding",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=10,
        help="Seconds between progress polls (default: 10)",
    )
    args = parser.parse_args()

    api_url = args.api_url.rstrip("/")

    # 1. Health check
    print(f"Checking API at {api_url}...")
    try:
        resp = requests.get(f"{api_url}/api/health", timeout=15)
        resp.raise_for_status()
        health = resp.json()
        print(f"  Status: {health.get('status', 'unknown')}")
        print(f"  Slides available: {health.get('slides_available', '?')}")
        print(f"  CUDA available: {health.get('cuda_available', '?')}")
    except Exception as e:
        print(f"ERROR: Cannot reach API at {api_url}: {e}")
        sys.exit(1)

    # 2. Get slide list
    if args.slide_ids:
        slide_ids = args.slide_ids
        print(f"\nTarget slides: {len(slide_ids)} (specified)")
    else:
        print("\nFetching slide list...")
        params = {"project_id": args.project_id} if args.project_id else None
        resp = requests.get(f"{api_url}/api/slides", params=params, timeout=30)
        resp.raise_for_status()
        slides = resp.json()
        if isinstance(slides, list):
            slide_ids = [s["slide_id"] for s in slides]
        else:
            slide_ids = [s["slide_id"] for s in slides.get("slides", [])]
        scope_label = f" for project '{args.project_id}'" if args.project_id else ""
        print(f"  Found {len(slide_ids)} slides{scope_label}")

    if not slide_ids:
        print("No slides found. Exiting.")
        sys.exit(0)

    if args.dry_run:
        print("\n=== DRY RUN ===")
        for i, sid in enumerate(slide_ids, 1):
            print(f"  {i:3d}. {sid}")
        print(f"\nWould re-embed {len(slide_ids)} slides at level {args.level}")
        sys.exit(0)

    # 3. Start batch embedding
    scope_label = f", project_id={args.project_id}" if args.project_id else ""
    print(f"\nStarting batch re-embed: {len(slide_ids)} slides, level={args.level}, "
          f"force={not args.no_force}, concurrency={args.concurrency}{scope_label}")
    print("=" * 60)

    payload = {
        "level": args.level,
        "force": not args.no_force,
        "slide_ids": slide_ids,
        "concurrency": args.concurrency,
        "project_id": args.project_id,
    }

    resp = requests.post(f"{api_url}/api/embed-slides/batch", json=payload, timeout=30)
    resp.raise_for_status()
    result = resp.json()

    task_id = result.get("batch_task_id")
    if not task_id:
        print(f"ERROR: No task_id returned: {result}")
        sys.exit(1)

    print(f"  Task ID: {task_id}")
    print(f"  Status: {result.get('status')}")
    print(f"  Message: {result.get('message')}")
    print()

    # 4. Poll for progress
    start_time = time.time()
    last_slide = ""

    while True:
        time.sleep(args.poll_interval)

        try:
            resp = requests.get(
                f"{api_url}/api/embed-slides/batch/status/{task_id}",
                timeout=15,
            )
            resp.raise_for_status()
            status = resp.json()
        except Exception as e:
            print(f"  [poll error: {e}]")
            continue

        task_status = status.get("status", "unknown")
        progress = status.get("progress", 0)
        completed = status.get("completed_slides", 0)
        total = status.get("total_slides", len(slide_ids))
        current = status.get("current_slide_id", "")
        message = status.get("message", "")
        elapsed = time.time() - start_time

        # Print progress
        if current != last_slide:
            last_slide = current
            eta_str = ""
            if completed > 0 and completed < total:
                per_slide = elapsed / completed
                remaining = (total - completed) * per_slide
                eta_min = remaining / 60
                eta_str = f" | ETA: {eta_min:.0f}m"

            print(
                f"  [{elapsed/60:.1f}m] {completed}/{total} ({progress:.1f}%) "
                f"- {current[:30]}{eta_str}"
            )

        if task_status in ("completed", "failed", "cancelled"):
            break

    # 5. Print final summary
    print()
    print("=" * 60)
    print(f"RESULT: {task_status.upper()}")
    elapsed = time.time() - start_time
    print(f"Total time: {elapsed/60:.1f} minutes")

    summary = status.get("summary", {})
    if summary:
        print(f"  Succeeded: {summary.get('succeeded', '?')}")
        print(f"  Failed: {summary.get('failed', '?')}")
        print(f"  Skipped: {summary.get('skipped', '?')}")
        print(f"  Total patches: {summary.get('total_patches', '?'):,}")

    # Print any failed slides
    results = status.get("results", [])
    failed = [r for r in results if r.get("status") == "failed"]
    if failed:
        print(f"\nFailed slides ({len(failed)}):")
        for r in failed:
            print(f"  - {r['slide_id']}: {r.get('error', 'unknown error')}")

    if task_status == "completed":
        print("\n✅ Batch re-embed completed successfully!")
        sys.exit(0)
    else:
        print(f"\n❌ Batch re-embed {task_status}")
        sys.exit(1)


if __name__ == "__main__":
    main()
