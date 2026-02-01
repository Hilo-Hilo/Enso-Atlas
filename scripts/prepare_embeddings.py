#!/usr/bin/env python3
"""
Prepare embeddings for the entire dataset.

This script:
1. Loads all WSIs from the data directory
2. Extracts patches using the WSI processor
3. Generates embeddings using Path Foundation
4. Saves embeddings to cache for training
"""

import os
import sys
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from rich.console import Console
from rich.progress import Progress, TaskID

console = Console()


def process_slide(
    slide_path: Path,
    output_dir: Path,
    config_path: Path,
) -> dict:
    """Process a single slide and save embeddings."""
    from enso_atlas.config import AtlasConfig
    from enso_atlas.wsi.processor import WSIProcessor
    from enso_atlas.embedding.embedder import PathFoundationEmbedder
    
    config = AtlasConfig.from_yaml(str(config_path))
    
    # Check if already processed
    output_path = output_dir / f"{slide_path.stem}.npy"
    if output_path.exists():
        return {"slide": slide_path.name, "status": "skipped", "cached": True}
    
    try:
        # Initialize processors
        wsi_processor = WSIProcessor(config.wsi)
        embedder = PathFoundationEmbedder(config.embedding)
        
        # Extract patches
        patches, coords = wsi_processor.extract_patches(slide_path)
        
        if len(patches) == 0:
            return {"slide": slide_path.name, "status": "failed", "error": "No patches extracted"}
        
        # Generate embeddings
        import numpy as np
        embeddings = embedder.embed(patches, cache_key=str(slide_path))
        
        # Save embeddings
        np.save(output_path, embeddings)
        
        # Save coordinates
        coords_path = output_dir / f"{slide_path.stem}_coords.npy"
        np.save(coords_path, np.array(coords))
        
        return {
            "slide": slide_path.name,
            "status": "success",
            "n_patches": len(patches),
            "embedding_shape": embeddings.shape,
        }
        
    except Exception as e:
        return {"slide": slide_path.name, "status": "failed", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Prepare embeddings for dataset")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing WSI files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/embeddings"),
        help="Output directory for embeddings",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/default.yaml"),
        help="Config file",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="**/*.svs",
        help="Glob pattern for slide files",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (GPU bottleneck usually limits this)",
    )
    
    args = parser.parse_args()
    
    # Find all slides
    slides = list(args.data_dir.glob(args.pattern))
    console.print(f"[bold]Found {len(slides)} slides to process[/]")
    
    if len(slides) == 0:
        console.print("[yellow]No slides found. Check the data directory and pattern.[/]")
        return
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process slides
    results = {"success": 0, "skipped": 0, "failed": 0}
    
    with Progress() as progress:
        task = progress.add_task("[cyan]Processing slides...", total=len(slides))
        
        for slide in slides:
            result = process_slide(slide, args.output_dir, args.config)
            
            status = result["status"]
            results[status] = results.get(status, 0) + 1
            
            if status == "success":
                console.print(f"[green]✓[/] {result['slide']} ({result['n_patches']} patches)")
            elif status == "skipped":
                console.print(f"[yellow]○[/] {result['slide']} (cached)")
            else:
                console.print(f"[red]✗[/] {result['slide']}: {result.get('error', 'Unknown error')}")
            
            progress.update(task, advance=1)
    
    console.print(f"\n[bold]Processing complete![/]")
    console.print(f"  Success: {results['success']}")
    console.print(f"  Skipped: {results['skipped']}")
    console.print(f"  Failed: {results['failed']}")
    console.print(f"\nEmbeddings saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
