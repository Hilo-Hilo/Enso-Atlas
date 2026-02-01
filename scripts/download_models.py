#!/usr/bin/env python3
"""
Download required models for Enso Atlas.

Downloads:
- Path Foundation (google/path-foundation)
- MedGemma 1.5 4B (google/medgemma-4b-it)

Requires HuggingFace Hub access with appropriate permissions.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from huggingface_hub import snapshot_download, login
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

MODELS = {
    "path-foundation": {
        "repo_id": "google/path-foundation",
        "description": "Path Foundation - Histopathology embedding model",
    },
    "medgemma": {
        "repo_id": "google/medgemma-4b-it", 
        "description": "MedGemma 1.5 4B - Medical LLM for reporting",
    },
}


def download_model(repo_id: str, cache_dir: Path) -> None:
    """Download a model from HuggingFace Hub."""
    console.print(f"Downloading: [bold]{repo_id}[/]")
    
    snapshot_download(
        repo_id=repo_id,
        cache_dir=str(cache_dir),
        local_dir=str(cache_dir / repo_id.replace("/", "--")),
    )
    
    console.print(f"[green]✓ Downloaded {repo_id}[/]")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Download models for Enso Atlas")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("models"),
        help="Directory to cache models",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()) + ["all"],
        default=["all"],
        help="Models to download",
    )
    parser.add_argument(
        "--token",
        type=str,
        help="HuggingFace token (or set HF_TOKEN env var)",
    )
    
    args = parser.parse_args()
    
    # Setup cache directory
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Login to HuggingFace if token provided
    token = args.token or os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
        console.print("[green]Logged in to HuggingFace Hub[/]")
    
    # Determine which models to download
    if "all" in args.models:
        models_to_download = list(MODELS.values())
    else:
        models_to_download = [MODELS[m] for m in args.models]
    
    console.print(f"\n[bold]Downloading {len(models_to_download)} models to {args.cache_dir}[/]\n")
    
    for model in models_to_download:
        console.print(f"\n{model['description']}")
        try:
            download_model(model["repo_id"], args.cache_dir)
        except Exception as e:
            console.print(f"[red]Error downloading {model['repo_id']}: {e}[/]")
            console.print("[yellow]Note: Some models require HuggingFace access approval.[/]")
            console.print("[yellow]Visit the model page and accept the license terms.[/]")
    
    console.print("\n[bold green]Model download complete![/]")
    console.print(f"Models cached in: {args.cache_dir}")


if __name__ == "__main__":
    main()
