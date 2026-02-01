#!/usr/bin/env python3
"""
Download the Ovarian Bevacizumab Response Dataset.

Dataset: "Histopathological whole slide image dataset for classification of 
treatment effectiveness to ovarian cancer"

Source: https://www.nature.com/articles/s41597-022-01127-6
Figshare: https://figshare.com/collections/Histopathological_whole_slide_image_dataset_for_classification_of_treatment_effectiveness_to_ovarian_cancer/5837857

Contains:
- 288 de-identified H&E WSIs from 78 patients
- Binary labels: effective vs invalid treatment response
"""

import os
import sys
from pathlib import Path
import zipfile
import requests
from tqdm import tqdm

from rich.console import Console

console = Console()

# Dataset metadata
DATASET_INFO = {
    "name": "Ovarian Bevacizumab Response Dataset",
    "source": "https://www.nature.com/articles/s41597-022-01127-6",
    "figshare_collection": "5837857",
    # Direct download links from Figshare (may need to be updated)
    "files": [
        # Note: These are placeholder URLs - actual download may require Figshare API
        # or manual download from the collection page
    ]
}


def download_file(url: str, dest: Path, desc: str = None) -> None:
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest, 'wb') as f:
        with tqdm(
            total=total_size,
            unit='iB',
            unit_scale=True,
            desc=desc or dest.name,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                pbar.update(size)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Ovarian Bevacizumab Response Dataset")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/ovarian_bevacizumab"),
        help="Directory to store dataset",
    )
    
    args = parser.parse_args()
    
    console.print("[bold]Ovarian Bevacizumab Response Dataset[/]")
    console.print(f"Source: {DATASET_INFO['source']}\n")
    
    # Create data directory
    args.data_dir.mkdir(parents=True, exist_ok=True)
    
    console.print("[yellow]⚠️ Automatic download not fully implemented.[/]")
    console.print("\nTo download the dataset manually:")
    console.print("1. Visit the Figshare collection:")
    console.print(f"   https://figshare.com/collections/_/{DATASET_INFO['figshare_collection']}")
    console.print("2. Download all WSI files")
    console.print(f"3. Extract to: {args.data_dir}")
    console.print("\nExpected structure:")
    console.print("  data/ovarian_bevacizumab/")
    console.print("  ├── effective/   # Responder slides")
    console.print("  ├── invalid/     # Non-responder slides")
    console.print("  └── clinical_info.csv")
    
    # Create placeholder structure
    (args.data_dir / "effective").mkdir(exist_ok=True)
    (args.data_dir / "invalid").mkdir(exist_ok=True)
    
    # Create sample labels CSV
    labels_path = args.data_dir / "labels.csv"
    if not labels_path.exists():
        labels_path.write_text("slide_id,label,patient_id,treatment\n")
        console.print(f"\n[green]Created empty labels template: {labels_path}[/]")
    
    console.print("\n[bold]Dataset preparation notes:[/]")
    console.print("- Total slides: 288 (from 78 patients)")
    console.print("- Use patient-level splits to avoid data leakage")
    console.print("- Labels: effective=1, invalid=0")
    
    console.print("\n[green]Data directory created: {args.data_dir}[/]")


if __name__ == "__main__":
    main()
