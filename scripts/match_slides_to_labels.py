#!/usr/bin/env python3
"""Match downloaded slides to clinical labels."""
import os
import sys
from pathlib import Path
import pandas as pd
import re

def normalize_slide_name(name):
    """Normalize slide name for matching."""
    # Remove extension
    name = name.replace(".svs", "")
    # Remove -Y suffix (PathDB artifact)
    name = re.sub(r"-Y$", "", name)
    # Remove -N suffix
    name = re.sub(r"-N$", "", name)
    # Normalize spacing
    name = name.replace(" ", "").replace("-", "")
    return name.upper()

def main():
    data_dir = Path("data/ovarian_bev")
    slides_dir = data_dir / "slides"
    clinical_path = data_dir / "clinical.csv"
    
    # Load clinical data
    df = pd.read_csv(clinical_path)
    clinical_slides = set(df["slide_filename"].tolist())
    
    # Get downloaded slides
    downloaded = list(slides_dir.glob("*.svs"))
    downloaded_names = {f.name for f in downloaded if f.stat().st_size > 0}
    
    print(f"Clinical slides: {len(clinical_slides)}")
    print(f"Downloaded slides: {len(downloaded_names)}")
    
    # Exact matches
    exact_matches = clinical_slides & downloaded_names
    print(f"Exact matches: {len(exact_matches)}")
    
    # Try fuzzy matching for remaining
    unmatched_clinical = clinical_slides - exact_matches
    unmatched_downloaded = downloaded_names - exact_matches
    
    # Create normalized lookup
    norm_to_clinical = {normalize_slide_name(s): s for s in unmatched_clinical}
    norm_to_downloaded = {normalize_slide_name(s): s for s in unmatched_downloaded}
    
    fuzzy_matches = []
    for norm, clinical_name in norm_to_clinical.items():
        if norm in norm_to_downloaded:
            fuzzy_matches.append((clinical_name, norm_to_downloaded[norm]))
    
    print(f"Fuzzy matches: {len(fuzzy_matches)}")
    
    if fuzzy_matches:
        print("\nFuzzy match examples:")
        for c, d in fuzzy_matches[:5]:
            print(f"  {c} -> {d}")
    
    # Create rename map if needed
    if fuzzy_matches:
        print("\nCreating symlinks for fuzzy matches...")
        for clinical_name, downloaded_name in fuzzy_matches:
            src = slides_dir / downloaded_name
            dst = slides_dir / clinical_name
            if src.exists() and not dst.exists():
                os.symlink(src.name, dst)
                print(f"  Linked: {clinical_name}")
    
    # Final count
    final_matched = len(exact_matches) + len(fuzzy_matches)
    print(f"\nTotal matched: {final_matched} / {len(clinical_slides)}")
    print(f"Missing: {len(clinical_slides) - final_matched}")

if __name__ == "__main__":
    main()
