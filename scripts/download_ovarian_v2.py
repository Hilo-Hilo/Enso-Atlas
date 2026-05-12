#!/usr/bin/env python3
"""Download Ovarian Bevacizumab slides with rate limiting and retries."""
import sys
sys.path.insert(0, "/home/hansonwen/med-gemma-hackathon/venv/lib/python3.12/site-packages")

import os
import time
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from tcia_utils import pathdb

def download_file(url, filepath, max_retries=3, timeout=600):
    """Download a single file with retries."""
    for attempt in range(max_retries):
        try:
            r = requests.get(url, stream=True, timeout=timeout, allow_redirects=True)
            r.raise_for_status()
            
            with open(filepath, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
            
            actual_size = os.path.getsize(filepath)
            if actual_size > 0:
                return True, actual_size
            else:
                os.remove(filepath)
                time.sleep(2)  # Wait before retry
        except Exception as e:
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except:
                    pass
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                return False, str(e)
    return False, "Max retries exceeded"

def main():
    output_dir = Path("data/ovarian_bev/slides")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Getting image list...")
    images = pathdb.getImages(16, format="df")
    print(f"Total images in collection: {len(images)}")
    
    # Find files that need downloading
    to_download = []
    for _, row in images.iterrows():
        url = row["imageUrl"]
        filename = url.split("/")[-1]
        filepath = output_dir / filename
        
        if not filepath.exists() or filepath.stat().st_size == 0:
            to_download.append((url, filepath))
    
    print(f"Files to download: {len(to_download)}")
    
    if not to_download:
        print("All files already downloaded!")
        return
    
    # Sequential download with progress - more reliable
    success = 0
    failed = []
    total_bytes = 0
    
    with tqdm(total=len(to_download), desc="Downloading", unit="file") as pbar:
        for url, fp in to_download:
            ok, result = download_file(url, fp)
            if ok:
                success += 1
                total_bytes += result
                pbar.set_postfix({
                    "file": fp.name[:20],
                    "size": f"{result/1e6:.1f}MB",
                    "total": f"{total_bytes/1e9:.1f}GB"
                })
            else:
                failed.append((fp.name, result))
            pbar.update(1)
    
    print(f"\nDownload complete!")
    print(f"Success: {success}")
    print(f"Failed: {len(failed)}")
    print(f"Total downloaded: {total_bytes/1e9:.2f} GB")
    
    if failed:
        print("\nFailed files:")
        for name, error in failed[:20]:
            print(f"  {name}: {error}")

if __name__ == "__main__":
    main()
