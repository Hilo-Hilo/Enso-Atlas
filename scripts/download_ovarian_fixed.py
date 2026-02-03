#!/usr/bin/env python3
"""Download Ovarian Bevacizumab slides with robust error handling."""
import sys
sys.path.insert(0, "/home/hansonwen/med-gemma-hackathon/venv/lib/python3.12/site-packages")

import os
import time
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm
from tcia_utils import pathdb

stats_lock = Lock()
stats = {"success": 0, "failed": 0, "bytes": 0}

def download_file(args):
    url, filepath = args
    max_retries = 3
    timeout = 600  # 10 min per file
    
    for attempt in range(max_retries):
        try:
            # Use HTTPS directly to avoid redirect
            https_url = url.replace("http://", "https://")
            r = requests.get(https_url, stream=True, timeout=timeout)
            r.raise_for_status()
            
            with open(filepath, "wb") as f:
                for chunk in r.iter_content(chunk_size=131072):
                    if chunk:
                        f.write(chunk)
            
            actual_size = os.path.getsize(filepath)
            if actual_size > 1000:  # At least 1KB
                with stats_lock:
                    stats["success"] += 1
                    stats["bytes"] += actual_size
                return True, filepath.name, actual_size
            else:
                os.remove(filepath)
                time.sleep(5)
        except Exception as e:
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except:
                    pass
            if attempt < max_retries - 1:
                time.sleep(10 * (attempt + 1))
    
    with stats_lock:
        stats["failed"] += 1
    return False, filepath.name, "Failed after retries"

def main():
    output_dir = Path("data/ovarian_bev/slides")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Getting image list...")
    images = pathdb.getImages(16, format="df")
    print(f"Total images: {len(images)}")
    
    to_download = []
    for _, row in images.iterrows():
        url = row["imageUrl"]
        filename = url.split("/")[-1]
        filepath = output_dir / filename
        
        if not filepath.exists() or filepath.stat().st_size < 1000:
            to_download.append((url, filepath))
    
    print(f"Files to download: {len(to_download)}")
    
    if not to_download:
        print("All files already downloaded!")
        return
    
    failed = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(download_file, args): args for args in to_download}
        
        with tqdm(total=len(to_download), desc="Downloading") as pbar:
            for future in as_completed(futures):
                ok, name, result = future.result()
                if ok:
                    total_gb = stats["bytes"] / 1e9
                    pbar.set_postfix(file=name[:15], total=f"{total_gb:.1f}GB")
                else:
                    failed.append((name, result))
                pbar.update(1)
    
    success_count = stats["success"]
    total_gb = stats["bytes"] / 1e9
    print(f"\nDone! Success: {success_count}, Failed: {len(failed)}, Total: {total_gb:.1f}GB")
    if failed:
        print("Failed files:")
        for name, err in failed[:20]:
            print(f"  {name}: {err}")

if __name__ == "__main__":
    main()
