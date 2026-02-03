#!/usr/bin/env python3
"""Download all Ovarian Bevacizumab Response slides from TCIA PathDB."""
import sys
sys.path.insert(0, '/home/hansonwen/med-gemma-hackathon/venv/lib/python3.12/site-packages')

from tcia_utils import pathdb
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def main():
    print('Getting image list for collection 16 (Ovarian Bevacizumab Response)...')
    images = pathdb.getImages(16, format='df')
    print(f'Total images to download: {len(images)}')
    
    # Download all images with 8 workers
    print('Starting download (this will take several hours for 253GB)...')
    pathdb.downloadImages(
        images, 
        path='data/ovarian_bev/slides/',
        max_workers=8,  # Increase parallelism
        number=0  # 0 = all images
    )
    print('Download complete!')

if __name__ == '__main__':
    main()
