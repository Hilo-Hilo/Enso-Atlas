#!/usr/bin/env python3
"""
Download the Ovarian Bevacizumab Response Dataset from TCIA.

Dataset: "Histopathological whole slide image dataset for classification of 
treatment effectiveness to ovarian cancer"

Source: https://doi.org/10.1038/s41597-022-01127-6
TCIA Collection: https://www.cancerimagingarchive.net/collection/ovarian-bevacizumab-response/

Contains:
- 288 de-identified H&E WSIs from 78 patients (~253.8 GB)
- Binary labels: effective vs invalid treatment response
- License: CC BY 4.0

Note: WSI bulk download requires IBM Aspera. This script downloads metadata/labels only.
"""

import os
import sys
from pathlib import Path
import urllib.request
import json
import csv

# TCIA endpoints
PATHDB_CSV_URL = "https://pathdb.cancerimagingarchive.net/system/files/collectionmetadata/202405/Ovarian%20Bevacizumab%20Response_05-28-2024.csv"
CLINICAL_XLSX_URL = "https://www.cancerimagingarchive.net/wp-content/uploads/Final-patient_list.xlsx"
CA125_XLSX_URL = "https://www.cancerimagingarchive.net/wp-content/uploads/new_CA125-data_20230207.xlsx"


def download_file(url: str, dest: Path, desc: str = None) -> bool:
    """Download a file with progress indication."""
    try:
        print(f"Downloading: {desc or url}")
        urllib.request.urlretrieve(url, str(dest))
        print(f"  -> Saved to: {dest}")
        return True
    except Exception as e:
        print(f"  -> Error: {e}")
        return False


def parse_ca125_xlsx_simple(xlsx_path: Path) -> list:
    """
    Parse CA125 XLSX to extract labels.
    Uses zipfile to read XLSX (no pandas/openpyxl required).
    """
    import zipfile
    import xml.etree.ElementTree as ET
    
    records = []
    
    try:
        with zipfile.ZipFile(xlsx_path) as z:
            # Read shared strings
            shared_strings = []
            if 'xl/sharedStrings.xml' in z.namelist():
                with z.open('xl/sharedStrings.xml') as f:
                    tree = ET.parse(f)
                    ns = {'': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
                    for si in tree.findall('.//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}si'):
                        text_parts = []
                        for t in si.iter('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t'):
                            if t.text:
                                text_parts.append(t.text)
                        shared_strings.append(''.join(text_parts))
            
            # Read sheet1
            with z.open('xl/worksheets/sheet1.xml') as f:
                tree = ET.parse(f)
                rows = tree.findall('.//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}row')
                
                headers = []
                for i, row in enumerate(rows):
                    cells = row.findall('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}c')
                    values = []
                    for cell in cells:
                        v = cell.find('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v')
                        t = cell.get('t')
                        if v is not None and v.text:
                            if t == 's':  # shared string
                                idx = int(v.text)
                                values.append(shared_strings[idx] if idx < len(shared_strings) else '')
                            else:
                                values.append(v.text)
                        else:
                            values.append('')
                    
                    if i == 0:
                        headers = values
                    else:
                        if len(values) >= 4:
                            records.append(dict(zip(headers[:len(values)], values)))
    except Exception as e:
        print(f"Warning: Could not parse XLSX: {e}")
    
    return records


def generate_labels_csv(data_dir: Path, ca125_records: list) -> Path:
    """Generate labels.csv from CA125 data."""
    labels_path = data_dir / "labels.csv"
    
    with open(labels_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['slide_id', 'patient_id', 'label', 'label_name', 'ca125_before', 'ca125_after'])
        
        for rec in ca125_records:
            patient_id = rec.get('Patient ID', rec.get('patient_id', ''))
            image_no = rec.get('Image No.', rec.get('image_no', ''))
            treatment = rec.get('Treatment effect', rec.get('treatment_effect', ''))
            ca125_before = rec.get('CA-125 before', rec.get('ca125_before', ''))
            ca125_after = rec.get('CA-125 after', rec.get('ca125_after', ''))
            
            # Normalize
            slide_id = str(image_no).replace('.svs', '').strip()
            label = 1 if treatment.lower() == 'effective' else 0
            label_name = 'effective' if label == 1 else 'invalid'
            
            if slide_id and patient_id:
                writer.writerow([slide_id, patient_id, label, label_name, ca125_before, ca125_after])
    
    print(f"Generated labels.csv with {len(ca125_records)} entries")
    return labels_path


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Ovarian Bevacizumab Response Dataset metadata")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/ovarian_bevacizumab"),
        help="Directory to store dataset",
    )
    parser.add_argument(
        "--skip-xlsx",
        action="store_true",
        help="Skip XLSX download (if already present)",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Ovarian Bevacizumab Response Dataset (TCIA)")
    print("=" * 60)
    print(f"\nSource: https://doi.org/10.1038/s41597-022-01127-6")
    print(f"TCIA: https://www.cancerimagingarchive.net/collection/ovarian-bevacizumab-response/")
    print(f"\nDataset info:")
    print(f"  - 78 patients, 288 WSIs")
    print(f"  - Total size: ~253.8 GB")
    print(f"  - License: CC BY 4.0")
    
    # Create data directory
    args.data_dir.mkdir(parents=True, exist_ok=True)
    (args.data_dir / "wsi").mkdir(exist_ok=True)
    
    print(f"\n[1/3] Downloading PathDB metadata CSV...")
    pathdb_csv = args.data_dir / "pathdb_metadata.csv"
    download_file(PATHDB_CSV_URL, pathdb_csv, "PathDB slide metadata")
    
    print(f"\n[2/3] Downloading CA125 labels XLSX...")
    ca125_xlsx = args.data_dir / "ca125_labels.xlsx"
    if not args.skip_xlsx or not ca125_xlsx.exists():
        download_file(CA125_XLSX_URL, ca125_xlsx, "CA125 treatment labels")
    
    print(f"\n[3/3] Generating labels.csv...")
    if ca125_xlsx.exists():
        records = parse_ca125_xlsx_simple(ca125_xlsx)
        if records:
            generate_labels_csv(args.data_dir, records)
        else:
            print("Warning: Could not parse CA125 XLSX. Creating empty labels.csv template.")
            with open(args.data_dir / "labels.csv", 'w') as f:
                f.write("slide_id,patient_id,label,label_name,ca125_before,ca125_after\n")
    
    print("\n" + "=" * 60)
    print("METADATA DOWNLOAD COMPLETE")
    print("=" * 60)
    
    print(f"\nFiles saved to: {args.data_dir}/")
    print(f"  - pathdb_metadata.csv (slide IDs + patient mapping)")
    print(f"  - ca125_labels.xlsx (treatment effect labels)")
    print(f"  - labels.csv (generated, ready for training)")
    
    print("\nWSI DOWNLOAD (253.8 GB) requires IBM Aspera:")
    print("1. Visit: https://www.cancerimagingarchive.net/collection/ovarian-bevacizumab-response/")
    print("2. Click 'Download' under 'Tissue Slide Images'")
    print("3. Follow Aspera/Faspex instructions")
    print(f"4. Extract WSIs to: {args.data_dir}/wsi/")
    
    print("\nExpected WSI folder structure:")
    print("  data/ovarian_bevacizumab/wsi/")
    print("    ├── effective/  # or eXX folders")
    print("    └── invalid/    # or inX folders")


if __name__ == "__main__":
    main()
