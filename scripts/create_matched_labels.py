#!/usr/bin/env python3
"""Match embedding filenames to clinical data labels."""

import os
import pandas as pd
from pathlib import Path

# Paths - use /workspace in container
base_dir = Path("/workspace/data/tcga_full")
embeddings_dir = base_dir / "embeddings"
clinical_path = base_dir / "clinical_priority.csv"
output_path = base_dir / "matched_labels.csv"

# Load clinical data
clinical_df = pd.read_csv(clinical_path)
print(f"Clinical data: {len(clinical_df)} patients")

# Get unique patient IDs with platinum status
clinical_df = clinical_df[clinical_df["platinum_sensitive"].notna()]
print(f"Patients with platinum status: {len(clinical_df)}")

# Create lookup: patient_id -> platinum_sensitive
patient_lookup = dict(zip(clinical_df["patient_id"], clinical_df["platinum_sensitive"]))

# Get embedding files (excluding coords)
embedding_files = [f for f in os.listdir(embeddings_dir) 
                   if f.endswith(".npy") and "coords" not in f]
print(f"Embedding files: {len(embedding_files)}")

# Match embeddings to clinical data
matched = []
unmatched_patients = set()

for emb_file in sorted(embedding_files):
    # Extract patient ID: TCGA-04-1331-01A-01-BS1.xxx.npy -> TCGA-04-1331
    parts = emb_file.split("-")
    patient_id = "-".join(parts[:3])
    
    # Get slide_id (filename without .npy)
    slide_id = emb_file.replace(".npy", "")
    
    if patient_id in patient_lookup:
        status = patient_lookup[patient_id]
        label = 1 if status == "sensitive" else 0
        matched.append({
            "slide_id": slide_id,
            "patient_id": patient_id,
            "label": label,
            "platinum_status": status
        })
    else:
        unmatched_patients.add(patient_id)

print(f"Matched slides: {len(matched)}")
print(f"Unmatched patients: {len(unmatched_patients)}")
if unmatched_patients:
    print(f"  Examples: {list(unmatched_patients)[:5]}")

# Count by label
df = pd.DataFrame(matched)
print(f"\nLabel distribution:")
print(df["label"].value_counts())

# Save
df.to_csv(output_path, index=False)
print(f"\nSaved to: {output_path}")
