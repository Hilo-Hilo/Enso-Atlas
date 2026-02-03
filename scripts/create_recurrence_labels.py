#!/usr/bin/env python3
"""Create recurrence labels file from clinical data."""

import pandas as pd
import os
from pathlib import Path

# Load clinical data
clinical = pd.read_csv('/app/data/tcga_full/clinical_priority.csv')
print(f"Total patients in clinical data: {len(clinical)}")

# Parse dfs_status - extract numeric value
def parse_dfs_status(val):
    if pd.isna(val):
        return None
    if isinstance(val, str):
        if val.startswith('1:'):
            return 1
        elif val.startswith('0:'):
            return 0
    return None

clinical['recurrence'] = clinical['dfs_status'].apply(parse_dfs_status)

# Filter to those with recurrence data
clinical_with_recurrence = clinical[clinical['recurrence'].notna()].copy()
print(f"Patients with recurrence data: {len(clinical_with_recurrence)}")

# Get embedding files
embed_dir = Path('/app/data/tcga_full/embeddings')
embed_files = [f for f in os.listdir(embed_dir) if f.endswith('.npy') and not f.endswith('_coords.npy')]
print(f"Total embedding files: {len(embed_files)}")

# Extract patient IDs from embedding files
def get_patient_id(filename):
    # TCGA-04-1331-01A-01-BS1.uuid.npy -> TCGA-04-1331
    parts = filename.split('-')
    if len(parts) >= 3:
        return '-'.join(parts[:3])
    return None

# Create mapping
records = []
for embed_file in embed_files:
    patient_id = get_patient_id(embed_file)
    if patient_id and patient_id in clinical_with_recurrence['patient_id'].values:
        row = clinical_with_recurrence[clinical_with_recurrence['patient_id'] == patient_id].iloc[0]
        records.append({
            'slide_id': embed_file.replace('.npy', ''),
            'patient_id': patient_id,
            'label': int(row['recurrence']),
            'dfs_months': row['dfs_months'] if pd.notna(row['dfs_months']) else ''
        })

labels_df = pd.DataFrame(records)
print(f"\nSlides with recurrence labels: {len(labels_df)}")
print(f"Unique patients: {labels_df['patient_id'].nunique()}")
print(f"\nLabel distribution:")
print(labels_df['label'].value_counts())
print(f"\nClass ratio: {labels_df['label'].value_counts()[0] / labels_df['label'].value_counts()[1]:.2f}:1 (0:1)")

# Save
output_path = '/app/data/tcga_full/recurrence_labels.csv'
labels_df.to_csv(output_path, index=False)
print(f"\nSaved to {output_path}")
