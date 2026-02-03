#!/usr/bin/env python3
"""Prepare Ovarian Bevacizumab Response dataset for training.

Creates:
- clinical.csv: Merged clinical data with labels
- splits.csv: Patient-level train/val/test splits (70/15/15)
"""
import sys
sys.path.insert(0, '/home/hansonwen/med-gemma-hackathon/venv/lib/python3.12/site-packages')

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import json

def load_and_merge_clinical_data(data_dir):
    """Load clinical data from both Excel sheets and merge."""
    xl = pd.ExcelFile(data_dir / 'clinical_ca125.xlsx')
    
    # Load both sheets
    effective = pd.read_excel(xl, sheet_name='Ovary.effective-162')
    invalid = pd.read_excel(xl, sheet_name='Ovary.invalid-126')
    
    # Standardize columns
    effective.columns = ['No', 'patient_id', 'treatment_effect', 'slide_filename', 'ca125_before', 'ca125_after']
    invalid.columns = ['No', 'patient_id', 'treatment_effect', 'slide_filename', 'ca125_before', 'ca125_after']
    
    # Combine
    clinical = pd.concat([effective, invalid], ignore_index=True)
    
    # Add binary label: effective=1, invalid=0
    clinical['label'] = (clinical['treatment_effect'] == 'effective').astype(int)
    
    # Extract slide_id (without .svs)
    clinical['slide_id'] = clinical['slide_filename'].str.replace('.svs', '', regex=False)
    
    return clinical

def create_patient_level_splits(clinical, seed=42):
    """Create stratified patient-level train/val/test splits.
    
    70% train, 15% val, 15% test
    Stratified by patient-level response (majority vote if multiple slides).
    """
    # Get patient-level labels (majority vote)
    patient_labels = clinical.groupby('patient_id')['label'].agg(
        lambda x: 1 if x.mean() >= 0.5 else 0
    ).reset_index()
    patient_labels.columns = ['patient_id', 'patient_label']
    
    # Split patients
    patients = patient_labels['patient_id'].values
    labels = patient_labels['patient_label'].values
    
    # First split: 70% train, 30% temp
    train_patients, temp_patients, train_labels, temp_labels = train_test_split(
        patients, labels, test_size=0.30, stratify=labels, random_state=seed
    )
    
    # Second split: 50% of temp = 15% val, 15% test
    val_patients, test_patients, _, _ = train_test_split(
        temp_patients, temp_labels, test_size=0.50, stratify=temp_labels, random_state=seed
    )
    
    # Create split mapping
    split_map = {}
    for p in train_patients:
        split_map[p] = 'train'
    for p in val_patients:
        split_map[p] = 'val'
    for p in test_patients:
        split_map[p] = 'test'
    
    return split_map

def main():
    data_dir = Path('/home/hansonwen/med-gemma-hackathon/data/ovarian_bev')
    
    print('Loading clinical data...')
    clinical = load_and_merge_clinical_data(data_dir)
    print(f'Total slides: {len(clinical)}')
    print(f'Unique patients: {clinical["patient_id"].nunique()}')
    print(f'Label distribution: {clinical["label"].value_counts().to_dict()}')
    
    print('\nCreating patient-level splits...')
    split_map = create_patient_level_splits(clinical)
    clinical['split'] = clinical['patient_id'].map(split_map)
    
    # Summary
    print('\n=== Split Summary ===')
    for split in ['train', 'val', 'test']:
        subset = clinical[clinical['split'] == split]
        n_patients = subset['patient_id'].nunique()
        n_slides = len(subset)
        n_effective = (subset['label'] == 1).sum()
        n_invalid = (subset['label'] == 0).sum()
        print(f'{split:5s}: {n_patients:2d} patients, {n_slides:3d} slides '
              f'(effective: {n_effective}, invalid: {n_invalid}, '
              f'ratio: {n_effective/n_slides*100:.1f}%)')
    
    # Save
    output_cols = ['patient_id', 'slide_id', 'slide_filename', 'label', 
                   'treatment_effect', 'ca125_before', 'ca125_after', 'split']
    clinical[output_cols].to_csv(data_dir / 'clinical.csv', index=False)
    print(f'\nSaved clinical.csv with {len(clinical)} rows')
    
    # Also save patient-level splits for reference
    patient_splits = clinical.groupby('patient_id').agg({
        'label': 'first',
        'split': 'first'
    }).reset_index()
    patient_splits.to_csv(data_dir / 'patient_splits.csv', index=False)
    print(f'Saved patient_splits.csv with {len(patient_splits)} patients')
    
    # Save stats
    stats = {
        'total_slides': len(clinical),
        'total_patients': int(clinical['patient_id'].nunique()),
        'effective_slides': int((clinical['label'] == 1).sum()),
        'invalid_slides': int((clinical['label'] == 0).sum()),
        'splits': {}
    }
    for split in ['train', 'val', 'test']:
        subset = clinical[clinical['split'] == split]
        stats['splits'][split] = {
            'patients': int(subset['patient_id'].nunique()),
            'slides': len(subset),
            'effective': int((subset['label'] == 1).sum()),
            'invalid': int((subset['label'] == 0).sum())
        }
    
    with open(data_dir / 'dataset_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print('Saved dataset_stats.json')

if __name__ == '__main__':
    main()
