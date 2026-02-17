#!/usr/bin/env python3
"""
Setup TCGA-BRCA project for Enso Atlas.
Downloads slide metadata, matches with clinical data, generates labels.
Uses tumor stage (early vs advanced) as primary endpoint.
"""

import json
import sys
from pathlib import Path
import requests

GDC_FILES_URL = "https://api.gdc.cancer.gov/files"
GDC_CASES_URL = "https://api.gdc.cancer.gov/cases"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "brca"


def get_all_brca_slides():
    """Get ALL TCGA-BRCA diagnostic slide file IDs."""
    filters = {
        "op": "and",
        "content": [
            {"op": "=", "content": {"field": "cases.project.project_id", "value": "TCGA-BRCA"}},
            {"op": "=", "content": {"field": "data_type", "value": "Slide Image"}},
            {"op": "=", "content": {"field": "experimental_strategy", "value": "Diagnostic Slide"}},
        ],
    }

    all_hits = []
    offset = 0
    while True:
        params = {
            "filters": json.dumps(filters),
            "fields": "file_id,file_name,file_size,cases.submitter_id",
            "size": 500,
            "from": offset,
            "format": "json",
        }
        r = requests.get(GDC_FILES_URL, params=params)
        r.raise_for_status()
        data = r.json()["data"]
        hits = data["hits"]
        if not hits:
            break
        all_hits.extend(hits)
        offset += len(hits)
        if offset >= data["pagination"]["total"]:
            break

    print(f"Found {len(all_hits)} BRCA diagnostic slides total")
    return all_hits


def get_clinical_data():
    """Get clinical data: vital status, days_to_death, tumor stage."""
    filters = json.dumps({
        "op": "=",
        "content": {"field": "project.project_id", "value": "TCGA-BRCA"},
    })

    all_cases = []
    offset = 0
    while True:
        params = {
            "filters": filters,
            "fields": "submitter_id,demographic.vital_status,demographic.days_to_death,diagnoses.days_to_last_follow_up,diagnoses.tumor_stage,diagnoses.ajcc_pathologic_stage,diagnoses.morphology",
            "size": 500,
            "from": offset,
            "format": "json",
        }
        r = requests.get(GDC_CASES_URL, params=params)
        r.raise_for_status()
        data = r.json()["data"]
        hits = data["hits"]
        if not hits:
            break
        all_cases.extend(hits)
        offset += len(hits)
        if offset >= data["pagination"]["total"]:
            break

    print(f"Retrieved clinical data for {len(all_cases)} BRCA cases")
    return all_cases


def compute_stage_labels(cases):
    """
    Binary classification: Early stage (I/II) vs Advanced stage (III/IV).
    Uses AJCC pathologic stage.
    """
    labels = {}
    for case in cases:
        case_id = case["submitter_id"]
        for dx in case.get("diagnoses", []):
            stage = dx.get("ajcc_pathologic_stage") or dx.get("tumor_stage") or ""
            stage = stage.lower().strip()
            if not stage or stage in ("not reported", "unknown", "stage x"):
                continue
            # Map to binary: Stage I/II = 0 (early), Stage III/IV = 1 (advanced)
            if any(s in stage for s in ["stage i", "stage ia", "stage ib", "stage ii"]):
                if "stage iii" not in stage:  # avoid matching "stage iiia" etc
                    labels[case_id] = 0  # early
            if any(s in stage for s in ["stage iii", "stage iv"]):
                labels[case_id] = 1  # advanced
    return labels


def compute_survival_labels(cases, years=5):
    """Compute N-year survival labels."""
    threshold_days = years * 365.25
    labels = {}
    for case in cases:
        case_id = case["submitter_id"]
        demo = case.get("demographic", {})
        vital = demo.get("vital_status", "")
        days_to_death = demo.get("days_to_death")
        days_to_follow_up = None
        for dx in case.get("diagnoses", []):
            dtf = dx.get("days_to_last_follow_up")
            if dtf is not None:
                if days_to_follow_up is None or dtf > days_to_follow_up:
                    days_to_follow_up = dtf
        if vital == "Dead" and days_to_death is not None:
            labels[case_id] = 1 if days_to_death > threshold_days else 0
        elif vital == "Alive" and days_to_follow_up is not None:
            if days_to_follow_up >= threshold_days:
                labels[case_id] = 1
    return labels


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Get all slides
    slides = get_all_brca_slides()

    # Build case -> slides mapping
    case_to_slides = {}
    for s in slides:
        case_id = s.get("cases", [{}])[0].get("submitter_id", "unknown")
        if case_id not in case_to_slides:
            case_to_slides[case_id] = []
        case_to_slides[case_id].append(s)

    print(f"Unique cases: {len(case_to_slides)}")

    # Get clinical data
    clinical = get_clinical_data()

    # Compute all label sets
    stage_labels = compute_stage_labels(clinical)
    surv5y_labels = compute_survival_labels(clinical, years=5)
    surv3y_labels = compute_survival_labels(clinical, years=3)

    pos = sum(v == 1 for v in stage_labels.values())
    neg = sum(v == 0 for v in stage_labels.values())
    print(f"Stage labels: {len(stage_labels)} cases ({neg} early, {pos} advanced, {100*pos/max(len(stage_labels),1):.0f}% advanced)")

    # Select slides: prefer cases that have stage labels, pick 1 slide per case, prefer smaller
    selected_slides = []
    for case_id in stage_labels:
        if case_id in case_to_slides:
            # Pick smallest slide for this case
            case_slides = sorted(case_to_slides[case_id], key=lambda s: s["file_size"])
            selected_slides.append(case_slides[0])

    # If we still need more, add unlabeled cases too (for embedding demonstration)
    remaining = set(case_to_slides.keys()) - set(stage_labels.keys())
    for case_id in sorted(remaining):
        if len(selected_slides) >= 120:
            break
        case_slides = sorted(case_to_slides[case_id], key=lambda s: s["file_size"])
        selected_slides.append(case_slides[0])

    # Sort by size and cap at 120
    selected_slides.sort(key=lambda s: s["file_size"])
    selected_slides = selected_slides[:120]

    total_gb = sum(s["file_size"] for s in selected_slides) / 1e9
    print(f"\nSelected {len(selected_slides)} slides ({total_gb:.1f} GB)")

    # Generate slide-level labels
    def save_labels(label_dict, cases_dict, filename, label_name):
        slide_labels = {}
        for s in selected_slides:
            case_id = s.get("cases", [{}])[0].get("submitter_id", "unknown")
            if case_id in label_dict:
                slide_id = s["file_name"].replace(".svs", "")
                slide_labels[slide_id] = label_dict[case_id]
        pos = sum(v == 1 for v in slide_labels.values())
        neg = sum(v == 0 for v in slide_labels.values())
        total = len(slide_labels)
        print(f"  {label_name}: {total} slides ({pos} pos, {neg} neg, {100*pos/max(total,1):.0f}% pos)")
        with open(OUTPUT_DIR / filename, "w") as f:
            json.dump(slide_labels, f, indent=2)
        return total

    print("\nLabel distributions for selected slides:")
    save_labels(stage_labels, case_to_slides, "stage_labels.json", "Stage (early/advanced)")
    save_labels(surv5y_labels, case_to_slides, "survival_5y_labels.json", "5Y survival")
    save_labels(surv3y_labels, case_to_slides, "survival_3y_labels.json", "3Y survival")

    # Save download manifest
    manifest = [{"id": s["file_id"], "filename": s["file_name"], "size_mb": round(s["file_size"]/1e6)} for s in selected_slides]
    with open(OUTPUT_DIR / "download_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    with open(OUTPUT_DIR / "file_ids.txt", "w") as f:
        for s in selected_slides:
            f.write(s["file_id"] + "\n")

    print(f"\nOutputs in {OUTPUT_DIR}")
    print(f"  download_manifest.json ({len(manifest)} files)")
    print(f"  file_ids.txt (for gdc-client)")
    print(f"  stage_labels.json")
    print(f"  survival_5y_labels.json")
    print(f"  survival_3y_labels.json")


if __name__ == "__main__":
    main()
