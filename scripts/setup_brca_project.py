#!/usr/bin/env python3
"""
Setup TCGA-BRCA project for Enso Atlas.

Downloads slide metadata, matches with clinical data, and generates labels.
Primary endpoint: tumor stage (early vs advanced).

By default this script writes to a project-scoped dataset root under
`data/projects/<project_id>` (default project_id: brca-stage).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests
import yaml

GDC_FILES_URL = "https://api.gdc.cancer.gov/files"
GDC_CASES_URL = "https://api.gdc.cancer.gov/cases"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROJECT_ID = "brca-stage"
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "projects.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare BRCA project metadata/labels")
    parser.add_argument(
        "--project-id",
        default=DEFAULT_PROJECT_ID,
        help=f"Project ID in projects.yaml (default: {DEFAULT_PROJECT_ID})",
    )
    parser.add_argument(
        "--projects-config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config/projects.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory (defaults to dataset root from project config)",
    )
    parser.add_argument(
        "--allow-non-modular-output",
        action="store_true",
        help="Allow output outside data/projects/<project_id> (not recommended)",
    )
    return parser.parse_args()


def _resolve_repo_relative(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (REPO_ROOT / p)


def _project_root_from_dataset_path(project_id: str, path_str: str) -> Path | None:
    """Return absolute data/projects/<project_id> root for a dataset path, if present."""
    abs_path = _resolve_repo_relative(path_str).resolve()
    marker = ("data", "projects", project_id)
    parts = abs_path.parts
    for idx in range(len(parts) - 2):
        if parts[idx: idx + 3] == marker:
            return Path(*parts[: idx + 3])
    return None


def resolve_output_dir(
    project_id: str,
    projects_config: Path,
    output_override: Path | None,
) -> Path:
    """Resolve output root from CLI override or project config."""
    if output_override is not None:
        return output_override if output_override.is_absolute() else (REPO_ROOT / output_override)

    if projects_config.exists():
        with open(projects_config, "r") as f:
            raw = yaml.safe_load(f) or {}
        proj = (raw.get("projects") or {}).get(project_id)
        if proj:
            dataset = proj.get("dataset") or {}
            roots = set()
            for field in ("slides_dir", "embeddings_dir", "labels_file"):
                path_val = dataset.get(field)
                if not path_val:
                    continue
                root = _project_root_from_dataset_path(project_id, path_val)
                if root is None:
                    raise ValueError(
                        f"Project '{project_id}' has non-modular dataset.{field} path in "
                        f"{projects_config}: {path_val}"
                    )
                roots.add(root)

            if len(roots) > 1:
                sorted_roots = ", ".join(str(p) for p in sorted(roots))
                raise ValueError(
                    f"Project '{project_id}' has inconsistent dataset roots in {projects_config}: "
                    f"{sorted_roots}"
                )
            if roots:
                return roots.pop()

    # Fallback for projects not yet in config
    return REPO_ROOT / "data" / "projects" / project_id


def validate_modular_output_dir(project_id: str, output_dir: Path) -> None:
    """Fail fast if output path violates per-project modular layout."""
    expected_root = (REPO_ROOT / "data" / "projects" / project_id).resolve()
    out = output_dir.resolve()
    if out != expected_root:
        raise ValueError(
            f"Output directory {out} is not project-scoped for '{project_id}'. "
            f"Expected: {expected_root}"
        )


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
    filters = json.dumps(
        {
            "op": "=",
            "content": {"field": "project.project_id", "value": "TCGA-BRCA"},
        }
    )

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
    args = parse_args()
    output_dir = resolve_output_dir(args.project_id, args.projects_config, args.output_dir)
    if not args.allow_non_modular_output:
        validate_modular_output_dir(args.project_id, output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using output directory: {output_dir}")

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
    print(
        f"Stage labels: {len(stage_labels)} cases "
        f"({neg} early, {pos} advanced, {100*pos/max(len(stage_labels),1):.0f}% advanced)"
    )

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
    def save_labels(label_dict, filename, label_name):
        slide_labels = {}
        for s in selected_slides:
            case_id = s.get("cases", [{}])[0].get("submitter_id", "unknown")
            if case_id in label_dict:
                slide_id = s["file_name"].replace(".svs", "")
                slide_labels[slide_id] = label_dict[case_id]
        pos_count = sum(v == 1 for v in slide_labels.values())
        neg_count = sum(v == 0 for v in slide_labels.values())
        total = len(slide_labels)
        print(
            f"  {label_name}: {total} slides "
            f"({pos_count} pos, {neg_count} neg, {100*pos_count/max(total,1):.0f}% pos)"
        )
        with open(output_dir / filename, "w") as f:
            json.dump(slide_labels, f, indent=2)
        return total

    print("\nLabel distributions for selected slides:")
    save_labels(stage_labels, "stage_labels.json", "Stage (early/advanced)")
    save_labels(surv5y_labels, "survival_5y_labels.json", "5Y survival")
    save_labels(surv3y_labels, "survival_3y_labels.json", "3Y survival")

    # Save download manifest
    manifest = [
        {"id": s["file_id"], "filename": s["file_name"], "size_mb": round(s["file_size"] / 1e6)}
        for s in selected_slides
    ]
    with open(output_dir / "download_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    with open(output_dir / "file_ids.txt", "w") as f:
        for s in selected_slides:
            f.write(s["file_id"] + "\n")

    print(f"\nOutputs in {output_dir}")
    print(f"  download_manifest.json ({len(manifest)} files)")
    print("  file_ids.txt (for gdc-client)")
    print("  stage_labels.json")
    print("  survival_5y_labels.json")
    print("  survival_3y_labels.json")


if __name__ == "__main__":
    main()
