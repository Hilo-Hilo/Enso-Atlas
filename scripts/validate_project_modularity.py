#!/usr/bin/env python3
"""Validate per-project data path modularity constraints.

Fails with non-zero exit when project dataset paths leak into shared/legacy layouts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

# Make `src/` importable when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from enso_atlas.api.projects import PROJECTS_DATA_ROOT, ProjectRegistry  # noqa: E402


def validate_project_paths(config_path: Path) -> List[str]:
    errors: List[str] = []

    registry = ProjectRegistry(config_path)
    projects = registry.list_projects()
    if not projects:
        return [f"No projects found in {config_path}"]

    # 1) Per-project dataset path checks
    for pid, project in projects.items():
        errors.extend(project.validate_dataset_modularity())

        expected_root = PROJECTS_DATA_ROOT / pid
        slides_dir = Path(project.dataset.slides_dir)
        embeddings_dir = Path(project.dataset.embeddings_dir)
        labels_file = Path(project.dataset.labels_file)

        if slides_dir != expected_root / "slides":
            errors.append(
                f"{pid}: slides_dir must be '{expected_root / 'slides'}', got '{slides_dir}'"
            )

        valid_embeddings = {
            expected_root / "embeddings",
            expected_root / "embeddings" / "level0",
        }
        if embeddings_dir not in valid_embeddings:
            errors.append(
                f"{pid}: embeddings_dir must be one of {sorted(str(p) for p in valid_embeddings)}, "
                f"got '{embeddings_dir}'"
            )

        valid_label_suffixes = {".csv", ".json"}
        if labels_file.parent != expected_root:
            errors.append(
                f"{pid}: labels_file must be under '{expected_root}', got '{labels_file}'"
            )
        if labels_file.suffix.lower() not in valid_label_suffixes:
            errors.append(
                f"{pid}: labels_file must end with .csv or .json, got '{labels_file}'"
            )

    # 2) Ensure no projects share exact dataset paths
    seen = {}
    for pid, project in projects.items():
        for field_name, value in (
            ("slides_dir", project.dataset.slides_dir),
            ("embeddings_dir", project.dataset.embeddings_dir),
            ("labels_file", project.dataset.labels_file),
        ):
            key = (field_name, value)
            if key in seen and seen[key] != pid:
                errors.append(
                    f"Path collision: {field_name}='{value}' shared by projects '{seen[key]}' and '{pid}'"
                )
            else:
                seen[key] = pid

    return errors


def validate_code_patterns(repo_root: Path) -> List[str]:
    """Guard against reintroducing known hardcoded legacy paths in modular files."""
    errors: List[str] = []

    checks = {
        "scripts/setup_luad_project.py": [
            '"data" / "luad"',
            "data/luad",
        ],
        "scripts/setup_brca_project.py": [
            '"data" / "brca"',
            "data/brca",
        ],
        "src/enso_atlas/api/project_routes.py": [
            'f"data/{body.id}/slides"',
            'f"data/{body.id}/embeddings',
            'f"data/{body.id}/labels',
        ],
        "src/enso_atlas/api/projects.py": [
            'slides_dir: str = "data/slides"',
            'embeddings_dir: str = "data/embeddings/level0"',
            'labels_file: str = "data/labels.csv"',
        ],
    }

    for rel_path, forbidden_snippets in checks.items():
        path = repo_root / rel_path
        if not path.exists():
            errors.append(f"Missing file for modularity check: {rel_path}")
            continue

        text = path.read_text(encoding="utf-8")
        for snippet in forbidden_snippets:
            if snippet in text:
                errors.append(
                    f"Legacy path pattern found in {rel_path}: {snippet}"
                )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate project data-path modularity")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/projects.yaml"),
        help="Path to project config YAML",
    )
    parser.add_argument(
        "--skip-code-scan",
        action="store_true",
        help="Skip static code-pattern checks",
    )
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else (REPO_ROOT / args.config)

    errors = validate_project_paths(config_path)
    if not args.skip_code_scan:
        errors.extend(validate_code_patterns(REPO_ROOT))

    if errors:
        print("❌ Project modularity validation failed:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("✅ Project modularity validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
