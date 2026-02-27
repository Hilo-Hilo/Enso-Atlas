#!/usr/bin/env python3
"""Validate project modularity invariants for config and code.

Fails with non-zero exit when project dataset paths leak into shared/legacy
layouts or when known legacy path patterns reappear in modularized files.
"""

from __future__ import annotations

import argparse
from importlib.util import module_from_spec, spec_from_file_location
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
PROJECTS_MODULE_PATH = SRC_DIR / "enso_atlas" / "api" / "projects.py"


def _load_projects_module():
    spec = spec_from_file_location("enso_atlas_api_projects", PROJECTS_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load project config module from {PROJECTS_MODULE_PATH}")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_projects_module = _load_projects_module()
PROJECTS_DATA_ROOT = _projects_module.PROJECTS_DATA_ROOT
ProjectRegistry = _projects_module.ProjectRegistry


def _resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (REPO_ROOT / path)


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def _symlink_components(path: Path) -> List[Path]:
    """Return symlink components along an absolute path (existing prefixes only)."""
    components = path.parts
    if not components:
        return []

    cursor = Path(components[0])
    symlinks: List[Path] = []
    if cursor.is_symlink():
        symlinks.append(cursor)

    for part in components[1:]:
        cursor = cursor / part
        if not cursor.exists() and not cursor.is_symlink():
            break
        if cursor.is_symlink():
            symlinks.append(cursor)

    return symlinks


def _embedding_root(path_value: str) -> Path:
    emb_path = _resolve_repo_path(path_value)
    return emb_path.parent if emb_path.name == "level0" else emb_path


def _format_path_list(paths: List[Path], *, limit: int = 3) -> str:
    if not paths:
        return "[]"
    preview = [str(p) for p in paths[:limit]]
    suffix = " ..." if len(paths) > limit else ""
    return f"[{', '.join(preview)}{suffix}]"


def _collect_embedding_ids(emb_dir: Path) -> Tuple[Set[str], Set[str]]:
    """Return (embedding_ids, coord_ids) from top-level *.npy in a directory."""
    embedding_ids: Set[str] = set()
    coord_ids: Set[str] = set()

    if not emb_dir.exists() or not emb_dir.is_dir():
        return embedding_ids, coord_ids

    for npy_file in emb_dir.glob("*.npy"):
        stem = npy_file.stem
        if stem.endswith("_coords"):
            coord_ids.add(stem[: -len("_coords")])
        else:
            embedding_ids.add(stem)

    return embedding_ids, coord_ids


def _format_set_preview(values: Set[str], *, limit: int = 5) -> str:
    if not values:
        return "[]"
    items = sorted(values)
    preview = items[:limit]
    suffix = " ..." if len(items) > limit else ""
    return f"[{', '.join(preview)}{suffix}]"


def validate_embedding_layout(config_path: Path) -> List[str]:
    """Validate level-0 embedding synchronization for project-scoped datasets.

    Guardrail intent:
    - if project root embeddings exist, level0/ must also exist and mirror slide IDs
    - every embedding file should have a matching *_coords.npy pair in both locations
    """
    errors: List[str] = []

    registry = ProjectRegistry(config_path)
    projects = registry.list_projects()
    if not projects:
        return [f"No projects found in {config_path}"]

    for pid, project in projects.items():
        configured_embeddings_dir = _resolve_repo_path(project.dataset.embeddings_dir)

        if configured_embeddings_dir.name == "level0":
            root_embeddings_dir = configured_embeddings_dir.parent
            level0_dir = configured_embeddings_dir
        else:
            root_embeddings_dir = configured_embeddings_dir
            level0_dir = configured_embeddings_dir / "level0"

        if not root_embeddings_dir.exists():
            errors.append(
                f"{pid}: root embeddings directory missing: '{root_embeddings_dir}'"
            )
            continue

        root_emb_ids, root_coord_ids = _collect_embedding_ids(root_embeddings_dir)
        level0_emb_ids, level0_coord_ids = _collect_embedding_ids(level0_dir)

        if root_emb_ids != root_coord_ids:
            missing_coords = root_emb_ids - root_coord_ids
            missing_embeddings = root_coord_ids - root_emb_ids
            if missing_coords:
                errors.append(
                    f"{pid}: root embeddings missing coords for {len(missing_coords)} slides "
                    f"{_format_set_preview(missing_coords)}"
                )
            if missing_embeddings:
                errors.append(
                    f"{pid}: root coords missing embeddings for {len(missing_embeddings)} slides "
                    f"{_format_set_preview(missing_embeddings)}"
                )

        if root_emb_ids and not level0_dir.exists():
            errors.append(
                f"{pid}: level0 directory missing while root embeddings exist: '{level0_dir}'"
            )
            continue

        if level0_emb_ids != level0_coord_ids:
            missing_coords = level0_emb_ids - level0_coord_ids
            missing_embeddings = level0_coord_ids - level0_emb_ids
            if missing_coords:
                errors.append(
                    f"{pid}: level0 embeddings missing coords for {len(missing_coords)} slides "
                    f"{_format_set_preview(missing_coords)}"
                )
            if missing_embeddings:
                errors.append(
                    f"{pid}: level0 coords missing embeddings for {len(missing_embeddings)} slides "
                    f"{_format_set_preview(missing_embeddings)}"
                )

        # Synchronization check (root is canonical source in current deployment)
        if root_emb_ids and root_emb_ids != level0_emb_ids:
            only_in_root = root_emb_ids - level0_emb_ids
            only_in_level0 = level0_emb_ids - root_emb_ids
            errors.append(
                f"{pid}: level0 sync mismatch (root={len(root_emb_ids)}, level0={len(level0_emb_ids)}; "
                f"only_root={len(only_in_root)} {_format_set_preview(only_in_root)}; "
                f"only_level0={len(only_in_level0)} {_format_set_preview(only_in_level0)})"
            )

    return errors


def validate_project_paths(config_path: Path) -> List[str]:
    errors: List[str] = []

    registry = ProjectRegistry(config_path)
    projects = registry.list_projects()
    top_level_classification_models = registry.classification_models
    has_top_level_classification_catalog = bool(top_level_classification_models)
    if not projects:
        return [f"No projects found in {config_path}"]

    # 1) Per-project dataset path checks
    for pid, project in projects.items():
        errors.extend(project.validate_dataset_modularity())

        expected_root = PROJECTS_DATA_ROOT / pid
        expected_root_abs = _resolve_repo_path(str(expected_root))
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

        valid_labels = {
            expected_root / "labels.csv",
            expected_root / "labels.json",
        }
        if labels_file not in valid_labels:
            errors.append(
                f"{pid}: labels_file must be one of {sorted(str(p) for p in valid_labels)}, "
                f"got '{labels_file}'"
            )

        # Symlink leakage guardrail: even valid-looking repo-relative paths must
        # resolve inside the project's nominal root (non-resolved path).
        for field_name, value in (
            ("slides_dir", project.dataset.slides_dir),
            ("embeddings_dir", project.dataset.embeddings_dir),
            ("labels_file", project.dataset.labels_file),
        ):
            configured_abs = _resolve_repo_path(value)
            resolved = configured_abs.resolve(strict=False)
            if not _is_relative_to(resolved, expected_root_abs):
                symlink_chain = _symlink_components(configured_abs)
                symlink_hint = ""
                if symlink_chain:
                    symlink_hint = f"; symlink components={_format_path_list(symlink_chain)}"
                errors.append(
                    f"{pid}: dataset.{field_name} resolves outside expected project root "
                    f"'{expected_root_abs}': '{configured_abs}' -> '{resolved}'{symlink_hint}"
                )

        project_classification_models = project.classification_models or []
        if (
            project_classification_models
            and project.prediction_target not in project_classification_models
        ):
            errors.append(
                f"{pid}: prediction_target '{project.prediction_target}' must be listed in "
                "project.classification_models when that list is non-empty"
            )

        if has_top_level_classification_catalog:
            if project.prediction_target not in top_level_classification_models:
                errors.append(
                    f"{pid}: prediction_target '{project.prediction_target}' not found in "
                    "top-level classification_models catalog"
                )

            for model_id in project_classification_models:
                if model_id not in top_level_classification_models:
                    errors.append(
                        f"{pid}: project.classification_models includes '{model_id}' not found in "
                        "top-level classification_models catalog"
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

    # 3) Cross-project contamination guardrail for embeddings roots.
    resolved_embedding_roots: Dict[Path, str] = {}
    for pid, project in projects.items():
        root = _embedding_root(project.dataset.embeddings_dir).resolve(strict=False)
        owner = resolved_embedding_roots.get(root)
        if owner is None:
            resolved_embedding_roots[root] = pid
            continue

        if owner != pid:
            errors.append(
                "Resolved embeddings root collision: "
                f"projects '{owner}' and '{pid}' share '{root}' "
                "(cross-project contamination risk)"
            )

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
    parser.add_argument(
        "--check-embedding-layout",
        action="store_true",
        help=(
            "Validate per-project level0 embedding synchronization against root embeddings "
            "(useful after embedding refreshes or migrations)"
        ),
    )
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else (REPO_ROOT / args.config)

    errors = validate_project_paths(config_path)
    if not args.skip_code_scan:
        errors.extend(validate_code_patterns(REPO_ROOT))
    if args.check_embedding_layout:
        errors.extend(validate_embedding_layout(config_path))

    if errors:
        print("Project modularity validation failed:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("Project modularity validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
