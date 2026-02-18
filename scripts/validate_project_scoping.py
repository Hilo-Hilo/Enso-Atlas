#!/usr/bin/env python3
"""Lightweight backend regression checks for project-scoped routing/model resolution."""

from pathlib import Path


MAIN_PY = Path(__file__).resolve().parents[1] / "src" / "enso_atlas" / "api" / "main.py"


def require(src: str, needle: str, label: str) -> None:
    if needle not in src:
        raise AssertionError(f"[FAIL] {label}\nMissing snippet: {needle}")
    print(f"[OK] {label}")


def main() -> None:
    src = MAIN_PY.read_text()

    require(
        src,
        "allowed_ids = await _resolve_project_model_ids(project_id)",
        "Model list endpoint uses centralized project model resolver",
    )
    require(
        src,
        "allowed_model_ids = await _resolve_project_model_ids(request.project_id)",
        "Multi-model analysis uses centralized project model resolver",
    )
    require(
        src,
        "project_id=request.project_id",
        "Async batch API forwards project_id into background worker",
    )
    require(
        src,
        "project_id: Optional[str] = None,",
        "Background batch worker accepts project_id parameter",
    )
    require(
        src,
        "slide_path = resolve_slide_path(slide_id, project_id=project_id)",
        "Heatmap/model-heatmap resolve WSI path with project scope",
    )
    require(
        src,
        "_require_project(project_id)\n        result = get_slide_and_dz(slide_id, project_id=project_id)",
        "DZI/tiles validate project_id before WSI lookup",
    )
    require(
        src,
        "proj_cfg = _require_project(project_id)",
        "Project-aware endpoints fail fast on unknown project_id",
    )
    require(
        src,
        "_report_embeddings_dir = _resolve_project_embeddings_dir(",
        "Report generation resolves project-scoped embeddings dir",
    )

    print("\nAll project-scoping validation checks passed.")


if __name__ == "__main__":
    main()
