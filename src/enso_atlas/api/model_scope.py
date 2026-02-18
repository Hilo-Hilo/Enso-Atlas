"""Helpers for project-scoped model access.

These helpers centralize how we resolve which classification models are valid
for a project so endpoints (e.g. /api/models and /api/heatmap/{slide}/{model})
stay coherent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable, Mapping, Optional, Sequence


@dataclass(frozen=True)
class ProjectModelScope:
    """Resolved model-scope information for a project."""

    project_exists: bool
    allowed_model_ids: set[str]


async def resolve_project_model_scope(
    project_id: str,
    *,
    project_registry: Any,
    get_project_models: Callable[[str], Awaitable[Iterable[str]]],
    logger: Any = None,
) -> ProjectModelScope:
    """Resolve allowed model IDs for a project.

    Resolution order (for allowed models):
    1) DB project_models assignments
    2) YAML project.classification_models fallback

    Project existence is true if either:
    - project exists in registry, or
    - DB has model assignments for the project
    """

    project_cfg = project_registry.get_project(project_id) if project_registry else None

    db_model_ids: list[str] = []
    try:
        db_model_ids = [mid for mid in (await get_project_models(project_id)) if mid]
    except Exception as exc:  # pragma: no cover - logging side effect
        if logger:
            logger.warning(f"DB model query failed for {project_id}: {exc}")

    project_exists = bool(project_cfg) or bool(db_model_ids)
    if not project_exists:
        return ProjectModelScope(project_exists=False, allowed_model_ids=set())

    allowed = set(db_model_ids)
    if not allowed and project_cfg and getattr(project_cfg, "classification_models", None):
        allowed = {mid for mid in project_cfg.classification_models if mid}

    return ProjectModelScope(project_exists=True, allowed_model_ids=allowed)


def filter_models_for_scope(
    models: Sequence[Mapping[str, Any]],
    allowed_model_ids: set[str],
) -> list[Mapping[str, Any]]:
    """Filter model metadata rows to a set of allowed model IDs."""

    if not allowed_model_ids:
        return []

    filtered: list[Mapping[str, Any]] = []
    for model in models:
        model_id = model.get("id", model.get("model_id"))
        if model_id in allowed_model_ids:
            filtered.append(model)
    return filtered


def is_model_allowed_for_scope(model_id: str, scope: ProjectModelScope) -> bool:
    """Return whether model_id is allowed under a resolved project scope."""

    return scope.project_exists and model_id in scope.allowed_model_ids
