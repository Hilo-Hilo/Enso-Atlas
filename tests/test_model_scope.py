import asyncio
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_SCOPE_PATH = REPO_ROOT / "src" / "enso_atlas" / "api" / "model_scope.py"
_spec = importlib.util.spec_from_file_location("model_scope", MODEL_SCOPE_PATH)
assert _spec and _spec.loader, "Failed to load model_scope module"
_model_scope = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _model_scope
_spec.loader.exec_module(_model_scope)

filter_models_for_scope = _model_scope.filter_models_for_scope
is_model_allowed_for_scope = _model_scope.is_model_allowed_for_scope
resolve_project_model_scope = _model_scope.resolve_project_model_scope


@dataclass
class _FakeProject:
    classification_models: list[str]


class _FakeRegistry:
    def __init__(self, projects: dict[str, _FakeProject]):
        self._projects = projects

    def get_project(self, project_id: str):
        return self._projects.get(project_id)


async def _db_models_factory(data: dict[str, list[str]], project_id: str) -> list[str]:
    return data.get(project_id, [])


def test_resolve_project_model_scope_falls_back_to_yaml_when_db_empty():
    registry = _FakeRegistry({"lung-stage": _FakeProject(classification_models=["lung_stage"])})

    async def get_project_models(project_id: str):
        return await _db_models_factory({}, project_id)

    scope = asyncio.run(
        resolve_project_model_scope(
            "lung-stage",
            project_registry=registry,
            get_project_models=get_project_models,
        )
    )

    assert scope.project_exists is True
    assert scope.allowed_model_ids == {"lung_stage"}


def test_resolve_project_model_scope_prefers_db_assignments_when_present():
    registry = _FakeRegistry({"lung-stage": _FakeProject(classification_models=["lung_stage"])})

    async def get_project_models(project_id: str):
        return await _db_models_factory({"lung-stage": ["platinum_sensitivity"]}, project_id)

    scope = asyncio.run(
        resolve_project_model_scope(
            "lung-stage",
            project_registry=registry,
            get_project_models=get_project_models,
        )
    )

    assert scope.project_exists is True
    assert scope.allowed_model_ids == {"platinum_sensitivity"}


def test_resolve_project_model_scope_unknown_project():
    registry = _FakeRegistry({})

    async def get_project_models(project_id: str):
        return await _db_models_factory({}, project_id)

    scope = asyncio.run(
        resolve_project_model_scope(
            "unknown-project",
            project_registry=registry,
            get_project_models=get_project_models,
        )
    )

    assert scope.project_exists is False
    assert scope.allowed_model_ids == set()


def test_resolve_project_model_scope_treats_db_assignments_as_existing_project():
    registry = _FakeRegistry({})

    async def get_project_models(project_id: str):
        return await _db_models_factory({"db-only-project": ["tumor_grade"]}, project_id)

    scope = asyncio.run(
        resolve_project_model_scope(
            "db-only-project",
            project_registry=registry,
            get_project_models=get_project_models,
        )
    )

    assert scope.project_exists is True
    assert scope.allowed_model_ids == {"tumor_grade"}


def test_filter_models_for_scope_filters_by_id_and_model_id_fields():
    models = [
        {"id": "lung_stage", "name": "Lung Stage"},
        {"id": "survival_5y", "name": "Survival 5Y"},
        {"model_id": "tumor_grade", "name": "Tumor Grade"},
    ]

    filtered = filter_models_for_scope(models, {"lung_stage", "tumor_grade"})

    assert filtered == [
        {"id": "lung_stage", "name": "Lung Stage"},
        {"model_id": "tumor_grade", "name": "Tumor Grade"},
    ]


def test_is_model_allowed_for_scope_rejects_ovarian_model_for_lung_scope():
    scope = asyncio.run(
        resolve_project_model_scope(
            "lung-stage",
            project_registry=_FakeRegistry({"lung-stage": _FakeProject(classification_models=["lung_stage"])}),
            get_project_models=lambda _pid: _db_models_factory({}, _pid),
        )
    )

    assert is_model_allowed_for_scope("lung_stage", scope) is True
    assert is_model_allowed_for_scope("survival_5y", scope) is False
