"""
Config-driven Project System for Enso Atlas.

Supports multiple cancer types / prediction targets via a YAML configuration.
Each project defines its own dataset paths, model configs, classes, and thresholds.

Usage:
    registry = ProjectRegistry("config/projects.yaml")
    project = registry.get_project("ovarian-platinum")
    print(project.name, project.classes)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DatasetConfig:
    """Dataset paths and labels configuration for a project."""
    slides_dir: str = "data/slides"
    embeddings_dir: str = "data/embeddings/level0"
    labels_file: str = "data/labels.csv"
    label_column: str = "platinum_sensitivity"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DatasetConfig":
        return cls(
            slides_dir=d.get("slides_dir", cls.slides_dir),
            embeddings_dir=d.get("embeddings_dir", cls.embeddings_dir),
            labels_file=d.get("labels_file", cls.labels_file),
            label_column=d.get("label_column", cls.label_column),
        )


@dataclass
class ModelsConfig:
    """Model configuration for a project."""
    embedder: str = "path-foundation"
    mil_architecture: str = "transmil"
    mil_checkpoint: str = "models/transmil_best.pt"
    report_generator: str = "medgemma-4b"
    semantic_search: str = "medsiglip"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelsConfig":
        return cls(
            embedder=d.get("embedder", cls.embedder),
            mil_architecture=d.get("mil_architecture", cls.mil_architecture),
            mil_checkpoint=d.get("mil_checkpoint", cls.mil_checkpoint),
            report_generator=d.get("report_generator", cls.report_generator),
            semantic_search=d.get("semantic_search", cls.semantic_search),
        )


@dataclass
class ProjectConfig:
    """Full configuration for a single project (cancer type / prediction target)."""
    id: str
    name: str
    cancer_type: str
    prediction_target: str
    classes: List[str] = field(default_factory=lambda: ["resistant", "sensitive"])
    positive_class: str = "sensitive"
    description: str = ""
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    threshold: float = 0.5
    threshold_config: Optional[str] = None

    @classmethod
    def from_dict(cls, project_id: str, d: Dict[str, Any]) -> "ProjectConfig":
        dataset_raw = d.get("dataset", {})
        models_raw = d.get("models", {})
        return cls(
            id=project_id,
            name=d.get("name", project_id),
            cancer_type=d.get("cancer_type", "unknown"),
            prediction_target=d.get("prediction_target", "unknown"),
            classes=d.get("classes", ["resistant", "sensitive"]),
            positive_class=d.get("positive_class", "sensitive"),
            description=d.get("description", ""),
            dataset=DatasetConfig.from_dict(dataset_raw),
            models=ModelsConfig.from_dict(models_raw),
            threshold=float(d.get("threshold", 0.5)),
            threshold_config=d.get("threshold_config"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize project config to a JSON-safe dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "cancer_type": self.cancer_type,
            "prediction_target": self.prediction_target,
            "classes": self.classes,
            "positive_class": self.positive_class,
            "description": self.description,
            "dataset": {
                "slides_dir": self.dataset.slides_dir,
                "embeddings_dir": self.dataset.embeddings_dir,
                "labels_file": self.dataset.labels_file,
                "label_column": self.dataset.label_column,
            },
            "models": {
                "embedder": self.models.embedder,
                "mil_architecture": self.models.mil_architecture,
                "mil_checkpoint": self.models.mil_checkpoint,
                "report_generator": self.models.report_generator,
                "semantic_search": self.models.semantic_search,
            },
            "threshold": self.threshold,
            "threshold_config": self.threshold_config,
        }


# ---------------------------------------------------------------------------
# Project Registry
# ---------------------------------------------------------------------------


class ProjectRegistry:
    """
    Loads and manages all project configurations from a YAML file.

    Usage:
        registry = ProjectRegistry("config/projects.yaml")
        project = registry.get_project("ovarian-platinum")
    """

    def __init__(self, config_path: str | Path = "config/projects.yaml"):
        self._config_path = Path(config_path)
        self._projects: Dict[str, ProjectConfig] = {}
        self._default_project_id: Optional[str] = None
        self._load()

    def _load(self):
        """Load projects from the YAML config file."""
        if not self._config_path.exists():
            logger.warning(f"Projects config not found: {self._config_path}")
            return

        with open(self._config_path, "r") as f:
            raw = yaml.safe_load(f) or {}

        projects_raw = raw.get("projects", {})
        if not projects_raw:
            logger.warning(f"No projects defined in {self._config_path}")
            return

        for project_id, project_data in projects_raw.items():
            try:
                project = ProjectConfig.from_dict(project_id, project_data)
                self._projects[project_id] = project
                logger.info(f"Loaded project: {project_id} ({project.name})")
            except Exception as e:
                logger.error(f"Failed to load project '{project_id}': {e}")

        # Default project: env var > first project in YAML
        env_default = os.environ.get("DEFAULT_PROJECT")
        if env_default and env_default in self._projects:
            self._default_project_id = env_default
        elif self._projects:
            self._default_project_id = next(iter(self._projects))

        logger.info(
            f"ProjectRegistry loaded {len(self._projects)} project(s), "
            f"default: {self._default_project_id}"
        )

    def get_project(self, project_id: str) -> Optional[ProjectConfig]:
        """Get a project by ID. Returns None if not found."""
        return self._projects.get(project_id)

    def list_projects(self) -> Dict[str, ProjectConfig]:
        """Return all projects as {id: ProjectConfig}."""
        return dict(self._projects)

    def get_default_project(self) -> Optional[ProjectConfig]:
        """Return the default project config."""
        if self._default_project_id:
            return self._projects.get(self._default_project_id)
        return None

    @property
    def default_project_id(self) -> Optional[str]:
        return self._default_project_id

    def save(self):
        """Write current project configurations back to the YAML config file."""
        projects_raw = {}
        for pid, proj in self._projects.items():
            d = proj.to_dict()
            # Remove 'id' since it is the key in the YAML mapping
            d.pop("id", None)
            projects_raw[pid] = d

        data = {"projects": projects_raw}
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._config_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Saved {len(self._projects)} project(s) to {self._config_path}")

    def add_project(self, project_id: str, config: Dict[str, Any]) -> ProjectConfig:
        """Add a new project to the registry and persist to YAML.

        Args:
            project_id: Unique identifier for the project.
            config: Dictionary of project configuration values.

        Returns:
            The newly created ProjectConfig.

        Raises:
            ValueError: If a project with the given ID already exists.
        """
        if project_id in self._projects:
            raise ValueError(f"Project '{project_id}' already exists")

        project = ProjectConfig.from_dict(project_id, config)
        self._projects[project_id] = project

        # If this is the first project, make it the default
        if self._default_project_id is None:
            self._default_project_id = project_id

        self.save()
        return project

    def update_project(self, project_id: str, updates: Dict[str, Any]) -> ProjectConfig:
        """Update an existing project with partial data and persist to YAML.

        Args:
            project_id: ID of the project to update.
            updates: Dictionary of fields to update (partial).

        Returns:
            The updated ProjectConfig.

        Raises:
            KeyError: If the project does not exist.
        """
        if project_id not in self._projects:
            raise KeyError(f"Project '{project_id}' not found")

        existing = self._projects[project_id].to_dict()
        existing.pop("id", None)

        # Merge top-level keys; nested dicts (dataset, models) are merged one level deep
        for key, value in updates.items():
            if key == "id":
                continue
            if key in ("dataset", "models") and isinstance(value, dict):
                existing.setdefault(key, {}).update(value)
            else:
                existing[key] = value

        project = ProjectConfig.from_dict(project_id, existing)
        self._projects[project_id] = project
        self.save()
        return project

    def remove_project(self, project_id: str) -> None:
        """Remove a project from the registry and persist to YAML.

        Does NOT delete any data files on disk.

        Args:
            project_id: ID of the project to remove.

        Raises:
            KeyError: If the project does not exist.
        """
        if project_id not in self._projects:
            raise KeyError(f"Project '{project_id}' not found")

        del self._projects[project_id]

        # Update default if we just removed it
        if self._default_project_id == project_id:
            if self._projects:
                self._default_project_id = next(iter(self._projects))
            else:
                self._default_project_id = None

        self.save()
        logger.info(f"Removed project '{project_id}' from registry")

    def reload(self):
        """Reload projects from the config file."""
        self._projects.clear()
        self._default_project_id = None
        self._load()
