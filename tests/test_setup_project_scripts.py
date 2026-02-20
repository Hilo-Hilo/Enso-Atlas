from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(rel_path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCRIPTS = (
    _load_script_module("scripts/setup_luad_project.py", "setup_luad_project"),
    _load_script_module("scripts/setup_brca_project.py", "setup_brca_project"),
)


def _write_projects_config(tmpdir: str, config_text: str) -> Path:
    config_path = Path(tmpdir) / "projects.yaml"
    config_path.write_text(config_text, encoding="utf-8")
    return config_path


def test_resolve_output_dir_uses_project_root_from_config():
    with tempfile.TemporaryDirectory() as td:
        cfg = _write_projects_config(
            td,
            """
projects:
  demo-proj:
    dataset:
      slides_dir: data/projects/demo-proj/slides
      embeddings_dir: data/projects/demo-proj/embeddings/level0
      labels_file: data/projects/demo-proj/labels.json
""".strip(),
        )

        expected = (REPO_ROOT / "data" / "projects" / "demo-proj").resolve()
        for module in SCRIPTS:
            resolved = module.resolve_output_dir("demo-proj", cfg, None)
            assert resolved.resolve() == expected


def test_resolve_output_dir_rejects_non_modular_config_paths():
    with tempfile.TemporaryDirectory() as td:
        cfg = _write_projects_config(
            td,
            """
projects:
  demo-proj:
    dataset:
      slides_dir: data/slides
      embeddings_dir: data/embeddings
      labels_file: data/labels.csv
""".strip(),
        )

        for module in SCRIPTS:
            try:
                module.resolve_output_dir("demo-proj", cfg, None)
                assert False, "Expected ValueError for non-modular dataset paths"
            except ValueError as exc:
                assert "non-modular" in str(exc)


def test_resolve_output_dir_rejects_inconsistent_dataset_roots():
    with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as external:
        external_labels = Path(external) / "data" / "projects" / "demo-proj" / "labels.csv"
        cfg = _write_projects_config(
            td,
            f"""
projects:
  demo-proj:
    dataset:
      slides_dir: data/projects/demo-proj/slides
      embeddings_dir: data/projects/demo-proj/embeddings
      labels_file: {external_labels}
""".strip(),
        )

        for module in SCRIPTS:
            try:
                module.resolve_output_dir("demo-proj", cfg, None)
                assert False, "Expected ValueError for inconsistent dataset roots"
            except ValueError as exc:
                assert "inconsistent dataset roots" in str(exc)
