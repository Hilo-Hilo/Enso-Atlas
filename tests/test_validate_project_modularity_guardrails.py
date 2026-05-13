import sys
import textwrap
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = REPO_ROOT / "scripts" / "validate_project_modularity.py"


def _load_module(name: str, path: Path):
    spec = spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


validator_module = _load_module("validate_project_modularity_guardrails_under_test", VALIDATOR_PATH)


def _write_projects_config(tmp_path: Path, projects: dict[str, dict[str, str]]) -> Path:
    project_blocks = []
    for pid, ds in projects.items():
        project_blocks.append(
            textwrap.dedent(
                f"""
                {pid}:
                  name: {pid}
                  cancer_type: Demo Cancer
                  prediction_target: demo_target
                  dataset:
                    slides_dir: {ds["slides_dir"]}
                    embeddings_dir: {ds["embeddings_dir"]}
                    labels_file: {ds["labels_file"]}
                """
            ).rstrip()
        )

    cfg_text = "projects:\n" + textwrap.indent("\n".join(project_blocks), "  ") + "\n"
    cfg_path = tmp_path / "projects.yaml"
    cfg_path.write_text(cfg_text, encoding="utf-8")
    return cfg_path


def test_validate_project_paths_allows_distinct_non_symlinked_projects(tmp_path: Path, monkeypatch):
    cfg = _write_projects_config(
        tmp_path,
        {
            "proj-a": {
                "slides_dir": "data/projects/proj-a/slides",
                "embeddings_dir": "data/projects/proj-a/embeddings",
                "labels_file": "data/projects/proj-a/labels.csv",
            },
            "proj-b": {
                "slides_dir": "data/projects/proj-b/slides",
                "embeddings_dir": "data/projects/proj-b/embeddings",
                "labels_file": "data/projects/proj-b/labels.json",
            },
        },
    )

    monkeypatch.setattr(validator_module, "REPO_ROOT", tmp_path)
    errors = validator_module.validate_project_paths(cfg)

    assert errors == []


def test_validate_project_paths_flags_symlinked_project_data_leakage(tmp_path: Path, monkeypatch):
    cfg = _write_projects_config(
        tmp_path,
        {
            "proj-a": {
                "slides_dir": "data/projects/proj-a/slides",
                "embeddings_dir": "data/projects/proj-a/embeddings",
                "labels_file": "data/projects/proj-a/labels.csv",
            },
        },
    )

    data_projects = tmp_path / "data" / "projects"
    data_projects.mkdir(parents=True, exist_ok=True)

    shared_root = tmp_path / "shared" / "global-proj-a"
    (shared_root / "slides").mkdir(parents=True, exist_ok=True)
    (shared_root / "embeddings").mkdir(parents=True, exist_ok=True)
    (shared_root / "labels.csv").write_text("slide,label\n", encoding="utf-8")

    (data_projects / "proj-a").symlink_to(shared_root, target_is_directory=True)

    monkeypatch.setattr(validator_module, "REPO_ROOT", tmp_path)
    errors = validator_module.validate_project_paths(cfg)

    assert any(
        "dataset.embeddings_dir resolves outside expected project root" in err for err in errors
    )
    assert any("symlink components=" in err for err in errors)


def test_validate_project_paths_flags_shared_resolved_embedding_roots(tmp_path: Path, monkeypatch):
    cfg = _write_projects_config(
        tmp_path,
        {
            "proj-a": {
                "slides_dir": "data/projects/proj-a/slides",
                "embeddings_dir": "data/projects/proj-a/embeddings",
                "labels_file": "data/projects/proj-a/labels.csv",
            },
            "proj-b": {
                "slides_dir": "data/projects/proj-b/slides",
                "embeddings_dir": "data/projects/proj-b/embeddings",
                "labels_file": "data/projects/proj-b/labels.csv",
            },
        },
    )

    data_projects = tmp_path / "data" / "projects"
    data_projects.mkdir(parents=True, exist_ok=True)

    shared_root = tmp_path / "shared" / "global-shared"
    (shared_root / "slides").mkdir(parents=True, exist_ok=True)
    (shared_root / "embeddings").mkdir(parents=True, exist_ok=True)
    (shared_root / "labels.csv").write_text("slide,label\n", encoding="utf-8")

    (data_projects / "proj-a").symlink_to(shared_root, target_is_directory=True)
    (data_projects / "proj-b").symlink_to(shared_root, target_is_directory=True)

    monkeypatch.setattr(validator_module, "REPO_ROOT", tmp_path)
    errors = validator_module.validate_project_paths(cfg)

    assert any("Resolved embeddings root collision" in err for err in errors)
    assert any("cross-project contamination risk" in err for err in errors)
