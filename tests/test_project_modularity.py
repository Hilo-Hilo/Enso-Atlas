from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import subprocess
import sys
import textwrap

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
VALIDATOR_PATH = REPO_ROOT / "scripts" / "validate_project_modularity.py"
PROJECTS_MODULE_PATH = SRC_DIR / "enso_atlas" / "api" / "projects.py"


def _load_module(name: str, path: Path):
    spec = spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


projects_module = _load_module("enso_atlas_api_projects", PROJECTS_MODULE_PATH)
validator_module = _load_module("validate_project_modularity", VALIDATOR_PATH)

PROJECTS_DATA_ROOT = projects_module.PROJECTS_DATA_ROOT
ProjectConfig = projects_module.ProjectConfig
ProjectRegistry = projects_module.ProjectRegistry


def _run_validator(config_path: Path, *, skip_code_scan: bool = False) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(VALIDATOR_PATH), "--config", str(config_path)]
    if skip_code_scan:
        cmd.append("--skip-code-scan")
    return subprocess.run(cmd, check=False, capture_output=True, text=True, cwd=REPO_ROOT)


def _write_project_config(tmp_path: Path, *, slides_dir: str, embeddings_dir: str, labels_file: str) -> Path:
    cfg = tmp_path / "projects.yaml"
    cfg.write_text(
        textwrap.dedent(
            f"""
            projects:
              demo-proj:
                name: Demo
                cancer_type: Demo Cancer
                prediction_target: demo_target
                dataset:
                  slides_dir: {slides_dir}
                  embeddings_dir: {embeddings_dir}
                  labels_file: {labels_file}
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return cfg


def test_project_config_defaults_are_project_scoped():
    proj = ProjectConfig.from_dict(
        "demo-proj",
        {
            "name": "Demo",
            "cancer_type": "Demo Cancer",
            "prediction_target": "demo_target",
        },
    )

    expected_root = PROJECTS_DATA_ROOT / "demo-proj"
    assert proj.dataset.slides_dir == str(expected_root / "slides")
    assert proj.dataset.embeddings_dir == str(expected_root / "embeddings")
    assert proj.dataset.labels_file == str(expected_root / "labels.csv")


def test_existing_projects_keep_expected_paths():
    registry = ProjectRegistry(Path("config/projects.yaml"))

    ovarian = registry.get_project("ovarian-platinum")
    assert ovarian is not None
    assert ovarian.dataset.slides_dir == "data/projects/ovarian-platinum/slides"
    assert ovarian.dataset.embeddings_dir == "data/projects/ovarian-platinum/embeddings"
    assert ovarian.dataset.labels_file == "data/projects/ovarian-platinum/labels.csv"

    lung = registry.get_project("lung-stage")
    assert lung is not None
    assert lung.dataset.slides_dir == "data/projects/lung-stage/slides"
    assert lung.dataset.embeddings_dir == "data/projects/lung-stage/embeddings"
    assert lung.dataset.labels_file == "data/projects/lung-stage/labels.json"


def test_modularity_validation_checks_pass_for_repo_config():
    result = _run_validator(REPO_ROOT / "config" / "projects.yaml")
    assert result.returncode == 0, result.stdout + "\n" + result.stderr


def test_modularity_validation_fails_for_paths_outside_project_root(tmp_path: Path):
    cfg = _write_project_config(
        tmp_path,
        slides_dir="data/slides",
        embeddings_dir="data/projects/demo-proj/embeddings",
        labels_file="data/projects/demo-proj/labels.csv",
    )

    result = _run_validator(cfg, skip_code_scan=True)
    output = result.stdout + "\n" + result.stderr

    assert result.returncode != 0
    assert "dataset.slides_dir='data/slides' is outside expected 'data/projects/demo-proj/..." in output


def test_modularity_validation_fails_for_inconsistent_dataset_relationships(tmp_path: Path):
    cfg = _write_project_config(
        tmp_path,
        slides_dir="data/projects/demo-proj/slides",
        embeddings_dir="data/projects/demo-proj/custom_embeddings",
        labels_file="data/projects/demo-proj/targets.csv",
    )

    result = _run_validator(cfg, skip_code_scan=True)
    output = result.stdout + "\n" + result.stderr

    assert result.returncode != 0
    assert "embeddings_dir must be one of" in output
    assert "labels_file must be one of" in output


def test_legacy_pattern_scan_detects_forbidden_snippets(tmp_path: Path):
    files = {
        "scripts/setup_luad_project.py": "# clean file\n",
        "scripts/setup_brca_project.py": "# clean file\n",
        "src/enso_atlas/api/project_routes.py": "# clean file\n",
        "src/enso_atlas/api/projects.py": "# clean file\n",
    }

    for rel_path, content in files.items():
        path = tmp_path / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    assert validator_module.validate_code_patterns(tmp_path) == []

    (tmp_path / "src/enso_atlas/api/project_routes.py").write_text(
        'path = f"data/{body.id}/slides"\n',
        encoding="utf-8",
    )

    errors = validator_module.validate_code_patterns(tmp_path)
    assert any("Legacy path pattern found in src/enso_atlas/api/project_routes.py" in e for e in errors)
