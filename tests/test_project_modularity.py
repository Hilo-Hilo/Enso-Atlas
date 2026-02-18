from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from enso_atlas.api.projects import PROJECTS_DATA_ROOT, ProjectConfig, ProjectRegistry


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
    result = subprocess.run(
        ["python", "scripts/validate_project_modularity.py"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
