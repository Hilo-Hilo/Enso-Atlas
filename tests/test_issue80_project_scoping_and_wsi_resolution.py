from pathlib import Path

from src.enso_atlas.api.slide_resolution import (
    filter_project_candidate_slide_ids,
    load_slide_ids_from_labels_file,
    resolve_slide_path_in_dirs,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_project_membership_filter_keeps_exact_and_unique_base_matches():
    candidates = {
        "TCGA-OV-0001.aaa111",
        "TCGA-OV-0002.bbb222",
        "TCGA-LUAD-1001.ccc333",
    }
    allowed = {
        "TCGA-OV-0001",  # base-only label style
        "TCGA-OV-0002.bbb222",  # exact match style
    }

    filtered = filter_project_candidate_slide_ids(candidates, allowed)

    assert filtered == ["TCGA-OV-0001.aaa111", "TCGA-OV-0002.bbb222"]


def test_project_membership_filter_rejects_ambiguous_base_matches():
    candidates = {
        "TCGA-OV-0003.uuidA",
        "TCGA-OV-0003.uuidB",
    }
    allowed = {"TCGA-OV-0003"}

    filtered = filter_project_candidate_slide_ids(candidates, allowed)

    assert filtered == []


def test_load_slide_ids_from_labels_file_supports_csv_and_json(tmp_path: Path):
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text("slide_id,label\nS1,1\nS2,0\n", encoding="utf-8")

    csv_slide_file_path = tmp_path / "labels_slide_file.csv"
    csv_slide_file_path.write_text(
        "slide_file,treatment_response\nS3.svs,responder\nS4.SVS,non-responder\n",
        encoding="utf-8",
    )

    json_path = tmp_path / "labels.json"
    json_path.write_text('{"S5": "early", "S6": "advanced"}', encoding="utf-8")

    assert load_slide_ids_from_labels_file(csv_path) == {"S1", "S2"}
    assert load_slide_ids_from_labels_file(csv_slide_file_path) == {"S3", "S4"}
    assert load_slide_ids_from_labels_file(json_path) == {"S5", "S6"}


def test_resolve_slide_path_in_dirs_uses_safe_uuid_fallback(tmp_path: Path):
    slide_dir = tmp_path / "slides"
    slide_dir.mkdir()

    exact = slide_dir / "CASE-A.svs"
    exact.write_text("x", encoding="utf-8")
    assert resolve_slide_path_in_dirs("CASE-A", [slide_dir], [".svs"]) == exact

    fallback = slide_dir / "CASE-B.12345.svs"
    fallback.write_text("x", encoding="utf-8")
    assert resolve_slide_path_in_dirs("CASE-B", [slide_dir], [".svs"]) == fallback

    fallback2 = slide_dir / "CASE-C.uuidY.svs"
    fallback2.write_text("x", encoding="utf-8")
    assert resolve_slide_path_in_dirs("CASE-C.uuidX", [slide_dir], [".svs"]) == fallback2


def test_resolve_slide_path_in_dirs_returns_none_on_ambiguous_fallback(tmp_path: Path):
    slide_dir = tmp_path / "slides"
    slide_dir.mkdir()
    (slide_dir / "CASE-D.a.svs").write_text("x", encoding="utf-8")
    (slide_dir / "CASE-D.b.svs").write_text("x", encoding="utf-8")

    assert resolve_slide_path_in_dirs("CASE-D", [slide_dir], [".svs"]) is None


def test_list_slides_uses_authoritative_membership_guard_without_embedding_fallback():
    src = (REPO_ROOT / "src" / "enso_atlas" / "api" / "slide_list_routes.py").read_text(
        encoding="utf-8"
    )

    assert "include_embedding_fallback=False" in src
    assert "filter_project_candidate_slide_ids(" in src
    assert "slide scoping guard filtered" in src
