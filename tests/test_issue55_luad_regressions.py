from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_model_picker_prunes_stale_model_ids_after_project_switch():
    src = _read("frontend/src/components/panels/ModelPicker.tsx")

    assert "Prune stale/duplicate selected model IDs" in src
    assert "const availableModelIds = new Set(models.map((m) => m.id));" in src
    assert "if (prunedSelection.length !== selectedModels.length)" in src


def test_level0_embedding_resolution_is_dense_only_no_sparse_fallback():
    src = _read("src/enso_atlas/api/main.py")

    assert "def _candidate_embedding_dirs_for_level(" in src
    assert "Level 0 is strict dense-only" in src
    assert (
        'candidate_dirs.extend([analysis_embeddings_dir / "level0", analysis_embeddings_dir])'
        not in src
    )
    assert (
        'candidate_dirs.extend([batch_embeddings_dir / "level0", batch_embeddings_dir])' not in src
    )


def test_heatmap_and_multi_model_paths_share_level0_embedding_resolver():
    sources = "\n".join(
        _read(path)
        for path in [
            "src/enso_atlas/api/heatmap_routes.py",
            "src/enso_atlas/api/model_heatmap_routes.py",
            "src/enso_atlas/api/multi_model_analysis_routes.py",
        ]
    )

    resolver_calls = sources.count("emb_path, searched_dirs = resolve_embedding_path(")
    resolver_calls += sources.count("emb_path, searched_dirs = _resolve_embedding_path(")
    assert resolver_calls >= 3
    assert 'coord_path = emb_path.with_name(f"{slide_id}_coords.npy")' in sources
    assert sources.count('"error": "LEVEL0_EMBEDDINGS_REQUIRED"') >= 3
