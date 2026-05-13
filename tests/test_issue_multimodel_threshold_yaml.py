from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _slice_between(src: str, start: str, end: str) -> str:
    i = src.index(start)
    j = src.index(end, i)
    return src[i:j]


def test_analyze_multi_cached_path_recomputes_label_and_confidence_from_current_threshold():
    src = _read("src/enso_atlas/api/multi_model_analysis_routes.py")
    cached_block = _slice_between(
        src,
        "for row in cached:",
        "if cached_predictions:",
    )

    assert (
        'current_threshold = cfg.get("decision_threshold", cfg.get("threshold", 0.5))'
        in cached_block
    )
    assert "cached_eval = score_to_prediction(" in cached_block
    assert 'score=row.get("score", 0.0)' in cached_block
    assert "decision_threshold=current_threshold" in cached_block
    assert '"label": cached_eval["label"]' in cached_block
    assert '"confidence": min(cached_eval["confidence"], 0.99)' in cached_block


def test_model_prediction_response_exposes_decision_threshold_field():
    src = _read("src/enso_atlas/api/schemas.py")
    model_prediction_block = _slice_between(
        src,
        "class ModelPrediction(BaseModel):",
        "class MultiModelRequest(BaseModel):",
    )

    assert "decision_threshold: float = 0.5" in model_prediction_block


def test_yaml_classification_model_decision_threshold_is_loaded_into_model_configs():
    src = _read("scripts/multi_model_inference.py")

    assert '"decision_threshold": m.get("decision_threshold", m.get("threshold", 0.5))' in src
    assert '_cfg["decision_threshold"] = _resolve_decision_threshold(_model_id, _cfg)' in src


def test_fresh_analyze_multi_cache_write_persists_threshold_value():
    src = _read("src/enso_atlas/api/multi_model_analysis_routes.py")

    assert "threshold=pred.decision_threshold" in src
