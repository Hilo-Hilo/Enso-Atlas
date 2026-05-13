"""Prediction threshold and confidence helpers."""

from __future__ import annotations

from typing import Any


def score_to_prediction(
    score: float,
    decision_threshold: float,
    positive_label: str,
    negative_label: str,
) -> dict[str, Any]:
    """Convert raw score + threshold into label/confidence for API responses."""
    try:
        score_f = float(score)
    except Exception:
        score_f = 0.0
    score_f = float(min(1.0, max(0.0, score_f)))

    try:
        threshold = float(decision_threshold)
    except Exception:
        threshold = 0.5
    threshold = float(min(0.99, max(0.01, threshold)))

    pos_label = str(positive_label or "Positive")
    neg_label = str(negative_label or "Negative")
    is_positive = score_f >= threshold
    label = pos_label if is_positive else neg_label

    if is_positive:
        denom = max(1e-6, 1.0 - threshold)
        confidence = (score_f - threshold) / denom
    else:
        denom = max(1e-6, threshold)
        confidence = (threshold - score_f) / denom
    confidence = float(min(max(confidence, 0.0), 0.99))

    return {
        "score": score_f,
        "decision_threshold": threshold,
        "label": label,
        "confidence": confidence,
    }
