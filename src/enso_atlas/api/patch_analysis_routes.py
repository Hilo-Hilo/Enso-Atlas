"""Few-shot patch classification and outlier-detection routes."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException

from .schemas import (
    OutlierDetectionResponse,
    OutlierPatch,
    PatchClassificationItem,
    PatchClassifyRequest,
    PatchClassifyResponse,
)

AuditLogger = Callable[[str, str | None, str, dict[str, Any] | None], None]


def create_patch_analysis_router(
    *,
    embeddings_dir_provider: Callable[[], Path],
    log_audit_event: AuditLogger,
) -> APIRouter:
    """Create routes for patch-level model-free analysis workflows."""
    router = APIRouter()

    def _slide_embedding_paths(slide_id: str) -> tuple[Path, Path]:
        embeddings_dir = embeddings_dir_provider()
        return embeddings_dir / f"{slide_id}.npy", embeddings_dir / f"{slide_id}_coords.npy"

    @router.post("/api/slides/{slide_id}/patch-classify", response_model=PatchClassifyResponse)
    async def classify_patches(slide_id: str, request: PatchClassifyRequest):
        """Few-shot patch classification using logistic regression on embeddings."""
        from sklearn.linear_model import LogisticRegression

        if len(request.classes) < 2:
            raise HTTPException(status_code=400, detail="At least 2 classes are required")

        for cls_name, indices in request.classes.items():
            if len(indices) < 1:
                raise HTTPException(
                    status_code=400,
                    detail=f"Class '{cls_name}' needs at least 1 example patch",
                )

        emb_path, coord_path = _slide_embedding_paths(slide_id)
        if not emb_path.exists():
            raise HTTPException(
                status_code=404, detail=f"Embeddings not found for slide {slide_id}"
            )
        if not coord_path.exists():
            raise HTTPException(
                status_code=404, detail=f"Coordinates not found for slide {slide_id}"
            )

        embeddings_data = np.load(emb_path).astype(np.float32)
        coords = np.load(coord_path)
        n_patches = len(embeddings_data)

        if n_patches == 0:
            raise HTTPException(status_code=400, detail="Slide has no patch embeddings")

        for cls_name, indices in request.classes.items():
            for idx in indices:
                if idx < 0 or idx >= n_patches:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Patch index {idx} out of range for class '{cls_name}' (slide has {n_patches} patches)",
                    )

        class_names = sorted(request.classes.keys())
        train_indices: list[int] = []
        train_labels: list[str] = []
        for cls_name in class_names:
            for idx in request.classes[cls_name]:
                train_indices.append(idx)
                train_labels.append(cls_name)

        X_train = embeddings_data[train_indices]
        y_train = np.array(train_labels)

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(embeddings_data)
        y_proba = clf.predict_proba(embeddings_data)
        proba_classes = list(clf.classes_)

        predictions = []
        class_counts: dict[str, int] = {c: 0 for c in class_names}
        for i in range(n_patches):
            pred_class = str(y_pred[i])
            proba_dict = {
                str(proba_classes[j]): round(float(y_proba[i][j]), 4)
                for j in range(len(proba_classes))
            }
            confidence = float(max(y_proba[i]))
            class_counts[pred_class] = class_counts.get(pred_class, 0) + 1
            predictions.append(
                PatchClassificationItem(
                    patch_idx=i,
                    x=int(coords[i][0]),
                    y=int(coords[i][1]),
                    predicted_class=pred_class,
                    confidence=round(confidence, 4),
                    probabilities=proba_dict,
                )
            )

        heatmap_data = []
        for i in range(n_patches):
            heatmap_data.append(
                {
                    "x": int(coords[i][0]),
                    "y": int(coords[i][1]),
                    "class_idx": class_names.index(str(y_pred[i])),
                    "confidence": round(float(max(y_proba[i])), 4),
                }
            )

        accuracy_estimate = None
        if len(train_indices) > len(class_names):
            correct = 0
            for leave_out in range(len(train_indices)):
                loo_indices = train_indices[:leave_out] + train_indices[leave_out + 1 :]
                loo_labels = list(train_labels[:leave_out]) + list(train_labels[leave_out + 1 :])
                if len(set(loo_labels)) < 2:
                    continue
                X_loo = embeddings_data[loo_indices]
                y_loo = np.array(loo_labels)
                clf_loo = LogisticRegression(max_iter=1000, random_state=42)
                clf_loo.fit(X_loo, y_loo)
                pred = clf_loo.predict(embeddings_data[[train_indices[leave_out]]])[0]
                if pred == train_labels[leave_out]:
                    correct += 1
            if len(train_indices) > 0:
                accuracy_estimate = round(correct / len(train_indices), 4)

        log_audit_event(
            "patch_classification",
            slide_id,
            "clinician",
            {
                "classes": class_names,
                "training_examples": len(train_indices),
                "total_patches": n_patches,
                "accuracy_estimate": accuracy_estimate,
            },
        )

        return PatchClassifyResponse(
            slide_id=slide_id,
            classes=class_names,
            total_patches=n_patches,
            predictions=predictions,
            class_counts=class_counts,
            accuracy_estimate=accuracy_estimate,
            heatmap_data=heatmap_data,
        )

    @router.post(
        "/api/slides/{slide_id}/outlier-detection", response_model=OutlierDetectionResponse
    )
    async def detect_outlier_tissue(slide_id: str, threshold: float = 2.0):
        """Detect tissue patches that are far from the slide embedding centroid."""
        emb_path, coord_path = _slide_embedding_paths(slide_id)
        if not emb_path.exists():
            raise HTTPException(
                status_code=404, detail=f"Embeddings not found for slide {slide_id}"
            )
        if not coord_path.exists():
            raise HTTPException(
                status_code=404, detail=f"Coordinates not found for slide {slide_id}"
            )

        embeddings_data = np.load(emb_path).astype(np.float32)
        coords = np.load(coord_path)

        if len(embeddings_data) == 0:
            raise HTTPException(status_code=400, detail="Slide has no patch embeddings")

        centroid = np.mean(embeddings_data, axis=0)
        distances = np.linalg.norm(embeddings_data - centroid, axis=1)

        mean_dist = float(np.mean(distances))
        std_dist = float(np.std(distances))
        cutoff = mean_dist + threshold * std_dist
        outlier_mask = distances > cutoff

        outlier_patches = []
        for idx in np.where(outlier_mask)[0]:
            z = (float(distances[idx]) - mean_dist) / std_dist if std_dist > 0 else 0.0
            outlier_patches.append(
                OutlierPatch(
                    patch_idx=int(idx),
                    x=int(coords[idx][0]),
                    y=int(coords[idx][1]),
                    distance=float(distances[idx]),
                    z_score=round(z, 3),
                )
            )
        outlier_patches.sort(key=lambda p: p.distance, reverse=True)

        d_min = float(distances.min())
        d_max = float(distances.max())
        if d_max - d_min > 0:
            scores = (distances - d_min) / (d_max - d_min)
        else:
            scores = np.zeros_like(distances)

        heatmap_data = []
        for i in range(len(coords)):
            heatmap_data.append(
                {
                    "x": int(coords[i][0]),
                    "y": int(coords[i][1]),
                    "score": round(float(scores[i]), 4),
                }
            )

        log_audit_event(
            "outlier_detection",
            slide_id,
            "clinician",
            {
                "threshold": threshold,
                "total_patches": len(embeddings_data),
                "outlier_count": len(outlier_patches),
            },
        )

        return OutlierDetectionResponse(
            slide_id=slide_id,
            outlier_patches=outlier_patches,
            total_patches=len(embeddings_data),
            outlier_count=len(outlier_patches),
            mean_distance=round(mean_dist, 4),
            std_distance=round(std_dist, 4),
            threshold=threshold,
            heatmap_data=heatmap_data,
        )

    @router.get("/api/slides/{slide_id}/patch-coords")
    async def get_patch_coords(slide_id: str):
        """Return patch coordinates for spatial selection in the viewer."""
        _, coord_path = _slide_embedding_paths(slide_id)
        if not coord_path.exists():
            raise HTTPException(
                status_code=404, detail=f"No coordinates found for slide {slide_id}"
            )
        coords = np.load(coord_path)
        return {
            "slide_id": slide_id,
            "count": len(coords),
            "coords": coords.tolist(),
        }

    return router
