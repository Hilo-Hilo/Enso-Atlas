"""Standalone slide attention heatmap route."""

import io
import time
from collections.abc import Callable
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from PIL import Image


def create_heatmap_router(
    *,
    classifier_provider: Callable[[], Any],
    evidence_generator_provider: Callable[[], Any],
    resolve_project_embeddings_dir: Callable[..., Path],
    resolve_embedding_path: Callable[..., tuple[Path | None, list[Path]]],
    resolve_slide_dims_cached: Callable[..., tuple[tuple[int, int], str]],
    path_signature: Callable[[Path | None], str | None],
    log_timing: Callable[..., float],
    logger: Any,
) -> APIRouter:
    router = APIRouter()
    cpu_heatmap_model_cache: dict[str, Any] = {"signature": None, "model": None}
    cpu_heatmap_model_lock = Lock()

    @router.get("/api/heatmap/{slide_id}")
    async def get_heatmap(
        slide_id: str,
        level: int = Query(
            default=2,
            ge=0,
            le=4,
            description="Downsample level: 0=2048px (highest detail), 2=512px (default), 4=128px (fastest)",
        ),
        smooth: bool = Query(
            default=True,
            description="Apply Gaussian blur for smooth interpolation (True) or show sharp patch tiles (False)",
        ),
        blur: int = Query(
            default=31, ge=3, le=101, description="Blur kernel size (odd number, higher=smoother)"
        ),
        project_id: str | None = Query(
            default=None, description="Project ID to scope embeddings lookup"
        ),
    ):
        """Get a PNG attention heatmap for a slide."""
        started_at = time.perf_counter()
        level_sizes = {0: 2048, 1: 1024, 2: 512, 3: 256, 4: 128}
        thumbnail_size = level_sizes.get(level, 512)

        classifier = classifier_provider()
        evidence_gen = evidence_generator_provider()
        if classifier is None or evidence_gen is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        project_requested = project_id is not None
        heatmap_embeddings_dir = resolve_project_embeddings_dir(
            project_id,
            require_exists=project_requested,
        )

        emb_path, searched_dirs = resolve_embedding_path(
            slide_id,
            level=0,
            project_id=project_id,
            base_embeddings_dir=heatmap_embeddings_dir,
        )
        if emb_path is None:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "LEVEL0_EMBEDDINGS_REQUIRED",
                    "message": f"Level 0 (full resolution) embeddings do not exist for slide {slide_id}. Generate embeddings first using /api/embed-slide with level=0.",
                    "needs_embedding": True,
                    "slide_id": slide_id,
                    "level": 0,
                    "project_id": project_id,
                    "searched_dirs": [str(d) for d in searched_dirs],
                },
            )

        coord_path = emb_path.with_name(f"{slide_id}_coords.npy")
        embeddings = np.load(emb_path)

        patch_size = 224
        if not coord_path.exists():
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "COORDS_REQUIRED_FOR_HEATMAP",
                    "slide_id": slide_id,
                    "project_id": project_id,
                    "message": "Patch coordinates are missing for this slide; regenerate/recover *_coords.npy before rendering heatmap.",
                },
            )

        coords_arr = np.load(coord_path).astype(np.int64, copy=False)
        coords = [tuple(map(int, c)) for c in coords_arr]

        def cpu_predict(embs):
            """Predict using cached CPU model to avoid per-request model reloads."""
            import torch

            from enso_atlas.mil.clam import LegacyCLAMModel

            x = torch.from_numpy(embs).float()
            model_path = Path(__file__).parent.parent.parent.parent / "models" / "clam_attention.pt"
            model_sig = path_signature(model_path) or "no_checkpoint"

            model = cpu_heatmap_model_cache.get("model")
            cached_sig = cpu_heatmap_model_cache.get("signature")
            if model is None or cached_sig != model_sig:
                with cpu_heatmap_model_lock:
                    model = cpu_heatmap_model_cache.get("model")
                    cached_sig = cpu_heatmap_model_cache.get("signature")
                    if model is None or cached_sig != model_sig:
                        model = LegacyCLAMModel(input_dim=384, hidden_dim=256)
                        if model_path.exists():
                            checkpoint = torch.load(
                                model_path, map_location=torch.device("cpu"), weights_only=False
                            )
                            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                                state_dict = checkpoint["model_state_dict"]
                            else:
                                state_dict = checkpoint
                            model.load_state_dict(state_dict)
                        model.eval()
                        cpu_heatmap_model_cache["model"] = model
                        cpu_heatmap_model_cache["signature"] = model_sig

            with torch.no_grad():
                prob, attention = model(x, return_attention=True)

            return prob.item(), attention.numpy(), model_sig

        try:
            _score, attention, model_signature = cpu_predict(embeddings)
        except Exception as exc:
            logger.error("CPU prediction failed: %s", exc)
            raise HTTPException(status_code=500, detail=f"Heatmap generation failed: {str(exc)}")

        slide_dims, dims_source = resolve_slide_dims_cached(
            slide_id,
            project_id=project_id,
            coord_path=coord_path,
            coords_arr=coords_arr,
            patch_size=patch_size,
        )

        slide_w, slide_h = slide_dims
        if slide_w >= slide_h:
            thumb_w = thumbnail_size
            thumb_h = max(1, int(round(thumbnail_size * slide_h / slide_w)))
        else:
            thumb_h = thumbnail_size
            thumb_w = max(1, int(round(thumbnail_size * slide_w / slide_h)))

        logger.info(
            "Heatmap thumbnail size: %dx%d (preserving aspect ratio of %dx%d)",
            thumb_w,
            thumb_h,
            slide_w,
            slide_h,
        )

        heatmap = evidence_gen.create_heatmap(
            attention, coords, slide_dims, (thumb_w, thumb_h), smooth=smooth, blur_kernel=blur
        )

        img = Image.fromarray(heatmap)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        response = StreamingResponse(
            buf,
            media_type="image/png",
            headers={
                "Content-Disposition": f"inline; filename={slide_id}_heatmap.png",
                "X-Slide-Width": str(slide_dims[0]),
                "X-Slide-Height": str(slide_dims[1]),
                "Access-Control-Expose-Headers": "X-Slide-Width, X-Slide-Height",
            },
        )
        log_timing(
            "api.heatmap.slide",
            started_at,
            slide_id=slide_id,
            project_id=project_id,
            dims_source=dims_source,
            thumb_width=thumb_w,
            thumb_height=thumb_h,
            attention_patches=len(coords),
            model_signature=model_signature,
        )
        return response

    return router
