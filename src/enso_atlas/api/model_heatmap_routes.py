"""Model-specific attention heatmap route."""

import asyncio
import hashlib
import io
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image

from .heatmap_grid import compute_heatmap_grid_coverage


def create_model_heatmap_router(
    *,
    multi_model_provider: Callable[[], Any],
    evidence_generator_provider: Callable[[], Any],
    model_configs_provider: Callable[[], dict[str, Any]],
    resolve_project_model_scope_cached: Callable[[str], Any],
    require_model_allowed_for_scope: Callable[..., None],
    resolve_project_model_ids: Callable[[str | None], Any],
    resolve_project_embeddings_dir: Callable[..., Path],
    resolve_embedding_path: Callable[..., tuple[Path | None, list[Path]]],
    resolve_slide_dims_cached: Callable[..., tuple[tuple[int, int], str]],
    model_checkpoint_signatures: dict[str, str],
    project_root: Path,
    log_timing: Callable[..., float],
    logger: Any,
) -> APIRouter:
    router = APIRouter()
    MODEL_CONFIGS = model_configs_provider()
    _resolve_project_model_scope_cached = resolve_project_model_scope_cached
    _resolve_project_model_ids = resolve_project_model_ids
    _resolve_project_embeddings_dir = resolve_project_embeddings_dir
    _resolve_embedding_path = resolve_embedding_path
    _resolve_slide_dims_cached = resolve_slide_dims_cached
    _project_root = project_root
    _log_timing = log_timing

    @router.get("/api/heatmap/{slide_id}/{model_id}")
    async def get_model_heatmap(
        slide_id: str,
        model_id: str,
        alpha_power: float = 0.2,
        smooth: bool = Query(
            default=False,
            description="Apply Gaussian blur interpolation for denser visualization (False keeps truthful patch-grid rendering)",
        ),
        refresh: bool = Query(
            default=False,
            description="Force regeneration and bypass disk cache for this request",
        ),
        analysis_run_id: str | None = Query(
            default=None,
            description="Optional analysis run nonce; when present heatmap is regenerated",
        ),
        project_id: str | None = Query(
            default=None, description="Project ID to scope embeddings lookup"
        ),
    ):
        """Get a model-specific attention heatmap for one slide.

        When ``project_id`` is provided, the model must be assigned to that
        project and embeddings are loaded from project-scoped paths.
        """
        started_at = time.perf_counter()
        nonlocal MODEL_CONFIGS
        multi_model_inference = multi_model_provider()
        evidence_gen = evidence_generator_provider()
        MODEL_CONFIGS = model_configs_provider()
        if multi_model_inference is None:
            raise HTTPException(status_code=503, detail="Multi-model inference not initialized")

        if model_id not in MODEL_CONFIGS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model: {model_id}. Available: {list(MODEL_CONFIGS.keys())}",
            )

        scoped_allowed_model_ids: set[str] | None = None
        if project_id:
            scope = await _resolve_project_model_scope_cached(project_id)
            require_model_allowed_for_scope(model_id, scope, project_id=project_id)
            scoped_allowed_model_ids = scope.allowed_model_ids

        project_requested = project_id is not None
        _model_heatmap_embeddings_dir = _resolve_project_embeddings_dir(
            project_id,
            require_exists=project_requested,
        )

        emb_path, searched_dirs = _resolve_embedding_path(
            slide_id,
            level=0,
            project_id=project_id,
            base_embeddings_dir=_model_heatmap_embeddings_dir,
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

        # Coordinates are required for truthful attention localization.
        # Synthetic fallback grids create misleading overlays when embeddings were
        # generated after tissue filtering without persisted coords.
        if not coord_path.exists():
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "COORDS_REQUIRED_FOR_HEATMAP",
                    "slide_id": slide_id,
                    "project_id": project_id,
                    "message": "Patch coordinates are missing for this slide; regenerate/recover *_coords.npy before rendering attention heatmap.",
                },
            )

        def _resolve_model_checkpoint_path_and_signature(
            model_identifier: str,
        ) -> tuple[Path | None, str]:
            """Return checkpoint path + signature used for cache invalidation."""
            model_cfg = MODEL_CONFIGS.get(model_identifier, {})
            model_dir = model_cfg.get("model_dir")
            if not model_dir:
                return None, "model_dir_missing"

            checkpoint_path = _project_root / "outputs" / str(model_dir) / "best_model.pt"
            if not checkpoint_path.exists():
                return checkpoint_path, "checkpoint_missing"

            stat = checkpoint_path.stat()
            signature = f"{int(stat.st_mtime_ns)}_{int(stat.st_size)}"
            return checkpoint_path, signature

        force_refresh = bool(refresh or (analysis_run_id and analysis_run_id.strip()))
        _checkpoint_path, checkpoint_signature = _resolve_model_checkpoint_path_and_signature(
            model_id
        )
        previous_signature = model_checkpoint_signatures.get(model_id)
        checkpoint_changed = (
            previous_signature is not None and checkpoint_signature != previous_signature
        )
        model_checkpoint_signatures[model_id] = checkpoint_signature

        if checkpoint_changed and multi_model_inference is not None:
            # Ensure in-memory model is reloaded from updated checkpoint.
            try:
                if hasattr(multi_model_inference, "models"):
                    multi_model_inference.models.pop(model_id, None)
                if hasattr(multi_model_inference, "model_configs"):
                    multi_model_inference.model_configs.pop(model_id, None)
                logger.info(
                    "Detected checkpoint update for %s (%s -> %s); forcing model reload",
                    model_id,
                    previous_signature,
                    checkpoint_signature,
                )
            except Exception as reload_err:
                logger.warning(
                    f"Failed to clear cached model {model_id} after checkpoint update: {reload_err}"
                )
            force_refresh = True

        if force_refresh:
            logger.info(
                "Forcing model heatmap regeneration for %s/%s (refresh=%s, analysis_run_id=%s, checkpoint=%s)",
                slide_id,
                model_id,
                refresh,
                analysis_run_id,
                checkpoint_signature,
            )

        # Clamp/normalize alpha and cache aggressively by alpha value.
        # This keeps sensitivity changes responsive after first request.
        alpha_power = float(min(1.5, max(0.1, alpha_power)))
        alpha_key = f"{alpha_power:.2f}"

        cache_dir = emb_path.parent / "heatmap_cache"
        cache_dir.mkdir(exist_ok=True)
        mode_suffix = "smooth" if smooth else "truthful"
        cache_suffix = project_id if project_id else "global"
        checkpoint_suffix = hashlib.sha1(checkpoint_signature.encode("utf-8")).hexdigest()[:10]

        emb_stat = emb_path.stat()
        coord_stat = coord_path.stat()
        data_sig_raw = f"{emb_stat.st_mtime_ns}:{emb_stat.st_size}:{coord_stat.st_mtime_ns}:{coord_stat.st_size}"
        data_signature = hashlib.sha1(data_sig_raw.encode("utf-8")).hexdigest()[:12]

        cache_path = cache_dir / (
            f"{cache_suffix}_{slide_id}_{model_id}_{mode_suffix}_{checkpoint_suffix}_{data_signature}_a{alpha_key}_v7.png"
        )
        attention_cache_path = cache_dir / (
            f"{cache_suffix}_{slide_id}_{model_id}_{checkpoint_suffix}_{data_signature}_attn_v1.npy"
        )

        if not force_refresh:
            cached_path_to_serve: Path | None = None
            if cache_path.exists():
                cached_path_to_serve = cache_path

            if cached_path_to_serve is not None:
                # Serve cached heatmap — still need slide dims for alignment headers.
                _slide_dims, dims_source = _resolve_slide_dims_cached(
                    slide_id,
                    project_id=project_id,
                    coord_path=coord_path,
                    coords_arr=None,
                    patch_size=224,
                )
                _coverage = compute_heatmap_grid_coverage(
                    _slide_dims[0], _slide_dims[1], patch_size=224
                )
                logger.info(
                    "Serving cached heatmap for %s/%s (checkpoint=%s, alpha=%s)",
                    slide_id,
                    model_id,
                    checkpoint_signature,
                    alpha_key,
                )
                response: FileResponse | StreamingResponse = FileResponse(
                    str(cached_path_to_serve),
                    media_type="image/png",
                    headers={
                        "X-Model-Id": model_id,
                        "X-Model-Name": MODEL_CONFIGS[model_id]["display_name"],
                        "X-Slide-Width": str(_slide_dims[0]),
                        "X-Slide-Height": str(_slide_dims[1]),
                        "X-Coverage-Width": str(_coverage.coverage_width),
                        "X-Coverage-Height": str(_coverage.coverage_height),
                        "Cache-Control": "no-store, max-age=0",
                        "Pragma": "no-cache",
                        "Expires": "0",
                        "Access-Control-Expose-Headers": "X-Model-Id, X-Model-Name, X-Slide-Width, X-Slide-Height, X-Coverage-Width, X-Coverage-Height",
                    },
                )
                _log_timing(
                    "api.heatmap.model",
                    started_at,
                    slide_id=slide_id,
                    model_id=model_id,
                    project_id=project_id,
                    cache_hit=True,
                    attention_cache_hit=attention_cache_path.exists(),
                    dims_source=dims_source,
                    force_refresh=force_refresh,
                    alpha=alpha_key,
                )
                return response

        patch_size = 224
        coords_arr = np.load(coord_path).astype(np.int64, copy=False)

        attention: np.ndarray | None = None
        attention_cache_hit = False
        if attention_cache_path.exists() and not force_refresh:
            try:
                attention = np.load(attention_cache_path)
                if attention.shape[0] != coords_arr.shape[0]:
                    logger.warning(
                        "Cached attention length mismatch for %s/%s (attn=%s coords=%s); recomputing",
                        slide_id,
                        model_id,
                        attention.shape[0],
                        coords_arr.shape[0],
                    )
                    attention = None
                else:
                    attention_cache_hit = True
                    logger.info(
                        "Loaded cached attention for %s/%s (checkpoint=%s)",
                        slide_id,
                        model_id,
                        checkpoint_signature,
                    )
            except Exception as attn_err:
                logger.warning(
                    f"Failed to load cached attention for {slide_id}/{model_id}: {attn_err}"
                )
                attention = None

        # Get prediction with attention from specific model (only when not cached)
        if attention is None:
            embeddings = np.load(emb_path)
            try:
                result = multi_model_inference.predict_single(
                    embeddings, model_id, return_attention=True
                )
                attention = result.get("attention")

                if attention is None:
                    raise HTTPException(
                        status_code=500, detail="Model did not return attention weights"
                    )

                attention = np.array(attention)
                try:
                    np.save(attention_cache_path, attention)
                except Exception as cache_err:
                    logger.warning(
                        f"Failed to cache attention for {slide_id}/{model_id}: {cache_err}"
                    )

            except Exception as e:
                logger.error(f"Model inference failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # If this request had to run model inference for attention, prewarm sibling
        # model attentions for this slide/project in the background so heatmap model
        # switching becomes near-instant after the first model load.
        if not attention_cache_hit and project_id and not force_refresh:
            try:
                allowed_model_ids = scoped_allowed_model_ids
                if allowed_model_ids is None:
                    allowed_model_ids = await _resolve_project_model_ids(project_id)
                sibling_model_ids = sorted(
                    mid for mid in (allowed_model_ids or set()) if mid != model_id
                )

                if sibling_model_ids:
                    data_signature_bg = data_signature
                    cache_suffix_bg = cache_suffix
                    emb_path_bg = emb_path
                    cache_dir_bg = cache_dir
                    checkpoint_resolver_bg = _resolve_model_checkpoint_path_and_signature
                    model_inference_bg = multi_model_inference
                    slide_id_bg = slide_id
                    project_id_bg = project_id

                    def _prewarm_sibling_attention_caches() -> None:
                        try:
                            embeddings_bg: np.ndarray | None = None
                            warmed = 0
                            for sib_mid in sibling_model_ids:
                                _, sib_sig = checkpoint_resolver_bg(sib_mid)
                                sib_suffix = hashlib.sha1(sib_sig.encode("utf-8")).hexdigest()[:10]
                                sib_attn_cache = cache_dir_bg / (
                                    f"{cache_suffix_bg}_{slide_id_bg}_{sib_mid}_{sib_suffix}_{data_signature_bg}_attn_v1.npy"
                                )
                                if sib_attn_cache.exists():
                                    continue

                                if embeddings_bg is None:
                                    embeddings_bg = np.load(emb_path_bg)

                                try:
                                    sib_result = model_inference_bg.predict_single(
                                        embeddings_bg,
                                        sib_mid,
                                        return_attention=True,
                                    )
                                    sib_attention = sib_result.get("attention")
                                    if sib_attention is None:
                                        continue
                                    np.save(
                                        sib_attn_cache, np.asarray(sib_attention, dtype=np.float32)
                                    )
                                    warmed += 1
                                except Exception as sib_err:
                                    logger.warning(
                                        "Sibling attention prewarm failed for %s/%s (%s): %s",
                                        slide_id_bg,
                                        sib_mid,
                                        project_id_bg,
                                        sib_err,
                                    )

                            if warmed:
                                logger.info(
                                    "Prewarmed %d sibling attention caches for %s (project=%s)",
                                    warmed,
                                    slide_id_bg,
                                    project_id_bg,
                                )
                        except Exception as prewarm_err:
                            logger.warning(
                                "Background sibling attention prewarm failed for %s (project=%s): %s",
                                slide_id_bg,
                                project_id_bg,
                                prewarm_err,
                            )

                    asyncio.create_task(asyncio.to_thread(_prewarm_sibling_attention_caches))
            except Exception as prewarm_setup_err:
                logger.warning(
                    "Could not schedule sibling attention prewarm for %s (project=%s): %s",
                    slide_id,
                    project_id,
                    prewarm_setup_err,
                )

        # Per-slide dynamic range normalization for color mapping:
        # map min attention -> 0 (blue) and max attention -> 1 (red)
        # for this specific slide/model pair.
        try:
            attention = np.asarray(attention, dtype=np.float32)
            finite_mask = np.isfinite(attention)
            if not finite_mask.any():
                raise ValueError("Attention contains no finite values")
            att_values = attention[finite_mask]
            att_min = float(att_values.min())
            att_max = float(att_values.max())
            if att_max > att_min:
                attention = (attention - att_min) / (att_max - att_min)
                attention = np.clip(attention, 0.0, 1.0)
            else:
                attention = np.zeros_like(attention, dtype=np.float32)

            logger.info(
                "Heatmap attention normalized per slide/model: slide=%s model=%s min=%.6g max=%.6g",
                slide_id,
                model_id,
                att_min,
                att_max,
            )
        except Exception as norm_err:
            logger.warning(
                "Attention normalization fallback for %s/%s: %s",
                slide_id,
                model_id,
                norm_err,
            )
            attention = np.asarray(attention, dtype=np.float32)

        # Generate heatmap image using EvidenceGenerator for proper coordinate scaling
        try:
            slide_dims, dims_source = _resolve_slide_dims_cached(
                slide_id,
                project_id=project_id,
                coord_path=coord_path,
                coords_arr=coords_arr,
                patch_size=patch_size,
            )

            # Generate a patch-resolution heatmap: 1 pixel = 1 patch (224x224).
            # This produces crisp discrete patches when rendered with image-rendering: pixelated.
            coords_list = [tuple(map(int, c)) for c in coords_arr]
            _coverage = compute_heatmap_grid_coverage(
                slide_dims[0], slide_dims[1], patch_size=patch_size
            )
            grid_w = _coverage.grid_width
            grid_h = _coverage.grid_height

            logger.info(
                f"Model heatmap patch-resolution: {grid_w}x{grid_h} (1 pixel per {patch_size}px patch)"
            )

            heatmap_rgba = evidence_gen.create_heatmap(
                attention,
                coords_list,
                slide_dims,
                thumbnail_size=(grid_w, grid_h),
                smooth=smooth,
                blur_kernel=31 if smooth else 1,
                alpha_power=alpha_power,
            )

            # Convert RGBA to PNG
            img = Image.fromarray(heatmap_rgba, mode="RGBA")

            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)

            # Save to disk cache for subsequent requests (all alpha values).
            try:
                with open(cache_path, "wb") as f:
                    f.write(buf.getvalue())
                logger.info(f"Cached heatmap to {cache_path}")
            except Exception as cache_err:
                logger.warning(f"Failed to cache heatmap: {cache_err}")

            response = StreamingResponse(
                buf,
                media_type="image/png",
                headers={
                    "X-Model-Id": model_id,
                    "X-Model-Name": MODEL_CONFIGS[model_id]["display_name"],
                    "X-Slide-Width": str(slide_dims[0]),
                    "X-Slide-Height": str(slide_dims[1]),
                    "X-Coverage-Width": str(_coverage.coverage_width),
                    "X-Coverage-Height": str(_coverage.coverage_height),
                    "Cache-Control": "no-store, max-age=0",
                    "Pragma": "no-cache",
                    "Expires": "0",
                    "Access-Control-Expose-Headers": "X-Model-Id, X-Model-Name, X-Slide-Width, X-Slide-Height, X-Coverage-Width, X-Coverage-Height",
                },
            )
            _log_timing(
                "api.heatmap.model",
                started_at,
                slide_id=slide_id,
                model_id=model_id,
                project_id=project_id,
                cache_hit=False,
                attention_cache_hit=attention_cache_hit,
                dims_source=dims_source,
                force_refresh=force_refresh,
                alpha=alpha_key,
                smooth=smooth,
            )
            return response

        except Exception as e:
            logger.error(f"Heatmap generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Heatmap generation failed: {e}")

    return router
