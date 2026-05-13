"""MedSigLIP semantic patch-search API route."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from PIL import Image

from .schemas import SemanticSearchRequest, SemanticSearchResponse, SemanticSearchResult

logger = logging.getLogger(__name__)


def create_semantic_search_router(
    *,
    medsiglip_embedder_provider: Callable[[], Any],
    slide_siglip_embeddings_provider: Callable[[], dict[str, Any]],
    classifier_provider: Callable[[], Any],
    resolve_project_embeddings_dir: Callable[..., Path],
    resolve_semantic_siglip_plan: Callable[..., tuple[str, str]],
    semantic_allow_on_the_fly_siglip_provider: Callable[[], bool],
    semantic_on_the_fly_max_patches_provider: Callable[[], int],
    get_slide_and_dz: Callable[..., Any],
    resolve_slide_path: Callable[..., Path | None],
    normalize_coords_to_level0: Callable[..., tuple[Any, int]],
    infer_patch_size_from_coords: Callable[..., int],
    classify_tissue_type: Callable[[int, int, int | None], dict],
) -> APIRouter:
    router = APIRouter()

    @router.post("/api/semantic-search", response_model=SemanticSearchResponse)
    async def semantic_search(request: SemanticSearchRequest):
        """
        Search patches by text query using MedSigLIP when available.

        Falls back to query-aware tissue-type search when SigLIP caches are
        unavailable and on-the-fly embedding is disabled or ineligible.
        """
        medsiglip_embedder = medsiglip_embedder_provider()
        if medsiglip_embedder is None:
            raise HTTPException(status_code=503, detail="MedSigLIP embedder not initialized")

        slide_id = request.slide_id

        project_requested = request.project_id is not None
        search_embeddings_dir = resolve_project_embeddings_dir(
            request.project_id,
            require_exists=project_requested,
        )

        emb_path = search_embeddings_dir / f"{slide_id}.npy"
        coord_path = search_embeddings_dir / f"{slide_id}_coords.npy"

        if not emb_path.exists():
            raise HTTPException(status_code=404, detail=f"Slide {slide_id} not found")

        coords = None
        if coord_path.exists():
            try:
                coords = np.load(coord_path, allow_pickle=False)
            except Exception as e:
                logger.warning("Failed to load patch coordinates for %s: %s", slide_id, e)
                coords = None

        slide_siglip_embeddings = slide_siglip_embeddings_provider()
        siglip_cache_key = f"{slide_id}_siglip"
        siglip_cache_path = search_embeddings_dir / "medsiglip_cache" / f"{slide_id}_siglip.npy"
        siglip_coords_path = (
            search_embeddings_dir / "medsiglip_cache" / f"{slide_id}_siglip_coords.npy"
        )
        siglip_coords = None
        siglip_embeddings = None
        use_siglip_search = False
        cache_source = None

        if siglip_cache_key in slide_siglip_embeddings:
            cached = slide_siglip_embeddings.get(siglip_cache_key)
            if isinstance(cached, np.ndarray) and cached.size > 0:
                siglip_embeddings = cached
                cache_source = "memory"
            else:
                logger.warning("Ignoring invalid in-memory MedSigLIP cache for %s", slide_id)
                slide_siglip_embeddings.pop(siglip_cache_key, None)

        if siglip_embeddings is None and siglip_cache_path.exists():
            try:
                siglip_embeddings = np.load(siglip_cache_path, allow_pickle=False)
                slide_siglip_embeddings[siglip_cache_key] = siglip_embeddings
                cache_source = "disk"
            except Exception as e:
                logger.warning(
                    "Failed to load MedSigLIP cache for %s (%s); using fallback.",
                    slide_id,
                    e,
                )
                siglip_embeddings = None

        if siglip_embeddings is not None:
            use_siglip_search = True
            if siglip_coords_path.exists():
                try:
                    siglip_coords = np.load(siglip_coords_path, allow_pickle=False)
                except Exception as e:
                    logger.warning(
                        "Failed to load MedSigLIP coords for %s (%s); using PF coords.",
                        slide_id,
                        e,
                    )
                    siglip_coords = None
            logger.info("Using %s MedSigLIP cache for %s", cache_source, slide_id)
        else:
            patch_count = int(len(coords)) if coords is not None else None
            plan_mode, plan_reason = resolve_semantic_siglip_plan(
                has_cached_siglip=False,
                allow_on_the_fly=semantic_allow_on_the_fly_siglip_provider(),
                patch_count=patch_count,
                max_patches=semantic_on_the_fly_max_patches_provider(),
            )

            if plan_mode == "on-the-fly":
                wsi_result = get_slide_and_dz(slide_id, project_id=request.project_id)

                if wsi_result is None:
                    plan_reason = "wsi-unavailable"
                    logger.warning(
                        "Skipping on-the-fly MedSigLIP for %s: WSI unavailable. Using fallback.",
                        slide_id,
                    )
                else:
                    try:
                        slide_obj, _ = wsi_result
                        patch_coords = np.asarray(coords)
                        if patch_coords.ndim != 2 or patch_coords.shape[1] < 2:
                            raise ValueError(f"Invalid coord array shape: {patch_coords.shape}")
                        patch_coords = patch_coords[:, :2]
                        patch_size = 224

                        logger.info(
                            "Computing MedSigLIP embeddings on-the-fly for %s (%d patches)",
                            slide_id,
                            int(len(patch_coords)),
                        )

                        patches = []
                        for i, (x, y) in enumerate(patch_coords):
                            try:
                                region = slide_obj.read_region(
                                    (int(x), int(y)), 0, (patch_size, patch_size)
                                )
                                if region.mode == "RGBA":
                                    background = Image.new("RGB", region.size, (255, 255, 255))
                                    background.paste(region, mask=region.split()[3])
                                    region = background
                                elif region.mode != "RGB":
                                    region = region.convert("RGB")
                                patches.append(np.array(region))
                            except Exception as e:
                                logger.warning("Failed to extract patch %s: %s", i, e)
                                patches.append(
                                    np.ones((patch_size, patch_size, 3), dtype=np.uint8) * 255
                                )

                        if patches:
                            siglip_embeddings = medsiglip_embedder.embed_patches(
                                patches=patches,
                                cache_key=slide_id,
                                show_progress=True,
                            )
                            slide_siglip_embeddings[siglip_cache_key] = siglip_embeddings
                            use_siglip_search = siglip_embeddings is not None
                            logger.info(
                                "Computed and cached MedSigLIP embeddings for %s: %s",
                                slide_id,
                                siglip_embeddings.shape,
                            )
                    except Exception as e:
                        logger.warning(
                            "On-the-fly MedSigLIP embedding failed for %s: %s", slide_id, e
                        )
                        siglip_embeddings = None
                        plan_reason = "on-the-fly-error"
            elif plan_reason == "on-the-fly-disabled":
                logger.info(
                    "No MedSigLIP cache for %s; on-the-fly embedding disabled "
                    "(set ENSO_SEMANTIC_ALLOW_ON_THE_FLY_SIGLIP=1 to enable). "
                    "Using fallback.",
                    slide_id,
                )
            elif plan_reason == "too-many-patches":
                logger.warning(
                    "Skipping on-the-fly MedSigLIP for %s: %d patches exceeds limit %d "
                    "(set ENSO_SEMANTIC_ON_THE_FLY_MAX_PATCHES to override).",
                    slide_id,
                    patch_count,
                    semantic_on_the_fly_max_patches_provider(),
                )
            elif plan_reason in {"missing-coordinates", "empty-coordinates", "invalid-patch-count"}:
                logger.warning(
                    "Skipping on-the-fly MedSigLIP for %s: patch coordinates unavailable (%s). Using fallback.",
                    slide_id,
                    plan_reason,
                )

            if siglip_embeddings is None:
                use_siglip_search = False
                logger.info(
                    "No MedSigLIP embeddings available for %s (reason=%s), using fallback",
                    slide_id,
                    plan_reason,
                )

        pf_embeddings = np.load(emb_path)

        if siglip_coords is not None:
            slide_dims = None
            slide_path = resolve_slide_path(slide_id, project_id=request.project_id)
            if slide_path is not None and slide_path.exists():
                try:
                    import openslide

                    with openslide.OpenSlide(str(slide_path)) as slide_obj:
                        slide_dims = (int(slide_obj.dimensions[0]), int(slide_obj.dimensions[1]))
                except Exception as e:
                    logger.debug(
                        "Could not read slide dimensions for semantic coord normalization: %s", e
                    )
            siglip_coords, siglip_scale = normalize_coords_to_level0(
                siglip_coords,
                slide_dims=slide_dims,
                patch_size=224,
            )
            if siglip_scale > 1:
                logger.info(
                    "Normalized SigLIP coordinates to level-0 for %s (x%s)", slide_id, siglip_scale
                )

        attention_weights = None
        classifier = classifier_provider()
        if classifier is not None:
            try:
                _, attention_weights = classifier.predict(pf_embeddings)
            except Exception as e:
                logger.warning(
                    "Could not compute attention weights for semantic search (fallback continues): %s",
                    e,
                )
                attention_weights = None

        num_patches = (
            len(siglip_embeddings) if siglip_embeddings is not None else len(pf_embeddings)
        )
        effective_coords = siglip_coords if siglip_coords is not None else coords
        patch_size_level0 = infer_patch_size_from_coords(effective_coords, default_patch_size=224)

        metadata: list[dict[str, Any]] = []
        for i in range(num_patches):
            meta: dict[str, Any] = {
                "index": i,
                "patch_size": int(patch_size_level0),
            }
            if effective_coords is not None and i < len(effective_coords):
                meta["coordinates"] = [int(effective_coords[i][0]), int(effective_coords[i][1])]
            if attention_weights is not None and i < len(attention_weights):
                meta["attention_weight"] = float(attention_weights[i])
            metadata.append(meta)

        try:
            if use_siglip_search and siglip_embeddings is not None:
                search_results = medsiglip_embedder.search(
                    query=request.query,
                    top_k=request.top_k,
                    embeddings=siglip_embeddings,
                    metadata=metadata,
                )
                model_used = medsiglip_embedder.config.model_id
            else:
                query_lower = request.query.lower()

                tissue_keywords = {
                    "tumor": [
                        "tumor",
                        "tumour",
                        "cancer",
                        "malignant",
                        "neoplastic",
                        "atypical",
                        "carcinoma",
                    ],
                    "stroma": [
                        "stroma",
                        "stromal",
                        "fibrous",
                        "connective",
                        "collagen",
                        "desmoplastic",
                        "fibroblast",
                    ],
                    "necrosis": ["necrosis", "necrotic", "dead", "dying", "debris", "coagulative"],
                    "inflammatory": [
                        "inflammatory",
                        "inflammation",
                        "lymphocyte",
                        "lymphocytic",
                        "immune",
                        "infiltrate",
                        "til",
                        "plasma",
                    ],
                    "normal": ["normal", "benign", "healthy"],
                    "artifact": ["artifact", "blur", "fold", "pen", "marker", "bubble"],
                }

                matching_types = set()
                for tissue_type, keywords in tissue_keywords.items():
                    if any(k in query_lower for k in keywords):
                        matching_types.add(tissue_type)

                logger.info(
                    "Semantic search fallback: query='%s' matched_types=%s",
                    request.query,
                    sorted(matching_types),
                )

                scored: list[dict[str, Any]] = []
                for i in range(num_patches):
                    if coords is not None and i < len(coords):
                        patch_x, patch_y = int(coords[i][0]), int(coords[i][1])
                    else:
                        patch_x, patch_y = 0, 0

                    tissue_info = classify_tissue_type(patch_x, patch_y, int(i))
                    patch_tissue = tissue_info["tissue_type"]
                    tissue_conf = float(tissue_info["confidence"])

                    attn = (
                        float(attention_weights[i])
                        if attention_weights is not None and i < len(attention_weights)
                        else 0.5
                    )

                    if matching_types:
                        if patch_tissue in matching_types:
                            score = 0.75 * tissue_conf + 0.25 * attn
                        else:
                            score = 0.05 * attn
                    else:
                        score = 0.35 * tissue_conf + 0.65 * attn

                    scored.append(
                        {
                            "patch_index": int(i),
                            "similarity_score": float(score),
                            "metadata": metadata[i] if i < len(metadata) else {},
                        }
                    )

                scored.sort(key=lambda r: float(r["similarity_score"]), reverse=True)
                search_results = scored[: min(request.top_k, len(scored))]
                model_used = "tissue-type-fallback"

        except Exception as e:
            logger.error("Semantic search failed: %s", e)
            raise HTTPException(status_code=500, detail=f"Semantic search failed: {str(e)}")

        results = []
        for r in search_results:
            results.append(
                SemanticSearchResult(
                    patch_index=r["patch_index"],
                    similarity_score=r["similarity_score"],
                    coordinates=r.get("metadata", {}).get("coordinates"),
                    patch_size=r.get("metadata", {}).get("patch_size"),
                    attention_weight=r.get("metadata", {}).get("attention_weight"),
                )
            )

        return SemanticSearchResponse(
            slide_id=slide_id,
            query=request.query,
            results=results,
            embedding_model=model_used,
        )

    return router
