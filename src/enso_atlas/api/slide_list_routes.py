"""Slide listing route and project-scoped flat-file inventory logic."""

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, Query

from .schemas import PatientContext, SlideDimensions, SlideInfo

SlideLister = Callable[..., Awaitable[list[SlideInfo]]]


def build_slide_lister(
    *,
    db_module: Any,
    db_available_provider: Callable[[], bool],
    logger: Any,
    require_project: Callable[[str | None], Any],
    has_wsi_file: Callable[[str, str | None], bool],
    resolve_dataset_path: Callable[[str | Path], Path],
    project_labels_path: Callable[[str | None], Path | None],
    data_root_provider: Callable[[], Path],
    embeddings_dir_provider: Callable[[], Path],
    slides_dir_provider: Callable[[], Path],
    available_slides_provider: Callable[[], list[str]],
    project_slide_ids: Callable[..., Awaitable[set[str] | None]],
    filter_project_candidate_slide_ids: Callable[[set[str], set[str]], list[str]],
    resolve_slide_path: Callable[[str, str | None], Path | None],
) -> SlideLister:
    flat_file_cache: dict[str, tuple[float, list[SlideInfo]]] = {}

    async def list_slides(
        project_id: str | None = None,
        include_metadata: bool = False,
    ) -> list[SlideInfo]:
        """List all available slides with patient context."""
        import json
        import time

        proj_cfg = require_project(project_id)
        embeddings_dir = embeddings_dir_provider()
        slides_dir = slides_dir_provider()
        data_root = data_root_provider()

        # For project-scoped requests we intentionally bypass DB rows because
        # patch counts can lag behind active re-embedding jobs. Flat-file scan
        # reads live embedding arrays from the project's directory.
        if db_available_provider() and not project_id:
            try:
                t0 = time.time()
                rows = await db_module.get_all_slides()
                db_slides = []
                for row in rows:
                    patient_ctx = None
                    if any(
                        row.get(k)
                        for k in ("age", "sex", "stage", "grade", "prior_lines", "histology")
                    ):
                        patient_ctx = PatientContext(
                            age=row.get("age"),
                            sex=row.get("sex"),
                            stage=row.get("stage"),
                            grade=row.get("grade"),
                            prior_lines=row.get("prior_lines"),
                            histology=row.get("histology"),
                        )
                    db_slides.append(
                        SlideInfo(
                            slide_id=row["slide_id"],
                            patient_id=row.get("patient_id"),
                            has_wsi=has_wsi_file(row["slide_id"], project_id),
                            has_embeddings=row.get("has_embeddings", False),
                            has_level0_embeddings=row.get("has_level0_embeddings", False),
                            label=row.get("label"),
                            num_patches=row.get("num_patches"),
                            patient=patient_ctx,
                            dimensions=SlideDimensions(
                                width=row.get("width") or 0,
                                height=row.get("height") or 0,
                            ),
                            mpp=row.get("mpp"),
                            magnification=row.get("magnification") or "40x",
                        )
                    )
                elapsed_ms = (time.time() - t0) * 1000
                logger.info(
                    "/api/slides returned %d slides from DB in %.0fms",
                    len(db_slides),
                    elapsed_ms,
                )
                return db_slides
            except Exception as exc:
                logger.warning("DB query failed, falling back to flat-file scan: %s", exc)

        # Cache flat-file scan results per project + metadata mode for 60s.
        _cache_key = f"{project_id or '__global__'}|meta={int(include_metadata)}"
        _cached = flat_file_cache.get(_cache_key)
        if _cached and (time.time() - _cached[0]) < 60:
            logger.info(
                "Returning cached flat-file scan for %s (%d slides)",
                _cache_key,
                len(_cached[1]),
            )
            return _cached[1]

        fallback_embeddings_dir = embeddings_dir
        fallback_slides_dir = slides_dir
        if proj_cfg:
            fallback_embeddings_dir = resolve_dataset_path(proj_cfg.dataset.embeddings_dir)
            fallback_slides_dir = resolve_dataset_path(proj_cfg.dataset.slides_dir)

        slides: list[SlideInfo] = []
        if proj_cfg:
            labels_path = project_labels_path(project_id)
        else:
            labels_path = data_root / "labels.csv"
            if not labels_path.exists():
                labels_path = data_root / "tcga_full" / "labels.csv"
            if not labels_path.exists():
                labels_path = embeddings_dir.parent / "labels.csv"

        slide_data: dict[str, dict[str, Any]] = {}

        if labels_path and labels_path.exists():
            if labels_path.suffix.lower() == ".json":
                try:
                    with open(labels_path) as f:
                        labels_json = json.load(f)
                    if isinstance(labels_json, dict):
                        for sid, label in labels_json.items():
                            slide_data[str(sid)] = {"label": str(label), "patient": None}
                except Exception as exc:
                    logger.warning("Could not parse labels JSON %s: %s", labels_path, exc)
            else:
                import csv

                with open(labels_path) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if "slide_id" in row:
                            sid = row["slide_id"]
                            label = row.get("label", "")
                        else:
                            slide_file = row.get("slide_file", "")
                            sid = slide_file.replace(".svs", "").replace(".SVS", "")
                            response = row.get("treatment_response", "")
                            label = (
                                "1"
                                if response == "responder"
                                else "0"
                                if response == "non-responder"
                                else ""
                            )

                        if sid:
                            patient_ctx = None
                            if any(
                                k in row
                                for k in [
                                    "age",
                                    "sex",
                                    "stage",
                                    "grade",
                                    "prior_treatments",
                                    "histology",
                                ]
                            ):
                                try:
                                    age_val = row.get("age")
                                    prior_val = row.get("prior_treatments")
                                    patient_ctx = PatientContext(
                                        age=int(age_val) if age_val else None,
                                        sex=row.get("sex") or None,
                                        stage=row.get("stage") or None,
                                        grade=row.get("grade") or None,
                                        prior_lines=int(prior_val) if prior_val else None,
                                        histology=row.get("histology") or None,
                                    )
                                except (ValueError, TypeError):
                                    patient_ctx = None

                            slide_data[sid] = {"label": label, "patient": patient_ctx}

        if proj_cfg:
            if not fallback_embeddings_dir.exists():
                logger.info(
                    "Project '%s' embeddings dir missing (%s); returning 0 slides (no global fallback)",
                    project_id,
                    fallback_embeddings_dir,
                )
                flat_file_cache[_cache_key] = (time.time(), [])
                return []

            project_slide_ids_set: set[str] = set()
            candidate_emb_dirs = [fallback_embeddings_dir]
            if fallback_embeddings_dir.name != "level0":
                candidate_emb_dirs.append(fallback_embeddings_dir / "level0")

            for emb_dir in candidate_emb_dirs:
                if not emb_dir.exists():
                    continue
                for path in sorted(emb_dir.glob("*.npy")):
                    if not path.name.endswith("_coords.npy"):
                        project_slide_ids_set.add(path.stem)

            authoritative_ids = await project_slide_ids(
                project_id,
                include_embedding_fallback=False,
            )

            if authoritative_ids:
                filtered_ids = set(
                    filter_project_candidate_slide_ids(
                        project_slide_ids_set,
                        authoritative_ids,
                    )
                )

                slide_exts = {".svs", ".tiff", ".tif", ".ndpi", ".mrxs", ".vms", ".scn"}
                project_wsi_ids: set[str] = set()
                if fallback_slides_dir.exists():
                    for slide_file in fallback_slides_dir.iterdir():
                        if slide_file.is_file() and slide_file.suffix.lower() in slide_exts:
                            project_wsi_ids.add(slide_file.stem)

                if project_wsi_ids:
                    filtered_ids.update(
                        filter_project_candidate_slide_ids(
                            project_slide_ids_set,
                            project_wsi_ids,
                        )
                    )

                removed = len(project_slide_ids_set) - len(filtered_ids)
                if removed > 0:
                    logger.warning(
                        "Project '%s' slide scoping guard filtered %d cross-project IDs",
                        project_id,
                        removed,
                    )

                fallback_slide_ids = sorted(filtered_ids)
            else:
                fallback_slide_ids = sorted(project_slide_ids_set)
        else:
            fallback_slide_ids = available_slides_provider()

        for slide_id in fallback_slide_ids:
            emb_path = fallback_embeddings_dir / f"{slide_id}.npy"
            if proj_cfg and not emb_path.exists():
                level0_emb_path = fallback_embeddings_dir / "level0" / f"{slide_id}.npy"
                if level0_emb_path.exists():
                    emb_path = level0_emb_path

            num_patches = None
            data = slide_data.get(slide_id, {})
            dims = SlideDimensions()
            mpp = None
            slide_path = resolve_slide_path(slide_id, project_id)

            if include_metadata:
                if emb_path.exists():
                    try:
                        emb = np.load(emb_path)
                        num_patches = len(emb)
                    except Exception:
                        pass

                if slide_path is not None and slide_path.exists():
                    try:
                        import openslide

                        with openslide.OpenSlide(str(slide_path)) as slide:
                            dims = SlideDimensions(
                                width=slide.dimensions[0],
                                height=slide.dimensions[1],
                            )
                            mpp_x = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
                            if mpp_x:
                                mpp = float(mpp_x)
                    except Exception as exc:
                        logger.warning("Could not read slide %s: %s", slide_id, exc)
                elif num_patches is not None and num_patches > 0:
                    import math

                    grid_side = int(math.ceil(math.sqrt(num_patches)))
                    estimated_px = grid_side * 256
                    dims = SlideDimensions(width=estimated_px, height=estimated_px)

            has_level0 = False
            if fallback_embeddings_dir.name == "level0":
                emb_check = fallback_embeddings_dir / f"{slide_id}.npy"
                has_level0 = emb_check.exists()
            else:
                level0_dir = fallback_embeddings_dir / "level0"
                if level0_dir.exists():
                    has_level0 = (level0_dir / f"{slide_id}.npy").exists()
                elif proj_cfg:
                    emb_check = fallback_embeddings_dir / f"{slide_id}.npy"
                    has_level0 = emb_check.exists()

            slides.append(
                SlideInfo(
                    slide_id=slide_id,
                    has_wsi=(slide_path is not None and slide_path.exists()),
                    has_embeddings=True,
                    has_level0_embeddings=has_level0,
                    label=data.get("label"),
                    num_patches=num_patches,
                    patient=data.get("patient"),
                    dimensions=dims,
                    mpp=mpp,
                )
            )

        flat_file_cache[_cache_key] = (time.time(), slides)
        logger.info("Flat-file scan for %s: %d slides (cached for 60s)", _cache_key, len(slides))
        return slides

    return list_slides


def create_slide_list_router(*, list_slides: SlideLister) -> APIRouter:
    router = APIRouter()

    @router.get("/api/slides", response_model=list[SlideInfo])
    async def list_slides_endpoint(
        project_id: str | None = Query(None, description="Filter slides by project"),
        include_metadata: bool = Query(
            default=False,
            description="Include expensive per-slide metadata (num_patches, dimensions, mpp).",
        ),
    ):
        return await list_slides(project_id=project_id, include_metadata=include_metadata)

    return router
