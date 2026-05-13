"""
Enso Atlas API - FastAPI backend for project-scoped pathology workflows.

This module serves slide analysis, report generation, embeddings, and WSI
visualization endpoints. Most data-facing routes accept an optional
``project_id`` that scopes dataset paths, model visibility, and labels to the
active project configuration.
"""

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from . import audit as audit_store
from . import database as db
from .analysis_routes import create_analysis_router
from .annotation_routes import create_annotation_router
from .async_batch_analysis_routes import create_async_batch_analysis_router
from .batch_analysis_routes import create_batch_analysis_router
from .batch_embed_tasks import batch_embed_manager
from .batch_tasks import batch_task_manager
from .chat_routes import create_chat_router
from .core_routes import create_core_router
from .database_routes import create_database_router
from .embedding_generation_routes import create_embedding_generation_router
from .embedding_routes import create_embedding_router
from .embedding_tasks import task_manager
from .feature_status_routes import create_feature_status_router
from .group_routes import create_group_router
from .heatmap_routes import create_heatmap_router
from .history_routes import router as history_router
from .model_heatmap_routes import create_model_heatmap_router
from .model_routes import create_model_router
from .model_scope import (
    require_model_allowed_for_scope,
    resolve_project_model_scope,
)
from .multi_model_analysis_routes import create_multi_model_analysis_router
from .patch_analysis_routes import create_patch_analysis_router
from .patch_coordinates import (
    infer_patch_size_from_coords as _infer_patch_size_from_coords,
)
from .patch_coordinates import (
    normalize_coords_to_level0 as _normalize_coords_to_level0,
)
from .pdf_routes import create_pdf_router
from .perf_observability import (
    InMemoryLatencyTracker,
    RequestLatencyRecord,
    normalize_perf_path,
    should_track_path,
)
from .prediction import score_to_prediction
from .project_routes import router as project_router
from .project_routes import set_registry as set_project_registry
from .projects import ProjectRegistry
from .report_routes import create_report_router
from .report_tasks import report_task_manager
from .runtime_config import env_flag as _env_flag
from .runtime_config import env_int as _env_int
from .semantic_guardrails import resolve_semantic_siglip_plan as _resolve_semantic_siglip_plan
from .semantic_search_routes import create_semantic_search_router
from .similar_routes import create_similar_router
from .slide_list_routes import build_slide_lister, create_slide_list_router
from .slide_metadata import SlideMetadataManager, create_metadata_router
from .slide_qc_routes import create_slide_qc_router
from .slide_resolution import (
    filter_project_candidate_slide_ids as _filter_project_candidate_slide_ids,
)
from .slide_resolution import (
    load_slide_ids_from_labels_file as _load_slide_ids_from_labels_file,
)
from .slide_resolution import (
    resolve_slide_path_in_dirs as _resolve_slide_path_in_dirs,
)
from .slide_search_routes import create_slide_search_router
from .slide_status_routes import create_slide_status_router
from .slide_viewer_routes import create_slide_viewer_router
from .state import ApiRuntimeState
from .task_status_routes import create_task_status_router
from .tissue_routes import create_tissue_router
from .uncertainty_routes import create_uncertainty_router
from .visual_search_routes import create_visual_search_router
from .wsi_service import WsiService

# PDF Export
try:
    from .pdf_export import generate_pdf_report, generate_report_pdf

    PDF_EXPORT_AVAILABLE = True
except ImportError:
    PDF_EXPORT_AVAILABLE = False
    generate_pdf_report = None
    generate_report_pdf = None

# Configure logging to show INFO level for our module (Python defaults to WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Multi-model TransMIL inference
import sys

_project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_project_root))
try:
    from scripts.multi_model_inference import MODEL_CONFIGS, MultiModelInference

    MULTI_MODEL_AVAILABLE = True
except ImportError:
    logger.warning("MultiModelInference not available - multi-model endpoints disabled")
    MULTI_MODEL_AVAILABLE = False
    MultiModelInference = None
    MODEL_CONFIGS = {}

# Agent workflow for multi-step analysis
try:
    from ..agent.routes import router as agent_router
    from ..agent.routes import set_agent_workflow
    from ..agent.workflow import AgentWorkflow

    AGENT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Agent workflow not available: {e}")
    AGENT_AVAILABLE = False
    AgentWorkflow = None
    agent_router = None
    set_agent_workflow = None


# Analysis history and audit storage live in src/enso_atlas/api/audit.py.
MAX_HISTORY_SIZE = audit_store.MAX_HISTORY_SIZE
analysis_history = audit_store.analysis_history
audit_log = audit_store.audit_log


def get_timestamp() -> str:
    """Get current ISO timestamp."""
    return audit_store.get_timestamp()


# Track server startup time for uptime calculation
_STARTUP_TIME = time.time()
# Cache CUDA probe once at startup; avoid per-request GPU checks in health endpoint.
_CUDA_AVAILABLE_AT_STARTUP: bool | None = None


def log_audit_event(
    event_type: str,
    slide_id: str | None = None,
    user_id: str = "clinician",
    details: dict[str, Any] | None = None,
):
    """Log an audit event for compliance tracking."""
    audit_store.log_audit_event(
        event_type,
        slide_id=slide_id,
        user_id=user_id,
        details=details,
        logger=logger,
    )


def save_analysis_to_history(
    slide_id: str,
    prediction: str,
    score: float,
    confidence: float,
    patches_analyzed: int,
    top_evidence: list[dict[str, Any]],
    similar_cases: list[dict[str, Any]],
    user_id: str = "clinician",
) -> dict[str, Any]:
    """Save analysis result to history and return the entry."""
    return audit_store.save_analysis_to_history(
        slide_id=slide_id,
        prediction=prediction,
        score=score,
        confidence=confidence,
        patches_analyzed=patches_analyzed,
        top_evidence=top_evidence,
        similar_cases=similar_cases,
        user_id=user_id,
        logger=logger,
    )


def _check_cuda() -> bool:
    """Check if CUDA is available."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def create_app(
    embeddings_dir: Path = Path("data/embeddings"),
    model_path: Path = Path("models/clam_attention.pt"),
    enable_cors: bool = True,
) -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="Enso Atlas API",
        description="On-Prem Pathology Evidence Engine for Treatment-Response Insight",
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
    )

    # CORS middleware for frontend development
    if enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[
                "*",
                "http://localhost:3000",
                "http://localhost:3001",
                "http://localhost:7860",
                "http://100.111.126.23:3000",
                "http://100.111.126.23:8003",
                "http://100.111.126.23:3002",
            ],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Load models on startup
    classifier = None
    evidence_gen = None
    embedder = None
    medsiglip_embedder = None
    reporter = None
    decision_support = None  # Clinical decision support engine
    multi_model_inference = None  # Multi-model TransMIL inference
    chat_manager = None  # Initialized after startup resources are loaded
    slide_siglip_embeddings = {}  # Cache for MedSigLIP embeddings per slide
    available_slides = []
    slide_labels = {}  # Cache for slide labels (slide_id -> label string)
    model_checkpoint_signatures: dict[
        str, str
    ] = {}  # model_id -> checkpoint signature for hot-reload/cache invalidation
    db_available = False  # Whether PostgreSQL is connected and populated
    project_registry: ProjectRegistry | None = None  # Loaded at startup from config/projects.yaml
    # Slide-level mean-embedding FAISS index (cosine similarity)
    slide_mean_index = None  # faiss.IndexFlatIP over L2-normalized mean embeddings
    slide_mean_ids: list[str] = []
    slide_mean_meta: dict[str, dict] = {}  # slide_id -> metadata (n_patches, label, patient, etc.)
    # Directories (may be updated at startup if we fall back to demo data)
    # Data root is always "data/" regardless of embeddings subdirectory (e.g. data/embeddings/level0)
    _data_root: Path = Path("data")
    slides_dir: Path = _data_root / "slides"
    runtime_state = ApiRuntimeState(
        embeddings_dir=embeddings_dir,
        model_path=model_path,
        data_root=_data_root,
        slides_dir=slides_dir,
    )
    app.state.runtime = runtime_state
    app.include_router(history_router)

    perf_enabled = _env_flag("ENSO_PERF_ENABLED", default=True)
    perf_summary_enabled = _env_flag("ENSO_PERF_SUMMARY_ENABLED", default=perf_enabled)
    perf_max_samples = _env_int("ENSO_PERF_MAX_SAMPLES", default=2000, minimum=100, logger=logger)
    perf_route_limit = _env_int("ENSO_PERF_ROUTE_LIMIT", default=25, minimum=1, logger=logger)
    perf_tracker = InMemoryLatencyTracker(max_samples=perf_max_samples)
    perf_logger = logging.getLogger("enso_atlas.perf")

    # Semantic-search safety guardrails:
    # on-the-fly SigLIP embedding can take many minutes on large slides and
    # block single-worker API responsiveness. Keep disabled by default.
    semantic_allow_on_the_fly_siglip = _env_flag(
        "ENSO_SEMANTIC_ALLOW_ON_THE_FLY_SIGLIP",
        default=False,
    )
    semantic_on_the_fly_max_patches = _env_int(
        "ENSO_SEMANTIC_ON_THE_FLY_MAX_PATCHES",
        default=1024,
        minimum=256,
        maximum=4096,
        logger=logger,
    )

    if perf_enabled:

        @app.middleware("http")
        async def request_timing_middleware(request: Request, call_next):
            request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
            request.state.request_id = request_id
            started_at = time.perf_counter()

            response = None
            status_code = 500

            try:
                response = await call_next(request)
                status_code = response.status_code
            except Exception:
                status_code = 500
                raise
            finally:
                route_obj = request.scope.get("route")
                route_path = getattr(route_obj, "path", None) or request.url.path
                normalized_path = normalize_perf_path(route_path)
                duration_ms = (time.perf_counter() - started_at) * 1000.0

                if should_track_path(normalized_path):
                    perf_tracker.add(
                        RequestLatencyRecord(
                            method=request.method.upper(),
                            path=normalized_path,
                            status=status_code,
                            duration_ms=duration_ms,
                            request_id=request_id,
                            ts_unix=time.time(),
                        )
                    )
                    perf_logger.info(
                        "request_timing %s",
                        json.dumps(
                            {
                                "method": request.method.upper(),
                                "path": normalized_path,
                                "status": status_code,
                                "duration_ms": round(duration_ms, 3),
                                "request_id": request_id,
                            },
                            separators=(",", ":"),
                        ),
                    )

            if response is not None:
                response.headers.setdefault("X-Request-ID", request_id)
                return response

            raise RuntimeError("request_timing_middleware reached unexpected state")

    logger.info(
        "Perf observability enabled=%s summary_enabled=%s max_samples=%d",
        perf_enabled,
        perf_summary_enabled,
        perf_max_samples,
    )

    # Hot-path memoization caches (small TTL to avoid stale scope semantics).
    _PROJECT_SCOPE_CACHE_TTL_S = 15.0
    _SLIDE_DIMS_CACHE_TTL_S = 300.0
    _project_model_scope_cache: dict[str, dict[str, Any]] = {}
    _labels_slide_ids_cache: dict[str, dict[str, Any]] = {}
    _slide_dims_cache: dict[str, dict[str, Any]] = {}

    def _classifier_threshold(default: float = 0.5) -> float:
        """Return a safe numeric decision threshold for binary predictions."""
        raw_threshold = getattr(classifier, "threshold", None)
        if raw_threshold is None:
            return float(default)
        try:
            return float(raw_threshold)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid classifier threshold %r; falling back to %.3f",
                raw_threshold,
                default,
            )
            return float(default)

    def _resolve_dataset_path(path_str: str | Path) -> Path:
        """Resolve a dataset path from config (repo-relative or absolute)."""
        p = Path(path_str)
        if p.is_absolute():
            return p
        return _data_root.parent / p

    def _project_labels_path(project_id: str | None) -> Path | None:
        """Resolve the configured labels file for a project, if available."""
        if not project_id or not project_registry:
            return None
        proj_cfg = project_registry.get_project(project_id)
        if not proj_cfg or not getattr(proj_cfg, "dataset", None):
            return None
        try:
            return _resolve_dataset_path(proj_cfg.dataset.labels_file)
        except Exception:
            return None

    def _log_timing(event: str, started_at: float, **fields: Any) -> float:
        """Emit a structured timing log and return elapsed milliseconds."""
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        payload = {"event": event, "duration_ms": round(elapsed_ms, 2), **fields}
        logger.info("PERF %s", json.dumps(payload, sort_keys=True, default=str))
        return elapsed_ms

    def _path_signature(path: Path | None) -> str | None:
        """Return a cheap file signature for cache invalidation."""
        if path is None or not path.exists():
            return None
        try:
            st = path.stat()
            return f"{path}:{int(st.st_mtime_ns)}:{int(st.st_size)}"
        except Exception:
            return None

    def _load_slide_ids_from_labels_file_cached(labels_path: Path | None) -> set[str]:
        """Memoize labels-file slide IDs by file signature to avoid repeated I/O."""
        if labels_path is None or not labels_path.exists():
            return set()

        cache_key = str(labels_path)
        signature = _path_signature(labels_path)
        cached = _labels_slide_ids_cache.get(cache_key)
        if cached and cached.get("signature") == signature:
            return set(cached.get("slide_ids", ()))

        slide_ids = _load_slide_ids_from_labels_file(labels_path)
        _labels_slide_ids_cache[cache_key] = {
            "signature": signature,
            "slide_ids": tuple(sorted(slide_ids)),
            "ts": time.time(),
        }
        return slide_ids

    def resolve_slide_path(slide_id: str, project_id: str | None = None) -> Path | None:
        """Resolve slide file path across possible slide directories.

        If project_id is provided, search ONLY that project's configured slides_dir
        for deterministic project-scoped behavior.
        """
        candidates_dirs: list[Path] = []

        if project_id:
            if not project_registry:
                return None
            proj_cfg = project_registry.get_project(project_id)
            if not proj_cfg:
                return None
            try:
                candidates_dirs.append(_resolve_dataset_path(proj_cfg.dataset.slides_dir))
            except Exception:
                return None
        else:
            # Global/default common locations
            candidates_dirs.extend(
                [
                    slides_dir,
                    _data_root / "tcga_full" / "slides",
                    _data_root / "ovarian_bev" / "slides",
                    _data_root / "demo" / "slides",
                    _data_root / "slides",
                ]
            )

            if project_registry:
                try:
                    for _pid, _cfg in project_registry.list_projects().items():
                        p = _resolve_dataset_path(_cfg.dataset.slides_dir)
                        if p not in candidates_dirs:
                            candidates_dirs.append(p)
                except Exception:
                    pass

        exts = [".svs", ".tiff", ".tif", ".ndpi", ".mrxs", ".vms", ".scn"]
        return _resolve_slide_path_in_dirs(
            slide_id,
            candidates_dirs,
            exts,
            logger_obj=logger,
        )

    def has_wsi_file(slide_id: str, project_id: str | None = None) -> bool:
        return resolve_slide_path(slide_id, project_id=project_id) is not None

    def _require_project(project_id: str | None):
        """Validate and return project config when project_id is supplied."""
        if not project_id:
            return None
        if not project_registry:
            raise HTTPException(
                status_code=503,
                detail="Project registry not available",
            )
        proj_cfg = project_registry.get_project(project_id)
        if not proj_cfg:
            raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")
        return proj_cfg

    def _resolve_project_embeddings_dir(
        project_id: str | None,
        *,
        require_exists: bool = False,
    ) -> Path:
        """Resolve embeddings dir for a project, defaulting to global embeddings_dir."""
        proj_cfg = _require_project(project_id)
        if not proj_cfg:
            return embeddings_dir

        proj_emb_dir = Path(proj_cfg.dataset.embeddings_dir)
        if not proj_emb_dir.is_absolute():
            proj_emb_dir = _data_root.parent / proj_emb_dir

        if require_exists and not proj_emb_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Embeddings directory not found for project '{project_id}'",
            )
        return proj_emb_dir

    async def _resolve_project_model_scope_cached(project_id: str):
        """Resolve project model scope with short-lived in-memory memoization."""
        now = time.time()
        cached = _project_model_scope_cache.get(project_id)
        if cached and (now - float(cached.get("ts", 0.0))) < _PROJECT_SCOPE_CACHE_TTL_S:
            return cached["scope"]

        started_at = time.perf_counter()
        scope = await resolve_project_model_scope(
            project_id,
            project_registry=project_registry,
            get_project_models=db.get_project_models,
            logger=logger,
        )
        _project_model_scope_cache[project_id] = {
            "scope": scope,
            "ts": now,
        }
        _log_timing(
            "project_model_scope.resolve",
            started_at,
            project_id=project_id,
            cache_hit=False,
            allowed_count=len(scope.allowed_model_ids),
            project_exists=scope.project_exists,
        )
        return scope

    def _slide_dims_from_coords(
        coords_arr: np.ndarray | None, *, patch_size: int = 224
    ) -> tuple[int, int]:
        """Derive slide dimensions from coordinates when WSI metadata is unavailable."""
        if coords_arr is None:
            return (patch_size, patch_size)

        arr = np.asarray(coords_arr)
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
            return (patch_size, patch_size)

        x_max = int(arr[:, 0].max()) + patch_size
        y_max = int(arr[:, 1].max()) + patch_size
        return (x_max, y_max)

    def _resolve_slide_dims_cached(
        slide_id: str,
        *,
        project_id: str | None,
        coord_path: Path | None = None,
        coords_arr: np.ndarray | None = None,
        patch_size: int = 224,
    ) -> tuple[tuple[int, int], str]:
        """Resolve slide dimensions with lightweight memoization."""
        cache_key = f"{project_id or '__global__'}::{slide_id}"
        now = time.time()

        slide_path = resolve_slide_path(slide_id, project_id=project_id)
        slide_sig = _path_signature(slide_path)
        coord_sig = _path_signature(coord_path)
        expected_signature = slide_sig or coord_sig

        cached = _slide_dims_cache.get(cache_key)
        if cached and (now - float(cached.get("ts", 0.0))) < _SLIDE_DIMS_CACHE_TTL_S:
            if cached.get("signature") == expected_signature:
                dims = cached.get("dims")
                if isinstance(dims, (tuple, list)) and len(dims) == 2:
                    return (int(dims[0]), int(dims[1])), "cache"

        if slide_path is not None and slide_path.exists():
            try:
                import openslide

                with openslide.OpenSlide(str(slide_path)) as slide:
                    dims = (int(slide.dimensions[0]), int(slide.dimensions[1]))
                _slide_dims_cache[cache_key] = {
                    "dims": dims,
                    "signature": slide_sig,
                    "source": "wsi",
                    "ts": now,
                }
                return dims, "wsi"
            except Exception as e:
                logger.warning(
                    "Could not read slide dimensions for %s/%s: %s", project_id, slide_id, e
                )

        if coords_arr is None and coord_path is not None and coord_path.exists():
            try:
                coords_arr = np.load(coord_path).astype(np.int64, copy=False)
            except Exception as e:
                logger.warning(
                    "Could not load coords for dimension fallback %s/%s: %s",
                    project_id,
                    slide_id,
                    e,
                )

        dims = _slide_dims_from_coords(coords_arr, patch_size=patch_size)
        _slide_dims_cache[cache_key] = {
            "dims": dims,
            "signature": expected_signature,
            "source": "coords",
            "ts": now,
        }
        return dims, "coords"

    def _candidate_embedding_dirs_for_level(
        base_embeddings_dir: Path,
        *,
        level: int,
        project_id: str | None,
    ) -> list[Path]:
        """Resolve candidate embedding directories for a request.

        Level 0 is strict dense-only: we only check explicit ``level0`` directories
        (project-level and global-level). We intentionally do not fall back to flat
        project embedding roots for level 0 because those roots may contain sparse
        embeddings, which causes silent dense/sparse mismatches.
        """
        project_requested = project_id is not None
        candidates: list[Path] = []

        def _add(path: Path):
            if path not in candidates:
                candidates.append(path)

        if level == 0:
            if base_embeddings_dir.name == "level0":
                _add(base_embeddings_dir)
            else:
                _add(base_embeddings_dir / "level0")

            if not project_requested:
                global_level0 = (
                    embeddings_dir if embeddings_dir.name == "level0" else embeddings_dir / "level0"
                )
                _add(global_level0)
                _add(_data_root / "embeddings" / "level0")
        else:
            _add(base_embeddings_dir)
            if base_embeddings_dir.name != "level1":
                _add(base_embeddings_dir / "level1")

            if not project_requested:
                _add(embeddings_dir)
                _add(_data_root / "embeddings" / "level1")

        return candidates

    def _resolve_embedding_path(
        slide_id: str,
        *,
        level: int,
        project_id: str | None,
        base_embeddings_dir: Path | None = None,
    ) -> tuple[Path | None, list[Path]]:
        """Return ``(embedding_path, searched_dirs)`` for a slide request.

        Uses project-scoped embedding roots when ``project_id`` is provided,
        then checks level-specific candidate directories in deterministic order.
        """
        project_requested = project_id is not None
        resolved_base = base_embeddings_dir or _resolve_project_embeddings_dir(
            project_id,
            require_exists=project_requested,
        )
        candidate_dirs = _candidate_embedding_dirs_for_level(
            resolved_base,
            level=level,
            project_id=project_id,
        )

        for d in candidate_dirs:
            emb_path = d / f"{slide_id}.npy"
            if emb_path.exists():
                return emb_path, candidate_dirs

        return None, candidate_dirs

    def _resolve_project_label_pair(
        project_id: str | None,
        *,
        positive_default: str,
        negative_default: str,
        uppercase: bool = False,
    ) -> tuple[str, str]:
        """Resolve positive/negative labels for a project with sensible defaults."""
        pos_label = positive_default
        neg_label = negative_default

        proj_cfg = _require_project(project_id)
        if proj_cfg and proj_cfg.classes:
            pos_label = proj_cfg.positive_class if proj_cfg.positive_class else proj_cfg.classes[-1]
            neg_candidates = [c for c in proj_cfg.classes if c.lower() != pos_label.lower()]
            neg_label = neg_candidates[0] if neg_candidates else negative_default

        if uppercase:
            pos_label = pos_label.upper()
            neg_label = neg_label.upper()

        return pos_label, neg_label

    async def _project_slide_ids(
        project_id: str | None,
        *,
        include_embedding_fallback: bool = True,
    ) -> set[str] | None:
        """Resolve allowed slide IDs for a project.

        Resolution order:
        1) project_slides junction table (authoritative)
        2) project labels file (authoritative flat-file fallback)
        3) project embeddings directory (optional inventory fallback)

        Returns ``None`` when ``project_id`` is not provided.
        """
        if not project_id:
            return None

        started_at = time.perf_counter()
        _require_project(project_id)

        if db_available:
            try:
                assigned = [sid for sid in (await db.get_project_slides(project_id)) if sid]
                if assigned:
                    _log_timing(
                        "project_slide_scope.resolve",
                        started_at,
                        project_id=project_id,
                        source="project_slides",
                        count=len(assigned),
                    )
                    return set(assigned)
            except Exception as e:
                logger.warning(f"DB project_slides query failed for {project_id}: {e}")

        labels_ids = _load_slide_ids_from_labels_file_cached(_project_labels_path(project_id))
        if labels_ids:
            _log_timing(
                "project_slide_scope.resolve",
                started_at,
                project_id=project_id,
                source="labels",
                count=len(labels_ids),
            )
            return labels_ids

        if not include_embedding_fallback:
            _log_timing(
                "project_slide_scope.resolve",
                started_at,
                project_id=project_id,
                source="none",
                count=0,
            )
            return set()

        proj_emb_dir = _resolve_project_embeddings_dir(project_id, require_exists=True)
        if not proj_emb_dir.exists():
            _log_timing(
                "project_slide_scope.resolve",
                started_at,
                project_id=project_id,
                source="embeddings_missing",
                count=0,
            )
            return set()

        slide_ids = set()
        for f in proj_emb_dir.glob("*.npy"):
            if not f.name.endswith("_coords.npy"):
                slide_ids.add(f.stem)
        _log_timing(
            "project_slide_scope.resolve",
            started_at,
            project_id=project_id,
            source="embeddings",
            count=len(slide_ids),
        )
        return slide_ids

    async def _batch_embed_inventory_slide_ids(project_id: str | None) -> list[str]:
        """Build batch-embed slide inventory with project-aware scoping."""
        proj_cfg = _require_project(project_id)

        slide_ids: set[str] = set()

        authoritative_slide_ids = await _project_slide_ids(project_id)
        if authoritative_slide_ids:
            slide_ids.update(authoritative_slide_ids)

        base_embeddings_dir = _resolve_project_embeddings_dir(project_id, require_exists=False)
        for emb_dir in _candidate_embedding_dirs_for_level(
            base_embeddings_dir,
            level=0,
            project_id=project_id,
        ):
            if not emb_dir.exists() or not emb_dir.is_dir():
                continue
            for emb_file in emb_dir.glob("*.npy"):
                if emb_file.name.endswith("_coords.npy"):
                    continue
                slide_ids.add(emb_file.stem)

        slide_dirs: list[Path] = []
        if proj_cfg:
            try:
                slide_dirs.append(_resolve_dataset_path(proj_cfg.dataset.slides_dir))
            except Exception:
                pass
        else:
            slide_dirs.extend(
                [
                    slides_dir,
                    _data_root / "tcga_full" / "slides",
                    _data_root / "ovarian_bev" / "slides",
                    _data_root / "demo" / "slides",
                    _data_root / "slides",
                ]
            )
            if project_registry:
                try:
                    for _, cfg in project_registry.list_projects().items():
                        cand = _resolve_dataset_path(cfg.dataset.slides_dir)
                        if cand not in slide_dirs:
                            slide_dirs.append(cand)
                except Exception:
                    pass

        slide_exts = {".svs", ".tiff", ".tif", ".ndpi", ".mrxs", ".vms", ".scn"}
        for slide_dir in slide_dirs:
            if not slide_dir.exists() or not slide_dir.is_dir():
                continue
            try:
                for slide_file in slide_dir.iterdir():
                    if slide_file.is_file() and slide_file.suffix.lower() in slide_exts:
                        slide_ids.add(slide_file.stem)
            except Exception:
                continue

        return sorted(slide_ids)

    def _similar_case_slide_id(candidate: Any) -> str | None:
        """Extract slide_id from similar-case payloads (flat or metadata-nested)."""
        if not isinstance(candidate, dict):
            return None

        sid = candidate.get("slide_id")
        if sid:
            return str(sid)

        meta = candidate.get("metadata")
        if isinstance(meta, dict):
            meta_sid = meta.get("slide_id")
            if meta_sid:
                return str(meta_sid)

        return None

    async def _resolve_project_model_ids(project_id: str | None) -> set[str] | None:
        """Resolve the allowed model IDs for a project.

        Returns ``None`` when no ``project_id`` is supplied. When a project is
        supplied, resolution is delegated to ``resolve_project_model_scope``:
        1) DB ``project_models`` assignments
        2) ``config/projects.yaml`` ``classification_models`` fallback

        Raises ``HTTPException(404)`` for unknown project IDs.
        """
        if not project_id:
            return None

        scope = await _resolve_project_model_scope_cached(project_id)
        if not scope.project_exists:
            raise HTTPException(status_code=404, detail=f"Unknown project_id: {project_id}")

        return scope.allowed_model_ids

    def _active_batch_embed_info() -> dict[str, Any] | None:
        """Return lightweight info for the active batch embedding task, if any."""
        active = batch_embed_manager.get_active_task()
        if not active:
            return None
        return {
            "task_id": active.task_id,
            "status": active.status.value,
            "progress": round(active.progress, 1),
            "current_slide_index": active.current_slide_index,
            "total_slides": active.total_slides,
            "current_slide_id": active.current_slide_id,
            "message": active.message,
        }

    async def load_models():
        nonlocal \
            classifier, \
            evidence_gen, \
            embedder, \
            medsiglip_embedder, \
            reporter, \
            decision_support, \
            multi_model_inference, \
            chat_manager, \
            available_slides, \
            slide_labels, \
            slides_dir, \
            embeddings_dir, \
            slide_mean_index, \
            slide_mean_ids, \
            slide_mean_meta, \
            project_registry
        global _CUDA_AVAILABLE_AT_STARTUP

        from ..config import EmbeddingConfig, EvidenceConfig, MILConfig
        from ..embedding.embedder import PathFoundationEmbedder
        from ..embedding.medsiglip import MedSigLIPConfig, MedSigLIPEmbedder
        from ..evidence.generator import EvidenceGenerator
        from ..mil.clam import create_classifier
        from ..reporting.decision_support import ClinicalDecisionSupport
        from ..reporting.medgemma import MedGemmaReporter, ReportingConfig

        def _count_embeddings(p: Path) -> int:
            if not p.exists() or not p.is_dir():
                return 0
            return sum(1 for f in p.glob("*.npy") if not f.name.endswith("_coords.npy"))

        def _count_slides(p: Path) -> int:
            if not p.exists() or not p.is_dir():
                return 0
            exts = {".svs", ".tiff", ".tif", ".ndpi", ".mrxs", ".vms", ".scn"}
            return sum(1 for f in p.iterdir() if f.is_file() and f.suffix.lower() in exts)

        if _CUDA_AVAILABLE_AT_STARTUP is None:
            _CUDA_AVAILABLE_AT_STARTUP = _check_cuda()

        primary_embeddings_dir = embeddings_dir
        primary_slides_dir = slides_dir
        primary_n = _count_embeddings(primary_embeddings_dir)
        primary_s = _count_slides(primary_slides_dir)
        logger.info(f"Embeddings dir: {primary_embeddings_dir} (npy={primary_n})")
        logger.info(f"Slides dir: {primary_slides_dir} (slides={primary_s})")

        if primary_n == 0:
            demo_embeddings_dir = primary_embeddings_dir.parent / "demo" / "embeddings"
            demo_slides_dir = demo_embeddings_dir.parent / "slides"
            demo_n = _count_embeddings(demo_embeddings_dir)
            demo_s = _count_slides(demo_slides_dir)
            if demo_n > 0:
                logger.warning(
                    f"No embeddings found in {primary_embeddings_dir}; falling back to demo embeddings at {demo_embeddings_dir} (npy={demo_n})"
                )
                embeddings_dir = demo_embeddings_dir
                slides_dir = demo_slides_dir
                logger.info(f"Using embeddings dir: {embeddings_dir}")
                logger.info(f"Using slides dir: {slides_dir} (slides={demo_s})")
            else:
                logger.warning(
                    f"No embeddings found in {primary_embeddings_dir} and no demo embeddings found at {demo_embeddings_dir}."
                )

        # Load MIL classifier (architecture from env or default to clam)
        import os as _os

        mil_arch = _os.environ.get("MIL_ARCHITECTURE", "clam")
        mil_threshold_str = _os.environ.get("MIL_THRESHOLD", "")
        mil_threshold = float(mil_threshold_str) if mil_threshold_str else None
        threshold_cfg_path = _os.environ.get(
            "MIL_THRESHOLD_CONFIG",
            str(model_path.parent / "threshold_config.json"),
        )

        # Resolve model checkpoint: use architecture-specific file if it exists
        mil_model_path_env = _os.environ.get("MIL_MODEL_PATH", "")
        if mil_model_path_env:
            mil_model_path = Path(mil_model_path_env)
        elif mil_arch == "transmil":
            candidate = model_path.parent / "transmil_best.pt"
            mil_model_path = candidate if candidate.exists() else model_path
        else:
            mil_model_path = model_path

        config = MILConfig(
            input_dim=384,
            hidden_dim=512,
            architecture=mil_arch,
            threshold=mil_threshold,
            threshold_config_path=threshold_cfg_path,
        )
        classifier = create_classifier(config)
        if mil_model_path.exists():
            classifier.load(mil_model_path)
            logger.info(
                "Loaded MIL model (%s) from %s  [threshold=%.4f]",
                mil_arch,
                mil_model_path,
                classifier.threshold,
            )

        # Initialize multi-model TransMIL inference
        multi_model_inference = None
        if MULTI_MODEL_AVAILABLE:
            outputs_dir = Path(__file__).parent.parent.parent.parent / "outputs"
            if outputs_dir.exists():
                try:
                    multi_model_inference = MultiModelInference(
                        models_dir=outputs_dir,
                        device="auto",
                        load_all=True,
                    )
                    logger.info(
                        f"Multi-model inference initialized with {len(multi_model_inference.models)} models"
                    )
                except Exception as e:
                    logger.warning(f"Failed to initialize multi-model inference: {e}")
            else:
                logger.warning(f"Outputs directory not found: {outputs_dir}")
        else:
            logger.warning("Multi-model inference not available (missing dependencies)")

        # Setup evidence generator
        evidence_config = EvidenceConfig()
        evidence_gen = EvidenceGenerator(evidence_config)

        # Setup embedder (lazy-loaded on first use)
        embedding_config = EmbeddingConfig()
        embedder = PathFoundationEmbedder(embedding_config)

        # Setup MedGemma reporter (lazy-loaded on first use)
        reporting_config = ReportingConfig()
        reporter = MedGemmaReporter(reporting_config)
        logger.info("MedGemma reporter initialized, warming up model...")

        # Warm up MedGemma model during startup to avoid timeout on first call
        # This runs a test inference to ensure CUDA kernels are loaded
        if reporter is not None:
            try:
                logger.info(
                    "Starting MedGemma warmup (may take ~60-120s for GPU kernel compilation)..."
                )
                warmup_start = time.time()
                await asyncio.wait_for(
                    asyncio.to_thread(reporter._warmup_inference),
                    timeout=180.0,
                )
                warmup_duration = time.time() - warmup_start
                logger.info(f"MedGemma reporter warmed up successfully in {warmup_duration:.1f}s")
            except asyncio.TimeoutError:
                warmup_duration = time.time() - warmup_start
                logger.warning(
                    f"MedGemma warmup timed out after {warmup_duration:.1f}s, continuing startup"
                )
            except Exception as e:
                warmup_duration = time.time() - warmup_start
                logger.warning(f"MedGemma warmup failed after {warmup_duration:.1f}s: {e}")

        # Setup clinical decision support engine
        decision_support = ClinicalDecisionSupport()
        logger.info("Clinical decision support engine initialized")

        # Setup MedSigLIP embedder for semantic search
        # Share GPU with MedGemma — SigLIP is ~800MB fp16, fits alongside MedGemma 4B.
        # GPU makes semantic search 10-50x faster (seconds vs minutes for 6000+ patches).
        siglip_config = MedSigLIPConfig(
            cache_dir=str(embeddings_dir / "medsiglip_cache"),
            device="auto",
        )
        medsiglip_embedder = MedSigLIPEmbedder(siglip_config)
        # Load MedSigLIP model on startup to enable semantic search immediately
        try:
            logger.info("Loading MedSigLIP model on startup...")
            medsiglip_embedder._load_model()
            logger.info("MedSigLIP model loaded successfully")
        except Exception as e:
            logger.warning(f"MedSigLIP model loading failed: {e}")

        # Find available slides and build FAISS index
        all_embeddings = []
        all_metadata = []

        if embeddings_dir.exists():
            for f in sorted(embeddings_dir.glob("*.npy")):
                if not f.name.endswith("_coords.npy"):
                    slide_id = f.stem
                    available_slides.append(slide_id)

                    # Load embeddings for FAISS index
                    embs = np.load(f)
                    all_embeddings.append(embs)
                    all_metadata.append(
                        {
                            "slide_id": slide_id,
                            "n_patches": len(embs),
                        }
                    )

            logger.info(f"Found {len(available_slides)} slides with embeddings")

            # Build FAISS index for similarity search
            if all_embeddings:
                evidence_gen.build_reference_index(all_embeddings, all_metadata)
                logger.info(f"Built FAISS index with {len(all_embeddings)} slides")

            # Build slide-level mean-embedding FAISS index for similar-case retrieval
            try:
                import faiss

                means = []
                slide_mean_ids = []
                slide_mean_meta = {}
                for embs, meta in zip(all_embeddings, all_metadata):
                    sid = meta.get("slide_id")
                    if sid is None:
                        continue
                    if embs is None or len(embs) == 0:
                        continue
                    mean = np.asarray(embs, dtype=np.float32).mean(axis=0)
                    mean = mean / (np.linalg.norm(mean) + 1e-12)
                    means.append(mean)
                    slide_mean_ids.append(sid)
                    slide_mean_meta[sid] = {
                        "slide_id": sid,
                        "n_patches": int(meta.get("n_patches", len(embs))),
                    }
                if means:
                    mat = np.vstack(means).astype(np.float32)
                    slide_mean_index = faiss.IndexFlatIP(mat.shape[1])
                    slide_mean_index.add(mat)
                    logger.info(f"Built slide-mean FAISS index with {len(slide_mean_ids)} slides")
                else:
                    slide_mean_index = None
                    logger.warning("No slide means available to build slide-mean FAISS index")
            except Exception as e:
                slide_mean_index = None
                logger.warning(f"Failed to build slide-mean FAISS index: {e}")

        # Load slide labels from labels.csv for similar case retrieval
        # Check multiple label files (primary + tcga_full)
        label_files = [
            _data_root / "labels.csv",
            _data_root / "tcga_full" / "labels.csv",
            embeddings_dir.parent / "labels.csv",
        ]
        # Build a prefix->full_slide_id lookup for matching short names to full IDs
        prefix_to_slide_ids: dict[str, list[str]] = {}
        for sid in available_slides:
            # Extract prefix before UUID: "TCGA-04-1331-01A-01-BS1.uuid" -> "TCGA-04-1331-01A-01-BS1"
            parts = sid.split(".")
            if len(parts) >= 2:
                prefix = parts[0]
            else:
                prefix = sid
            prefix_to_slide_ids.setdefault(prefix, []).append(sid)

        for labels_path in label_files:
            if not labels_path.exists():
                continue
            import csv

            with open(labels_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "slide_id" in row:
                        sid = row["slide_id"]
                        label_val = row.get("label", "")
                    else:
                        slide_file = row.get("slide_file", "")
                        sid = slide_file.replace(".svs", "").replace(".SVS", "")
                        response = row.get("treatment_response", "")
                        label_val = (
                            "responder"
                            if response == "responder"
                            else "non-responder"
                            if response == "non-responder"
                            else ""
                        )

                    # Normalize label format
                    if label_val == "1":
                        label_val = "responder"
                    elif label_val == "0":
                        label_val = "non-responder"
                    # Also check platinum_status column
                    if not label_val:
                        platinum = row.get("platinum_status", "")
                        if platinum == "sensitive":
                            label_val = "responder"
                        elif platinum == "resistant":
                            label_val = "non-responder"

                    if not (sid and label_val):
                        continue

                    # Direct match (full slide ID in CSV)
                    if sid in available_slides or "." in sid:
                        slide_labels[sid] = label_val
                    # Prefix match: short slide name -> all matching full IDs
                    for full_sid in prefix_to_slide_ids.get(sid, []):
                        if full_sid not in slide_labels:
                            slide_labels[full_sid] = label_val

        logger.info(f"Loaded labels for {len(slide_labels)} slides")

        # Attach labels to slide-mean metadata
        try:
            for sid, lab in slide_labels.items():
                if sid in slide_mean_meta:
                    slide_mean_meta[sid]["label"] = lab
        except Exception as e:
            logger.warning(f"Failed to attach labels to slide-mean metadata: {e}")

        # Initialize PostgreSQL database (creates tables, populates from flat files on first run)
        nonlocal db_available
        try:
            logger.info("Initializing PostgreSQL database...")
            await db.init_schema()
            already_populated = await db.is_populated()
            if not already_populated:
                logger.info(
                    "Database is empty, populating from flat files (this may take a few minutes on first run)..."
                )
                await db.populate_from_flat_files(
                    data_root=_data_root,
                    embeddings_dir=embeddings_dir,
                )
            else:
                logger.info("Database already populated, skipping flat-file import")
            db_available = True
            logger.info("PostgreSQL database ready")
        except Exception as e:
            logger.warning(f"PostgreSQL not available, falling back to flat-file mode: {e}")
            db_available = False

        # Load project registry from YAML config
        try:
            _projects_yaml = Path("config/projects.yaml")
            if _projects_yaml.exists():
                _project_registry = ProjectRegistry(_projects_yaml)
                project_registry = _project_registry
                set_project_registry(_project_registry)
                logger.info(
                    f"Project registry loaded: {list(_project_registry.list_projects().keys())}"
                )
                # Sync projects to database
                if db_available:
                    try:
                        await db.populate_projects_from_registry(_project_registry)
                    except Exception as e:
                        logger.warning(f"Failed to sync projects to database: {e}")
            else:
                logger.warning("config/projects.yaml not found, project system disabled")
        except Exception as e:
            logger.warning(f"Failed to load project registry: {e}")

        runtime_state.embeddings_dir = embeddings_dir
        runtime_state.slides_dir = slides_dir
        runtime_state.classifier = classifier
        runtime_state.evidence_generator = evidence_gen
        runtime_state.embedder = embedder
        runtime_state.medsiglip_embedder = medsiglip_embedder
        runtime_state.reporter = reporter
        runtime_state.decision_support = decision_support
        runtime_state.multi_model_inference = multi_model_inference
        runtime_state.project_registry = project_registry
        runtime_state.db_available = db_available
        runtime_state.available_slides = available_slides
        runtime_state.slide_labels = slide_labels
        runtime_state.slide_siglip_embeddings = slide_siglip_embeddings
        runtime_state.model_checkpoint_signatures = model_checkpoint_signatures
        runtime_state.slide_mean_index = slide_mean_index
        runtime_state.slide_mean_ids = slide_mean_ids
        runtime_state.slide_mean_meta = slide_mean_meta

        # Initialize agent workflow now that models and indexes are ready
        if AGENT_AVAILABLE:
            try:
                agent_workflow = AgentWorkflow(
                    embeddings_dir=embeddings_dir,
                    multi_model_inference=multi_model_inference,
                    evidence_generator=evidence_gen,
                    medgemma_reporter=reporter,
                    medsiglip_embedder=medsiglip_embedder,
                    slide_labels=slide_labels,
                    slide_mean_index=slide_mean_index,
                    slide_mean_ids=slide_mean_ids,
                    slide_mean_meta=slide_mean_meta,
                )
                set_agent_workflow(agent_workflow)
                logger.info("Agent workflow initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize agent workflow: {e}")

        try:
            from ..llm.chat import ChatManager

            chat_manager = ChatManager(
                embeddings_dir=embeddings_dir,
                multi_model_inference=multi_model_inference,
                evidence_generator=evidence_gen,
                medgemma_reporter=reporter,
                slide_labels=slide_labels,
                slide_mean_index=slide_mean_index,
                slide_mean_ids=slide_mean_ids,
                slide_mean_meta=slide_mean_meta,
            )
            logger.info("ChatManager initialized for RAG-based chat")
        except Exception as e:
            chat_manager = None
            logger.warning(f"Failed to initialize ChatManager: {e}")

    async def shutdown():
        """Clean up resources on shutdown."""
        try:
            await db.close_pool()
        except Exception:
            pass

    @asynccontextmanager
    async def lifespan_context(_app: FastAPI):
        await load_models()
        try:
            yield
        finally:
            await shutdown()

    app.router.lifespan_context = lifespan_context

    def _build_health_payload() -> dict[str, Any]:
        """Build health response from live runtime state."""
        slides_by_project = {}
        if project_registry and hasattr(project_registry, "list_projects"):
            projects_dict = project_registry.list_projects()  # Dict[str, ProjectConfig]
            for pid in projects_dict:
                try:
                    proj_cfg = projects_dict[pid]
                    if proj_cfg and hasattr(proj_cfg, "dataset") and proj_cfg.dataset:
                        proj_emb_dir = Path(proj_cfg.dataset.embeddings_dir)
                        if not proj_emb_dir.is_absolute():
                            proj_emb_dir = _data_root.parent / proj_emb_dir
                        if proj_emb_dir.exists():
                            slides_by_project[pid] = len(
                                [
                                    f
                                    for f in proj_emb_dir.glob("*.npy")
                                    if not f.name.endswith("_coords.npy")
                                ]
                            )
                except Exception:
                    pass
        total_slides = (
            sum(slides_by_project.values()) if slides_by_project else len(available_slides)
        )
        active_batch_embedding = _active_batch_embed_info()
        return {
            "status": "healthy",
            "version": "0.1.0",
            "model_loaded": classifier is not None,
            "cuda_available": bool(_CUDA_AVAILABLE_AT_STARTUP),
            "slides_available": total_slides,
            "slides_by_project": slides_by_project if slides_by_project else None,
            "db_available": db_available,
            "uptime": time.time() - _STARTUP_TIME,
            "active_batch_embedding": active_batch_embedding,
        }

    app.include_router(
        create_core_router(
            health_provider=_build_health_payload,
            perf_enabled=perf_enabled,
            perf_summary_enabled=perf_summary_enabled,
            perf_max_samples=perf_max_samples,
            perf_route_limit=perf_route_limit,
            perf_tracker=perf_tracker,
        )
    )
    app.include_router(
        create_task_status_router(
            embedding_task_manager=task_manager,
            batch_task_manager=batch_task_manager,
            report_task_manager=report_task_manager,
            batch_embed_manager=batch_embed_manager,
            resolve_project_embeddings_dir=_resolve_project_embeddings_dir,
            log_audit_event=log_audit_event,
        )
    )
    app.include_router(
        create_feature_status_router(
            embedder_provider=lambda: embedder,
            medsiglip_embedder_provider=lambda: medsiglip_embedder,
            slide_siglip_embeddings_provider=lambda: slide_siglip_embeddings,
            evidence_generator_provider=lambda: evidence_gen,
            available_slides_provider=lambda: available_slides,
            check_cuda=_check_cuda,
            require_project=_require_project,
            project_slide_ids=_project_slide_ids,
        )
    )
    app.include_router(
        create_visual_search_router(
            require_project=_require_project,
            project_slide_ids=_project_slide_ids,
            resolve_project_embeddings_dir=_resolve_project_embeddings_dir,
            resolve_embedding_path=_resolve_embedding_path,
            evidence_generator_provider=lambda: evidence_gen,
            slide_labels_provider=lambda: slide_labels,
            log_audit_event=log_audit_event,
        )
    )
    app.include_router(
        create_uncertainty_router(
            classifier_provider=lambda: classifier,
            embeddings_dir_provider=lambda: embeddings_dir,
            log_audit_event=log_audit_event,
        )
    )
    app.include_router(
        create_similar_router(
            resolve_project_embeddings_dir=_resolve_project_embeddings_dir,
            project_slide_ids=_project_slide_ids,
            slide_mean_index_provider=lambda: slide_mean_index,
            slide_mean_ids_provider=lambda: slide_mean_ids,
            slide_mean_meta_provider=lambda: slide_mean_meta,
            slide_labels_provider=lambda: slide_labels,
        )
    )
    app.include_router(create_embedding_router(embedder_provider=lambda: embedder))
    app.include_router(create_slide_qc_router(embeddings_dir_provider=lambda: embeddings_dir))

    list_slides = build_slide_lister(
        db_module=db,
        db_available_provider=lambda: db_available,
        logger=logger,
        require_project=_require_project,
        has_wsi_file=has_wsi_file,
        resolve_dataset_path=_resolve_dataset_path,
        project_labels_path=_project_labels_path,
        data_root_provider=lambda: _data_root,
        embeddings_dir_provider=lambda: embeddings_dir,
        slides_dir_provider=lambda: slides_dir,
        available_slides_provider=lambda: available_slides,
        project_slide_ids=_project_slide_ids,
        filter_project_candidate_slide_ids=_filter_project_candidate_slide_ids,
        resolve_slide_path=resolve_slide_path,
    )
    app.include_router(create_slide_list_router(list_slides=list_slides))

    app.include_router(create_slide_search_router(list_slides=list_slides))

    app.include_router(
        create_database_router(
            db_module=db,
            db_available_provider=lambda: db_available,
            data_root_provider=lambda: _data_root,
            embeddings_dir_provider=lambda: embeddings_dir,
        )
    )

    # Tissue type constants for classification
    TISSUE_TYPES = ["tumor", "stroma", "necrosis", "inflammatory", "normal", "artifact"]
    TISSUE_DESCRIPTIONS = {
        "tumor": "Region appears to contain tumor tissue with atypical cellular morphology",
        "stroma": "Region shows stromal tissue with fibrous connective tissue patterns",
        "necrosis": "Region displays necrotic tissue with cell death indicators",
        "inflammatory": "Region contains inflammatory infiltrate with immune cell presence",
        "normal": "Region appears to contain normal tissue architecture",
        "artifact": "Region may contain processing artifacts or technical issues",
    }

    def classify_tissue_type(x: int, y: int, patch_index: int | None = None) -> dict:
        """Classify tissue type of a region. Mock implementation - deterministic based on coordinates."""
        # Use patch_index if provided for more consistent results, otherwise use coordinates
        if patch_index is not None:
            idx = patch_index % len(TISSUE_TYPES)
        else:
            idx = (x + y) % len(TISSUE_TYPES)
        tissue_type = TISSUE_TYPES[idx]
        # Generate confidence based on hash for variety (0.70 - 0.95 range)
        confidence = 0.70 + ((x * 7 + y * 13) % 26) / 100.0
        return {
            "tissue_type": tissue_type,
            "confidence": round(confidence, 2),
            "description": TISSUE_DESCRIPTIONS[tissue_type],
        }

    app.include_router(create_tissue_router(classify_tissue_type=classify_tissue_type))
    app.include_router(
        create_analysis_router(
            classifier_provider=lambda: classifier,
            evidence_generator_provider=lambda: evidence_gen,
            resolve_project_embeddings_dir=_resolve_project_embeddings_dir,
            classifier_threshold=_classifier_threshold,
            resolve_project_label_pair=_resolve_project_label_pair,
            classify_tissue_type=classify_tissue_type,
            project_slide_ids=_project_slide_ids,
            slide_mean_index_provider=lambda: slide_mean_index,
            slide_mean_ids_provider=lambda: slide_mean_ids,
            slide_mean_meta_provider=lambda: slide_mean_meta,
            slide_labels_provider=lambda: slide_labels,
            similar_case_slide_id=_similar_case_slide_id,
            save_analysis_to_history=save_analysis_to_history,
        )
    )
    app.include_router(
        create_batch_analysis_router(
            classifier_provider=lambda: classifier,
            resolve_project_embeddings_dir=_resolve_project_embeddings_dir,
            resolve_project_label_pair=_resolve_project_label_pair,
            classifier_threshold=_classifier_threshold,
            log_audit_event=log_audit_event,
        )
    )

    app.include_router(
        create_async_batch_analysis_router(
            classifier_provider=lambda: classifier,
            multi_model_inference_provider=lambda: multi_model_inference,
            model_configs_provider=lambda: MODEL_CONFIGS,
            batch_task_manager=batch_task_manager,
            resolve_project_embeddings_dir=_resolve_project_embeddings_dir,
            resolve_embedding_path=_resolve_embedding_path,
            resolve_project_model_ids=_resolve_project_model_ids,
            resolve_project_label_pair=_resolve_project_label_pair,
            classifier_threshold=_classifier_threshold,
            log_audit_event=log_audit_event,
            logger=logger,
        )
    )

    app.include_router(
        create_report_router(
            classifier_provider=lambda: classifier,
            reporter_provider=lambda: reporter,
            decision_support_provider=lambda: decision_support,
            evidence_generator_provider=lambda: evidence_gen,
            report_task_manager=report_task_manager,
            require_project=_require_project,
            project_labels_path=_project_labels_path,
            data_root_provider=lambda: _data_root,
            embeddings_dir_provider=lambda: embeddings_dir,
            resolve_project_embeddings_dir=_resolve_project_embeddings_dir,
            resolve_embedding_path=_resolve_embedding_path,
            resolve_project_label_pair=_resolve_project_label_pair,
            classifier_threshold=_classifier_threshold,
            project_slide_ids=_project_slide_ids,
            similar_case_slide_id=_similar_case_slide_id,
            logger=logger,
        )
    )

    app.include_router(
        create_pdf_router(
            pdf_export_available_provider=lambda: PDF_EXPORT_AVAILABLE,
            generate_report_pdf_provider=lambda: generate_report_pdf,
            generate_pdf_report_provider=lambda: generate_pdf_report,
            embeddings_dir_provider=lambda: embeddings_dir,
            classifier_provider=lambda: classifier,
            evidence_generator_provider=lambda: evidence_gen,
            logger=logger,
            log_audit_event=log_audit_event,
        )
    )

    app.include_router(
        create_heatmap_router(
            classifier_provider=lambda: classifier,
            evidence_generator_provider=lambda: evidence_gen,
            resolve_project_embeddings_dir=_resolve_project_embeddings_dir,
            resolve_embedding_path=_resolve_embedding_path,
            resolve_slide_dims_cached=_resolve_slide_dims_cached,
            path_signature=_path_signature,
            log_timing=_log_timing,
            logger=logger,
        )
    )

    # WSI / DZI Tile Serving
    # Cache for OpenSlide objects and DeepZoom generators
    logger.info(f"Slides directory: {slides_dir}")
    wsi_service = WsiService(resolve_slide_path=resolve_slide_path, logger=logger)

    def get_slide_and_dz(slide_id: str, project_id: str | None = None):
        """Route adapter for the shared WSI service."""
        return wsi_service.get_slide_and_dz(slide_id, project_id=project_id)

    app.include_router(
        create_semantic_search_router(
            medsiglip_embedder_provider=lambda: medsiglip_embedder,
            slide_siglip_embeddings_provider=lambda: slide_siglip_embeddings,
            classifier_provider=lambda: classifier,
            resolve_project_embeddings_dir=_resolve_project_embeddings_dir,
            resolve_semantic_siglip_plan=_resolve_semantic_siglip_plan,
            semantic_allow_on_the_fly_siglip_provider=lambda: semantic_allow_on_the_fly_siglip,
            semantic_on_the_fly_max_patches_provider=lambda: semantic_on_the_fly_max_patches,
            get_slide_and_dz=get_slide_and_dz,
            resolve_slide_path=resolve_slide_path,
            normalize_coords_to_level0=_normalize_coords_to_level0,
            infer_patch_size_from_coords=_infer_patch_size_from_coords,
            classify_tissue_type=classify_tissue_type,
        )
    )

    # Thumbnail cache directory
    thumbnail_cache_dir = embeddings_dir / "thumbnail_cache"
    thumbnail_cache_dir.mkdir(parents=True, exist_ok=True)

    app.include_router(
        create_slide_viewer_router(
            require_project=_require_project,
            get_slide_and_dz=get_slide_and_dz,
            resolve_project_embeddings_dir=_resolve_project_embeddings_dir,
            resolve_slide_path=resolve_slide_path,
            infer_patch_size_from_coords=_infer_patch_size_from_coords,
            normalize_coords_to_level0=_normalize_coords_to_level0,
            embeddings_dir_provider=lambda: embeddings_dir,
            thumbnail_cache_dir=thumbnail_cache_dir,
        )
    )

    app.include_router(
        create_slide_status_router(
            db_module=db,
            logger=logger,
            require_project=_require_project,
            project_slide_ids=_project_slide_ids,
            resolve_project_model_ids=_resolve_project_model_ids,
        )
    )

    app.include_router(
        create_annotation_router(
            db_module=db,
            logger=logger,
            log_audit_event=log_audit_event,
        )
    )

    app.include_router(
        create_model_router(
            multi_model_provider=lambda: multi_model_inference,
            resolve_project_model_ids=_resolve_project_model_ids,
            log_timing=_log_timing,
        )
    )

    # ====== Multi-Model Analysis Endpoints ======

    app.include_router(
        create_embedding_generation_router(
            task_manager=task_manager,
            batch_embed_manager=batch_embed_manager,
            resolve_project_embeddings_dir=_resolve_project_embeddings_dir,
            resolve_slide_path=resolve_slide_path,
            require_project=_require_project,
            batch_embed_inventory_slide_ids=_batch_embed_inventory_slide_ids,
            logger=logger,
        )
    )

    app.include_router(
        create_multi_model_analysis_router(
            multi_model_provider=lambda: multi_model_inference,
            model_configs_provider=lambda: MODEL_CONFIGS,
            resolve_project_model_ids=_resolve_project_model_ids,
            active_batch_embed_info=_active_batch_embed_info,
            resolve_project_embeddings_dir=_resolve_project_embeddings_dir,
            resolve_embedding_path=_resolve_embedding_path,
            score_to_prediction=score_to_prediction,
            db_module=db,
            log_audit_event=log_audit_event,
            project_root=_project_root,
            logger=logger,
        )
    )

    app.include_router(
        create_model_heatmap_router(
            multi_model_provider=lambda: multi_model_inference,
            evidence_generator_provider=lambda: evidence_gen,
            model_configs_provider=lambda: MODEL_CONFIGS,
            resolve_project_model_scope_cached=_resolve_project_model_scope_cached,
            require_model_allowed_for_scope=require_model_allowed_for_scope,
            resolve_project_model_ids=_resolve_project_model_ids,
            resolve_project_embeddings_dir=_resolve_project_embeddings_dir,
            resolve_embedding_path=_resolve_embedding_path,
            resolve_slide_dims_cached=_resolve_slide_dims_cached,
            model_checkpoint_signatures=model_checkpoint_signatures,
            project_root=_project_root,
            log_timing=_log_timing,
            logger=logger,
        )
    )

    # Register slide metadata API
    metadata_path = _data_root / "slide_metadata.json"
    if not metadata_path.exists():
        metadata_path = embeddings_dir.parent / "slide_metadata.json"
    metadata_manager = SlideMetadataManager(metadata_path)

    def get_available_slide_ids():
        return list(available_slides.keys())

    metadata_router = create_metadata_router(metadata_manager, get_available_slide_ids)
    app.include_router(metadata_router)
    app.include_router(create_group_router(metadata_manager))

    # Project system routes (config-driven multi-cancer support)
    app.include_router(project_router)

    # Agent workflow routes (workflow instance is initialized during startup, after models are loaded)
    if AGENT_AVAILABLE:
        try:
            app.include_router(agent_router)
            logger.info("Agent workflow routes registered")
        except Exception as e:
            logger.warning(f"Failed to register agent workflow routes: {e}")
    else:
        logger.warning("Agent workflow not available - skipping registration")

    app.include_router(create_chat_router(lambda: chat_manager))
    logger.info("Chat API endpoints registered")

    app.include_router(
        create_patch_analysis_router(
            embeddings_dir_provider=lambda: embeddings_dir,
            log_audit_event=log_audit_event,
        )
    )

    return app


# Default app instance — prefer level0 embeddings when available
import os as _os_app

_emb_dir = Path(_os_app.environ.get("EMBEDDINGS_DIR", "data/embeddings"))
# Auto-detect level0 subdirectory (full-resolution re-embeddings)
if not _os_app.environ.get("EMBEDDINGS_DIR") and (_emb_dir / "level0").is_dir():
    _emb_dir = _emb_dir / "level0"
app = create_app(embeddings_dir=_emb_dir)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
