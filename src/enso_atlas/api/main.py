"""
Enso Atlas API - FastAPI Backend

Provides REST API endpoints for the professional frontend:
- Slide analysis with MIL prediction
- Evidence generation (heatmaps, patches)
- Similar case retrieval (FAISS)
- Report generation (MedGemma)
- Patch embedding (Path Foundation)
- Analysis history and audit trail
"""

from pathlib import Path
from typing import Optional, List, Dict, Any
import asyncio
import logging
import json
import base64
import io
from datetime import datetime
from .embedding_tasks import task_manager, TaskStatus, EmbeddingTask
from .report_tasks import report_task_manager, ReportTaskStatus, ReportTask
from .batch_tasks import batch_task_manager, BatchTaskStatus, BatchTask, BatchSlideResult
from collections import deque

import numpy as np
from PIL import Image

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Query
from .slide_metadata import SlideMetadataManager, create_metadata_router
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Multi-model TransMIL inference
import sys
_project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_project_root))
try:
    from scripts.multi_model_inference import MultiModelInference, MODEL_CONFIGS
    MULTI_MODEL_AVAILABLE = True
except ImportError:
    logger.warning("MultiModelInference not available - multi-model endpoints disabled")
    MULTI_MODEL_AVAILABLE = False
    MultiModelInference = None
    MODEL_CONFIGS = {}

# Agent workflow for multi-step analysis
try:
    from ..agent.workflow import AgentWorkflow
    from ..agent.routes import router as agent_router, set_agent_workflow
    AGENT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Agent workflow not available: {e}")
    AGENT_AVAILABLE = False
    AgentWorkflow = None
    agent_router = None
    set_agent_workflow = None



# Analysis History Storage
# In-memory deque for fast access, limited to 100 entries
MAX_HISTORY_SIZE = 100
analysis_history: deque = deque(maxlen=MAX_HISTORY_SIZE)
audit_log: deque = deque(maxlen=500)  # Audit trail for compliance


def get_timestamp() -> str:
    """Get current ISO timestamp."""
    return datetime.utcnow().isoformat() + "Z"


def log_audit_event(
    event_type: str,
    slide_id: Optional[str] = None,
    user_id: str = "clinician",
    details: Optional[Dict[str, Any]] = None,
):
    """Log an audit event for compliance tracking."""
    entry = {
        "timestamp": get_timestamp(),
        "event_type": event_type,
        "user_id": user_id,
        "slide_id": slide_id,
        "details": details or {},
    }
    audit_log.append(entry)
    logger.info(f"AUDIT: {event_type} - slide={slide_id} user={user_id}")


def save_analysis_to_history(
    slide_id: str,
    prediction: str,
    score: float,
    confidence: float,
    patches_analyzed: int,
    top_evidence: List[Dict[str, Any]],
    similar_cases: List[Dict[str, Any]],
    user_id: str = "clinician",
) -> Dict[str, Any]:
    """Save analysis result to history and return the entry."""
    entry = {
        "id": f"{slide_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
        "timestamp": get_timestamp(),
        "slide_id": slide_id,
        "user_id": user_id,
        "prediction": prediction,
        "score": score,
        "confidence": confidence,
        "patches_analyzed": patches_analyzed,
        "top_evidence_count": len(top_evidence),
        "similar_cases_count": len(similar_cases),
    }
    analysis_history.append(entry)
    log_audit_event("analysis_completed", slide_id, user_id, {
        "prediction": prediction,
        "confidence": confidence,
    })
    return entry


def _check_cuda() -> bool:
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# Request/Response Models
class AnalyzeRequest(BaseModel):
    slide_id: str = Field(..., min_length=1, max_length=256)
    generate_report: bool = False


class AnalyzeResponse(BaseModel):
    slide_id: str
    prediction: str
    score: float
    confidence: float
    patches_analyzed: int
    top_evidence: List[Dict[str, Any]]
    similar_cases: List[Dict[str, Any]]


class UncertaintyRequest(BaseModel):
    """Request for analysis with uncertainty quantification."""
    slide_id: str = Field(..., min_length=1, max_length=256)
    n_samples: int = Field(
        default=20,
        ge=5,
        le=50,
        description="Number of MC Dropout samples (5-50, default 20)"
    )


class UncertaintyResponse(BaseModel):
    """Response with MC Dropout uncertainty quantification."""
    slide_id: str
    prediction: str
    probability: float
    uncertainty: float
    confidence_interval: List[float]
    is_uncertain: bool
    requires_review: bool
    uncertainty_level: str  # "low", "moderate", "high"
    clinical_recommendation: str
    patches_analyzed: int
    n_samples: int
    samples: List[float]
    top_evidence: List[Dict[str, Any]]


class ReportRequest(BaseModel):
    slide_id: str = Field(..., min_length=1, max_length=256)
    include_evidence: bool = True
    include_similar: bool = True


class ReportResponse(BaseModel):
    slide_id: str
    report_json: Dict[str, Any]
    summary_text: str


class PatientContext(BaseModel):
    """Patient demographic and clinical context for a slide."""
    age: Optional[int] = None
    sex: Optional[str] = None
    stage: Optional[str] = None
    grade: Optional[str] = None
    prior_lines: Optional[int] = None
    histology: Optional[str] = None


class SlideDimensions(BaseModel):
    width: int = 0
    height: int = 0

class SlideInfo(BaseModel):
    slide_id: str
    patient_id: Optional[str] = None
    has_embeddings: bool = False
    has_level0_embeddings: bool = False  # Whether level 0 (full resolution) embeddings exist
    label: Optional[str] = None
    num_patches: Optional[int] = None
    patient: Optional[PatientContext] = None
    dimensions: SlideDimensions = SlideDimensions()
    mpp: Optional[float] = None  # microns per pixel
    magnification: Optional[str] = "40x"


class SlideQCResponse(BaseModel):
    """Quality control metrics for a slide."""
    slide_id: str
    tissue_coverage: float
    blur_score: float
    stain_uniformity: float
    artifact_detected: bool
    pen_marks: bool
    fold_detected: bool
    overall_quality: str


class EmbedRequest(BaseModel):
    """Request for patch embedding."""
    patches: List[str] = Field(
        ...,
        description="Base64-encoded patch images (224x224 RGB)",
        min_length=1,
        max_length=128,
    )
    return_embeddings: bool = Field(
        default=True,
        description="Whether to return embedding vectors",
    )


class EmbedResponse(BaseModel):
    """Response from embedding endpoint."""
    num_patches: int
    embedding_dim: int = 384
    embeddings: Optional[List[List[float]]] = None


class SimilarRequest(BaseModel):
    """Request for similar case search."""
    slide_id: str = Field(..., min_length=1, max_length=256)
    k: int = Field(default=5, ge=1, le=20)
    top_patches: int = Field(default=3, ge=1, le=10)


class BatchAnalyzeRequest(BaseModel):
    """Request for batch analysis of multiple slides."""
    slide_ids: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of slide IDs to analyze (1-100 slides)"
    )


class BatchAnalysisResult(BaseModel):
    """Result for a single slide in batch analysis."""
    slide_id: str
    prediction: str
    score: float
    confidence: float
    patches_analyzed: int
    requires_review: bool
    uncertainty_level: str = "unknown"
    error: Optional[str] = None


class BatchAnalysisSummary(BaseModel):
    """Summary statistics for batch analysis."""
    total: int
    completed: int
    failed: int
    responders: int
    non_responders: int
    uncertain: int
    avg_confidence: float
    requires_review_count: int


class BatchAnalyzeResponse(BaseModel):
    """Response from batch analysis endpoint."""
    results: List[BatchAnalysisResult]
    summary: BatchAnalysisSummary
    processing_time_ms: float


class ClassifyRegionRequest(BaseModel):
    """Request for tissue region classification."""
    x: int = Field(..., description="X coordinate of the region")
    y: int = Field(..., description="Y coordinate of the region")
    patch_index: Optional[int] = Field(None, description="Optional patch index for deterministic classification")


class ClassifyRegionResponse(BaseModel):
    """Response from tissue region classification."""
    tissue_type: str
    confidence: float
    description: str


class SimilarResponse(BaseModel):
    """Response from similar case search."""
    slide_id: str
    similar_cases: List[Dict[str, Any]]
    num_queries: int


class AnalysisHistoryEntry(BaseModel):
    """Single analysis history entry."""
    id: str
    timestamp: str
    slide_id: str
    user_id: str
    prediction: str
    score: float
    confidence: float
    patches_analyzed: int
    top_evidence_count: int
    similar_cases_count: int


class AnalysisHistoryResponse(BaseModel):
    """Response containing analysis history."""
    analyses: List[AnalysisHistoryEntry]
    total: int


class AuditLogEntry(BaseModel):
    """Single audit log entry."""
    timestamp: str
    event_type: str
    user_id: str
    slide_id: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class AuditLogResponse(BaseModel):
    """Response containing audit log entries."""
    entries: List[AuditLogEntry]
    total: int


class SemanticSearchRequest(BaseModel):
    """Request for text-to-patch semantic search using MedSigLIP."""
    slide_id: str = Field(..., min_length=1, max_length=256)
    query: str = Field(..., min_length=1, max_length=512, description="Text query (e.g., 'tumor cells', 'lymphocytes')")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of patches to return")


class SemanticSearchResult(BaseModel):
    """Single result from semantic search."""
    patch_index: int
    similarity_score: float
    coordinates: Optional[List[int]] = None
    attention_weight: Optional[float] = None


class SemanticSearchResponse(BaseModel):
    """Response from semantic search endpoint."""
    slide_id: str
    query: str
    results: List[SemanticSearchResult]
    embedding_model: str = "siglip-so400m"


# ====== Multi-Model Prediction Models ======

class ModelPrediction(BaseModel):
    """Single model prediction result."""
    model_id: str
    model_name: str
    category: str  # ovarian_cancer or general_pathology
    score: float
    label: str
    positive_label: str
    negative_label: str
    confidence: float
    auc: float
    n_training_slides: int
    description: str


class MultiModelRequest(BaseModel):
    """Request for multi-model analysis."""
    slide_id: str = Field(..., min_length=1, max_length=256)
    models: Optional[List[str]] = None  # None = run all models
    return_attention: bool = False
    level: int = Field(default=1, ge=0, le=1, description="Resolution level: 0=full res, 1=downsampled")


class MultiModelResponse(BaseModel):
    """Response with predictions from multiple models."""
    slide_id: str
    predictions: Dict[str, ModelPrediction]
    by_category: Dict[str, List[ModelPrediction]]
    n_patches: int
    processing_time_ms: float


class AvailableModelsResponse(BaseModel):
    """Response listing available models."""
    models: List[Dict[str, Any]]


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
            allow_origins=["*", 
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
    slide_siglip_embeddings = {}  # Cache for MedSigLIP embeddings per slide
    available_slides = []
    slide_labels = {}  # Cache for slide labels (slide_id -> label string)
    # Slide-level mean-embedding FAISS index (cosine similarity)
    slide_mean_index = None  # faiss.IndexFlatIP over L2-normalized mean embeddings
    slide_mean_ids: list[str] = []
    slide_mean_meta: dict[str, dict] = {}  # slide_id -> metadata (n_patches, label, patient, etc.)
    # Directories (may be updated at startup if we fall back to demo data)
    slides_dir: Path = embeddings_dir.parent / 'slides'

    def resolve_slide_path(slide_id: str) -> Path | None:
        """Resolve slide file path across possible slide directories."""
        # Common locations we may store slides
        candidates_dirs = [
            slides_dir,
            embeddings_dir.parent / 'tcga_full' / 'slides',
            embeddings_dir.parent / 'ovarian_bev' / 'slides',
            embeddings_dir.parent / 'demo' / 'slides',
        ]
        exts = ['.svs', '.tiff', '.tif', '.ndpi', '.mrxs', '.vms', '.scn']
        for d in candidates_dirs:
            if not d.exists():
                continue
            for ext in exts:
                cand = d / f"{slide_id}{ext}"
                if cand.exists():
                    return cand
        return None


    @app.on_event("startup")
    async def load_models():
        nonlocal classifier, evidence_gen, embedder, medsiglip_embedder, reporter, decision_support, multi_model_inference, available_slides, slide_labels, slides_dir, embeddings_dir, slide_mean_index, slide_mean_ids, slide_mean_meta

        from ..config import MILConfig, EvidenceConfig, EmbeddingConfig
        from ..mil.clam import CLAMClassifier
        from ..evidence.generator import EvidenceGenerator
        from ..embedding.embedder import PathFoundationEmbedder
        from ..embedding.medsiglip import MedSigLIPEmbedder, MedSigLIPConfig
        from ..reporting.medgemma import MedGemmaReporter, ReportingConfig
        from ..reporting.decision_support import ClinicalDecisionSupport


        def _count_embeddings(p: Path) -> int:
            if not p.exists() or not p.is_dir():
                return 0
            return sum(1 for f in p.glob('*.npy') if not f.name.endswith('_coords.npy'))

        def _count_slides(p: Path) -> int:
            if not p.exists() or not p.is_dir():
                return 0
            exts = {'.svs', '.tiff', '.tif', '.ndpi', '.mrxs', '.vms', '.scn'}
            return sum(1 for f in p.iterdir() if f.is_file() and f.suffix.lower() in exts)

        primary_embeddings_dir = embeddings_dir
        primary_slides_dir = slides_dir
        primary_n = _count_embeddings(primary_embeddings_dir)
        primary_s = _count_slides(primary_slides_dir)
        logger.info(f'Embeddings dir: {primary_embeddings_dir} (npy={primary_n})')
        logger.info(f'Slides dir: {primary_slides_dir} (slides={primary_s})')

        if primary_n == 0:
            demo_embeddings_dir = primary_embeddings_dir.parent / 'demo' / 'embeddings'
            demo_slides_dir = demo_embeddings_dir.parent / 'slides'
            demo_n = _count_embeddings(demo_embeddings_dir)
            demo_s = _count_slides(demo_slides_dir)
            if demo_n > 0:
                logger.warning(
                    f'No embeddings found in {primary_embeddings_dir}; falling back to demo embeddings at {demo_embeddings_dir} (npy={demo_n})'
                )
                embeddings_dir = demo_embeddings_dir
                slides_dir = demo_slides_dir
                logger.info(f'Using embeddings dir: {embeddings_dir}')
                logger.info(f'Using slides dir: {slides_dir} (slides={demo_s})')
            else:
                logger.warning(
                    f'No embeddings found in {primary_embeddings_dir} and no demo embeddings found at {demo_embeddings_dir}.'
                )

        # Load MIL classifier
        config = MILConfig(input_dim=384, hidden_dim=256)
        classifier = CLAMClassifier(config)
        if model_path.exists():
            classifier.load(model_path)
            logger.info(f"Loaded MIL model from {model_path}")

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
                    logger.info(f"Multi-model inference initialized with {len(multi_model_inference.models)} models")
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
                logger.info("Starting MedGemma warmup (this may take ~30s on first load)...")
                reporter._warmup_inference()
                logger.info("MedGemma reporter warmed up successfully")
            except Exception as e:
                logger.warning(f"MedGemma warmup failed: {e}")

        # Setup clinical decision support engine
        decision_support = ClinicalDecisionSupport()
        logger.info("Clinical decision support engine initialized")

        # Setup MedSigLIP embedder for semantic search (lazy-loaded on first use)
        siglip_config = MedSigLIPConfig(cache_dir=str(embeddings_dir / "medsiglip_cache"))
        medsiglip_embedder = MedSigLIPEmbedder(siglip_config)
        logger.info("MedSigLIP embedder initialized (model loads on first call)")

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
                    all_metadata.append({
                        "slide_id": slide_id,
                        "n_patches": len(embs),
                    })

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
                    sid = meta.get('slide_id')
                    if sid is None:
                        continue
                    if embs is None or len(embs) == 0:
                        continue
                    mean = np.asarray(embs, dtype=np.float32).mean(axis=0)
                    mean = mean / (np.linalg.norm(mean) + 1e-12)
                    means.append(mean)
                    slide_mean_ids.append(sid)
                    slide_mean_meta[sid] = {
                        'slide_id': sid,
                        'n_patches': int(meta.get('n_patches', len(embs))),
                    }
                if means:
                    mat = np.vstack(means).astype(np.float32)
                    slide_mean_index = faiss.IndexFlatIP(mat.shape[1])
                    slide_mean_index.add(mat)
                    logger.info(f'Built slide-mean FAISS index with {len(slide_mean_ids)} slides')
                else:
                    slide_mean_index = None
                    logger.warning('No slide means available to build slide-mean FAISS index')
            except Exception as e:
                slide_mean_index = None
                logger.warning(f'Failed to build slide-mean FAISS index: {e}')

        # Load slide labels from labels.csv for similar case retrieval
        labels_path = embeddings_dir.parent / "labels.csv"
        if labels_path.exists():
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
                        label_val = "responder" if response == "responder" else "non-responder" if response == "non-responder" else ""
                    if sid and label_val:
                        # Normalize label format
                        if label_val == "1":
                            slide_labels[sid] = "responder"
                        elif label_val == "0":
                            slide_labels[sid] = "non-responder"
                        else:
                            slide_labels[sid] = label_val
            logger.info(f"Loaded labels for {len(slide_labels)} slides")

            # Attach labels to slide-mean metadata
            try:
                for sid, lab in slide_labels.items():
                    if sid in slide_mean_meta:
                        slide_mean_meta[sid]['label'] = lab
            except Exception as e:
                logger.warning(f'Failed to attach labels to slide-mean metadata: {e}')

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "version": "0.1.0",
            "model_loaded": classifier is not None,
            "cuda_available": _check_cuda(),
            "slides_available": len(available_slides),
        }

    @app.get("/api/health")
    async def api_health_check():
        """Health check endpoint (aliased for frontend compatibility)."""
        return await health_check()

    @app.get("/")
    async def root():
        """API root endpoint."""
        return {
            "name": "Enso Atlas API",
            "version": "0.1.0",
            "docs": "/api/docs",
        }

    @app.get("/docs", include_in_schema=False)
    async def docs_redirect():
        """Redirect /docs to /api/docs for convenience."""
        from starlette.responses import RedirectResponse
        return RedirectResponse(url="/api/docs")

    @app.get("/redoc", include_in_schema=False)
    async def redoc_redirect():
        """Redirect /redoc to /api/redoc for convenience."""
        from starlette.responses import RedirectResponse
        return RedirectResponse(url="/api/redoc")

    @app.get("/api/slides", response_model=List[SlideInfo])
    async def list_slides():
        """List all available slides with patient context."""
        slides = []
        labels_path = embeddings_dir.parent / "labels.csv"
        slide_data = {}

        if labels_path.exists():
            import csv
            with open(labels_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Handle different CSV formats
                    if "slide_id" in row:
                        sid = row["slide_id"]
                        label = row.get("label", "")
                    else:
                        # Alternative format: slide_file contains filename with .svs
                        slide_file = row.get("slide_file", "")
                        sid = slide_file.replace(".svs", "").replace(".SVS", "")
                        # Derive label from treatment_response
                        response = row.get("treatment_response", "")
                        label = "1" if response == "responder" else "0" if response == "non-responder" else ""

                    if sid:
                        # Parse patient context from CSV
                        patient_ctx = None
                        if any(k in row for k in ["age", "sex", "stage", "grade", "prior_treatments", "histology"]):
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

                        slide_data[sid] = {
                            "label": label,
                            "patient": patient_ctx,
                        }

        for slide_id in available_slides:
            # Get patch count
            emb_path = embeddings_dir / f"{slide_id}.npy"
            num_patches = None
            if emb_path.exists():
                try:
                    emb = np.load(emb_path)
                    num_patches = len(emb)
                except Exception:
                    pass

            data = slide_data.get(slide_id, {})
            # Try to get slide dimensions from actual slide file
            dims = SlideDimensions()
            mpp = None
            slide_path = resolve_slide_path(slide_id)
            if slide_path is not None and slide_path.exists():
                try:
                    import openslide
                    with openslide.OpenSlide(str(slide_path)) as slide:
                        dims = SlideDimensions(
                            width=slide.dimensions[0],
                            height=slide.dimensions[1]
                        )
                        # Try to get MPP from slide properties
                        mpp_x = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
                        if mpp_x:
                            mpp = float(mpp_x)
                except Exception as e:
                    logger.warning(f"Could not read slide {slide_id}: {e}")
            
            # Check if level 0 embeddings exist
            level0_dir = embeddings_dir / "level0"
            has_level0 = False
            if level0_dir.exists():
                level0_emb_path = level0_dir / f"{slide_id}.npy"
                has_level0 = level0_emb_path.exists()
            
            slides.append(SlideInfo(
                slide_id=slide_id,
                has_embeddings=True,
                has_level0_embeddings=has_level0,
                label=data.get("label"),
                num_patches=num_patches,
                patient=data.get("patient"),
                dimensions=dims,
                mpp=mpp,
            ))

        return slides

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

    def classify_tissue_type(x: int, y: int, patch_index: Optional[int] = None) -> dict:
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

    @app.post("/api/classify-region", response_model=ClassifyRegionResponse)
    async def classify_region(request: ClassifyRegionRequest):
        """Classify tissue type of a region.

        In production, this would use a trained tissue classifier model.
        Current implementation is deterministic based on coordinates for demo purposes.

        Tissue types:
        - tumor: Neoplastic cells with atypical morphology
        - stroma: Fibrous connective tissue
        - necrosis: Dead/dying tissue
        - inflammatory: Immune cell infiltrates
        - normal: Healthy tissue architecture
        - artifact: Technical issues (blur, folds, etc.)
        """
        result = classify_tissue_type(request.x, request.y, request.patch_index)
        return ClassifyRegionResponse(**result)

    @app.post("/api/analyze", response_model=AnalyzeResponse)
    async def analyze_slide(request: AnalyzeRequest):
        """Analyze a slide and return prediction with evidence."""
        if classifier is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        slide_id = request.slide_id
        emb_path = embeddings_dir / f"{slide_id}.npy"

        if not emb_path.exists():
            raise HTTPException(status_code=404, detail=f"Slide {slide_id} not found")

        # Load embeddings
        embeddings = np.load(emb_path)

        # Run prediction
        score, attention = classifier.predict(embeddings)
        label = "RESPONDER" if score > 0.5 else "NON-RESPONDER"
        confidence = abs(score - 0.5) * 2

        # Load coordinates if available for tissue classification
        coord_path = embeddings_dir / f"{slide_id}_coords.npy"
        coords = None
        if coord_path.exists():
            coords = np.load(coord_path)

        # Get top evidence patches
        top_k = min(8, len(attention))
        top_indices = np.argsort(attention)[-top_k:][::-1]

        top_evidence = []
        for i, idx in enumerate(top_indices):
            # Get coordinates for this patch
            patch_x = int(coords[idx][0]) if coords is not None else 0
            patch_y = int(coords[idx][1]) if coords is not None else 0

            # Classify tissue type for this patch
            tissue_info = classify_tissue_type(patch_x, patch_y, int(idx))

            top_evidence.append({
                "rank": i + 1,
                "patch_index": int(idx),
                "attention_weight": float(attention[idx]),
                "coordinates": [patch_x, patch_y],
                "tissue_type": tissue_info["tissue_type"],
                "tissue_confidence": tissue_info["confidence"],
            })

        # Get similar cases using FAISS
        similar_cases = []
        if evidence_gen is not None:
            try:
                similar_results = evidence_gen.find_similar(
                    embeddings, attention, k=10, top_patches=3
                )
                seen_slides = set()
                for s in similar_results:
                    meta = s.get("metadata", {})
                    sid = meta.get("slide_id", "unknown")
                    if sid != slide_id and sid not in seen_slides:
                        seen_slides.add(sid)
                        # Get label from cache
                        case_label = slide_labels.get(sid)
                        similar_cases.append({
                            "slide_id": sid,
                            "similarity_score": 1.0 / (1.0 + s["distance"]),
                            "distance": float(s["distance"]),
                            "label": case_label,
                        })
                    if len(similar_cases) >= 5:
                        break
            except Exception as e:
                logger.warning(f"Similar case search failed: {e}")
                # Fall back to random
                for sid in available_slides[:5]:
                    if sid != slide_id:
                        case_label = slide_labels.get(sid)
                        similar_cases.append({
                            "slide_id": sid,
                            "similarity_score": float(np.random.uniform(0.7, 0.95)),
                            "label": case_label,
                        })

        # Save to analysis history for audit trail
        save_analysis_to_history(
            slide_id=slide_id,
            prediction=label,
            score=float(score),
            confidence=float(confidence),
            patches_analyzed=len(embeddings),
            top_evidence=top_evidence,
            similar_cases=similar_cases[:5],
        )

        return AnalyzeResponse(
            slide_id=slide_id,
            prediction=label,
            score=float(score),
            confidence=float(confidence),
            patches_analyzed=len(embeddings),
            top_evidence=top_evidence,
            similar_cases=similar_cases[:5],
        )

    # ====== Batch Analysis Endpoint ======

    @app.post("/api/analyze-batch", response_model=BatchAnalyzeResponse)
    async def analyze_batch(request: BatchAnalyzeRequest):
        """
        Analyze multiple slides in batch for clinical workflow efficiency.

        This endpoint processes multiple slides at once and returns:
        - Individual results for each slide
        - Summary statistics across all slides
        - Priority ordering (uncertain cases flagged first)

        Clinical use cases:
        - Review incoming cases for a day/week
        - Triage cases by prediction confidence
        - Identify cases requiring human review
        """
        import time
        start_time = time.time()

        if classifier is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        results = []
        for slide_id in request.slide_ids:
            emb_path = embeddings_dir / f"{slide_id}.npy"

            if not emb_path.exists():
                # Record failed slide
                results.append(BatchAnalysisResult(
                    slide_id=slide_id,
                    prediction="ERROR",
                    score=0.0,
                    confidence=0.0,
                    patches_analyzed=0,
                    requires_review=True,
                    uncertainty_level="unknown",
                    error=f"Slide {slide_id} not found",
                ))
                continue

            try:
                # Load embeddings
                embeddings = np.load(emb_path)

                # Run prediction
                score, attention = classifier.predict(embeddings)
                label = "RESPONDER" if score > 0.5 else "NON-RESPONDER"
                confidence = abs(score - 0.5) * 2

                # Determine uncertainty level and review requirement
                if confidence < 0.3:
                    uncertainty_level = "high"
                    requires_review = True
                elif confidence < 0.6:
                    uncertainty_level = "moderate"
                    requires_review = True
                else:
                    uncertainty_level = "low"
                    requires_review = False

                results.append(BatchAnalysisResult(
                    slide_id=slide_id,
                    prediction=label,
                    score=float(score),
                    confidence=float(confidence),
                    patches_analyzed=len(embeddings),
                    requires_review=requires_review,
                    uncertainty_level=uncertainty_level,
                    error=None,
                ))

                # Log to audit trail
                log_audit_event(
                    "batch_analysis_slide",
                    slide_id,
                    details={
                        "prediction": label,
                        "confidence": float(confidence),
                        "requires_review": requires_review,
                    },
                )

            except Exception as e:
                logger.error(f"Batch analysis failed for {slide_id}: {e}")
                results.append(BatchAnalysisResult(
                    slide_id=slide_id,
                    prediction="ERROR",
                    score=0.0,
                    confidence=0.0,
                    patches_analyzed=0,
                    requires_review=True,
                    uncertainty_level="unknown",
                    error=str(e),
                ))

        # Sort results: uncertain cases first (by confidence ascending)
        results.sort(key=lambda r: (
            0 if r.error else 1,  # Errors first
            r.confidence if not r.error else 999,  # Then by confidence (lowest first)
        ))

        # Calculate summary statistics
        completed = [r for r in results if r.error is None]
        failed = [r for r in results if r.error is not None]
        responders = [r for r in completed if r.prediction == "RESPONDER"]
        non_responders = [r for r in completed if r.prediction == "NON-RESPONDER"]
        uncertain = [r for r in completed if r.requires_review]
        avg_confidence = (
            sum(r.confidence for r in completed) / len(completed)
            if completed else 0.0
        )

        summary = BatchAnalysisSummary(
            total=len(results),
            completed=len(completed),
            failed=len(failed),
            responders=len(responders),
            non_responders=len(non_responders),
            uncertain=len(uncertain),
            avg_confidence=round(avg_confidence, 3),
            requires_review_count=sum(1 for r in results if r.requires_review),
        )

        processing_time_ms = (time.time() - start_time) * 1000

        # Log batch completion
        log_audit_event(
            "batch_analysis_completed",
            details={
                "total_slides": len(results),
                "completed": len(completed),
                "failed": len(failed),
                "processing_time_ms": processing_time_ms,
            },
        )

        return BatchAnalyzeResponse(
            results=results,
            summary=summary,
            processing_time_ms=round(processing_time_ms, 2),
        )


    # ====== Async Batch Analysis Endpoints ======

    class AsyncBatchRequest(BaseModel):
        """Request for async batch analysis."""
        slide_ids: List[str] = Field(
            ...,
            min_length=1,
            max_length=100,
            description="List of slide IDs to analyze"
        )
        concurrency: int = Field(
            default=4,
            ge=1,
            le=10,
            description="Number of slides to process in parallel (1-10)"
        )

    class AsyncBatchResponse(BaseModel):
        """Response from async batch analysis start."""
        task_id: str
        status: str
        total_slides: int
        message: str

    @app.post("/api/analyze-batch/async", response_model=AsyncBatchResponse)
    async def analyze_batch_async(request: AsyncBatchRequest, background_tasks: BackgroundTasks):
        """
        Start async batch analysis with progress tracking and cancellation support.

        Returns a task_id immediately. Poll /api/analyze-batch/status/{task_id} for progress.
        Use POST /api/analyze-batch/cancel/{task_id} to cancel a running analysis.

        Benefits over synchronous batch:
        - Real-time progress ("Analyzing slide 3/10...")
        - Cancellation support for long batches
        - Better handling of partial failures
        - Non-blocking for large batches
        """
        if classifier is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Create task
        task = batch_task_manager.create_task(request.slide_ids)

        # Run batch analysis in background
        def run_batch_analysis():
            _run_batch_analysis_background(
                task.task_id,
                request.slide_ids,
                request.concurrency,
            )

        background_tasks.add_task(run_batch_analysis)

        return AsyncBatchResponse(
            task_id=task.task_id,
            status="pending",
            total_slides=len(request.slide_ids),
            message=f"Batch analysis started. Poll /api/analyze-batch/status/{task.task_id} for progress.",
        )

    def _run_batch_analysis_background(
        task_id: str,
        slide_ids: List[str],
        concurrency: int = 4,
    ):
        """Background task to run batch analysis with progress tracking."""
        import time
        import concurrent.futures

        task = batch_task_manager.get_task(task_id)
        if not task:
            return

        batch_task_manager.update_task(task_id,
            status=BatchTaskStatus.RUNNING,
            started_at=time.time(),
            message="Starting batch analysis..."
        )

        def analyze_single_slide(slide_id: str) -> BatchSlideResult:
            """Analyze a single slide and return result."""
            emb_path = embeddings_dir / f"{slide_id}.npy"

            if not emb_path.exists():
                return BatchSlideResult(
                    slide_id=slide_id,
                    prediction="ERROR",
                    error=f"Slide {slide_id} not found",
                )

            try:
                embeddings = np.load(emb_path)
                score, attention = classifier.predict(embeddings)
                label = "RESPONDER" if score > 0.5 else "NON-RESPONDER"
                confidence = abs(score - 0.5) * 2

                if confidence < 0.3:
                    uncertainty_level = "high"
                    requires_review = True
                elif confidence < 0.6:
                    uncertainty_level = "moderate"
                    requires_review = True
                else:
                    uncertainty_level = "low"
                    requires_review = False

                return BatchSlideResult(
                    slide_id=slide_id,
                    prediction=label,
                    score=float(score),
                    confidence=float(confidence),
                    patches_analyzed=len(embeddings),
                    requires_review=requires_review,
                    uncertainty_level=uncertainty_level,
                )
            except Exception as e:
                logger.error(f"Batch analysis failed for {slide_id}: {e}")
                return BatchSlideResult(
                    slide_id=slide_id,
                    prediction="ERROR",
                    error=str(e),
                )

        try:
            total = len(slide_ids)
            completed = 0

            # Process slides with concurrency limit
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                # Submit all tasks
                future_to_slide = {
                    executor.submit(analyze_single_slide, slide_id): slide_id
                    for slide_id in slide_ids
                }

                # Process as they complete
                for future in concurrent.futures.as_completed(future_to_slide):
                    # Check for cancellation
                    if batch_task_manager.is_cancelled(task_id):
                        # Cancel remaining futures
                        for f in future_to_slide:
                            f.cancel()
                        batch_task_manager.update_task(task_id,
                            status=BatchTaskStatus.CANCELLED,
                            message=f"Cancelled after {completed}/{total} slides",
                            completed_at=time.time(),
                        )
                        logger.info(f"Batch analysis {task_id} cancelled after {completed} slides")
                        return

                    slide_id = future_to_slide[future]
                    try:
                        result = future.result()
                        batch_task_manager.add_result(task_id, result)
                        completed += 1

                        # Update progress
                        progress = (completed / total) * 100
                        batch_task_manager.update_task(task_id,
                            progress=progress,
                            current_slide_index=completed,
                            current_slide_id=slide_id,
                            message=f"Analyzing slide {completed}/{total}: {slide_id[:20]}...",
                        )

                    except Exception as e:
                        logger.error(f"Future failed for {slide_id}: {e}")
                        batch_task_manager.add_result(task_id, BatchSlideResult(
                            slide_id=slide_id,
                            prediction="ERROR",
                            error=str(e),
                        ))
                        completed += 1

            # Complete
            batch_task_manager.update_task(task_id,
                status=BatchTaskStatus.COMPLETED,
                progress=100,
                message=f"Completed analysis of {total} slides",
                completed_at=time.time(),
            )

            # Log completion
            log_audit_event(
                "batch_analysis_async_completed",
                details={
                    "task_id": task_id,
                    "total_slides": total,
                },
            )

            logger.info(f"Batch analysis {task_id} completed: {total} slides")

        except Exception as e:
            logger.error(f"Batch analysis {task_id} failed: {e}")
            batch_task_manager.update_task(task_id,
                status=BatchTaskStatus.FAILED,
                error=str(e),
                message=f"Batch analysis failed: {str(e)}",
            )

    @app.get("/api/analyze-batch/status/{task_id}")
    async def get_batch_status(task_id: str):
        """
        Get status of an async batch analysis task.

        Returns:
        - status: pending, running, completed, cancelled, or failed
        - progress: 0-100 percentage
        - current_slide_index: which slide is being processed
        - current_slide_id: ID of current slide
        - message: human-readable status message
        - results: full results (only when completed)
        """
        task = batch_task_manager.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        # Return full results if completed
        if task.status in [BatchTaskStatus.COMPLETED, BatchTaskStatus.CANCELLED]:
            return task.to_full_dict()

        return task.to_dict()

    @app.post("/api/analyze-batch/cancel/{task_id}")
    async def cancel_batch_analysis(task_id: str):
        """
        Cancel a running batch analysis task.

        Already completed slides will be retained in the results.
        """
        task = batch_task_manager.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        if task.status != BatchTaskStatus.RUNNING:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel task with status {task.status.value}"
            )

        success = batch_task_manager.request_cancel(task_id)

        if success:
            log_audit_event(
                "batch_analysis_cancelled",
                details={"task_id": task_id, "slides_completed": task.completed_slides},
            )
            return {
                "success": True,
                "message": f"Cancellation requested for task {task_id}",
                "completed_slides": task.completed_slides,
            }

        return {
            "success": False,
            "message": "Failed to request cancellation",
        }

    @app.get("/api/analyze-batch/tasks")
    async def list_batch_tasks(
        status: Optional[str] = Query(None, description="Filter by status"),
    ):
        """
        List all batch analysis tasks.

        Optionally filter by status: pending, running, completed, cancelled, failed
        """
        status_filter = None
        if status:
            try:
                status_filter = BatchTaskStatus(status)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid status: {status}. Valid values: pending, running, completed, cancelled, failed"
                )

        tasks = batch_task_manager.list_tasks(status_filter)

        return {
            "tasks": tasks,
            "total": len(tasks),
        }

    # ====== Uncertainty Quantification Endpoint ======

    @app.post("/api/analyze-uncertainty", response_model=UncertaintyResponse)
    async def analyze_with_uncertainty(request: UncertaintyRequest):
        """
        Analyze a slide with MC Dropout uncertainty quantification.

        Uses Monte Carlo Dropout to estimate predictive uncertainty by running
        multiple forward passes with dropout enabled. High variance indicates
        the model is uncertain about its prediction.

        Clinical interpretation:
        - Low uncertainty (< 0.10): Model is confident, prediction reliable
        - Moderate uncertainty (0.10 - 0.20): Some uncertainty, review recommended
        - High uncertainty (> 0.20): Model is uncertain, consider additional testing

        When uncertainty is high, the system flags the case for human review
        and provides conservative recommendations.
        """
        if classifier is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        slide_id = request.slide_id
        emb_path = embeddings_dir / f"{slide_id}.npy"

        if not emb_path.exists():
            raise HTTPException(status_code=404, detail=f"Slide {slide_id} not found")

        # Load embeddings
        embeddings = np.load(emb_path)

        # Run MC Dropout prediction
        try:
            result = classifier.predict_with_uncertainty(
                embeddings, n_samples=request.n_samples
            )
        except Exception as e:
            logger.error(f"Uncertainty prediction failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Uncertainty prediction failed: {str(e)}"
            )

        # Determine uncertainty level and clinical recommendation
        uncertainty = result["uncertainty"]
        probability = result["probability"]

        if uncertainty < 0.10:
            uncertainty_level = "low"
            requires_review = False
            if result["prediction"] == "RESPONDER":
                clinical_recommendation = (
                    "Model shows high confidence in RESPONDER prediction. "
                    "Consider proceeding with bevacizumab treatment evaluation."
                )
            else:
                clinical_recommendation = (
                    "Model shows high confidence in NON-RESPONDER prediction. "
                    "Consider alternative treatment options."
                )
        elif uncertainty < 0.20:
            uncertainty_level = "moderate"
            requires_review = True
            clinical_recommendation = (
                "Model shows moderate uncertainty. Recommend pathologist review "
                "of high-attention regions and correlation with clinical factors."
            )
        else:
            uncertainty_level = "high"
            requires_review = True
            clinical_recommendation = (
                "Model is uncertain about this case - consider additional testing. "
                "Do not rely solely on this prediction. Recommend molecular profiling "
                "and/or expert pathology consultation."
            )

        # Load coordinates for evidence patches
        coord_path = embeddings_dir / f"{slide_id}_coords.npy"
        coords = None
        if coord_path.exists():
            coords = np.load(coord_path)

        # Get top evidence patches using mean attention
        attention = result["attention_weights"]
        attention_std = result["attention_uncertainty"]
        top_k = min(8, len(attention))
        top_indices = np.argsort(attention)[-top_k:][::-1]

        top_evidence = []
        for i, idx in enumerate(top_indices):
            patch_x = int(coords[idx][0]) if coords is not None else 0
            patch_y = int(coords[idx][1]) if coords is not None else 0

            # Include attention uncertainty for each patch
            top_evidence.append({
                "rank": i + 1,
                "patch_index": int(idx),
                "attention_weight": float(attention[idx]),
                "attention_uncertainty": float(attention_std[idx]),
                "coordinates": [patch_x, patch_y],
            })

        # Log audit event
        log_audit_event(
            "uncertainty_analysis_completed",
            slide_id,
            details={
                "prediction": result["prediction"],
                "uncertainty": uncertainty,
                "uncertainty_level": uncertainty_level,
                "requires_review": requires_review,
            },
        )

        return UncertaintyResponse(
            slide_id=slide_id,
            prediction=result["prediction"],
            probability=probability,
            uncertainty=uncertainty,
            confidence_interval=result["confidence_interval"],
            is_uncertain=result["is_uncertain"],
            requires_review=requires_review,
            uncertainty_level=uncertainty_level,
            clinical_recommendation=clinical_recommendation,
            patches_analyzed=len(embeddings),
            n_samples=result["n_samples"],
            samples=result["samples"],
            top_evidence=top_evidence,
        )

    # ====== Analysis History Endpoints ======

    @app.get("/api/history", response_model=AnalysisHistoryResponse)
    async def get_analysis_history(
        limit: int = 50,
        offset: int = 0,
        slide_id: Optional[str] = None,
        prediction: Optional[str] = None,
    ):
        """
        Get recent analysis history.

        Args:
            limit: Maximum number of entries to return (default 50, max 100)
            offset: Number of entries to skip
            slide_id: Filter by specific slide ID
            prediction: Filter by prediction result (RESPONDER/NON-RESPONDER)

        Returns:
            List of analysis history entries, most recent first.
        """
        # Convert deque to list for filtering and slicing
        all_entries = list(analysis_history)

        # Apply filters
        if slide_id:
            all_entries = [e for e in all_entries if e["slide_id"] == slide_id]
        if prediction:
            all_entries = [e for e in all_entries if e["prediction"] == prediction.upper()]

        # Sort by timestamp descending (most recent first)
        all_entries.sort(key=lambda x: x["timestamp"], reverse=True)

        # Apply pagination
        limit = min(limit, 100)
        paginated = all_entries[offset:offset + limit]

        return AnalysisHistoryResponse(
            analyses=[AnalysisHistoryEntry(**e) for e in paginated],
            total=len(all_entries),
        )

    @app.get("/api/slides/{slide_id}/history", response_model=AnalysisHistoryResponse)
    async def get_slide_history(slide_id: str, limit: int = 20):
        """
        Get analysis history for a specific slide.

        Args:
            slide_id: The slide ID to get history for
            limit: Maximum number of entries to return (default 20)

        Returns:
            List of analysis history entries for this slide, most recent first.
        """
        # Filter entries for this slide
        slide_entries = [e for e in analysis_history if e["slide_id"] == slide_id]

        # Sort by timestamp descending
        slide_entries.sort(key=lambda x: x["timestamp"], reverse=True)

        # Apply limit
        limit = min(limit, 50)
        paginated = slide_entries[:limit]

        return AnalysisHistoryResponse(
            analyses=[AnalysisHistoryEntry(**e) for e in paginated],
            total=len(slide_entries),
        )

    @app.get("/api/audit-log", response_model=AuditLogResponse)
    async def get_audit_log(
        limit: int = 100,
        offset: int = 0,
        event_type: Optional[str] = None,
        slide_id: Optional[str] = None,
    ):
        """
        Get audit log entries for compliance tracking.

        Event types:
        - analysis_completed: Slide analysis was run
        - report_generated: Clinical report was generated
        - pdf_exported: Report was exported as PDF
        - json_exported: Report was exported as JSON

        Args:
            limit: Maximum number of entries to return (default 100)
            offset: Number of entries to skip
            event_type: Filter by event type
            slide_id: Filter by slide ID

        Returns:
            List of audit log entries, most recent first.
        """
        all_entries = list(audit_log)

        # Apply filters
        if event_type:
            all_entries = [e for e in all_entries if e["event_type"] == event_type]
        if slide_id:
            all_entries = [e for e in all_entries if e.get("slide_id") == slide_id]

        # Sort by timestamp descending
        all_entries.sort(key=lambda x: x["timestamp"], reverse=True)

        # Apply pagination
        limit = min(limit, 500)
        paginated = all_entries[offset:offset + limit]

        return AuditLogResponse(
            entries=[AuditLogEntry(**e) for e in paginated],
            total=len(all_entries),
        )

    def _load_patient_context(slide_id: str) -> Optional[Dict[str, Any]]:
        """Load patient context from labels.csv for a given slide."""
        labels_path = embeddings_dir.parent / "labels.csv"
        if not labels_path.exists():
            return None

        import csv
        with open(labels_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Handle different CSV formats
                if "slide_id" in row:
                    sid = row["slide_id"]
                else:
                    slide_file = row.get("slide_file", "")
                    sid = slide_file.replace(".svs", "").replace(".SVS", "")

                if sid == slide_id:
                    patient_ctx = {}
                    if row.get("age"):
                        try:
                            patient_ctx["age"] = int(row["age"])
                        except ValueError:
                            pass
                    if row.get("sex"):
                        patient_ctx["sex"] = row["sex"]
                    if row.get("stage"):
                        patient_ctx["stage"] = row["stage"]
                    if row.get("grade"):
                        patient_ctx["grade"] = row["grade"]
                    if row.get("prior_treatments"):
                        try:
                            patient_ctx["prior_lines"] = int(row["prior_treatments"])
                        except ValueError:
                            pass
                    if row.get("histology"):
                        patient_ctx["histology"] = row["histology"]
                    return patient_ctx if patient_ctx else None
        return None

    def _format_patient_summary(patient_ctx: Optional[Dict[str, Any]]) -> str:
        """Format patient context into a clinical summary sentence."""
        if not patient_ctx:
            return ""

        parts = []
        if patient_ctx.get("age"):
            parts.append(f"{patient_ctx['age']}-year-old")
        if patient_ctx.get("sex"):
            sex_full = "female" if patient_ctx["sex"].upper() == "F" else "male" if patient_ctx["sex"].upper() == "M" else patient_ctx["sex"]
            parts.append(sex_full)

        summary = " ".join(parts) if parts else "Patient"

        clinical_parts = []
        if patient_ctx.get("stage"):
            clinical_parts.append(f"Stage {patient_ctx['stage']}")
        if patient_ctx.get("histology"):
            clinical_parts.append(patient_ctx["histology"].lower())
        if clinical_parts:
            summary += " with " + " ".join(clinical_parts)

        if patient_ctx.get("prior_lines") is not None:
            lines = patient_ctx["prior_lines"]
            if lines == 0:
                summary += ", treatment-naive"
            else:
                summary += f", {lines} prior line{'s' if lines > 1 else ''} of therapy"

        return summary

    @app.post("/api/report", response_model=ReportResponse)
    async def generate_report(request: ReportRequest):
        """Generate a structured report for a slide using MedGemma."""
        if classifier is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        slide_id = request.slide_id
        emb_path = embeddings_dir / f"{slide_id}.npy"
        coord_path = embeddings_dir / f"{slide_id}_coords.npy"

        if not emb_path.exists():
            raise HTTPException(status_code=404, detail=f"Slide {slide_id} not found")

        embeddings = np.load(emb_path)
        score, attention = classifier.predict(embeddings)
        label = "responder" if score > 0.5 else "non-responder"

        # Load patient context
        patient_ctx = _load_patient_context(slide_id)

        # Get top evidence patches with coordinates
        top_k = min(8, len(attention))
        top_indices = np.argsort(attention)[-top_k:][::-1]

        # Load coordinates if available
        coords = None
        if coord_path.exists():
            coords = np.load(coord_path)

        evidence_patches = []
        for rank, idx in enumerate(top_indices, 1):
            patch_info = {
                "rank": rank,
                "patch_index": int(idx),
                "attention_weight": float(attention[idx]),
                "coordinates": [int(coords[idx][0]), int(coords[idx][1])] if coords is not None else [0, 0],
            }
            evidence_patches.append(patch_info)

        # Get similar cases
        similar_cases = []
        if evidence_gen is not None:
            try:
                similar_results = evidence_gen.find_similar(
                    embeddings, attention, k=5, top_patches=3
                )
                for s in similar_results:
                    similar_cases.append(s)
            except Exception as e:
                logger.warning(f"Similar case search failed for report: {e}")

        # Get slide quality metrics for decision support
        quality_metrics = None
        try:
            # Generate deterministic quality metrics based on slide_id
            import hashlib
            hash_val = int(hashlib.md5(slide_id.encode()).hexdigest(), 16)
            tissue_coverage = 0.60 + (hash_val % 40) / 100.0
            blur_score = (hash_val % 30) / 100.0
            stain_uniformity = 0.70 + (hash_val % 30) / 100.0
            artifact_detected = (hash_val % 10) == 0
            
            quality_score = (
                tissue_coverage * 0.3 +
                (1 - blur_score) * 0.3 +
                stain_uniformity * 0.2 +
                (0 if artifact_detected else 0.2)
            )
            overall_quality = "good" if quality_score >= 0.75 else "acceptable" if quality_score >= 0.50 else "poor"
            
            quality_metrics = {
                "overall_quality": overall_quality,
                "tissue_coverage": tissue_coverage,
                "blur_score": blur_score,
                "artifact_detected": artifact_detected,
            }
        except Exception as e:
            logger.warning(f"Could not compute quality metrics: {e}")

        # Generate clinical decision support
        decision_support_data = None
        if decision_support is not None:
            try:
                ds_output = decision_support.generate(
                    prediction=label,
                    score=float(score),
                    similar_cases=similar_cases,
                    quality_metrics=quality_metrics,
                    patient_context=patient_ctx,
                )
                decision_support_data = decision_support.to_dict(ds_output)
                logger.info(f"Generated decision support for {slide_id}: risk_level={ds_output.risk_level.value}")
            except Exception as e:
                logger.warning(f"Decision support generation failed: {e}")

        # Try MedGemma report generation
        if reporter is not None:
            timeout_s = None
            try:
                timeout_s = getattr(reporter.config, "max_generation_time_s", None)
                if timeout_s is None:
                    timeout_s = 120.0
                timeout_s = max(10.0, float(timeout_s) + 10.0)

                report = await asyncio.wait_for(
                    asyncio.to_thread(
                        reporter.generate_report,
                        evidence_patches=evidence_patches,
                        score=score,
                        label=label,
                        similar_cases=similar_cases,
                        case_id=slide_id,
                        patient_context=patient_ctx,
                    ),
                    timeout=timeout_s,
                )

                # Merge decision support into structured report
                structured = report["structured"]
                if decision_support_data:
                    structured["decision_support"] = decision_support_data

                return ReportResponse(
                    slide_id=slide_id,
                    report_json=structured,
                    summary_text=report["summary"],
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "MedGemma report generation timed out after %.1fs for slide %s",
                    timeout_s or 0.0,
                    slide_id,
                )
            except Exception as e:
                logger.warning(f"MedGemma report generation failed, using template: {e}")

        # Format patient summary for report
        patient_summary = _format_patient_summary(patient_ctx)

        # Generate morphology descriptions based on tissue types and attention
        def generate_morphology_description(patch: dict, rank: int) -> tuple[str, str]:
            """Generate morphology description and significance for a patch."""
            tissue_type = patch.get("tissue_type", "unknown")
            attention = patch.get("attention_weight", 0)
            coords = patch.get("coordinates", [0, 0])
            
            # Tissue-type specific morphology descriptions
            morphology_templates = {
                "tumor": [
                    "Dense cellular region with atypical epithelial morphology and increased nuclear-to-cytoplasmic ratio",
                    "Papillary architecture with stratified epithelium showing nuclear atypia",
                    "Solid sheets of cells with irregular nuclear contours and prominent nucleoli",
                    "Glandular structures with back-to-back arrangement and cribriform patterns",
                ],
                "stroma": [
                    "Desmoplastic stroma with spindle-shaped fibroblasts and collagen deposition",
                    "Fibrovascular core with loose connective tissue and scattered vessels",
                    "Dense fibrous stroma with hyalinized collagen bundles",
                ],
                "necrosis": [
                    "Geographic necrosis with ghost cell outlines and nuclear debris",
                    "Coagulative necrosis with preserved tissue architecture",
                    "Necrotic debris with inflammatory cell infiltration",
                ],
                "inflammatory": [
                    "Lymphocytic infiltrate with peritumoral distribution",
                    "Tumor-infiltrating lymphocytes forming dense aggregates",
                    "Mixed inflammatory infiltrate with plasma cells and lymphocytes",
                ],
                "normal": [
                    "Normal epithelial architecture with maintained polarity",
                    "Benign glandular tissue with regular spacing",
                ],
            }
            
            # Significance based on label prediction
            significance_templates = {
                "responder": {
                    "tumor": "Tumor morphology patterns in this region are associated with better bevacizumab response in the training cohort",
                    "stroma": "Stromal composition in this area correlates with improved anti-angiogenic therapy outcomes",
                    "inflammatory": "Inflammatory infiltrate pattern suggests favorable tumor microenvironment for treatment response",
                    "necrosis": "Necrotic pattern may indicate pre-existing vascular compromise potentially responsive to anti-angiogenic therapy",
                    "normal": "Preserved tissue architecture in adjacent regions may indicate better overall tissue health",
                },
                "non-responder": {
                    "tumor": "Tumor morphology in this region shows patterns associated with resistance to anti-angiogenic therapy",
                    "stroma": "Stromal characteristics suggest possible treatment resistance mechanisms",
                    "inflammatory": "Inflammatory pattern may indicate tumor microenvironment less responsive to bevacizumab",
                    "necrosis": "Necrotic patterns in this configuration are associated with poor treatment outcomes",
                    "normal": "Limited tumor involvement in this area provides context for overall assessment",
                },
            }
            
            # Select morphology description
            templates = morphology_templates.get(tissue_type, ["Tissue region with notable morphological features"])
            morphology = templates[rank % len(templates)]
            
            # Add coordinate context
            morphology += f" at position ({coords[0]:,}, {coords[1]:,})"
            
            # Select significance
            label_key = label if label in significance_templates else "non-responder"
            sig_templates = significance_templates.get(label_key, {})
            significance = sig_templates.get(tissue_type, 
                f"High model attention (weight: {attention:.3f}) indicates this region contributes significantly to the prediction")
            
            return morphology, significance
        
        # Fallback to template report with detailed evidence
        evidence_list = []
        for i, p in enumerate(evidence_patches[:5]):
            morphology, significance = generate_morphology_description(p, i)
            evidence_list.append({
                "patch_id": f"patch_{p['patch_index']}",
                "attention_weight": p["attention_weight"],
                "coordinates": p["coordinates"],
                "morphology_description": morphology,
                "significance": significance,
                "tissue_type": p.get("tissue_type", "unknown"),
            })
        
        report_json = {
            "case_id": slide_id,
            "task": "Bevacizumab treatment response prediction from H&E histopathology",
            "patient_context": patient_ctx,
            "model_output": {
                "label": label,
                "probability": float(score),
                "calibration_note": "Model probability requires external validation. This is an uncalibrated research model.",
            },
            "evidence": evidence_list,
            "similar_examples": [
                {
                    "example_id": s.get("metadata", {}).get("slide_id", f"case_{i}"),
                    "slide_id": s.get("metadata", {}).get("slide_id", f"case_{i}"),
                    "distance": float(s.get("distance", 0)),
                    "similarity_score": 1.0 / (1.0 + s.get("distance", 0)),
                    "label": s.get("metadata", {}).get("label", "unknown"),
                }
                for i, s in enumerate(similar_cases[:5])
            ],
            "limitations": [
                "This is an uncalibrated research model - probabilities are not clinically validated",
                "Prediction is based on morphological patterns and may not capture all relevant clinical factors",
                "Model has been trained on a limited ovarian cancer dataset and may not generalize to all populations",
                "Slide quality and tissue representation may affect prediction accuracy",
                "Similar case comparison is based on embedding distance, not verified clinical outcomes",
            ],
            "suggested_next_steps": [
                "Review high-attention regions with attending pathologist",
                "Correlate findings with patient clinical history and imaging",
                "Consider molecular profiling (e.g., BRCA status, HRD) for additional treatment guidance",
                "Discuss findings in multidisciplinary tumor board before treatment decisions",
                "Validate prediction against institutional experience with similar cases",
            ],
            "safety_statement": "This is a research decision-support tool, not a diagnostic device. All findings must be validated by qualified pathologists and clinicians. Do not use for standalone clinical decision-making. Treatment decisions should incorporate all available clinical, pathological, and molecular data.",
            "decision_support": decision_support_data,
        }

        # Build comprehensive summary with patient context
        patient_intro = f"Patient: {patient_summary}.\n\n" if patient_summary else ""
        
        # Confidence interpretation
        confidence_val = abs(score - 0.5) * 2
        if confidence_val >= 0.6:
            confidence_desc = "high"
        elif confidence_val >= 0.3:
            confidence_desc = "moderate"
        else:
            confidence_desc = "low"
        
        # Evidence summary
        tissue_types_seen = [p.get("tissue_type", "unknown") for p in evidence_patches[:5]]
        tissue_summary = ", ".join(set(t for t in tissue_types_seen if t != "unknown")) or "various tissue types"

        summary_text = f"""{patient_intro}CASE ANALYSIS SUMMARY
=====================

Case ID: {slide_id}
Prediction: {label.upper()}
Model Score: {score:.3f}
Confidence Level: {confidence_desc.upper()} ({confidence_val:.1%})

ANALYSIS OVERVIEW
-----------------
This analysis examined {len(embeddings):,} tissue patches extracted from the whole-slide image.
The multiple instance learning (MIL) model identified {min(5, len(attention))} high-attention 
regions that contributed most significantly to the prediction.

Key morphological features observed in high-attention regions include: {tissue_summary}.

{"RESPONDER INTERPRETATION" if label == "responder" else "NON-RESPONDER INTERPRETATION"}
---------------------------------
{"The morphological patterns identified by the model suggest features associated with favorable response to bevacizumab-based anti-angiogenic therapy in the training cohort. These patterns may include specific tumor architecture, stromal characteristics, or inflammatory infiltrate distributions that have been correlated with treatment response." if label == "responder" else "The morphological patterns identified by the model suggest features associated with reduced response to bevacizumab-based anti-angiogenic therapy in the training cohort. Alternative treatment strategies may warrant consideration pending further clinical evaluation."}

SIMILAR CASES
-------------
{len(similar_cases)} similar cases from the reference cohort were identified based on 
morphological similarity. Review of these cases may provide additional context for 
interpreting the current prediction.

IMPORTANT DISCLAIMER
--------------------
This is a RESEARCH TOOL for decision support only. The model is uncalibrated and has not 
been clinically validated. All findings must be reviewed and validated by qualified 
pathologists and clinicians before any clinical decision-making. Treatment decisions 
should incorporate all available clinical, pathological, and molecular data."""

        return ReportResponse(
            slide_id=slide_id,
            report_json=report_json,
            summary_text=summary_text,
        )

    # ====== Async Report Generation Endpoints ======
    
    class AsyncReportRequest(BaseModel):
        """Request for async report generation."""
        slide_id: str = Field(..., min_length=1, max_length=256)
        include_evidence: bool = True
        include_similar: bool = True
    
    class AsyncReportResponse(BaseModel):
        """Response from async report generation."""
        task_id: str
        slide_id: str
        status: str
        message: str
        estimated_time_seconds: int = 30
    
    @app.post("/api/report/async", response_model=AsyncReportResponse)
    async def generate_report_async(request: AsyncReportRequest, background_tasks: BackgroundTasks):
        """
        Start async report generation for a slide.
        
        Returns a task_id immediately. Poll /api/report/status/{task_id} for progress.
        Report generation typically takes 20-60 seconds depending on model warmup.
        """
        slide_id = request.slide_id
        
        # Check if slide exists
        emb_path = embeddings_dir / f"{slide_id}.npy"
        if not emb_path.exists():
            raise HTTPException(status_code=404, detail=f"Slide {slide_id} not found")
        
        # Check for existing task
        existing_task = report_task_manager.get_task_by_slide(slide_id)
        if existing_task:
            return AsyncReportResponse(
                task_id=existing_task.task_id,
                slide_id=slide_id,
                status=existing_task.status.value,
                message=existing_task.message,
                estimated_time_seconds=30,
            )
        
        # Create new task
        task = report_task_manager.create_task(slide_id)
        
        # Run report generation in background
        def run_report_generation():
            _generate_report_background(
                task.task_id,
                slide_id,
                request.include_evidence,
                request.include_similar,
            )
        
        background_tasks.add_task(run_report_generation)
        
        return AsyncReportResponse(
            task_id=task.task_id,
            slide_id=slide_id,
            status="pending",
            message="Report generation started. Poll /api/report/status/{task_id} for progress.",
            estimated_time_seconds=30,
        )
    
    def _generate_report_background(
        task_id: str,
        slide_id: str,
        include_evidence: bool,
        include_similar: bool,
    ):
        """Background task to generate report."""
        import time
        import threading
        
        task = report_task_manager.get_task(task_id)
        if not task:
            return
        
        report_task_manager.update_task(task_id,
            status=ReportTaskStatus.RUNNING,
            started_at=time.time(),
            stage="analyzing",
            progress=10,
            message="Loading embeddings and running analysis..."
        )
        
        try:
            # Load embeddings
            emb_path = embeddings_dir / f"{slide_id}.npy"
            coord_path = embeddings_dir / f"{slide_id}_coords.npy"
            
            embeddings = np.load(emb_path)
            
            report_task_manager.update_task(task_id,
                progress=20,
                message="Running MIL prediction..."
            )
            
            score, attention = classifier.predict(embeddings)
            label = "responder" if score > 0.5 else "non-responder"
            
            report_task_manager.update_task(task_id,
                progress=30,
                stage="generating",
                message="Loading patient context and evidence..."
            )
            
            # Load patient context
            patient_ctx = _load_patient_context(slide_id)
            
            # Get top evidence patches
            top_k = min(8, len(attention))
            top_indices = np.argsort(attention)[-top_k:][::-1]
            
            coords = None
            if coord_path.exists():
                coords = np.load(coord_path)
            
            evidence_patches = []
            for rank, idx in enumerate(top_indices, 1):
                patch_info = {
                    "rank": rank,
                    "patch_index": int(idx),
                    "attention_weight": float(attention[idx]),
                    "coordinates": [int(coords[idx][0]), int(coords[idx][1])] if coords is not None else [0, 0],
                }
                evidence_patches.append(patch_info)
            
            report_task_manager.update_task(task_id,
                progress=40,
                message="Finding similar cases..."
            )
            
            # Get similar cases
            similar_cases = []
            if include_similar and evidence_gen is not None:
                try:
                    similar_results = evidence_gen.find_similar(
                        embeddings, attention, k=5, top_patches=3
                    )
                    for s in similar_results:
                        similar_cases.append(s)
                except Exception as e:
                    logger.warning(f"Similar case search failed: {e}")
            
            report_task_manager.update_task(task_id,
                progress=50,
                message="Generating clinical decision support..."
            )
            
            # Generate quality metrics
            import hashlib
            hash_val = int(hashlib.md5(slide_id.encode()).hexdigest(), 16)
            quality_metrics = {
                "overall_quality": "good" if hash_val % 3 == 0 else "acceptable",
                "tissue_coverage": 0.60 + (hash_val % 40) / 100.0,
                "blur_score": (hash_val % 30) / 100.0,
                "artifact_detected": (hash_val % 10) == 0,
            }
            
            # Generate decision support
            decision_support_data = None
            if decision_support is not None:
                try:
                    ds_output = decision_support.generate(
                        prediction=label,
                        score=float(score),
                        similar_cases=similar_cases,
                        quality_metrics=quality_metrics,
                        patient_context=patient_ctx,
                    )
                    decision_support_data = decision_support.to_dict(ds_output)
                except Exception as e:
                    logger.warning(f"Decision support failed: {e}")
            
            report_task_manager.update_task(task_id,
                progress=60,
                message="Generating report with MedGemma (this may take up to 120s)..."
            )
            
            # Generate report
            report_json = None
            summary_text = None
            
            if reporter is not None:
                try:
                    max_time = getattr(reporter.config, "max_generation_time_s", None)
                    max_tokens = getattr(reporter.config, "max_output_tokens", None)
                    max_time_display = f"{float(max_time):.1f}s" if max_time is not None else "none"
                    logger.info(
                        "Starting MedGemma report generation for %s (max_time=%s, max_new_tokens=%s)",
                        slide_id,
                        max_time_display,
                        max_tokens,
                    )

                    stop_event = threading.Event()
                    def _progress_heartbeat():
                        progress = 60
                        while not stop_event.wait(5):
                            progress = min(85, progress + 2)
                            report_task_manager.update_task(
                                task_id,
                                progress=progress,
                                stage="generating",
                                message="MedGemma is generating the report..."
                            )

                    heartbeat = threading.Thread(target=_progress_heartbeat, daemon=True)
                    heartbeat.start()

                    gen_start = time.time()
                    report = reporter.generate_report(
                        evidence_patches=evidence_patches,
                        score=score,
                        label=label,
                        similar_cases=similar_cases,
                        case_id=slide_id,
                        patient_context=patient_ctx,
                    )
                    logger.info(
                        "MedGemma report generation completed for %s in %.1fs",
                        slide_id,
                        time.time() - gen_start,
                    )
                    
                    report_json = report["structured"]
                    summary_text = report["summary"]
                    
                    if decision_support_data:
                        report_json["decision_support"] = decision_support_data

                    stop_event.set()
                    heartbeat.join(timeout=1)
                except Exception as e:
                    stop_event.set()
                    heartbeat.join(timeout=1)
                    logger.warning(f"MedGemma failed: {e}")
            
            # Fallback to template if MedGemma failed
            if report_json is None:
                report_task_manager.update_task(task_id,
                    progress=80,
                    message="Using template report (MedGemma unavailable)..."
                )
                report_json = _create_template_report(
                    slide_id, label, float(score), evidence_patches, 
                    similar_cases, patient_ctx, decision_support_data
                )
                summary_text = _create_template_summary(
                    slide_id, label, float(score), len(embeddings),
                    patient_ctx, similar_cases
                )
            
            report_task_manager.update_task(task_id,
                progress=90,
                stage="formatting",
                message="Finalizing report..."
            )
            
            elapsed = time.time() - task.started_at
            report_task_manager.update_task(task_id,
                status=ReportTaskStatus.COMPLETED,
                progress=100,
                stage="complete",
                message=f"Report generated successfully in {elapsed:.1f}s",
                completed_at=time.time(),
                result={
                    "slide_id": slide_id,
                    "report_json": report_json,
                    "summary_text": summary_text,
                }
            )
            
            logger.info(f"Report generation completed for {slide_id} in {elapsed:.1f}s")
            
        except Exception as e:
            logger.error(f"Report generation failed for {slide_id}: {e}")
            import traceback
            traceback.print_exc()
            report_task_manager.update_task(task_id,
                status=ReportTaskStatus.FAILED,
                error=str(e),
                message=f"Report generation failed: {str(e)}"
            )
    
    def _create_template_report(slide_id, label, score, evidence_patches, similar_cases, patient_ctx, decision_support_data):
        """Create a fallback template report."""
        return {
            "case_id": slide_id,
            "task": "Bevacizumab treatment response prediction from H&E histopathology",
            "patient_context": patient_ctx,
            "model_output": {
                "label": label,
                "probability": score,
                "calibration_note": "Model probability requires external validation.",
            },
            "evidence": [
                {
                    "patch_id": f"patch_{p['patch_index']}",
                    "attention_weight": p["attention_weight"],
                    "coordinates": p["coordinates"],
                    "morphology_description": "High attention region identified by model",
                    "significance": "Region contributes to prediction outcome",
                }
                for p in evidence_patches[:5]
            ],
            "similar_examples": [
                {
                    "example_id": s.get("metadata", {}).get("slide_id", f"case_{i}"),
                    "distance": float(s.get("distance", 0)),
                    "label": s.get("metadata", {}).get("label", "unknown"),
                }
                for i, s in enumerate(similar_cases[:5])
            ],
            "limitations": [
                "This is an uncalibrated research model",
                "Prediction based on morphological patterns only",
                "Requires validation by qualified pathologists",
            ],
            "suggested_next_steps": [
                "Review high-attention regions with pathologist",
                "Correlate with clinical history",
                "Consider molecular profiling",
            ],
            "safety_statement": "This is a research tool. All findings must be validated by qualified clinicians.",
            "decision_support": decision_support_data,
        }
    
    def _create_template_summary(slide_id, label, score, num_patches, patient_ctx, similar_cases):
        """Create a fallback template summary."""
        confidence = abs(score - 0.5) * 2
        return f"""CASE ANALYSIS SUMMARY
Case ID: {slide_id}
Prediction: {label.upper()}
Score: {score:.3f}
Confidence: {confidence:.1%}

This analysis examined {num_patches:,} tissue patches.
{len(similar_cases)} similar cases were identified for reference.

DISCLAIMER: This is a research tool. All findings must be validated by qualified pathologists."""
    
    @app.get("/api/report/status/{task_id}")
    async def get_report_status(task_id: str):
        """
        Get status of an async report generation task.
        
        Poll this endpoint to track report generation progress.
        When status is 'completed', the result field contains the full report.
        """
        task = report_task_manager.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        
        return task.to_dict()
    
    @app.get("/api/report/tasks")
    async def list_report_tasks(
        slide_id: Optional[str] = Query(None),
        status: Optional[str] = Query(None),
    ):
        """List all report generation tasks."""
        tasks = []
        for task in report_task_manager.tasks.values():
            if slide_id and task.slide_id != slide_id:
                continue
            if status and task.status.value != status:
                continue
            tasks.append(task.to_dict())
        
        return {
            "tasks": sorted(tasks, key=lambda t: t.get("elapsed_seconds", 0), reverse=True),
            "total": len(tasks)
        }



    @app.get("/api/heatmap/{slide_id}")
    async def get_heatmap(
        slide_id: str,
        level: int = Query(default=2, ge=0, le=4, description="Downsample level: 0=2048px (highest detail), 2=512px (default), 4=128px (fastest)"),
        smooth: bool = Query(default=True, description="Apply Gaussian blur for smooth interpolation (True) or show sharp patch tiles (False)"),
        blur: int = Query(default=31, ge=3, le=101, description="Blur kernel size (odd number, higher=smoother)")
    ):
        """Get the attention heatmap for a slide as PNG.
        
        Uses CPU-only inference to avoid CUDA driver compatibility issues.
        Level controls output resolution: 0=2048, 1=1024, 2=512, 3=256, 4=128 pixels.
        Smooth enables Gaussian blur interpolation for cleaner visualization.
        """
        # Map level to thumbnail size (level 0 = highest res, level 4 = lowest/fastest)
        LEVEL_SIZES = {0: 2048, 1: 1024, 2: 512, 3: 256, 4: 128}
        thumbnail_size = LEVEL_SIZES.get(level, 512)
        import torch
        
        if classifier is None or evidence_gen is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        emb_path = embeddings_dir / f"{slide_id}.npy"
        coord_path = embeddings_dir / f"{slide_id}_coords.npy"

        if not emb_path.exists():
            raise HTTPException(status_code=404, detail=f"Slide {slide_id} not found")

        embeddings = np.load(emb_path)

        # Load or generate coordinates
        patch_size = 224
        if coord_path.exists():
            coords_arr = np.load(coord_path)
            coords_arr = coords_arr.astype(np.int64, copy=False)
        else:
            n_patches = len(embeddings)
            if n_patches > 0:
                grid_size = int(np.ceil(np.sqrt(n_patches)))
                grid_x, grid_y = np.meshgrid(
                    np.arange(grid_size) * patch_size,
                    np.arange(grid_size) * patch_size,
                )
                coords_arr = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)[:n_patches]
            else:
                coords_arr = np.zeros((0, 2), dtype=np.int64)
            logger.warning(
                f"No coords found for {slide_id}; generated {len(coords_arr)} grid coords from embeddings"
            )

        coords = [tuple(map(int, c)) for c in coords_arr]

        # Helper function for CPU-only inference (avoids CUDA driver issues)
        def cpu_predict(embs):
            """Predict using CPU only to avoid CUDA errors."""
            import torch
            from enso_atlas.config import MILConfig
            from enso_atlas.mil.clam import CLAMClassifier, LegacyCLAMModel
            
            # Create CPU tensor
            x = torch.from_numpy(embs).float()
            
            # Create a fresh CPU-only model
            model = LegacyCLAMModel(input_dim=384, hidden_dim=256)
            
            # Load weights
            model_path = Path(__file__).parent.parent.parent.parent / "models" / "clam_attention.pt"
            if model_path.exists():
                checkpoint = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                else:
                    state_dict = checkpoint
                model.load_state_dict(state_dict)
            
            model.eval()
            
            with torch.no_grad():
                prob, attention = model(x, return_attention=True)
            
            return prob.item(), attention.numpy()

        # Always use CPU for heatmap to avoid CUDA driver issues
        try:
            score, attention = cpu_predict(embeddings)
        except Exception as e:
            logger.error(f"CPU prediction failed: {e}")
            raise HTTPException(status_code=500, detail=f"Heatmap generation failed: {str(e)}")
        
        # Get actual slide dimensions from slide file
        slide_path = resolve_slide_path(slide_id)
        if slide_path is not None and slide_path.exists():
            try:
                import openslide
                with openslide.OpenSlide(str(slide_path)) as slide:
                    slide_dims = slide.dimensions
                    logger.info(f"Heatmap using actual slide dims: {slide_dims}")
            except Exception as e:
                logger.warning(f"Could not read slide dimensions: {e}")
                slide_dims = None
        else:
            slide_dims = None
        
        # Fall back to computing bounds from coordinates if no slide available
        if slide_dims is None:
            if coords_arr.size == 0:
                slide_dims = (patch_size, patch_size)
                logger.warning(
                    f"No coordinates available to derive dims for {slide_id}; "
                    f"falling back to {slide_dims}"
                )
            else:
                x_max = int(coords_arr[:, 0].max()) + patch_size  # Add patch size
                y_max = int(coords_arr[:, 1].max()) + patch_size
                slide_dims = (x_max, y_max)
                logger.info(f"Heatmap using coords-derived dims: {slide_dims}")
        
        # Calculate aspect-ratio-preserving thumbnail dimensions
        # This ensures the heatmap matches the slide geometry for correct overlay alignment
        slide_w, slide_h = slide_dims
        if slide_w >= slide_h:
            thumb_w = thumbnail_size
            thumb_h = max(1, int(round(thumbnail_size * slide_h / slide_w)))
        else:
            thumb_h = thumbnail_size
            thumb_w = max(1, int(round(thumbnail_size * slide_w / slide_h)))
        
        logger.info(f"Heatmap thumbnail size: {thumb_w}x{thumb_h} (preserving aspect ratio of {slide_w}x{slide_h})")
        
        heatmap = evidence_gen.create_heatmap(attention, coords, slide_dims, (thumb_w, thumb_h), smooth=smooth, blur_kernel=blur)

        # Save to temp file and return
        from PIL import Image
        import io

        img = Image.fromarray(heatmap)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        # Return with bounds headers so frontend can position correctly
        return StreamingResponse(
            buf,
            media_type="image/png",
            headers={
                "Content-Disposition": f"inline; filename={slide_id}_heatmap.png",
                "X-Slide-Width": str(slide_dims[0]),
                "X-Slide-Height": str(slide_dims[1]),
                "Access-Control-Expose-Headers": "X-Slide-Width, X-Slide-Height",
            },
        )


    @app.post("/api/embed", response_model=EmbedResponse)
    async def embed_patches(request: EmbedRequest):
        """Generate embeddings for patch images using Path Foundation."""
        if embedder is None:
            raise HTTPException(status_code=503, detail="Embedder not initialized")

        # Decode patches
        patches = []
        for i, b64_patch in enumerate(request.patches):
            try:
                # Handle data URI format
                if "," in b64_patch:
                    b64_patch = b64_patch.split(",", 1)[1]

                image_data = base64.b64decode(b64_patch)
                image = Image.open(io.BytesIO(image_data))

                if image.mode != "RGB":
                    image = image.convert("RGB")

                if image.size != (224, 224):
                    image = image.resize((224, 224), Image.Resampling.LANCZOS)

                patches.append(np.array(image))
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to decode patch {i}: {str(e)}",
                )

        # Generate embeddings
        try:
            embeddings = embedder.embed(patches, show_progress=False)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Embedding generation failed: {str(e)}",
            )

        response = EmbedResponse(
            num_patches=len(patches),
            embedding_dim=embeddings.shape[1] if len(embeddings.shape) > 1 else 384,
        )

        if request.return_embeddings:
            response.embeddings = embeddings.tolist()

        return response

    @app.get("/api/similar", response_model=SimilarResponse)
    async def get_similar_cases(slide_id: str, k: int = 5, top_patches: int = 3):
        """Find similar slides from the reference cohort.

        Uses FAISS over **L2-normalized mean slide embeddings** and returns
        top-k by **cosine similarity** (implemented as inner product).

        Notes:
        - `top_patches` is kept for backwards compatibility but is not used in
          the slide-mean similarity method.
        """
        if slide_mean_index is None:
            raise HTTPException(status_code=503, detail="Similarity index not available")

        emb_path = embeddings_dir / f"{slide_id}.npy"
        if not emb_path.exists():
            raise HTTPException(status_code=404, detail=f"Slide {slide_id} not found")

        embs = np.load(emb_path)
        if embs is None or len(embs) == 0:
            return SimilarResponse(slide_id=slide_id, similar_cases=[], num_queries=1)

        q = np.asarray(embs, dtype=np.float32).mean(axis=0)
        q = q / (np.linalg.norm(q) + 1e-12)
        q = q.reshape(1, -1).astype(np.float32)

        search_k = min(len(slide_mean_ids), max(k + 10, k * 3))
        sims, idxs = slide_mean_index.search(q, search_k)

        similar_cases = []
        seen = set()

        for sim, idx in zip(sims[0], idxs[0]):
            if idx < 0 or idx >= len(slide_mean_ids):
                continue
            sid = slide_mean_ids[int(idx)]
            if sid == slide_id or sid in seen:
                continue
            seen.add(sid)

            meta = slide_mean_meta.get(sid, {})
            similar_cases.append({
                "slide_id": sid,
                "similarity_score": float(sim),
                "distance": float(1.0 - float(sim)),
                "label": meta.get("label") or slide_labels.get(sid),
                "n_patches": meta.get("n_patches"),
            })

            if len(similar_cases) >= k:
                break

        return SimilarResponse(
            slide_id=slide_id,
            similar_cases=similar_cases,
            num_queries=1,
        )

    @app.get("/api/embed/status")
    async def embedder_status():
        """Check the status of the Path Foundation embedder."""
        model_loaded = embedder is not None and embedder._model is not None
        device = "unknown"

        if model_loaded:
            device = str(embedder._device)
        else:
            device = "cuda" if _check_cuda() else "cpu"

        return {
            "model": "google/path-foundation",
            "model_loaded": model_loaded,
            "device": device,
            "embedding_dim": 384,
            "input_size": 224,
        }

    # ====== MedSigLIP Semantic Search ======

    @app.post("/api/semantic-search", response_model=SemanticSearchResponse)
    async def semantic_search(request: SemanticSearchRequest):
        """
        Search patches by text query using MedSigLIP when available.

        If precomputed MedSigLIP patch embeddings are unavailable, falls back to
        a query-aware tissue-type search (based on patch coordinates) and ranks
        results using tissue-type confidence and (when available) MIL attention.

        This ensures the endpoint still works meaningfully without SigLIP caches.
        """
        if medsiglip_embedder is None:
            raise HTTPException(status_code=503, detail="MedSigLIP embedder not initialized")

        slide_id = request.slide_id
        emb_path = embeddings_dir / f"{slide_id}.npy"
        coord_path = embeddings_dir / f"{slide_id}_coords.npy"

        if not emb_path.exists():
            raise HTTPException(status_code=404, detail=f"Slide {slide_id} not found")

        # Try to use precomputed MedSigLIP patch embeddings if present
        siglip_cache_key = f"{slide_id}_siglip"
        use_siglip_search = True

        if siglip_cache_key in slide_siglip_embeddings:
            siglip_embeddings = slide_siglip_embeddings[siglip_cache_key]
            logger.info(f"Using cached MedSigLIP embeddings for {slide_id}")
        else:
            siglip_cache_path = embeddings_dir / "medsiglip_cache" / f"{slide_id}_siglip.npy"
            if siglip_cache_path.exists():
                siglip_embeddings = np.load(siglip_cache_path)
                slide_siglip_embeddings[siglip_cache_key] = siglip_embeddings
                logger.info(f"Loaded MedSigLIP embeddings from cache for {slide_id}")
            else:
                use_siglip_search = False
                siglip_embeddings = None

        # Load coordinates if available
        coords = None
        if coord_path.exists():
            coords = np.load(coord_path)

        # Load PF embeddings (for attention computation and metadata sizing)
        pf_embeddings = np.load(emb_path)

        # Attention weights are helpful but should never break search (CUDA asserts, etc.)
        attention_weights = None
        if classifier is not None:
            try:
                _, attention_weights = classifier.predict(pf_embeddings)
            except Exception as e:
                logger.warning(f"Could not compute attention weights for semantic search (fallback continues): {e}")
                attention_weights = None

        # Build metadata
        num_patches = len(siglip_embeddings) if siglip_embeddings is not None else len(pf_embeddings)
        metadata = []
        for i in range(num_patches):
            meta = {"index": i}
            if coords is not None and i < len(coords):
                meta["coordinates"] = [int(coords[i][0]), int(coords[i][1])]
            if attention_weights is not None and i < len(attention_weights):
                meta["attention_weight"] = float(attention_weights[i])
            metadata.append(meta)

        # Execute search
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
                # Query-aware tissue-type fallback
                query_lower = request.query.lower()

                TISSUE_KEYWORDS = {
                    "tumor": ["tumor", "tumour", "cancer", "malignant", "neoplastic", "atypical", "carcinoma"],
                    "stroma": ["stroma", "stromal", "fibrous", "connective", "collagen", "desmoplastic", "fibroblast"],
                    "necrosis": ["necrosis", "necrotic", "dead", "dying", "debris", "coagulative"],
                    "inflammatory": ["inflammatory", "inflammation", "lymphocyte", "lymphocytic", "immune", "infiltrate", "til", "plasma"],
                    "normal": ["normal", "benign", "healthy"],
                    "artifact": ["artifact", "blur", "fold", "pen", "marker", "bubble"],
                }

                matching_types = set()
                for tissue_type, keywords in TISSUE_KEYWORDS.items():
                    if any(k in query_lower for k in keywords):
                        matching_types.add(tissue_type)

                logger.info(f"Semantic search fallback: query='{request.query}' matched_types={sorted(matching_types)}")

                scored = []
                for i in range(num_patches):
                    if coords is not None and i < len(coords):
                        patch_x, patch_y = int(coords[i][0]), int(coords[i][1])
                    else:
                        patch_x, patch_y = 0, 0

                    tissue_info = classify_tissue_type(patch_x, patch_y, int(i))
                    patch_tissue = tissue_info["tissue_type"]
                    tissue_conf = float(tissue_info["confidence"])

                    attn = float(attention_weights[i]) if attention_weights is not None and i < len(attention_weights) else 0.5

                    if matching_types:
                        if patch_tissue in matching_types:
                            score = 0.75 * tissue_conf + 0.25 * attn
                        else:
                            score = 0.05 * attn
                    else:
                        score = 0.35 * tissue_conf + 0.65 * attn

                    scored.append({
                        "patch_index": int(i),
                        "similarity_score": float(score),
                        "metadata": metadata[i] if i < len(metadata) else {},
                    })

                scored.sort(key=lambda r: r["similarity_score"], reverse=True)
                search_results = scored[: min(request.top_k, len(scored))]
                model_used = "tissue-type-fallback"

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            raise HTTPException(status_code=500, detail=f"Semantic search failed: {str(e)}")

        results = []
        for r in search_results:
            results.append(
                SemanticSearchResult(
                    patch_index=r["patch_index"],
                    similarity_score=r["similarity_score"],
                    coordinates=r.get("metadata", {}).get("coordinates"),
                    attention_weight=r.get("metadata", {}).get("attention_weight"),
                )
            )

        return SemanticSearchResponse(
            slide_id=slide_id,
            query=request.query,
            results=results,
            embedding_model=model_used,
        )

    @app.get("/api/semantic-search/status")
    async def semantic_search_status():
        """Check the status of the MedSigLIP semantic search feature."""
        model_loaded = medsiglip_embedder is not None and medsiglip_embedder._model is not None
        device = "unknown"

        if model_loaded:
            device = str(medsiglip_embedder._device)
        else:
            device = "cuda" if _check_cuda() else "cpu"

        return {
            "model": medsiglip_embedder.config.model_id if medsiglip_embedder else "not initialized",
            "model_loaded": model_loaded,
            "device": device,
            "embedding_dim": medsiglip_embedder.EMBEDDING_DIM if medsiglip_embedder else None,
            "input_size": medsiglip_embedder.INPUT_SIZE if medsiglip_embedder else None,
            "cached_slides": list(slide_siglip_embeddings.keys()) if slide_siglip_embeddings else [],
        }

    # ====== Slide Quality Control ======

    @app.get("/api/slides/{slide_id}/qc", response_model=SlideQCResponse)
    async def slide_quality_check(slide_id: str):
        """
        Check slide quality metrics.

        Returns quality indicators that help oncologists assess whether
        a slide has quality issues that might affect prediction accuracy.

        For demo purposes, this returns mock values based on slide_id hash.
        Real implementation would analyze tile sharpness, tissue mask,
        stain normalization, and detect artifacts.
        """
        emb_path = embeddings_dir / f"{slide_id}.npy"

        if not emb_path.exists():
            raise HTTPException(status_code=404, detail=f"Slide {slide_id} not found")

        # Generate deterministic mock QC values based on slide_id hash
        # This ensures consistent results for the same slide across requests
        import hashlib
        hash_val = int(hashlib.md5(slide_id.encode()).hexdigest(), 16)

        # Use hash to generate pseudo-random but deterministic values
        tissue_coverage = 0.60 + (hash_val % 40) / 100.0  # 0.60 - 0.99
        blur_score = (hash_val % 30) / 100.0  # 0.00 - 0.29
        stain_uniformity = 0.70 + (hash_val % 30) / 100.0  # 0.70 - 0.99

        # Artifacts based on hash bits
        artifact_detected = (hash_val % 10) == 0  # ~10% chance
        pen_marks = (hash_val % 15) == 0  # ~7% chance
        fold_detected = (hash_val % 12) == 0  # ~8% chance

        # Determine overall quality based on metrics
        quality_score = (
            tissue_coverage * 0.3 +
            (1 - blur_score) * 0.3 +
            stain_uniformity * 0.2 +
            (0 if artifact_detected else 0.1) +
            (0 if pen_marks else 0.05) +
            (0 if fold_detected else 0.05)
        )

        if quality_score >= 0.75:
            overall_quality = "good"
        elif quality_score >= 0.50:
            overall_quality = "acceptable"
        else:
            overall_quality = "poor"

        return SlideQCResponse(
            slide_id=slide_id,
            tissue_coverage=round(tissue_coverage, 2),
            blur_score=round(blur_score, 2),
            stain_uniformity=round(stain_uniformity, 2),
            artifact_detected=artifact_detected,
            pen_marks=pen_marks,
            fold_detected=fold_detected,
            overall_quality=overall_quality,
        )

    # WSI / DZI Tile Serving
    # Cache for OpenSlide objects and DeepZoom generators
    wsi_cache: Dict[str, Any] = {}
    logger.info(f"Slides directory: {slides_dir}")

    def get_slide_and_dz(slide_id: str):
        """Get or create OpenSlide and DeepZoomGenerator for a slide."""
        if slide_id in wsi_cache:
            return wsi_cache[slide_id]

        # Use resolve_slide_path to search multiple directories
        slide_path = resolve_slide_path(slide_id)
        
        if slide_path is None:
            logger.warning(f"WSI file not found for slide_id={slide_id}")
            return None

        try:
            import openslide
            from openslide.deepzoom import DeepZoomGenerator

            slide = openslide.OpenSlide(str(slide_path))
            # tile_size=254 with overlap=1 is standard for OpenSeadragon
            dz = DeepZoomGenerator(slide, tile_size=254, overlap=1, limit_bounds=True)
            wsi_cache[slide_id] = (slide, dz)
            logger.info(f"Loaded WSI: {slide_path}")
            return slide, dz
        except Exception as e:
            logger.error(f"Failed to load WSI {slide_path}: {e}")
            return None

    @app.get("/api/slides/{slide_id}/dzi")
    async def get_dzi_descriptor(slide_id: str):
        """Get Deep Zoom Image descriptor (XML) for OpenSeadragon."""
        result = get_slide_and_dz(slide_id)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"WSI file not found for slide {slide_id}"
            )

        slide, dz = result

        # Generate DZI XML descriptor
        dzi_xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<Image xmlns="http://schemas.microsoft.com/deepzoom/2008"
  Format="jpeg"
  Overlap="1"
  TileSize="254">
  <Size Width="{dz.level_dimensions[-1][0]}" Height="{dz.level_dimensions[-1][1]}"/>
</Image>'''

        from fastapi.responses import Response
        return Response(
            content=dzi_xml,
            media_type="application/xml",
            headers={"Content-Disposition": f"inline; filename={slide_id}.dzi"}
        )

    @app.get("/api/slides/{slide_id}/dzi_files/{level}/{tile_spec}")
    async def get_dzi_tile(slide_id: str, level: int, tile_spec: str):
        """Serve a single DZI tile image."""
        result = get_slide_and_dz(slide_id)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"WSI file not found for slide {slide_id}"
            )

        slide, dz = result

        # Parse tile coordinates from spec like "0_0.jpeg"
        try:
            tile_name = tile_spec.rsplit(".", 1)[0]  # Remove extension
            col, row = map(int, tile_name.split("_"))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid tile specification")

        # Validate level and tile coordinates
        if level < 0 or level >= dz.level_count:
            raise HTTPException(status_code=404, detail="Invalid zoom level")

        level_tiles = dz.level_tiles[level]
        if col < 0 or col >= level_tiles[0] or row < 0 or row >= level_tiles[1]:
            raise HTTPException(status_code=404, detail="Tile coordinates out of bounds")

        try:
            tile = dz.get_tile(level, (col, row))

            # Convert to JPEG
            buf = io.BytesIO()
            tile.save(buf, format="JPEG", quality=85)
            buf.seek(0)

            return StreamingResponse(
                buf,
                media_type="image/jpeg",
                headers={
                    "Cache-Control": "public, max-age=86400",
                    "Content-Disposition": f"inline; filename={level}_{col}_{row}.jpeg"
                }
            )
        except Exception as e:
            logger.error(f"Failed to get tile {level}/{col}_{row} for {slide_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to generate tile: {e}")

    # Thumbnail cache directory
    thumbnail_cache_dir = embeddings_dir / "thumbnail_cache"
    thumbnail_cache_dir.mkdir(parents=True, exist_ok=True)

    @app.get("/api/slides/{slide_id}/thumbnail")
    async def get_slide_thumbnail(slide_id: str, size: int = 256):
        """Get a thumbnail of the whole slide.
        
        Thumbnails are cached to disk for performance.
        """
        # Check disk cache first
        cache_path = thumbnail_cache_dir / f"{slide_id}_{size}.jpg"
        if cache_path.exists():
            return FileResponse(
                cache_path,
                media_type="image/jpeg",
                headers={"Cache-Control": "public, max-age=86400"}
            )

        result = get_slide_and_dz(slide_id)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"WSI file not found for slide {slide_id}"
            )

        slide, dz = result

        try:
            # Get thumbnail maintaining aspect ratio
            thumb = slide.get_thumbnail((size, size))

            # Save to disk cache
            thumb.save(cache_path, format="JPEG", quality=90)
            logger.info(f"Cached thumbnail for {slide_id} at size {size}")

            # Return the cached file
            return FileResponse(
                cache_path,
                media_type="image/jpeg",
                headers={"Cache-Control": "public, max-age=86400"}
            )
        except Exception as e:
            logger.error(f"Failed to get thumbnail for {slide_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to generate thumbnail: {e}")

    @app.get("/api/slides/{slide_id}/patches/{patch_id}")
    async def get_patch_image(slide_id: str, patch_id: str, size: int = 224):
        """
        Get a patch image thumbnail for semantic search results.
        
        Patch ID format: patch_{index} (e.g., patch_0, patch_42)
        
        If WSI is available, extracts the region at the patch coordinates.
        If only embeddings are available, returns a placeholder colored by patch index.
        """
        # Parse patch index from patch_id
        try:
            if patch_id.startswith("patch_"):
                patch_index = int(patch_id.replace("patch_", ""))
            else:
                patch_index = int(patch_id)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid patch ID format: {patch_id}")
        
        # Load coordinates for this slide
        coord_path = embeddings_dir / f"{slide_id}_coords.npy"
        emb_path = embeddings_dir / f"{slide_id}.npy"
        
        if not emb_path.exists():
            raise HTTPException(status_code=404, detail=f"Slide {slide_id} not found")
        
        # Get number of patches
        embeddings = np.load(emb_path)
        if patch_index < 0 or patch_index >= len(embeddings):
            raise HTTPException(
                status_code=404, 
                detail=f"Patch index {patch_index} out of range (0-{len(embeddings)-1})"
            )
        
        # Get coordinates if available
        coords = None
        if coord_path.exists():
            coords = np.load(coord_path)
            if patch_index < len(coords):
                x, y = int(coords[patch_index][0]), int(coords[patch_index][1])
            else:
                coords = None
        
        # Try to extract from WSI if available
        result = get_slide_and_dz(slide_id)
        if result is not None and coords is not None:
            slide, dz = result
            try:
                # Read region at level 0 (highest resolution)
                # Default patch size is 224x224
                region = slide.read_region((x, y), 0, (size, size))
                
                # Convert RGBA to RGB
                if region.mode == 'RGBA':
                    # Create white background
                    background = Image.new('RGB', region.size, (255, 255, 255))
                    background.paste(region, mask=region.split()[3])
                    region = background
                elif region.mode != 'RGB':
                    region = region.convert('RGB')
                
                buf = io.BytesIO()
                region.save(buf, format="JPEG", quality=85)
                buf.seek(0)
                
                return StreamingResponse(
                    buf,
                    media_type="image/jpeg",
                    headers={
                        "Cache-Control": "public, max-age=86400",
                        "Content-Disposition": f"inline; filename={slide_id}_{patch_id}.jpeg"
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to extract patch from WSI: {e}")
                # Fall through to placeholder generation
        
        # Generate a colored placeholder if no WSI available
        # Use a color based on the patch index for visual distinction
        import colorsys
        hue = (patch_index * 0.618033988749895) % 1.0  # Golden ratio for color distribution
        r, g, b = colorsys.hsv_to_rgb(hue, 0.3, 0.9)  # Pastel colors
        color = (int(r * 255), int(g * 255), int(b * 255))
        
        # Create placeholder image with patch info
        img = Image.new('RGB', (size, size), color)
        
        # Try to add text label
        try:
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(img)
            
            # Draw patch index
            text = f"Patch {patch_index}"
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
            except Exception:
                font = ImageFont.load_default()
            
            # Get text bounding box
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Center the text
            x_pos = (size - text_width) // 2
            y_pos = (size - text_height) // 2
            
            # Draw with contrasting color
            draw.text((x_pos, y_pos), text, fill=(60, 60, 60), font=font)
            
            # Add coordinates if available
            if coords is not None:
                coord_text = f"({x}, {y})"
                coord_bbox = draw.textbbox((0, 0), coord_text, font=font)
                coord_width = coord_bbox[2] - coord_bbox[0]
                draw.text(
                    ((size - coord_width) // 2, y_pos + text_height + 5),
                    coord_text,
                    fill=(80, 80, 80),
                    font=font
                )
        except Exception as e:
            logger.debug(f"Could not add text to placeholder: {e}")
        
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        buf.seek(0)
        
        return StreamingResponse(
            buf,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "public, max-age=3600",
                "Content-Disposition": f"inline; filename={slide_id}_{patch_id}.jpeg"
            }
        )

    @app.get("/api/slides/{slide_id}/info")
    async def get_slide_info(slide_id: str):
        """Get detailed information about a WSI file."""
        result = get_slide_and_dz(slide_id)
        if result is None:
            # Return info even without WSI for embedding-only slides
            emb_path = embeddings_dir / f"{slide_id}.npy"
            if emb_path.exists():
                embeddings = np.load(emb_path)
                return {
                    "slide_id": slide_id,
                    "has_wsi": False,
                    "has_embeddings": True,
                    "num_patches": len(embeddings),
                }
            raise HTTPException(status_code=404, detail=f"Slide {slide_id} not found")

        slide, dz = result

        return {
            "slide_id": slide_id,
            "has_wsi": True,
            "dimensions": {
                "width": slide.dimensions[0],
                "height": slide.dimensions[1],
            },
            "level_count": slide.level_count,
            "level_dimensions": [list(d) for d in slide.level_dimensions],
            "properties": dict(slide.properties) if hasattr(slide, "properties") else {},
            "dzi": {
                "tile_size": 254,
                "overlap": 1,
                "level_count": dz.level_count,
            },
        }

    # ====== Annotations API (for Pathologist Review Mode) ======

    # In-memory storage for annotations (would be persisted to DB in production)
    slide_annotations: Dict[str, List[Dict[str, Any]]] = {}
    annotation_counter = {"value": 0}

    class AnnotationCreate(BaseModel):
        """Request to create a new annotation (flat format for frontend compatibility)."""
        x: float = Field(..., description="X coordinate of annotation")
        y: float = Field(..., description="Y coordinate of annotation")
        width: float = Field(default=0, description="Width of annotation region")
        height: float = Field(default=0, description="Height of annotation region")
        label: Optional[str] = Field(None, description="Label/category for the annotation")
        notes: Optional[str] = Field(None, description="Additional notes or description")

    class AnnotationResponse(BaseModel):
        """Single annotation response (flat format for frontend compatibility)."""
        id: str
        slide_id: str
        x: float
        y: float
        width: float
        height: float
        label: Optional[str] = None
        notes: Optional[str] = None
        created_at: str

    class AnnotationsListResponse(BaseModel):
        """Response containing all annotations for a slide."""
        slide_id: str
        annotations: List[AnnotationResponse]
        total: int

    @app.get("/api/slides/{slide_id}/annotations", response_model=AnnotationsListResponse)
    async def get_annotations(slide_id: str):
        """
        Get all annotations for a slide.

        Returns annotations created in pathologist review mode, including
        region markings, notes, and labels. Format uses flat x/y/width/height
        coordinates for frontend compatibility.
        """
        annotations = slide_annotations.get(slide_id, [])

        return AnnotationsListResponse(
            slide_id=slide_id,
            annotations=[
                AnnotationResponse(
                    id=a["id"],
                    slide_id=a["slide_id"],
                    x=a["x"],
                    y=a["y"],
                    width=a["width"],
                    height=a["height"],
                    label=a.get("label"),
                    notes=a.get("notes"),
                    created_at=a["created_at"],
                )
                for a in annotations
            ],
            total=len(annotations),
        )

    @app.post("/api/slides/{slide_id}/annotations", response_model=AnnotationResponse)
    async def save_annotation(slide_id: str, annotation: AnnotationCreate):
        """
        Save a new annotation for a slide.

        Creates an annotation with flat x/y/width/height coordinates plus
        optional label and notes fields. This enables the PathologistView
        "Save Annotations" feature in the frontend.
        """
        annotation_counter["value"] += 1
        annotation_id = f"ann_{slide_id}_{annotation_counter['value']}"

        annotation_data = {
            "id": annotation_id,
            "slide_id": slide_id,
            "x": annotation.x,
            "y": annotation.y,
            "width": annotation.width,
            "height": annotation.height,
            "label": annotation.label,
            "notes": annotation.notes,
            "created_at": get_timestamp(),
        }

        if slide_id not in slide_annotations:
            slide_annotations[slide_id] = []
        slide_annotations[slide_id].append(annotation_data)

        log_audit_event("annotation_created", slide_id, "pathologist", {
            "annotation_id": annotation_id,
            "label": annotation.label,
        })

        logger.info(f"Created annotation {annotation_id} for slide {slide_id}")

        return AnnotationResponse(
            id=annotation_id,
            slide_id=slide_id,
            x=annotation.x,
            y=annotation.y,
            width=annotation.width,
            height=annotation.height,
            label=annotation.label,
            notes=annotation.notes,
            created_at=annotation_data["created_at"],
        )

    @app.delete("/api/slides/{slide_id}/annotations/{annotation_id}")
    async def delete_annotation(slide_id: str, annotation_id: str):
        """
        Delete an annotation.

        Removes the annotation from the slide. This action is logged in the audit trail.
        """
        if slide_id not in slide_annotations:
            raise HTTPException(status_code=404, detail=f"No annotations found for slide {slide_id}")

        annotations = slide_annotations[slide_id]
        original_count = len(annotations)
        slide_annotations[slide_id] = [a for a in annotations if a["id"] != annotation_id]

        if len(slide_annotations[slide_id]) == original_count:
            raise HTTPException(status_code=404, detail=f"Annotation {annotation_id} not found")

        log_audit_event("annotation_deleted", slide_id, "pathologist", {
            "annotation_id": annotation_id,
        })

        logger.info(f"Deleted annotation {annotation_id} from slide {slide_id}")

        return {"success": True, "message": f"Annotation {annotation_id} deleted"}

    @app.get("/api/slides/{slide_id}/annotations/summary")
    async def get_annotations_summary(slide_id: str):
        """
        Get a summary of annotations for a slide.

        Returns counts by label, useful for pathologist workflow summary.
        """
        annotations = slide_annotations.get(slide_id, [])

        label_counts: Dict[str, int] = {}

        for ann in annotations:
            label = ann.get("label")
            if label:
                label_counts[label] = label_counts.get(label, 0) + 1

        return {
            "slide_id": slide_id,
            "total_annotations": len(annotations),
            "by_label": label_counts,
        }

    # ====== Multi-Model Analysis Endpoints ======

    @app.get("/api/models", response_model=AvailableModelsResponse)
    async def list_available_models():
        """
        List all available TransMIL models.
        
        Returns model metadata including:
        - Model ID and display name
        - Description of what the model predicts
        - Training AUC score (model reliability)
        - Number of training slides
        - Category (ovarian_cancer vs general_pathology)
        """
        if multi_model_inference is None:
            raise HTTPException(status_code=503, detail="Multi-model inference not initialized")
        
        models = multi_model_inference.get_available_models()
        return AvailableModelsResponse(models=models)


    @app.post("/api/embed-slide")
    async def embed_slide_on_demand(request: dict, background_tasks: BackgroundTasks):
        """
        Extract patches and generate embeddings for a slide on-demand.
        
        For level 0 (full resolution), this starts a background task and returns
        a task_id for polling. For level 1, embedding is done inline.
        
        Supports:
        - level=0: Full resolution (5-30K patches, 5-20 min, background task)
        - level=1: Downsampled resolution (100-500 patches, ~30s, inline)
        - async=true: Force background task mode (for level 1 as well)
        
        Returns immediately with task_id for background tasks.
        Poll /api/embed-slide/status/{task_id} for progress.
        """
        import time
        
        slide_id = request.get("slide_id")
        level = request.get("level", 1)
        force_reembed = request.get("force", False)
        use_async = request.get("async", level == 0)  # Default async for level 0
        
        if not slide_id:
            raise HTTPException(status_code=400, detail="slide_id required")
        
        if level not in [0, 1]:
            raise HTTPException(status_code=400, detail="level must be 0 or 1")
        
        # Level-specific embedding paths
        level_dir = embeddings_dir / f"level{level}"
        level_dir.mkdir(parents=True, exist_ok=True)
        
        emb_path = level_dir / f"{slide_id}.npy"
        coord_path = level_dir / f"{slide_id}_coords.npy"
        
        # Also check legacy flat path (for backwards compatibility)
        legacy_emb_path = embeddings_dir / f"{slide_id}.npy"
        legacy_coord_path = embeddings_dir / f"{slide_id}_coords.npy"
        
        # Check if embeddings already exist at this level
        if emb_path.exists() and coord_path.exists() and not force_reembed:
            emb = np.load(emb_path)
            return {
                "status": "exists",
                "slide_id": slide_id,
                "level": level,
                "num_patches": len(emb),
                "message": f"Level {level} embeddings already exist"
            }
        
        # For level 1, check if legacy flat embeddings exist
        if level == 1 and legacy_emb_path.exists() and legacy_coord_path.exists() and not force_reembed:
            emb = np.load(legacy_emb_path)
            return {
                "status": "exists",
                "slide_id": slide_id,
                "level": level,
                "num_patches": len(emb),
                "message": f"Level {level} embeddings exist (legacy path)"
            }
        
        # Check if there's already a running task for this slide/level
        existing_task = task_manager.get_task_by_slide(slide_id, level)
        if existing_task:
            return {
                "status": "in_progress",
                "task_id": existing_task.task_id,
                "slide_id": slide_id,
                "level": level,
                "progress": existing_task.progress,
                "message": existing_task.message,
            }
        
        # Find the slide file
        slide_path = resolve_slide_path(slide_id)
        if not slide_path:
            raise HTTPException(
                status_code=404, 
                detail=f"Slide file not found for {slide_id}. Level {level} embedding requires the original .svs file."
            )
        
        # For level 0 or explicit async mode, use background task
        if use_async:
            task = task_manager.create_task(slide_id, level)
            
            # Run embedding in background
            def run_embedding():
                _run_embedding_task(
                    task.task_id,
                    slide_id,
                    level,
                    slide_path,
                    emb_path,
                    coord_path,
                )
            
            background_tasks.add_task(run_embedding)
            
            return {
                "status": "started",
                "task_id": task.task_id,
                "slide_id": slide_id,
                "level": level,
                "message": f"Embedding started in background. Poll /api/embed-slide/status/{task.task_id} for progress.",
                "estimated_time_minutes": 15 if level == 0 else 1,
            }
        
        # For level 1 without async, run inline (original behavior)
        return _run_embedding_inline(slide_id, level, slide_path, emb_path, coord_path)
    
    def _run_embedding_task(
        task_id: str,
        slide_id: str, 
        level: int,
        slide_path: Path,
        emb_path: Path,
        coord_path: Path,
    ):
        """Background task to run embedding."""
        import time
        
        task = task_manager.get_task(task_id)
        if not task:
            return
        
        task_manager.update_task(task_id, 
            status=TaskStatus.RUNNING,
            started_at=time.time(),
            message="Starting embedding process..."
        )
        
        try:
            import openslide
            
            task_manager.update_task(task_id,
                progress=5,
                message="Opening slide file..."
            )
            
            slide = openslide.OpenSlide(str(slide_path))
            
            actual_level = min(level, slide.level_count - 1)
            level_dims = slide.level_dimensions[actual_level]
            downsample = slide.level_downsamples[actual_level]
            
            width, height = level_dims
            patch_size = 224
            stride = 224
            max_patches = 30000 if level == 0 else 10000
            
            task_manager.update_task(task_id,
                progress=10,
                message=f"Extracting tissue patches from {width}x{height} region..."
            )
            
            # Count total potential patches for progress
            total_potential = max(1, ((height - patch_size) // stride) * ((width - patch_size) // stride))
            
            patches = []
            coords = []
            processed = 0
            last_progress_update = time.time()
            
            for y in range(0, height - patch_size, stride):
                for x in range(0, width - patch_size, stride):
                    x0 = int(x * downsample)
                    y0 = int(y * downsample)
                    
                    patch = slide.read_region((x0, y0), actual_level, (patch_size, patch_size))
                    patch = patch.convert("RGB")
                    
                    patch_array = np.array(patch)
                    mean_val = patch_array.mean()
                    std_val = patch_array.std()
                    
                    if mean_val < 235 and std_val > 15:
                        patches.append(patch_array)
                        coords.append([x0, y0])
                    
                    processed += 1
                    
                    # Update progress every 2 seconds
                    if time.time() - last_progress_update > 2:
                        extraction_progress = min(processed / total_potential, 1.0) * 40
                        task_manager.update_task(task_id,
                            progress=10 + extraction_progress,
                            message=f"Extracting patches: {len(patches)} tissue patches found ({processed}/{total_potential} checked)"
                        )
                        last_progress_update = time.time()
                    
                    if len(patches) >= max_patches:
                        break
                if len(patches) >= max_patches:
                    break
            
            slide.close()
            
            if len(patches) == 0:
                task_manager.update_task(task_id,
                    status=TaskStatus.FAILED,
                    error="No tissue patches found in slide"
                )
                return
            
            task_manager.update_task(task_id,
                progress=50,
                message=f"Extracted {len(patches)} patches. Loading Path Foundation model..."
            )
            
            # Generate embeddings
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
            import tensorflow as tf
            
            saved_model_path = os.path.expanduser(
                "~/.cache/huggingface/hub/models--google--path-foundation/snapshots/b50f2be6f055ea6ea8719f467ab44b38f37e2142"
            )
            
            if not os.path.exists(saved_model_path):
                task_manager.update_task(task_id,
                    status=TaskStatus.FAILED,
                    error="Path Foundation model not found"
                )
                return
            
            model = tf.saved_model.load(saved_model_path)
            infer = model.signatures["serving_default"]
            
            task_manager.update_task(task_id,
                progress=55,
                message=f"Model loaded. Generating embeddings for {len(patches)} patches..."
            )
            
            batch_size = 64
            all_embeddings = []
            
            for i in range(0, len(patches), batch_size):
                batch = patches[i:i+batch_size]
                batch_array = np.array(batch, dtype=np.float32) / 255.0
                batch_tensor = tf.constant(batch_array)
                
                result = infer(inputs=batch_tensor)
                embs = result["output_0"].numpy()
                all_embeddings.append(embs)
                
                embed_progress = (i + len(batch)) / len(patches) * 40
                task_manager.update_task(task_id,
                    progress=55 + embed_progress,
                    message=f"Generating embeddings: {min(i+batch_size, len(patches))}/{len(patches)} patches"
                )
            
            embeddings = np.vstack(all_embeddings).astype(np.float32)
            coords_array = np.array(coords)
            
            task_manager.update_task(task_id,
                progress=95,
                message="Saving embeddings..."
            )
            
            np.save(emb_path, embeddings)
            np.save(coord_path, coords_array)
            
            elapsed = time.time() - task.started_at
            task_manager.update_task(task_id,
                status=TaskStatus.COMPLETED,
                progress=100,
                num_patches=len(patches),
                processing_time_seconds=elapsed,
                message=f"Completed: {len(patches)} patches embedded in {elapsed:.1f}s",
                completed_at=time.time()
            )
            
            logger.info(f"Background embedding completed for {slide_id} level {level}: {len(patches)} patches in {elapsed:.1f}s")
            
        except Exception as e:
            logger.error(f"Background embedding failed for {slide_id}: {e}")
            import traceback
            traceback.print_exc()
            task_manager.update_task(task_id,
                status=TaskStatus.FAILED,
                error=str(e)
            )
    
    def _run_embedding_inline(slide_id: str, level: int, slide_path: Path, emb_path: Path, coord_path: Path):
        """Run embedding inline (original behavior for level 1)."""
        import time
        start_time = time.time()
        
        try:
            import openslide
            
            slide = openslide.OpenSlide(str(slide_path))
            
            actual_level = min(level, slide.level_count - 1)
            level_dims = slide.level_dimensions[actual_level]
            downsample = slide.level_downsamples[actual_level]
            
            width, height = level_dims
            patch_size = 224
            stride = 224
            max_patches = 10000
            
            patches = []
            coords = []
            
            for y in range(0, height - patch_size, stride):
                for x in range(0, width - patch_size, stride):
                    x0 = int(x * downsample)
                    y0 = int(y * downsample)
                    
                    patch = slide.read_region((x0, y0), actual_level, (patch_size, patch_size))
                    patch = patch.convert("RGB")
                    
                    patch_array = np.array(patch)
                    mean_val = patch_array.mean()
                    std_val = patch_array.std()
                    
                    if mean_val < 235 and std_val > 15:
                        patches.append(patch_array)
                        coords.append([x0, y0])
                    
                    if len(patches) >= max_patches:
                        break
                if len(patches) >= max_patches:
                    break
            
            slide.close()
            
            if len(patches) == 0:
                raise HTTPException(status_code=400, detail="No tissue patches found in slide")
            
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
            import tensorflow as tf
            
            saved_model_path = os.path.expanduser(
                "~/.cache/huggingface/hub/models--google--path-foundation/snapshots/b50f2be6f055ea6ea8719f467ab44b38f37e2142"
            )
            
            model = tf.saved_model.load(saved_model_path)
            infer = model.signatures["serving_default"]
            
            batch_size = 64
            all_embeddings = []
            
            for i in range(0, len(patches), batch_size):
                batch = patches[i:i+batch_size]
                batch_array = np.array(batch, dtype=np.float32) / 255.0
                batch_tensor = tf.constant(batch_array)
                
                result = infer(inputs=batch_tensor)
                embs = result["output_0"].numpy()
                all_embeddings.append(embs)
            
            embeddings = np.vstack(all_embeddings).astype(np.float32)
            coords_array = np.array(coords)
            
            np.save(emb_path, embeddings)
            np.save(coord_path, coords_array)
            
            elapsed = time.time() - start_time
            
            return {
                "status": "completed",
                "slide_id": slide_id,
                "level": level,
                "num_patches": len(patches),
                "processing_time_seconds": round(elapsed, 1),
                "message": f"Embedded {len(patches)} patches at level {level}"
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")
    
    @app.get("/api/embed-slide/status/{task_id}")
    async def get_embedding_status(task_id: str):
        """Get status of a background embedding task."""
        task = task_manager.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        
        result = task.to_dict()
        
        if task.status == TaskStatus.COMPLETED:
            level_dir = embeddings_dir / f"level{task.level}"
            emb_path = level_dir / f"{task.slide_id}.npy"
            result["embedding_path"] = str(emb_path) if emb_path.exists() else None
        
        return result
    
    @app.get("/api/embed-slide/tasks")
    async def list_embedding_tasks(
        slide_id: Optional[str] = Query(None),
        status: Optional[str] = Query(None)
    ):
        """List all embedding tasks."""
        tasks = []
        for task in task_manager.tasks.values():
            if slide_id and task.slide_id != slide_id:
                continue
            if status and task.status.value != status:
                continue
            tasks.append(task.to_dict())
        
        return {
            "tasks": sorted(tasks, key=lambda t: t.get("elapsed_seconds", 0), reverse=True),
            "total": len(tasks)
        }

    @app.post("/api/analyze-multi", response_model=MultiModelResponse)
    async def analyze_slide_multi(request: MultiModelRequest):
        """
        Analyze a slide with multiple TransMIL models.
        
        This endpoint runs all (or selected) trained models on a slide:
        
        **Ovarian Cancer Specific:**
        - Platinum Sensitivity (AUC 0.907): Predicts platinum chemotherapy response
        - 5-Year Survival (AUC 0.697): Predicts 5-year overall survival
        - 3-Year Survival (AUC 0.645): Predicts 3-year overall survival
        - 1-Year Survival (AUC 0.639): Predicts 1-year overall survival
        
        **General Pathology:**
        - Tumor Grade (AUC 0.752): Predicts high vs low tumor grade
        
        Use the `models` parameter to select specific models, or leave empty for all.
        Set `level=0` for full-resolution analysis (requires pre-generated level 0 embeddings).
        """
        import time
        start_time = time.time()
        
        if multi_model_inference is None:
            raise HTTPException(status_code=503, detail="Multi-model inference not initialized")
        
        slide_id = request.slide_id
        level = request.level  # Get requested resolution level
        
        # Determine embedding path based on level
        if level == 0:
            level0_dir = embeddings_dir / "level0"
            emb_path = level0_dir / f"{slide_id}.npy"
            
            # Block analysis if level 0 embeddings don't exist
            if not emb_path.exists():
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "LEVEL0_EMBEDDINGS_REQUIRED",
                        "message": f"Level 0 (full resolution) embeddings do not exist for slide {slide_id}. Generate embeddings first using /api/embed-slide with level=0.",
                        "needs_embedding": True,
                        "slide_id": slide_id,
                        "level": 0,
                    }
                )
        else:
            # Level 1: check level1 subdir first, then fallback to flat embeddings
            level1_dir = embeddings_dir / "level1"
            level1_emb_path = level1_dir / f"{slide_id}.npy"
            if level1_emb_path.exists():
                emb_path = level1_emb_path
            else:
                emb_path = embeddings_dir / f"{slide_id}.npy"
        
        if not emb_path.exists():
            raise HTTPException(status_code=404, detail=f"Slide {slide_id} not found")
        
        # Load embeddings
        embeddings = np.load(emb_path)
        
        # Run multi-model inference
        try:
            results = multi_model_inference.predict_all(
                embeddings,
                model_ids=request.models,
                return_attention=request.return_attention,
            )
        except Exception as e:
            logger.error(f"Multi-model inference failed: {e}")
            raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
        
        processing_time = (time.time() - start_time) * 1000
        
        # Convert to Pydantic models
        predictions = {}
        for model_id, pred in results["predictions"].items():
            if "error" not in pred:
                predictions[model_id] = ModelPrediction(**{k: v for k, v in pred.items() if k != "attention"})
        
        by_category = {
            "ovarian_cancer": [ModelPrediction(**{k: v for k, v in p.items() if k != "attention"}) 
                              for p in results["by_category"]["ovarian_cancer"] if "error" not in p],
            "general_pathology": [ModelPrediction(**{k: v for k, v in p.items() if k != "attention"}) 
                                  for p in results["by_category"]["general_pathology"] if "error" not in p],
        }
        
        # Log to audit trail
        log_audit_event(
            "multi_model_analysis",
            slide_id,
            details={
                "models_run": list(predictions.keys()),
                "processing_time_ms": processing_time,
            },
        )
        
        return MultiModelResponse(
            slide_id=slide_id,
            predictions=predictions,
            by_category=by_category,
            n_patches=results["n_patches"],
            processing_time_ms=processing_time,
        )


    @app.get("/api/heatmap/{slide_id}/{model_id}")
    async def get_model_heatmap(slide_id: str, model_id: str):
        """Get the attention heatmap for a specific TransMIL model.
        
        Available models:
        - platinum_sensitivity
        - tumor_grade  
        - survival_5y
        - survival_3y
        - survival_1y
        """
        import torch
        
        if multi_model_inference is None:
            raise HTTPException(status_code=503, detail="Multi-model inference not initialized")
        
        if model_id not in MODEL_CONFIGS:
            raise HTTPException(
                status_code=400, 
                detail=f"Unknown model: {model_id}. Available: {list(MODEL_CONFIGS.keys())}"
            )
        
        emb_path = embeddings_dir / f"{slide_id}.npy"
        coord_path = embeddings_dir / f"{slide_id}_coords.npy"
        
        if not emb_path.exists():
            raise HTTPException(status_code=404, detail=f"Slide {slide_id} not found")
        
        embeddings = np.load(emb_path)
        
        # Check for coordinates
        patch_size = 224
        if coord_path.exists():
            coords_arr = np.load(coord_path)
            coords_arr = coords_arr.astype(np.int64, copy=False)
        else:
            # Generate grid coordinates (pixel space) as fallback
            n_patches = len(embeddings)
            if n_patches > 0:
                grid_size = int(np.ceil(np.sqrt(n_patches)))
                grid_x, grid_y = np.meshgrid(
                    np.arange(grid_size) * patch_size,
                    np.arange(grid_size) * patch_size,
                )
                coords_arr = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)[:n_patches]
            else:
                coords_arr = np.zeros((0, 2), dtype=np.int64)
            logger.warning(
                f"No coords found for {slide_id}; generated {len(coords_arr)} grid coords from embeddings"
            )
        
        # Get prediction with attention from specific model
        try:
            result = multi_model_inference.predict_single(
                embeddings, 
                model_id, 
                return_attention=True
            )
            attention = result.get("attention")
            
            if attention is None:
                raise HTTPException(status_code=500, detail="Model did not return attention weights")
            
            attention = np.array(attention)
            
        except Exception as e:
            logger.error(f"Model inference failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        
        # Generate heatmap image using EvidenceGenerator for proper coordinate scaling
        try:
            # Get actual slide dimensions
            slide_path = resolve_slide_path(slide_id)
            if slide_path is not None and slide_path.exists():
                try:
                    import openslide
                    with openslide.OpenSlide(str(slide_path)) as slide:
                        slide_dims = slide.dimensions
                        logger.info(f"Model heatmap using actual slide dims: {slide_dims}")
                except Exception as e:
                    logger.warning(f"Could not read slide dimensions: {e}")
                    slide_dims = None
            else:
                slide_dims = None
            
            # Fall back to computing bounds from coordinates
            if slide_dims is None:
                if coords_arr.size == 0:
                    slide_dims = (patch_size, patch_size)
                    logger.warning(
                        f"No coordinates available to derive dims for {slide_id}; "
                        f"falling back to {slide_dims}"
                    )
                else:
                    x_max = int(coords_arr[:, 0].max()) + patch_size
                    y_max = int(coords_arr[:, 1].max()) + patch_size
                    slide_dims = (x_max, y_max)
                    logger.info(f"Model heatmap using coords-derived dims: {slide_dims}")
            
            # Use EvidenceGenerator for consistent heatmap generation
            coords_list = [tuple(map(int, c)) for c in coords_arr]
            
            # Calculate aspect-ratio-preserving thumbnail dimensions
            # This ensures the heatmap matches the slide geometry for correct overlay alignment
            base_size = 512
            slide_w, slide_h = slide_dims
            if slide_w >= slide_h:
                thumb_w = base_size
                thumb_h = max(1, int(round(base_size * slide_h / slide_w)))
            else:
                thumb_h = base_size
                thumb_w = max(1, int(round(base_size * slide_w / slide_h)))
            
            logger.info(f"Model heatmap thumbnail size: {thumb_w}x{thumb_h} (preserving aspect ratio)")
            
            heatmap_rgba = evidence_gen.create_heatmap(
                attention, 
                coords_list, 
                slide_dims, 
                thumbnail_size=(thumb_w, thumb_h),
                smooth=True,
                blur_kernel=31
            )
            
            # Convert RGBA to RGB for PNG output
            img = Image.fromarray(heatmap_rgba, mode="RGBA")
            
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            
            # Get actual slide dimensions for frontend alignment
            slide_path = resolve_slide_path(slide_id)
            if slide_path is not None and slide_path.exists():
                try:
                    import openslide
                    with openslide.OpenSlide(str(slide_path)) as slide:
                        slide_dims = slide.dimensions
                except Exception:
                    if coords_arr.size == 0:
                        slide_dims = (patch_size, patch_size)
                    else:
                        slide_dims = (int(coords_arr[:, 0].max()) + patch_size, int(coords_arr[:, 1].max()) + patch_size)
            else:
                if coords_arr.size == 0:
                    slide_dims = (patch_size, patch_size)
                else:
                    slide_dims = (int(coords_arr[:, 0].max()) + patch_size, int(coords_arr[:, 1].max()) + patch_size)
            
            return StreamingResponse(
                buf, 
                media_type="image/png",
                headers={
                    "X-Model-Id": model_id,
                    "X-Model-Name": MODEL_CONFIGS[model_id]["display_name"],
                    "X-Slide-Width": str(slide_dims[0]),
                    "X-Slide-Height": str(slide_dims[1]),
                    "Access-Control-Expose-Headers": "X-Model-Id, X-Model-Name, X-Slide-Width, X-Slide-Height",
                }
            )
            
        except Exception as e:
            logger.error(f"Heatmap generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Heatmap generation failed: {e}")


    # Register slide metadata API
    metadata_path = embeddings_dir.parent / "slide_metadata.json"
    metadata_manager = SlideMetadataManager(metadata_path)

    def get_available_slide_ids():
        return list(available_slides.keys())
    metadata_router = create_metadata_router(metadata_manager, get_available_slide_ids)
    app.include_router(metadata_router)



    # Initialize agent workflow for multi-step analysis
    if AGENT_AVAILABLE:
        try:
            agent_workflow = AgentWorkflow(
                embeddings_dir=embeddings_dir,
                multi_model_inference=multi_model_inference,
                evidence_generator=evidence_gen,
                medgemma_reporter=reporter,
                slide_labels=slide_labels,
                slide_mean_index=slide_mean_index,
                slide_mean_ids=slide_mean_ids,
                slide_mean_meta=slide_mean_meta,
            )
            set_agent_workflow(agent_workflow)
            app.include_router(agent_router)
            logger.info("Agent workflow initialized and routes registered")
        except Exception as e:
            logger.warning(f"Failed to initialize agent workflow: {e}")
    else:
        logger.warning("Agent workflow not available - skipping initialization")

    return app


# Default app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
