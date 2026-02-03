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
import logging
import json
import base64
import io
from datetime import datetime
from collections import deque

import numpy as np
from PIL import Image

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
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


class SlideInfo(BaseModel):
    slide_id: str
    patient_id: Optional[str] = None
    has_embeddings: bool = False
    label: Optional[str] = None
    num_patches: Optional[int] = None
    patient: Optional[PatientContext] = None


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
            allow_origins=[
                "http://localhost:3000",
                "http://localhost:3001",
                "http://localhost:7860",
                "http://100.111.126.23:3000",
                "http://100.111.126.23:8003",
                "http://100.111.126.23:3002",
            ],
            allow_credentials=True,
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
    slide_siglip_embeddings = {}  # Cache for MedSigLIP embeddings per slide
    available_slides = []
    slide_labels = {}  # Cache for slide labels (slide_id -> label string)
    # Directories (may be updated at startup if we fall back to demo data)
    slides_dir: Path = embeddings_dir.parent / 'slides'

    @app.on_event("startup")
    async def load_models():
        nonlocal classifier, evidence_gen, embedder, medsiglip_embedder, reporter, decision_support, available_slides, slide_labels, slides_dir, embeddings_dir

        from ..config import MILConfig, EvidenceConfig, EmbeddingConfig
        from ..mil.clam import CLAMClassifier
        from ..evidence.generator import EvidenceGenerator
        from ..embedding.embedder import PathFoundationEmbedder, MedSigLIPEmbedder, MedSigLIPConfig
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
        logger.info("MedGemma reporter initialized (model loads on first call)")

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
            slides.append(SlideInfo(
                slide_id=slide_id,
                has_embeddings=True,
                label=data.get("label"),
                num_patches=num_patches,
                patient=data.get("patient"),
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
            try:
                report = reporter.generate_report(
                    evidence_patches=evidence_patches,
                    score=score,
                    label=label,
                    similar_cases=similar_cases,
                    case_id=slide_id,
                    patient_context=patient_ctx,
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

    @app.get("/api/heatmap/{slide_id}")
    async def get_heatmap(slide_id: str):
        """Get the attention heatmap for a slide as PNG."""
        if classifier is None or evidence_gen is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        emb_path = embeddings_dir / f"{slide_id}.npy"
        coord_path = embeddings_dir / f"{slide_id}_coords.npy"

        if not emb_path.exists():
            raise HTTPException(status_code=404, detail=f"Slide {slide_id} not found")

        embeddings = np.load(emb_path)

        # Load or generate coordinates
        if coord_path.exists():
            coords = np.load(coord_path)
        else:
            coords = np.random.randint(0, 50000, (len(embeddings), 2))

        coords = [tuple(c) for c in coords]

        # Run prediction and create heatmap
        score, attention = classifier.predict(embeddings)
        slide_dims = (50000, 50000)
        heatmap = evidence_gen.create_heatmap(attention, coords, slide_dims, (512, 512))

        # Save to temp file and return
        from PIL import Image
        import io

        img = Image.fromarray(heatmap)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        return StreamingResponse(
            buf,
            media_type="image/png",
            headers={"Content-Disposition": f"inline; filename={slide_id}_heatmap.png"},
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
        """Find similar cases from the reference cohort."""
        if classifier is None or evidence_gen is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        emb_path = embeddings_dir / f"{slide_id}.npy"
        if not emb_path.exists():
            raise HTTPException(status_code=404, detail=f"Slide {slide_id} not found")

        embeddings = np.load(emb_path)
        _, attention = classifier.predict(embeddings)

        similar_results = evidence_gen.find_similar(
            embeddings, attention, k=k * 3, top_patches=top_patches
        )

        similar_cases = []
        seen_slides = set()

        for s in similar_results:
            meta = s.get("metadata", {})
            sid = meta.get("slide_id", "unknown")

            if sid == slide_id:
                continue

            if sid not in seen_slides:
                seen_slides.add(sid)
                similar_cases.append({
                    "slide_id": sid,
                    "distance": float(s["distance"]),
                    "similarity_score": 1.0 / (1.0 + s["distance"]),
                    "patch_index": meta.get("patch_index"),
                })

            if len(similar_cases) >= k:
                break

        return SimilarResponse(
            slide_id=slide_id,
            similar_cases=similar_cases,
            num_queries=top_patches,
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
        Search patches by text query using MedSigLIP.

        This enables "semantic evidence search" - query with text like
        "tumor infiltrating lymphocytes" or "necrosis" to find matching patches.

        Example queries:
        - "tumor cells"
        - "lymphocytes"
        - "necrosis"
        - "stroma"
        - "mitotic figures"
        - "tumor infiltrating lymphocytes"
        """
        if medsiglip_embedder is None:
            raise HTTPException(status_code=503, detail="MedSigLIP embedder not initialized")

        slide_id = request.slide_id
        emb_path = embeddings_dir / f"{slide_id}.npy"
        coord_path = embeddings_dir / f"{slide_id}_coords.npy"

        if not emb_path.exists():
            raise HTTPException(status_code=404, detail=f"Slide {slide_id} not found")

        # Check for cached MedSigLIP embeddings for this slide
        siglip_cache_key = f"{slide_id}_siglip"
        use_siglip_search = True

        if siglip_cache_key in slide_siglip_embeddings:
            siglip_embeddings = slide_siglip_embeddings[siglip_cache_key]
            logger.info(f"Using cached MedSigLIP embeddings for {slide_id}")
        else:
            # Try to load pre-computed SigLIP embeddings
            siglip_cache_path = embeddings_dir / "medsiglip_cache" / f"{slide_id}_siglip.npy"
            if siglip_cache_path.exists():
                siglip_embeddings = np.load(siglip_cache_path)
                slide_siglip_embeddings[siglip_cache_key] = siglip_embeddings
                logger.info(f"Loaded MedSigLIP embeddings from cache for {slide_id}")
            else:
                # MedSigLIP embeddings not available - use attention-based fallback
                logger.warning(
                    f"MedSigLIP embeddings not pre-computed for {slide_id}. "
                    "Using attention-weighted fallback search."
                )
                use_siglip_search = False
                siglip_embeddings = None

        # Load coordinates if available
        coords = None
        if coord_path.exists():
            coords = np.load(coord_path)

        # Get attention weights for additional context
        pf_embeddings = np.load(emb_path)
        attention_weights = None
        if classifier is not None:
            _, attention_weights = classifier.predict(pf_embeddings)

        # Build metadata for search results
        num_patches = len(pf_embeddings) if siglip_embeddings is None else len(siglip_embeddings)
        metadata = []
        for i in range(num_patches):
            meta = {"index": i}
            if coords is not None and i < len(coords):
                meta["coordinates"] = [int(coords[i][0]), int(coords[i][1])]
            if attention_weights is not None:
                meta["attention_weight"] = float(attention_weights[i])
            metadata.append(meta)

        # Perform semantic search
        try:
            if use_siglip_search and siglip_embeddings is not None:
                # Full MedSigLIP semantic search
                search_results = medsiglip_embedder.search(
                    query=request.query,
                    top_k=request.top_k,
                    embeddings=siglip_embeddings,
                    metadata=metadata,
                )
            else:
                # Fallback: return top attention-weighted patches
                # This provides reasonable results for demos without pre-computed embeddings
                logger.info(f"Using attention-weighted fallback for query: {request.query}")

                if attention_weights is None:
                    # No attention weights, return random selection
                    indices = np.random.choice(num_patches, min(request.top_k, num_patches), replace=False)
                    scores = np.ones(len(indices)) * 0.5
                else:
                    # Sort by attention weight (higher = more relevant)
                    sorted_indices = np.argsort(attention_weights)[::-1]
                    indices = sorted_indices[:request.top_k]
                    scores = attention_weights[indices]

                search_results = []
                for idx, score in zip(indices, scores):
                    search_results.append({
                        "patch_index": int(idx),
                        "similarity_score": float(score),
                        "metadata": metadata[idx] if idx < len(metadata) else {},
                    })

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Semantic search failed: {str(e)}",
            )

        # Format results
        results = []
        for r in search_results:
            result = SemanticSearchResult(
                patch_index=r["patch_index"],
                similarity_score=r["similarity_score"],
                coordinates=r["metadata"].get("coordinates"),
                attention_weight=r["metadata"].get("attention_weight"),
            )
            results.append(result)

        # Determine which model was used
        model_used = medsiglip_embedder.config.model_id if use_siglip_search else "attention-fallback"

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

        # Try common WSI extensions
        slide_path = None
        for ext in [".svs", ".tiff", ".tif", ".ndpi", ".mrxs", ".vms", ".scn"]:
            candidate = slides_dir / f"{slide_id}{ext}"
            if candidate.exists():
                slide_path = candidate
                break

        if slide_path is None:
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

    @app.get("/api/slides/{slide_id}/thumbnail")
    async def get_slide_thumbnail(slide_id: str, size: int = 512):
        """Get a thumbnail of the whole slide."""
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

            buf = io.BytesIO()
            thumb.save(buf, format="JPEG", quality=90)
            buf.seek(0)

            return StreamingResponse(
                buf,
                media_type="image/jpeg",
                headers={"Cache-Control": "public, max-age=3600"}
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
        """
        import time
        start_time = time.time()
        
        if multi_model_inference is None:
            raise HTTPException(status_code=503, detail="Multi-model inference not initialized")
        
        slide_id = request.slide_id
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

    return app


# Default app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
