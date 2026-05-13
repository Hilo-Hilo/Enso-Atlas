"""Pydantic request/response schemas for the Enso Atlas API.

Keeping schemas outside route registration makes endpoint modules easier to
split, test, and reuse without importing the full FastAPI application factory.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AnalyzeRequest(BaseModel):
    slide_id: str = Field(..., min_length=1, max_length=256)
    generate_report: bool = False
    project_id: str | None = Field(None, description="Project ID to scope embeddings lookup")


class AnalyzeResponse(BaseModel):
    slide_id: str
    prediction: str
    score: float
    confidence: float
    patches_analyzed: int
    top_evidence: list[dict[str, Any]]
    similar_cases: list[dict[str, Any]]


class UncertaintyRequest(BaseModel):
    """Request for analysis with uncertainty quantification."""

    slide_id: str = Field(..., min_length=1, max_length=256)
    n_samples: int = Field(
        default=20,
        ge=5,
        le=50,
        description="Number of MC Dropout samples (5-50, default 20)",
    )


class UncertaintyResponse(BaseModel):
    """Response with MC Dropout uncertainty quantification."""

    slide_id: str
    prediction: str
    probability: float
    uncertainty: float
    confidence_interval: list[float]
    is_uncertain: bool
    requires_review: bool
    uncertainty_level: str
    clinical_recommendation: str
    patches_analyzed: int
    n_samples: int
    samples: list[float]
    top_evidence: list[dict[str, Any]]


class ReportRequest(BaseModel):
    slide_id: str = Field(..., min_length=1, max_length=256)
    include_evidence: bool = True
    include_similar: bool = True
    project_id: str | None = Field(
        default=None, description="Project ID to determine cancer type for report"
    )


class ReportResponse(BaseModel):
    slide_id: str
    report_json: dict[str, Any]
    summary_text: str


class PatientContext(BaseModel):
    """Patient demographic and clinical context for a slide."""

    age: int | None = None
    sex: str | None = None
    stage: str | None = None
    grade: str | None = None
    prior_lines: int | None = None
    histology: str | None = None


class SlideDimensions(BaseModel):
    width: int = 0
    height: int = 0


class SlideInfo(BaseModel):
    slide_id: str
    patient_id: str | None = None
    has_wsi: bool = False
    has_embeddings: bool = False
    has_level0_embeddings: bool = False
    label: str | None = None
    num_patches: int | None = None
    patient: PatientContext | None = None
    dimensions: SlideDimensions = SlideDimensions()
    mpp: float | None = None
    magnification: str | None = "40x"


class SlideRenameRequest(BaseModel):
    """Request to update a slide display name."""

    display_name: str | None = None


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

    patches: list[str] = Field(
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
    embeddings: list[list[float]] | None = None


class EmbedSlideRequest(BaseModel):
    """Request for project-aware on-demand slide embedding."""

    slide_id: str = Field(..., min_length=1, max_length=256)
    level: int = Field(
        default=0, ge=0, le=0, description="Resolution level fixed to 0 (dense full-resolution)"
    )
    force: bool = Field(default=False, description="Force re-embedding even if cached")
    async_mode: bool = Field(default=True, alias="async", description="Run embedding in background")
    project_id: str | None = Field(
        default=None, description="Project ID to scope slide + embeddings paths"
    )


class SimilarRequest(BaseModel):
    """Request for similar case search."""

    slide_id: str = Field(..., min_length=1, max_length=256)
    k: int = Field(default=5, ge=1, le=20)
    top_patches: int = Field(default=3, ge=1, le=10)


class BatchAnalyzeRequest(BaseModel):
    """Request for batch analysis of multiple slides."""

    slide_ids: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of slide IDs to analyze (1-100 slides)",
    )
    project_id: str | None = Field(default=None, description="Project ID to scope analysis")


class BatchAnalysisResult(BaseModel):
    """Result for a single slide in batch analysis."""

    slide_id: str
    prediction: str
    score: float
    confidence: float
    patches_analyzed: int
    requires_review: bool
    uncertainty_level: str = "unknown"
    error: str | None = None


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

    results: list[BatchAnalysisResult]
    summary: BatchAnalysisSummary
    processing_time_ms: float


class ClassifyRegionRequest(BaseModel):
    """Request for tissue region classification."""

    x: int = Field(..., description="X coordinate of the region")
    y: int = Field(..., description="Y coordinate of the region")
    patch_index: int | None = Field(
        None, description="Optional patch index for deterministic classification"
    )


class ClassifyRegionResponse(BaseModel):
    """Response from tissue region classification."""

    tissue_type: str
    confidence: float
    description: str


class SimilarResponse(BaseModel):
    """Response from similar case search."""

    slide_id: str
    similar_cases: list[dict[str, Any]]
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

    analyses: list[AnalysisHistoryEntry]
    total: int


class AuditLogEntry(BaseModel):
    """Single audit log entry."""

    timestamp: str
    event_type: str
    user_id: str
    slide_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class AuditLogResponse(BaseModel):
    """Response containing audit log entries."""

    entries: list[AuditLogEntry]
    total: int


class SemanticSearchRequest(BaseModel):
    """Request for text-to-patch semantic search using MedSigLIP."""

    slide_id: str = Field(..., min_length=1, max_length=256)
    query: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="Text query (e.g., 'tumor cells', 'lymphocytes')",
    )
    top_k: int = Field(default=10, ge=1, le=50, description="Number of patches to return")
    project_id: str | None = Field(
        default=None, description="Project ID to scope embeddings lookup"
    )


class SemanticSearchResult(BaseModel):
    """Single result from semantic search."""

    patch_index: int
    similarity_score: float
    coordinates: list[int] | None = None
    patch_size: int | None = None
    attention_weight: float | None = None


class SemanticSearchResponse(BaseModel):
    """Response from semantic search endpoint."""

    slide_id: str
    query: str
    results: list[SemanticSearchResult]
    embedding_model: str = "siglip-so400m"


class VisualSearchRequest(BaseModel):
    """Request for image-to-image visual similarity search using FAISS."""

    slide_id: str | None = Field(
        None, max_length=256, description="Source slide ID to look up patch embedding"
    )
    patch_index: int | None = Field(
        None, ge=0, description="Index of the patch in the source slide"
    )
    coordinates: list[int] | None = Field(None, description="[x, y] coordinates of the patch")
    patch_embedding: list[float] | None = Field(
        None, description="Direct embedding vector (384-dim)"
    )
    top_k: int = Field(default=10, ge=1, le=50, description="Number of similar patches to return")
    exclude_same_slide: bool = Field(
        default=True, description="Exclude patches from the same slide"
    )
    project_id: str | None = Field(
        default=None, description="Project ID to scope visual search candidates"
    )


class VisualSearchResultPatch(BaseModel):
    """Single similar patch result from visual search."""

    slide_id: str
    patch_index: int
    coordinates: list[int] | None = None
    distance: float
    similarity: float
    label: str | None = None
    thumbnail_url: str | None = None


class VisualSearchResponse(BaseModel):
    """Response from visual similarity search."""

    query_slide_id: str | None = None
    query_patch_index: int | None = None
    query_coordinates: list[int] | None = None
    results: list[VisualSearchResultPatch]
    total_patches_searched: int
    search_time_ms: float


class ModelPrediction(BaseModel):
    """Single model prediction result."""

    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    model_name: str
    category: str
    score: float
    decision_threshold: float = 0.5
    label: str
    positive_label: str
    negative_label: str
    confidence: float
    auc: float
    n_training_slides: int
    description: str
    warning: str | None = None


class MultiModelRequest(BaseModel):
    """Request for multi-model analysis."""

    model_config = ConfigDict(protected_namespaces=())

    slide_id: str = Field(..., min_length=1, max_length=256)
    models: list[str] | None = None
    project_id: str | None = Field(
        default=None, description="Project ID to scope models to project's classification_models"
    )
    return_attention: bool = False
    level: int = Field(
        default=0, ge=0, le=0, description="Resolution level is fixed to 0 (full resolution, dense)"
    )
    force: bool = Field(default=False, description="Bypass cache and force fresh analysis")


class MultiModelResponse(BaseModel):
    """Response with predictions from multiple models."""

    slide_id: str
    predictions: dict[str, ModelPrediction]
    by_category: dict[str, list[ModelPrediction]]
    n_patches: int
    processing_time_ms: float
    warnings: list[str] | None = None


class PdfExportRequest(BaseModel):
    """Request for PDF report export."""

    slide_id: str = Field(..., min_length=1, max_length=256)
    report_data: dict[str, Any] = Field(..., description="Structured report from MedGemma")
    prediction_data: dict[str, Any] = Field(..., description="Model prediction results")
    include_heatmap: bool = Field(default=True, description="Include attention heatmap image")
    include_evidence_patches: bool = Field(
        default=True, description="Include evidence patch images"
    )
    patient_context: dict[str, Any] | None = Field(
        default=None, description="Patient demographic info"
    )


class ReportPdfRequest(BaseModel):
    """Request body for the lightweight /api/report/pdf endpoint."""

    report: dict[str, Any] = Field(..., description="Report JSON from /api/report")
    case_id: str | None = Field(
        default=None, description="Case identifier (falls back to report.case_id)"
    )


class AvailableModelsResponse(BaseModel):
    """Response listing available models."""

    models: list[dict[str, Any]]


class PatchClassifyRequest(BaseModel):
    """Request for few-shot patch classification."""

    classes: dict[str, list[int]]


class AnnotationCreate(BaseModel):
    """Request to create a new annotation."""

    type: str = Field(
        default="rectangle",
        description="Annotation type: circle, rectangle, freehand, point, marker, note, measurement",
    )
    coordinates: dict[str, Any] = Field(
        default_factory=lambda: {"x": 0, "y": 0, "width": 0, "height": 0},
        description="Coordinates in image space",
    )
    text: str | None = Field(None, description="Annotation text (mapped to notes)")
    label: str | None = Field(None, description="Label/category for the annotation")
    notes: str | None = Field(None, description="Additional notes or description")
    color: str | None = Field(None, description="Display color")
    category: str | None = Field(None, description="Category: mitotic, tumor, stroma, etc.")


class AnnotationUpdate(BaseModel):
    """Request to update an annotation."""

    label: str | None = None
    notes: str | None = None
    color: str | None = None
    category: str | None = None


class PatchClassificationItem(BaseModel):
    """Single patch classification result."""

    patch_idx: int
    x: int
    y: int
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]


class PatchClassifyResponse(BaseModel):
    """Response from few-shot patch classification."""

    slide_id: str
    classes: list[str]
    total_patches: int
    predictions: list[PatchClassificationItem]
    class_counts: dict[str, int]
    accuracy_estimate: float | None
    heatmap_data: list[dict[str, Any]]


class OutlierPatch(BaseModel):
    """Single outlier patch result."""

    patch_idx: int
    x: int
    y: int
    distance: float
    z_score: float


class OutlierDetectionResponse(BaseModel):
    """Response from outlier tissue detection."""

    slide_id: str
    outlier_patches: list[OutlierPatch]
    total_patches: int
    outlier_count: int
    mean_distance: float
    std_distance: float
    threshold: float
    heatmap_data: list[dict[str, Any]]
