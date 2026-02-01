// Enso Atlas - API Client
// Backend communication utilities

import type {
  AnalysisRequest,
  AnalysisResponse,
  ReportRequest,
  SlideInfo,
  SlidesListResponse,
  StructuredReport,
  ApiError,
  SemanticSearchResponse,
  SlideQCMetrics,
  UncertaintyResult,
  Annotation,
  AnnotationRequest,
  AnnotationsResponse,
} from "@/types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Custom error class for API errors
export class AtlasApiError extends Error {
  code: string;
  details?: Record<string, unknown>;

  constructor(error: ApiError) {
    super(error.message);
    this.name = "AtlasApiError";
    this.code = error.code;
    this.details = error.details;
  }
}

// Generic fetch wrapper with error handling
async function fetchApi<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;

  const defaultHeaders: HeadersInit = {
    "Content-Type": "application/json",
  };

  const response = await fetch(url, {
    ...options,
    headers: {
      ...defaultHeaders,
      ...options.headers,
    },
  });

  if (!response.ok) {
    let errorData: ApiError;
    try {
      errorData = await response.json();
    } catch {
      errorData = {
        code: "UNKNOWN_ERROR",
        message: `HTTP ${response.status}: ${response.statusText}`,
      };
    }
    throw new AtlasApiError(errorData);
  }

  return response.json();
}

// API Client functions

// Backend patient context (snake_case from Python)
interface BackendPatientContext {
  age?: number;
  sex?: string;
  stage?: string;
  grade?: string;
  prior_lines?: number;
  histology?: string;
}

// Backend slide info (different from frontend type)
interface BackendSlideInfo {
  slide_id: string;
  patient_id?: string;
  has_embeddings: boolean;
  label?: string;
  num_patches?: number;
  patient?: BackendPatientContext;
}

/**
 * Fetch list of available slides
 */
export async function getSlides(): Promise<SlidesListResponse> {
  // Backend returns array with different schema, adapt it
  const slides = await fetchApi<BackendSlideInfo[]>("/api/slides");
  return {
    slides: slides.map(s => ({
      id: s.slide_id,
      filename: `${s.slide_id}.svs`,
      dimensions: { width: 0, height: 0 },
      magnification: 40,
      mpp: 0.25,
      createdAt: new Date().toISOString(),
      // Extended fields from backend
      label: s.label,
      hasEmbeddings: s.has_embeddings,
      numPatches: s.num_patches,
      // Patient context
      patient: s.patient ? {
        age: s.patient.age,
        sex: s.patient.sex,
        stage: s.patient.stage,
        grade: s.patient.grade,
        prior_lines: s.patient.prior_lines,
        histology: s.patient.histology,
      } : undefined,
    })),
    total: slides.length,
  };
}

/**
 * Get details for a specific slide
 */
export async function getSlide(slideId: string): Promise<SlideInfo> {
  return fetchApi<SlideInfo>(`/api/slides/${slideId}`);
}

// Backend analysis response
interface BackendAnalysisResponse {
  slide_id: string;
  prediction: string;
  score: number;
  confidence: number;
  patches_analyzed: number;
  top_evidence: Array<{
    rank: number;
    patch_index: number;
    attention_weight: number;
    coordinates?: [number, number];
    tissue_type?: string;
    tissue_confidence?: number;
  }>;
  similar_cases: Array<{
    slide_id: string;
    similarity_score: number;
    distance?: number;
  }>;
}

/**
 * Analyze a slide with the pathology model
 */
export async function analyzeSlide(
  request: AnalysisRequest
): Promise<AnalysisResponse> {
  const backend = await fetchApi<BackendAnalysisResponse>("/api/analyze", {
    method: "POST",
    body: JSON.stringify({ slide_id: request.slideId }),
  });
  
  // Adapt backend response to frontend format
  return {
    slideInfo: {
      id: backend.slide_id,
      filename: `${backend.slide_id}.svs`,
      dimensions: { width: 0, height: 0 },
      magnification: 40,
      mpp: 0.25,
      createdAt: new Date().toISOString(),
    },
    prediction: {
      label: backend.prediction,
      score: backend.score,
      confidence: backend.confidence,
      calibrationNote: "Model probability requires external validation.",
    },
    evidencePatches: backend.top_evidence.map(e => ({
      id: `patch_${e.patch_index}`,
      patchId: `patch_${e.patch_index}`,
      coordinates: {
        x: e.coordinates?.[0] ?? 0,
        y: e.coordinates?.[1] ?? 0,
        width: 224,
        height: 224,
        level: 0,
      },
      attentionWeight: e.attention_weight,
      thumbnailUrl: "",
      rank: e.rank,
      tissueType: e.tissue_type as import("@/types").TissueType | undefined,
      tissueConfidence: e.tissue_confidence,
    })),
    similarCases: backend.similar_cases.map(s => ({
      slideId: s.slide_id,
      similarity: s.similarity_score,
      label: "unknown",
      thumbnailUrl: `/api/heatmap/${s.slide_id}`,
    })),
    heatmap: {
      imageUrl: `/api/heatmap/${backend.slide_id}`,
      minValue: 0,
      maxValue: 1,
      colorScale: "viridis",
    },
    processingTimeMs: 0,
  };
}

/**
 * Generate a structured report for a slide
 */
export async function generateReport(
  request: ReportRequest
): Promise<StructuredReport> {
  return fetchApi<StructuredReport>("/api/report", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

/**
 * Get Deep Zoom Image (DZI) metadata for OpenSeadragon
 */
export function getDziUrl(slideId: string): string {
  return `${API_BASE_URL}/api/slides/${slideId}/dzi`;
}

/**
 * Get heatmap overlay image URL
 */
export function getHeatmapUrl(slideId: string): string {
  return `${API_BASE_URL}/api/slides/${slideId}/heatmap`;
}

/**
 * Get thumbnail URL for a slide
 */
export function getThumbnailUrl(slideId: string): string {
  return `${API_BASE_URL}/api/slides/${slideId}/thumbnail`;
}

/**
 * Get patch image URL
 */
export function getPatchUrl(slideId: string, patchId: string): string {
  return `${API_BASE_URL}/api/slides/${slideId}/patches/${patchId}`;
}

/**
 * Export report as PDF
 */
export async function exportReportPdf(slideId: string): Promise<Blob> {
  const response = await fetch(
    `${API_BASE_URL}/api/slides/${slideId}/report/pdf`,
    {
      method: "GET",
    }
  );

  if (!response.ok) {
    throw new AtlasApiError({
      code: "EXPORT_FAILED",
      message: "Failed to export report as PDF",
    });
  }

  return response.blob();
}

/**
 * Export report as JSON
 */
export async function exportReportJson(
  slideId: string
): Promise<StructuredReport> {
  return fetchApi<StructuredReport>(`/api/slides/${slideId}/report/json`);
}

/**
 * Health check for the backend API
 */
export async function healthCheck(): Promise<{ status: string; version: string }> {
  return fetchApi<{ status: string; version: string }>("/api/health");
}

/**
 * Semantic search for patches using MedSigLIP text embeddings
 */
export async function semanticSearch(
  slideId: string,
  query: string,
  topK: number = 5
): Promise<SemanticSearchResponse> {
  return fetchApi<SemanticSearchResponse>("/api/semantic-search", {
    method: "POST",
    body: JSON.stringify({
      slide_id: slideId,
      query,
      top_k: topK,
    }),
  });
}

// Backend QC response (snake_case)
interface BackendSlideQCResponse {
  slide_id: string;
  tissue_coverage: number;
  blur_score: number;
  stain_uniformity: number;
  artifact_detected: boolean;
  pen_marks: boolean;
  fold_detected: boolean;
  overall_quality: "poor" | "acceptable" | "good";
}

/**
 * Get quality control metrics for a slide
 */
export async function getSlideQC(slideId: string): Promise<SlideQCMetrics> {
  const backend = await fetchApi<BackendSlideQCResponse>(
    `/api/slides/${slideId}/qc`
  );

  return {
    slideId: backend.slide_id,
    tissueCoverage: backend.tissue_coverage,
    blurScore: backend.blur_score,
    stainUniformity: backend.stain_uniformity,
    artifactDetected: backend.artifact_detected,
    penMarks: backend.pen_marks,
    foldDetected: backend.fold_detected,
    overallQuality: backend.overall_quality,
  };
}

// Backend uncertainty response (snake_case)
interface BackendUncertaintyResponse {
  slide_id: string;
  prediction: string;
  probability: number;
  uncertainty: number;
  confidence_interval: [number, number];
  is_uncertain: boolean;
  requires_review: boolean;
  uncertainty_level: "low" | "moderate" | "high";
  clinical_recommendation: string;
  patches_analyzed: number;
  n_samples: number;
  samples: number[];
  top_evidence: Array<{
    rank: number;
    patch_index: number;
    attention_weight: number;
    attention_uncertainty: number;
    coordinates: [number, number];
  }>;
}

/**
 * Analyze a slide with MC Dropout uncertainty quantification
 */
export async function analyzeWithUncertainty(
  slideId: string,
  nSamples: number = 20
): Promise<UncertaintyResult> {
  const backend = await fetchApi<BackendUncertaintyResponse>(
    "/api/analyze-uncertainty",
    {
      method: "POST",
      body: JSON.stringify({
        slide_id: slideId,
        n_samples: nSamples,
      }),
    }
  );

  return {
    slideId: backend.slide_id,
    prediction: backend.prediction,
    probability: backend.probability,
    uncertainty: backend.uncertainty,
    confidenceInterval: backend.confidence_interval,
    isUncertain: backend.is_uncertain,
    requiresReview: backend.requires_review,
    uncertaintyLevel: backend.uncertainty_level,
    clinicalRecommendation: backend.clinical_recommendation,
    patchesAnalyzed: backend.patches_analyzed,
    nSamples: backend.n_samples,
    samples: backend.samples,
    topEvidence: backend.top_evidence.map((e) => ({
      rank: e.rank,
      patchIndex: e.patch_index,
      attentionWeight: e.attention_weight,
      attentionUncertainty: e.attention_uncertainty,
      coordinates: e.coordinates,
    })),
  };
}

// ====== Annotations API ======

// Backend annotation response (snake_case)
interface BackendAnnotation {
  id: string;
  slide_id: string;
  type: string;
  coordinates: {
    x: number;
    y: number;
    width: number;
    height: number;
    points?: Array<{ x: number; y: number }>;
  };
  text?: string;
  color?: string;
  category?: string;
  created_at: string;
  created_by?: string;
}

interface BackendAnnotationsResponse {
  slide_id: string;
  annotations: BackendAnnotation[];
  total: number;
}

/**
 * Get all annotations for a slide
 */
export async function getAnnotations(slideId: string): Promise<AnnotationsResponse> {
  const backend = await fetchApi<BackendAnnotationsResponse>(
    `/api/slides/${slideId}/annotations`
  );

  return {
    slideId: backend.slide_id,
    annotations: backend.annotations.map((a) => ({
      id: a.id,
      slideId: a.slide_id,
      type: a.type as Annotation["type"],
      coordinates: a.coordinates,
      text: a.text,
      color: a.color,
      category: a.category,
      createdAt: a.created_at,
      createdBy: a.created_by,
    })),
    total: backend.total,
  };
}

/**
 * Save a new annotation for a slide
 */
export async function saveAnnotation(
  slideId: string,
  annotation: AnnotationRequest
): Promise<Annotation> {
  const backend = await fetchApi<BackendAnnotation>(
    `/api/slides/${slideId}/annotations`,
    {
      method: "POST",
      body: JSON.stringify({
        type: annotation.type,
        coordinates: annotation.coordinates,
        text: annotation.text,
        color: annotation.color,
        category: annotation.category,
      }),
    }
  );

  return {
    id: backend.id,
    slideId: backend.slide_id,
    type: backend.type as Annotation["type"],
    coordinates: backend.coordinates,
    text: backend.text,
    color: backend.color,
    category: backend.category,
    createdAt: backend.created_at,
    createdBy: backend.created_by,
  };
}

/**
 * Delete an annotation
 */
export async function deleteAnnotation(
  slideId: string,
  annotationId: string
): Promise<void> {
  await fetchApi<{ success: boolean }>(
    `/api/slides/${slideId}/annotations/${annotationId}`,
    {
      method: "DELETE",
    }
  );
}
