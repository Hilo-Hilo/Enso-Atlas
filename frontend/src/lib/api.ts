// Enso Atlas - API Client
// Backend communication utilities with robust error handling

import type {
  Tag,
  Group,
  SlideFilters,
  SlideSearchResult,
  ExtendedSlideInfo,
  AnalysisRequest,
  AnalysisResponse,
  ReportRequest,
  SlideInfo,
  SlidesListResponse,
  StructuredReport,
  PatientContext,
  ApiError,
  SemanticSearchResponse,
  SlideQCMetrics,
  UncertaintyResult,
  Annotation,
  AnnotationRequest,
  AnnotationsResponse,
  BatchAnalyzeRequest,
  BatchAnalyzeResponse,
  BatchAnalysisResult,
  BatchAnalysisSummary,
  ModelPrediction,
  AvailableModel,
  MultiModelResponse,
  AvailableModelsResponse,
} from "@/types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://100.111.126.23:8003";

// Configuration for retry behavior
const RETRY_CONFIG = {
  maxRetries: 3,
  baseDelayMs: 1000,
  maxDelayMs: 10000,
  retryableStatusCodes: [408, 429, 500, 502, 503, 504],
};

// Default timeout for API requests (30 seconds)
const DEFAULT_TIMEOUT_MS = 30000;

// Custom error class for API errors
export class AtlasApiError extends Error {
  code: string;
  details?: Record<string, unknown>;
  statusCode?: number;
  isRetryable: boolean;
  isTimeout: boolean;
  isNetworkError: boolean;

  constructor(error: ApiError & { statusCode?: number; isTimeout?: boolean; isNetworkError?: boolean }) {
    super(error.message);
    this.name = "AtlasApiError";
    this.code = error.code;
    this.details = error.details;
    this.statusCode = error.statusCode;
    this.isTimeout = error.isTimeout || false;
    this.isNetworkError = error.isNetworkError || false;
    this.isRetryable = this.isTimeout || 
                       this.isNetworkError || 
                       (this.statusCode !== undefined && 
                        RETRY_CONFIG.retryableStatusCodes.includes(this.statusCode));
  }

  /**
   * Get a user-friendly error message
   */
  getUserMessage(): string {
    if (this.isTimeout) {
      return "The request timed out. Please try again.";
    }
    if (this.isNetworkError) {
      return "Unable to connect to the server. Please check your connection.";
    }
    if (this.statusCode === 429) {
      return "Too many requests. Please wait a moment and try again.";
    }
    if (this.statusCode && this.statusCode >= 500) {
      return "Server error. Our team has been notified.";
    }
    if (this.statusCode === 404) {
      return "The requested resource was not found.";
    }
    if (this.statusCode === 403) {
      return "You do not have permission to perform this action.";
    }
    if (this.statusCode === 401) {
      return "Authentication required. Please log in.";
    }
    return this.message || "An unexpected error occurred.";
  }
}

/**
 * Sleep utility for retry delays
 */
function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Calculate exponential backoff delay
 */
function getRetryDelay(attempt: number): number {
  const delay = RETRY_CONFIG.baseDelayMs * Math.pow(2, attempt);
  // Add jitter (0-25% of delay)
  const jitter = delay * Math.random() * 0.25;
  return Math.min(delay + jitter, RETRY_CONFIG.maxDelayMs);
}

/**
 * Create an AbortController with timeout
 */
function createTimeoutController(timeoutMs: number): { controller: AbortController; timeoutId: ReturnType<typeof setTimeout> } {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  return { controller, timeoutId };
}

/**
 * Generic fetch wrapper with error handling, retries, and timeout
 */
async function fetchApi<T>(
  endpoint: string,
  options: RequestInit = {},
  config: { 
    timeoutMs?: number; 
    retries?: number;
    skipRetry?: boolean;
  } = {}
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;
  const timeoutMs = config.timeoutMs || DEFAULT_TIMEOUT_MS;
  const maxRetries = config.skipRetry ? 0 : (config.retries ?? RETRY_CONFIG.maxRetries);

  const defaultHeaders: HeadersInit = {
    "Content-Type": "application/json",
  };

  let lastError: AtlasApiError | null = null;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    const { controller, timeoutId } = createTimeoutController(timeoutMs);

    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          ...defaultHeaders,
          ...options.headers,
        },
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        let errorData: ApiError;
        try {
          errorData = await response.json();
        } catch {
          errorData = {
            code: `HTTP_${response.status}`,
            message: `HTTP ${response.status}: ${response.statusText}`,
          };
        }
        
        const apiError = new AtlasApiError({
          ...errorData,
          statusCode: response.status,
        });

        // Only retry if error is retryable and we have retries left
        if (apiError.isRetryable && attempt < maxRetries) {
          lastError = apiError;
          const delay = getRetryDelay(attempt);
          console.warn(`[API] Retryable error on ${endpoint}, attempt ${attempt + 1}/${maxRetries + 1}, retrying in ${delay}ms...`);
          await sleep(delay);
          continue;
        }

        throw apiError;
      }

      return response.json();
    } catch (error) {
      clearTimeout(timeoutId);

      // Handle AbortController timeout
      if (error instanceof DOMException && error.name === "AbortError") {
        const timeoutError = new AtlasApiError({
          code: "TIMEOUT",
          message: `Request to ${endpoint} timed out after ${timeoutMs}ms`,
          isTimeout: true,
        });

        if (attempt < maxRetries) {
          lastError = timeoutError;
          const delay = getRetryDelay(attempt);
          console.warn(`[API] Timeout on ${endpoint}, attempt ${attempt + 1}/${maxRetries + 1}, retrying in ${delay}ms...`);
          await sleep(delay);
          continue;
        }

        throw timeoutError;
      }

      // Handle network errors
      if (error instanceof TypeError && error.message.includes("fetch")) {
        const networkError = new AtlasApiError({
          code: "NETWORK_ERROR",
          message: "Failed to connect to the server. Please check your network connection.",
          isNetworkError: true,
        });

        if (attempt < maxRetries) {
          lastError = networkError;
          const delay = getRetryDelay(attempt);
          console.warn(`[API] Network error on ${endpoint}, attempt ${attempt + 1}/${maxRetries + 1}, retrying in ${delay}ms...`);
          await sleep(delay);
          continue;
        }

        throw networkError;
      }

      // Re-throw AtlasApiError as-is
      if (error instanceof AtlasApiError) {
        throw error;
      }

      // Wrap unknown errors
      throw new AtlasApiError({
        code: "UNKNOWN_ERROR",
        message: error instanceof Error ? error.message : "An unknown error occurred",
      });
    }
  }

  // Should not reach here, but just in case
  throw lastError || new AtlasApiError({
    code: "UNKNOWN_ERROR",
    message: "Request failed after all retries",
  });
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
  has_level0_embeddings?: boolean;  // Whether level 0 (full res) embeddings exist
  label?: string;
  num_patches?: number;
  patient?: BackendPatientContext;
  dimensions?: { width: number; height: number };
  mpp?: number;
  magnification?: string;
}

interface BackendSlidesListResponse {
  slides: BackendSlideInfo[];
  total?: number;
  page?: number;
  per_page?: number;
}

/**
 * Fetch list of available slides
 */
export async function getSlides(params: { page?: number; perPage?: number } = {}): Promise<SlidesListResponse> {
  const query = new URLSearchParams();
  if (params.page !== undefined) query.set("page", String(params.page));
  if (params.perPage !== undefined) query.set("per_page", String(params.perPage));
  const endpoint = query.toString() ? `/api/slides?${query.toString()}` : "/api/slides";

  // Backend may return either an array or a paginated object.
  const backend = await fetchApi<BackendSlideInfo[] | BackendSlidesListResponse>(endpoint);

  const mapSlides = (items: BackendSlideInfo[]) =>
    items.map((s) => ({
      id: s.slide_id,
      filename: `${s.slide_id}.svs`,
      dimensions: s.dimensions ?? { width: 0, height: 0 },
      magnification: s.magnification ? parseInt(s.magnification.replace("x", ""), 10) : 40,
      mpp: s.mpp ?? 0.25,
      createdAt: new Date().toISOString(),
      // Extended fields from backend
      label: s.label,
      hasEmbeddings: s.has_embeddings,
      hasLevel0Embeddings: s.has_level0_embeddings ?? false, // Level 0 embedding status
      numPatches: s.num_patches,
      // Patient context
      patient: s.patient
        ? {
            age: s.patient.age,
            sex: s.patient.sex,
            stage: s.patient.stage,
            grade: s.patient.grade,
            prior_lines: s.patient.prior_lines,
            histology: s.patient.histology,
          }
        : undefined,
    }));

  if (Array.isArray(backend)) {
    const mapped = mapSlides(backend);
    if (params.page !== undefined || params.perPage !== undefined) {
      const page = params.page ?? 1;
      const perPage = params.perPage ?? mapped.length || 1;
      const start = (page - 1) * perPage;
      const end = start + perPage;
      return { slides: mapped.slice(start, end), total: mapped.length };
    }
    return { slides: mapped, total: mapped.length };
  }

  const mapped = mapSlides(backend.slides || []);
  const total = typeof backend.total === "number" ? backend.total : mapped.length;
  const hasPaginationParams = params.page !== undefined || params.perPage !== undefined;

  if (!hasPaginationParams && typeof backend.total === "number" && total > mapped.length) {
    const perPage = backend.per_page ?? mapped.length;
    const totalPages = perPage > 0 ? Math.ceil(total / perPage) : 1;

    if (totalPages > 1) {
      const pageRequests: Promise<BackendSlideInfo[] | BackendSlidesListResponse>[] = [];
      for (let page = 2; page <= totalPages; page += 1) {
        const pageParams = new URLSearchParams();
        pageParams.set("page", String(page));
        pageParams.set("per_page", String(perPage));
        pageRequests.push(
          fetchApi<BackendSlideInfo[] | BackendSlidesListResponse>(
            `/api/slides?${pageParams.toString()}`
          )
        );
      }

      const pageResponses = await Promise.all(pageRequests);
      const extraSlides: BackendSlideInfo[] = [];
      pageResponses.forEach((response) => {
        if (Array.isArray(response)) {
          extraSlides.push(...response);
        } else if (response.slides) {
          extraSlides.push(...response.slides);
        }
      });

      return {
        slides: mapped.concat(mapSlides(extraSlides)),
        total,
      };
    }
  }

  return { slides: mapped, total };
}

/**
 * Get details for a specific slide
 */
export async function getSlide(slideId: string): Promise<SlideInfo> {
  return fetchApi<SlideInfo>(`/api/slides/${encodeURIComponent(slideId)}`);
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
    label?: string;
  }>;
}

/**
 * Analyze a slide with the pathology model
 * Uses longer timeout for potentially slow inference
 */
export async function analyzeSlide(
  request: AnalysisRequest
): Promise<AnalysisResponse> {
  const backend = await fetchApi<BackendAnalysisResponse>(
    "/api/analyze",
    {
      method: "POST",
      body: JSON.stringify({ slide_id: request.slideId }),
    },
    { timeoutMs: 60000 } // 60 second timeout for analysis
  );
  
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
      distance: s.distance,
      label: s.label || undefined,
      thumbnailUrl: `${API_BASE_URL}/api/slides/${s.slide_id}/thumbnail?size=128`,
    })),
    heatmap: {
      imageUrl: `${API_BASE_URL}/api/heatmap/${backend.slide_id}`,
      minValue: 0,
      maxValue: 1,
      colorScale: "viridis",
    },
    processingTimeMs: 0,
  };
}

// Backend report response (different from frontend StructuredReport)
interface BackendReportResponse {
  slide_id: string;
  report_json: {
    case_id: string;
    task: string;
    patient_context?: {
      age?: number;
      sex?: string;
      stage?: string;
      grade?: string;
      prior_lines?: number;
      histology?: string;
    };
    model_output: {
      label: string;
      probability: number;
      calibration_note?: string;
    };
    evidence?: Array<{
      patch_id: string;
      attention_weight?: number;
      coordinates?: [number, number];
      morphology_description?: string;
      significance?: string;
    }>;
    similar_examples?: Array<{
      slide_id?: string;
      example_id?: string;
      distance?: number;
      similarity_score?: number;
      label?: string;
    }>;
    limitations?: string[];
    suggested_next_steps?: string[];
    safety_statement?: string;
    decision_support?: {
      risk_level: string;
      confidence_level: string;
      confidence_score: number;
      primary_recommendation: string;
      supporting_rationale?: string[];
      alternative_considerations?: string[];
      guideline_references?: Array<{
        source: string;
        section: string;
        recommendation: string;
        url?: string;
      }>;
      uncertainty_statement?: string;
      quality_warnings?: string[];
      suggested_workup?: string[];
      interpretation_note?: string;
      caveat?: string;
    };
  };
  summary_text: string;
}

/**
 * Generate a structured report for a slide
 * Uses longer timeout for MedGemma inference
 */
export async function generateReport(
  request: ReportRequest
): Promise<StructuredReport> {
  const backend = await fetchApi<BackendReportResponse>(
    "/api/report",
    {
      method: "POST",
      body: JSON.stringify({ slide_id: request.slideId }),
    },
    { timeoutMs: 90000 } // 90 second timeout for report generation
  );

  // Transform backend response to frontend StructuredReport format
  const reportJson = backend.report_json;
  const modelOutput = reportJson.model_output;
  
  // Transform evidence array
  const evidence = (reportJson.evidence || []).map((e, idx) => ({
    patchId: e.patch_id || `patch_${idx}`,
    coordsLevel0: (e.coordinates || [0, 0]) as [number, number],
    morphologyDescription: e.morphology_description || 
      `High-attention region at coordinates (${e.coordinates?.[0] || 0}, ${e.coordinates?.[1] || 0}) with attention weight ${(e.attention_weight || 0).toFixed(3)}`,
    whyThisPatchMatters: e.significance ||
      "This region shows morphological patterns associated with the predicted classification based on model attention analysis.",
  }));

  // Transform similar examples
  const similarExamples = (reportJson.similar_examples || []).map((s) => ({
    exampleId: s.example_id || s.slide_id || "unknown",
    label: s.label || "unknown",
    distance: s.distance ?? (1 - (s.similarity_score || 0)),
  }));

  // Build decision support if available
  const decisionSupport = reportJson.decision_support ? {
    risk_level: reportJson.decision_support.risk_level as import("@/types").RiskLevel,
    confidence_level: reportJson.decision_support.confidence_level as import("@/types").ConfidenceLevel,
    confidence_score: reportJson.decision_support.confidence_score,
    primary_recommendation: reportJson.decision_support.primary_recommendation,
    supporting_rationale: reportJson.decision_support.supporting_rationale || [],
    alternative_considerations: reportJson.decision_support.alternative_considerations || [],
    guideline_references: reportJson.decision_support.guideline_references || [],
    uncertainty_statement: reportJson.decision_support.uncertainty_statement || 
      "Prediction confidence should be interpreted in the context of slide quality and similar case evidence.",
    quality_warnings: reportJson.decision_support.quality_warnings || [],
    suggested_workup: reportJson.decision_support.suggested_workup || [],
    interpretation_note: reportJson.decision_support.interpretation_note ||
      "This is an AI-generated interpretation. Clinical correlation and expert review are essential.",
    caveat: reportJson.decision_support.caveat ||
      "This tool is for research purposes only and should not be used as the sole basis for clinical decisions.",
  } : undefined;

  // Default limitations if not provided
  const defaultLimitations = [
    "This is an uncalibrated research model - probabilities are not clinically validated",
    "Prediction is based on morphological patterns and may not capture all relevant clinical factors",
    "Model has been trained on a limited dataset and may not generalize to all populations",
    "Slide quality and tissue representation may affect prediction accuracy",
  ];

  // Default next steps if not provided
  const defaultNextSteps = [
    "Correlate findings with clinical history and other diagnostic tests",
    "Review high-attention regions identified by the model",
    "Consider molecular profiling for additional treatment guidance",
    "Discuss findings in multidisciplinary tumor board",
  ];

  return {
    caseId: reportJson.case_id || backend.slide_id,
    task: reportJson.task || "Bevacizumab treatment response prediction",
    generatedAt: new Date().toISOString(),
    patientContext: reportJson.patient_context as PatientContext | undefined,
    modelOutput: {
      label: modelOutput.label.toUpperCase(),
      score: modelOutput.probability,
      confidence: Math.abs(modelOutput.probability - 0.5) * 2,
      calibrationNote: modelOutput.calibration_note || 
        "Model probability requires external validation. This is not a clinically calibrated probability.",
    },
    evidence,
    similarExamples,
    limitations: reportJson.limitations?.length ? reportJson.limitations : defaultLimitations,
    suggestedNextSteps: reportJson.suggested_next_steps?.length ? reportJson.suggested_next_steps : defaultNextSteps,
    safetyStatement: reportJson.safety_statement || 
      "This is a research decision-support tool, not a diagnostic device. All findings must be validated by qualified pathologists and clinicians. Do not use for standalone clinical decision-making.",
    summary: backend.summary_text,
    decisionSupport,
  };
}

/**
 * Get Deep Zoom Image (DZI) metadata for OpenSeadragon
 */
export function getDziUrl(slideId: string): string {
  return `${API_BASE_URL}/api/slides/${encodeURIComponent(slideId)}/dzi`;
}

/**
 * Get heatmap overlay image URL
 * @param slideId - Slide identifier
 * @param modelId - Optional model ID for multi-model heatmaps
 * @param level - Downsample level (0-4): 0=2048px highest detail, 2=512px default, 4=128px fastest
 */
export function getHeatmapUrl(slideId: string, modelId?: string, level?: number): string {
  const levelParam = level !== undefined ? `?level=${level}` : '';
  if (modelId) {
    return `${API_BASE_URL}/api/heatmap/${encodeURIComponent(slideId)}/${encodeURIComponent(modelId)}${levelParam}`;
  }
  return `${API_BASE_URL}/api/heatmap/${encodeURIComponent(slideId)}${levelParam}`;
}

/**
 * Get thumbnail URL for a slide
 */
export function getThumbnailUrl(slideId: string): string {
  return `${API_BASE_URL}/api/slides/${encodeURIComponent(slideId)}/thumbnail`;
}

/**
 * Get patch image URL
 */
export function getPatchUrl(slideId: string, patchId: string): string {
  return `${API_BASE_URL}/api/slides/${encodeURIComponent(slideId)}/patches/${encodeURIComponent(patchId)}`;
}

/**
 * Export report as PDF
 */
export async function exportReportPdf(slideId: string): Promise<Blob> {
  const { controller, timeoutId } = createTimeoutController(60000);
  
  try {
    const response = await fetch(
      `${API_BASE_URL}/api/slides/${encodeURIComponent(slideId)}/report/pdf`,
      {
        method: "GET",
        signal: controller.signal,
      }
    );

    clearTimeout(timeoutId);

    if (!response.ok) {
      throw new AtlasApiError({
        code: "EXPORT_FAILED",
        message: "Failed to export report as PDF",
        statusCode: response.status,
      });
    }

    return response.blob();
  } catch (error) {
    clearTimeout(timeoutId);
    
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new AtlasApiError({
        code: "TIMEOUT",
        message: "PDF export timed out",
        isTimeout: true,
      });
    }
    
    if (error instanceof AtlasApiError) {
      throw error;
    }
    
    throw new AtlasApiError({
      code: "EXPORT_FAILED",
      message: error instanceof Error ? error.message : "Failed to export PDF",
    });
  }
}

/**
 * Export report as JSON
 */
export async function exportReportJson(
  slideId: string
): Promise<StructuredReport> {
  return fetchApi<StructuredReport>(`/api/slides/${encodeURIComponent(slideId)}/report/json`);
}

/**
 * Health check for the backend API
 * Uses short timeout and no retries for quick feedback
 */
export async function healthCheck(): Promise<{ status: string; version: string }> {
  return fetchApi<{ status: string; version: string }>(
    "/api/health",
    {},
    { timeoutMs: 15000, skipRetry: true }
  );
}

// Backend semantic search response (snake_case with similarity_score)
interface BackendSemanticSearchResult {
  patch_index: number;
  similarity_score: number;
  coordinates?: [number, number];
  attention_weight?: number;
}

interface BackendSemanticSearchResponse {
  slide_id: string;
  query: string;
  results: BackendSemanticSearchResult[];
  embedding_model: string;
}

/**
 * Semantic search for patches using MedSigLIP text embeddings
 */
export async function semanticSearch(
  slideId: string,
  query: string,
  topK: number = 5
): Promise<SemanticSearchResponse> {
  const backend = await fetchApi<BackendSemanticSearchResponse>(
    "/api/semantic-search",
    {
      method: "POST",
      body: JSON.stringify({
        slide_id: slideId,
        query,
        top_k: topK,
      }),
    },
    { timeoutMs: 30000 }
  );
  
  // Transform backend response to frontend format
  return {
    slide_id: backend.slide_id,
    query: backend.query,
    results: backend.results.map(r => ({
      patch_index: r.patch_index,
      similarity: r.similarity_score,  // Map similarity_score -> similarity
      coordinates: r.coordinates,
    })),
  };
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
    `/api/slides/${encodeURIComponent(slideId)}/qc`
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
 * Uses longer timeout for multiple forward passes
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
    },
    { timeoutMs: 120000 } // 2 minute timeout for uncertainty analysis
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
    `/api/slides/${encodeURIComponent(slideId)}/annotations`
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
    `/api/slides/${encodeURIComponent(slideId)}/annotations`,
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
    `/api/slides/${encodeURIComponent(slideId)}/annotations/${encodeURIComponent(annotationId)}`,
    {
      method: "DELETE",
    }
  );
}

// ====== Batch Analysis API ======

// Backend batch analysis response (snake_case)
interface BackendBatchAnalysisResult {
  slide_id: string;
  prediction: string;
  score: number;
  confidence: number;
  patches_analyzed: number;
  requires_review: boolean;
  uncertainty_level: string;
  error?: string;
}

interface BackendBatchAnalysisSummary {
  total: number;
  completed: number;
  failed: number;
  responders: number;
  non_responders: number;
  uncertain: number;
  avg_confidence: number;
  requires_review_count: number;
}

interface BackendBatchAnalyzeResponse {
  results: BackendBatchAnalysisResult[];
  summary: BackendBatchAnalysisSummary;
  processing_time_ms: number;
}

/**
 * Analyze multiple slides in batch for clinical workflow efficiency.
 * Returns individual results sorted by priority (uncertain cases first)
 * along with summary statistics.
 * 
 * Uses extended timeout for batch processing.
 */
export async function analyzeBatch(
  slideIds: string[]
): Promise<BatchAnalyzeResponse> {
  // Scale timeout based on number of slides (30s base + 10s per slide)
  const timeoutMs = Math.min(300000, 30000 + slideIds.length * 10000);
  
  const backend = await fetchApi<BackendBatchAnalyzeResponse>(
    "/api/analyze-batch",
    {
      method: "POST",
      body: JSON.stringify({
        slide_ids: slideIds,
      }),
    },
    { timeoutMs }
  );

  return {
    results: backend.results.map((r) => ({
      slideId: r.slide_id,
      prediction: r.prediction,
      score: r.score,
      confidence: r.confidence,
      patchesAnalyzed: r.patches_analyzed,
      requiresReview: r.requires_review,
      uncertaintyLevel: r.uncertainty_level as BatchAnalysisResult["uncertaintyLevel"],
      error: r.error,
    })),
    summary: {
      total: backend.summary.total,
      completed: backend.summary.completed,
      failed: backend.summary.failed,
      responders: backend.summary.responders,
      nonResponders: backend.summary.non_responders,
      uncertain: backend.summary.uncertain,
      avgConfidence: backend.summary.avg_confidence,
      requiresReviewCount: backend.summary.requires_review_count,
    },
    processingTimeMs: backend.processing_time_ms,
  };
}

/**
 * Export batch analysis results to CSV format
 */
export function exportBatchResultsToCsv(
  results: BatchAnalysisResult[],
  summary: BatchAnalysisSummary
): string {
  const headers = [
    "Slide ID",
    "Prediction",
    "Score",
    "Confidence",
    "Patches Analyzed",
    "Requires Review",
    "Uncertainty Level",
    "Error",
  ];

  const rows = results.map((r) => [
    r.slideId,
    r.prediction,
    r.score.toFixed(4),
    r.confidence.toFixed(4),
    r.patchesAnalyzed.toString(),
    r.requiresReview ? "Yes" : "No",
    r.uncertaintyLevel,
    r.error || "",
  ]);

  // Add summary section
  const summaryRows = [
    [],
    ["SUMMARY"],
    ["Total Slides", summary.total.toString()],
    ["Completed", summary.completed.toString()],
    ["Failed", summary.failed.toString()],
    ["Responders", summary.responders.toString()],
    ["Non-Responders", summary.nonResponders.toString()],
    ["Uncertain Cases", summary.uncertain.toString()],
    ["Avg Confidence", summary.avgConfidence.toFixed(3)],
    ["Requires Review", summary.requiresReviewCount.toString()],
  ];

  const csvContent = [
    headers.join(","),
    ...rows.map((row) => row.map((cell) => `"${cell}"`).join(",")),
    ...summaryRows.map((row) => row.join(",")),
  ].join("\n");

  return csvContent;
}

/**
 * Download batch results as CSV file
 */
export function downloadBatchCsv(
  results: BatchAnalysisResult[],
  summary: BatchAnalysisSummary,
  filename: string = "batch_analysis_results.csv"
): void {
  const csv = exportBatchResultsToCsv(results, summary);
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

// ====== Connection State Management ======

type ConnectionState = "connected" | "disconnected" | "connecting" | "error";
type ConnectionListener = (state: ConnectionState) => void;

const connectionListeners: Set<ConnectionListener> = new Set();
let currentConnectionState: ConnectionState = "disconnected";

/**
 * Subscribe to connection state changes
 */
export function onConnectionStateChange(listener: ConnectionListener): () => void {
  connectionListeners.add(listener);
  // Immediately call with current state
  listener(currentConnectionState);
  return () => connectionListeners.delete(listener);
}

/**
 * Update connection state and notify listeners
 */
function setConnectionState(state: ConnectionState): void {
  if (state !== currentConnectionState) {
    currentConnectionState = state;
    connectionListeners.forEach(listener => listener(state));
  }
}

/**
 * Get current connection state
 */
export function getConnectionState(): ConnectionState {
  return currentConnectionState;
}

/**
 * Check backend connection status with state management
 */
export async function checkConnection(): Promise<boolean> {
  setConnectionState("connecting");
  try {
    await healthCheck();
    setConnectionState("connected");
    return true;
  } catch (error) {
    if (error instanceof AtlasApiError && error.isNetworkError) {
      setConnectionState("disconnected");
    } else {
      setConnectionState("error");
    }
    return false;
  }
}
// Backend multi-model response (snake_case)
interface BackendModelPrediction {
  model_id: string;
  model_name: string;
  category: string;
  score: number;
  label: string;
  positive_label: string;
  negative_label: string;
  confidence: number;
  auc: number;
  n_training_slides: number;
  description: string;
}

interface BackendMultiModelResponse {
  slide_id: string;
  predictions: Record<string, BackendModelPrediction>;
  by_category: {
    ovarian_cancer: BackendModelPrediction[];
    general_pathology: BackendModelPrediction[];
  };
  n_patches: number;
  processing_time_ms: number;
}

interface BackendAvailableModel {
  id: string;
  name: string;
  description: string;
  auc: number;
  n_slides: number;
  category: string;
  positive_label: string;
  negative_label: string;
  available: boolean;
}

interface BackendAvailableModelsResponse {
  models: BackendAvailableModel[];
}

/**
 * Transform backend model prediction to frontend format
 */
function transformModelPrediction(backend: BackendModelPrediction): ModelPrediction {
  return {
    modelId: backend.model_id,
    modelName: backend.model_name,
    category: backend.category as 'ovarian_cancer' | 'general_pathology',
    score: backend.score,
    label: backend.label,
    positiveLabel: backend.positive_label,
    negativeLabel: backend.negative_label,
    confidence: backend.confidence,
    auc: backend.auc,
    nTrainingSlides: backend.n_training_slides,
    description: backend.description,
  };
}

/**
 * Get list of available TransMIL models
 */
export async function getAvailableModels(): Promise<AvailableModelsResponse> {
  const backend = await fetchApi<BackendAvailableModelsResponse>("/api/models");
  
  return {
    models: backend.models.map((m) => ({
      id: m.id,
      name: m.name,
      description: m.description,
      auc: m.auc,
      nSlides: m.n_slides,
      category: m.category as 'ovarian_cancer' | 'general_pathology',
      positiveLabel: m.positive_label,
      negativeLabel: m.negative_label,
      available: m.available,
    })),
  };
}

/**
 * Analyze a slide with multiple TransMIL models
 * Uses extended timeout for running multiple models
 */

/**
 * Embed a slide on-demand using DINOv2.
 * Called automatically before analysis if embeddings don't exist.
 */
export async function embedSlide(
  slideId: string,
  level: number = 1,
  force: boolean = false
): Promise<{ status: string; numPatches: number; message: string; level: number }> {
  const response = await fetchApi<{
    status: string;
    slide_id: string;
    level: number;
    num_patches: number;
    processing_time_seconds?: number;
    message: string;
  }>(
    "/api/embed-slide",
    {
      method: "POST",
      body: JSON.stringify({
        slide_id: slideId,
        level: level,
        force: force,
      }),
    },
    { timeoutMs: 1800000 } // 30 minute timeout for level 0 embedding
  );

  return {
    status: response.status,
    numPatches: response.num_patches,
    message: response.message,
    level: response.level,
  };
}

export async function analyzeSlideMultiModel(
  slideId: string,
  models?: string[],
  returnAttention: boolean = false,
  level: number = 1  // 0 = full resolution, 1 = downsampled
): Promise<MultiModelResponse> {
  const backend = await fetchApi<BackendMultiModelResponse>(
    "/api/analyze-multi",
    {
      method: "POST",
      body: JSON.stringify({
        slide_id: slideId,
        models: (models === undefined ? null : models),
        return_attention: returnAttention,
        level: level,
      }),
    },
    { timeoutMs: 120000 } // 2 minute timeout for multi-model analysis
  );

  // Transform predictions
  const predictions: Record<string, ModelPrediction> = {};
  for (const [modelId, pred] of Object.entries(backend.predictions)) {
    predictions[modelId] = transformModelPrediction(pred);
  }

  return {
    slideId: backend.slide_id,
    predictions,
    byCategory: {
      ovarianCancer: backend.by_category.ovarian_cancer.map(transformModelPrediction),
      generalPathology: backend.by_category.general_pathology.map(transformModelPrediction),
    },
    nPatches: backend.n_patches,
    processingTimeMs: backend.processing_time_ms,
  };
}


// Backend response types (snake_case)
interface BackendTag {
  name: string;
  color?: string;
  count: number;
}

interface BackendGroup {
  id: string;
  name: string;
  description?: string;
  slide_ids: string[];
  created_at: string;
  updated_at: string;
}

interface BackendSlideSearchResult {
  slides: BackendSlideInfo[];
  total: number;
  page: number;
  per_page: number;
  filters: SlideFilters;
}

// ====== Tags API ======

/**
 * Get all tags with their usage counts
 */
export async function getAllTags(): Promise<Tag[]> {
  const backend = await fetchApi<BackendTag[]>("/api/tags");
  return backend.map(t => ({
    name: t.name,
    color: t.color,
    count: t.count,
  }));
}

/**
 * Add tags to a slide
 */
export async function addTagsToSlide(slideId: string, tags: string[]): Promise<void> {
  await fetchApi<{ success: boolean }>(
    `/api/slides/${encodeURIComponent(slideId)}/tags`,
    {
      method: "POST",
      body: JSON.stringify({ tags }),
    }
  );
}

/**
 * Remove a tag from a slide
 */
export async function removeTagFromSlide(slideId: string, tag: string): Promise<void> {
  await fetchApi<{ success: boolean }>(
    `/api/slides/${encodeURIComponent(slideId)}/tags/${encodeURIComponent(tag)}`,
    {
      method: "DELETE",
    }
  );
}

// ====== Groups API ======

/**
 * Get all slide groups
 */
export async function getGroups(): Promise<Group[]> {
  const backend = await fetchApi<BackendGroup[]>("/api/groups");
  return backend.map(g => ({
    id: g.id,
    name: g.name,
    description: g.description,
    slideIds: g.slide_ids,
    createdAt: g.created_at,
    updatedAt: g.updated_at,
  }));
}

/**
 * Create a new slide group
 */
export async function createGroup(name: string, description?: string): Promise<Group> {
  const backend = await fetchApi<BackendGroup>(
    "/api/groups",
    {
      method: "POST",
      body: JSON.stringify({ name, description }),
    }
  );
  return {
    id: backend.id,
    name: backend.name,
    description: backend.description,
    slideIds: backend.slide_ids,
    createdAt: backend.created_at,
    updatedAt: backend.updated_at,
  };
}

/**
 * Update a slide group
 */
export async function updateGroup(groupId: string, updates: Partial<Group>): Promise<Group> {
  const backend = await fetchApi<BackendGroup>(
    `/api/groups/${encodeURIComponent(groupId)}`,
    {
      method: "PATCH",
      body: JSON.stringify({
        name: updates.name,
        description: updates.description,
        slide_ids: updates.slideIds,
      }),
    }
  );
  return {
    id: backend.id,
    name: backend.name,
    description: backend.description,
    slideIds: backend.slide_ids,
    createdAt: backend.created_at,
    updatedAt: backend.updated_at,
  };
}

/**
 * Delete a slide group
 */
export async function deleteGroup(groupId: string): Promise<void> {
  await fetchApi<{ success: boolean }>(
    `/api/groups/${encodeURIComponent(groupId)}`,
    {
      method: "DELETE",
    }
  );
}

/**
 * Add slides to a group
 */
export async function addSlidesToGroup(groupId: string, slideIds: string[]): Promise<void> {
  await fetchApi<{ success: boolean }>(
    `/api/groups/${encodeURIComponent(groupId)}/slides`,
    {
      method: "POST",
      body: JSON.stringify({ slide_ids: slideIds }),
    }
  );
}

/**
 * Remove a slide from a group
 */
export async function removeSlideFromGroup(groupId: string, slideId: string): Promise<void> {
  await fetchApi<{ success: boolean }>(
    `/api/groups/${encodeURIComponent(groupId)}/slides/${encodeURIComponent(slideId)}`,
    {
      method: "DELETE",
    }
  );
}

// ====== Search/Filter API ======

/**
 * Search and filter slides with pagination
 */
export async function searchSlides(filters: SlideFilters): Promise<SlideSearchResult> {
  const params = new URLSearchParams();
  
  if (filters.search) params.set("search", filters.search);
  if (filters.tags?.length) params.set("tags", filters.tags.join(","));
  if (filters.groupId) params.set("group_id", filters.groupId);
  if (filters.hasEmbeddings !== undefined) params.set("has_embeddings", String(filters.hasEmbeddings));
  if (filters.label) params.set("label", filters.label);
  if (filters.minPatches !== undefined) params.set("min_patches", String(filters.minPatches));
  if (filters.maxPatches !== undefined) params.set("max_patches", String(filters.maxPatches));
  if (filters.starred !== undefined) params.set("starred", String(filters.starred));
  if (filters.dateFrom) params.set("date_from", filters.dateFrom);
  if (filters.dateTo) params.set("date_to", filters.dateTo);
  if (filters.sortBy) params.set("sort_by", filters.sortBy);
  if (filters.sortOrder) params.set("sort_order", filters.sortOrder);
  if (filters.page !== undefined) params.set("page", String(filters.page));
  if (filters.perPage !== undefined) params.set("per_page", String(filters.perPage));

  const backend = await fetchApi<BackendSlideSearchResult>(
    `/api/slides/search?${params.toString()}`
  );

  return {
    slides: backend.slides.map(s => ({
      id: s.slide_id,
      filename: `${s.slide_id}.svs`,
      dimensions: s.dimensions ?? { width: 0, height: 0 },
      magnification: s.magnification ? parseInt(s.magnification.replace('x', ''), 10) : 40,
      mpp: s.mpp ?? 0.25,
      createdAt: new Date().toISOString(),
      label: s.label,
      hasEmbeddings: s.has_embeddings,
      numPatches: s.num_patches,
      patient: s.patient ? {
        age: s.patient.age,
        sex: s.patient.sex,
        stage: s.patient.stage,
        grade: s.patient.grade,
        prior_lines: s.patient.prior_lines,
        histology: s.patient.histology,
      } : undefined,
    })),
    total: backend.total,
    page: backend.page,
    perPage: backend.per_page,
    filters: backend.filters,
  };
}

// ====== Metadata API ======

/**
 * Update slide custom metadata
 */
export async function updateSlideMetadata(
  slideId: string, 
  metadata: Record<string, unknown>
): Promise<void> {
  await fetchApi<{ success: boolean }>(
    `/api/slides/${encodeURIComponent(slideId)}/metadata`,
    {
      method: "PATCH",
      body: JSON.stringify({ metadata }),
    }
  );
}

/**
 * Toggle slide star status
 * Returns the new starred state
 */
export async function toggleSlideStar(slideId: string): Promise<boolean> {
  const result = await fetchApi<{ starred: boolean }>(
    `/api/slides/${encodeURIComponent(slideId)}/star`,
    {
      method: "POST",
    }
  );
  return result.starred;
}

// ====== Bulk Operations API ======

/**
 * Add tags to multiple slides at once
 */
export async function bulkAddTags(slideIds: string[], tags: string[]): Promise<void> {
  await fetchApi<{ success: boolean; count: number }>(
    "/api/bulk/tags",
    {
      method: "POST",
      body: JSON.stringify({
        slide_ids: slideIds,
        tags,
      }),
    }
  );
}

/**
 * Add multiple slides to a group at once
 */
export async function bulkAddToGroup(slideIds: string[], groupId: string): Promise<void> {
  await fetchApi<{ success: boolean; count: number }>(
    "/api/bulk/group",
    {
      method: "POST",
      body: JSON.stringify({
        slide_ids: slideIds,
        group_id: groupId,
      }),
    }
  );
}

// ====== Embedding Task Polling ======

export interface EmbeddingTaskStatus {
  task_id: string;
  slide_id: string;
  level: number;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;  // 0-100
  message: string;
  num_patches: number;
  processing_time_seconds: number;
  error: string | null;
  elapsed_seconds: number;
  embedding_path?: string;
}

export interface EmbedSlideResponse {
  status: 'exists' | 'completed' | 'started' | 'in_progress';
  task_id?: string;
  slide_id: string;
  level: number;
  num_patches?: number;
  message: string;
  estimated_time_minutes?: number;
  progress?: number;
}

/**
 * Start embedding for a slide. For level 0, returns a task_id to poll.
 */
export async function embedSlideAsync(
  slideId: string,
  level: number = 1,
  force: boolean = false
): Promise<EmbedSlideResponse> {
  const response = await fetchApi<EmbedSlideResponse>(
    "/api/embed-slide",
    {
      method: "POST",
      body: JSON.stringify({
        slide_id: slideId,
        level: level,
        force: force,
        async: level === 0,  // Force async for level 0
      }),
    },
    { timeoutMs: 120000 }  // 2 min timeout for initial request
  );
  return response;
}

/**
 * Get status of an embedding task.
 */
export async function getEmbeddingTaskStatus(taskId: string): Promise<EmbeddingTaskStatus> {
  return fetchApi<EmbeddingTaskStatus>(
    `/api/embed-slide/status/${encodeURIComponent(taskId)}`,
    {},
    { timeoutMs: 10000, skipRetry: true }
  );
}

/**
 * Poll embedding task until completion or failure.
 * Calls onProgress callback with status updates.
 * Returns final status when done.
 */
export async function pollEmbeddingTask(
  taskId: string,
  onProgress?: (status: EmbeddingTaskStatus) => void,
  pollIntervalMs: number = 2000,
  maxWaitMs: number = 1800000  // 30 minutes max
): Promise<EmbeddingTaskStatus> {
  const startTime = Date.now();
  
  while (Date.now() - startTime < maxWaitMs) {
    try {
      const status = await getEmbeddingTaskStatus(taskId);
      
      if (onProgress) {
        onProgress(status);
      }
      
      if (status.status === 'completed' || status.status === 'failed') {
        return status;
      }
      
      // Wait before next poll
      await new Promise(resolve => setTimeout(resolve, pollIntervalMs));
    } catch (error) {
      // On network error, wait and retry
      console.warn('Polling error, retrying...', error);
      await new Promise(resolve => setTimeout(resolve, pollIntervalMs * 2));
    }
  }
  
  throw new AtlasApiError({
    code: 'TIMEOUT',
    message: `Embedding task ${taskId} did not complete within ${maxWaitMs / 60000} minutes`,
    isTimeout: true,
  });
}

/**
 * Embed a slide with automatic polling for level 0.
 * Returns when embedding is complete.
 */
export async function embedSlideWithPolling(
  slideId: string,
  level: number = 1,
  force: boolean = false,
  onProgress?: (status: { phase: string; progress: number; message: string }) => void
): Promise<{ status: string; numPatches: number; message: string; level: number }> {
  // Start embedding
  const response = await embedSlideAsync(slideId, level, force);
  
  // If already exists or completed inline, return immediately
  if (response.status === 'exists' || response.status === 'completed') {
    if (onProgress) {
      onProgress({
        phase: 'complete',
        progress: 100,
        message: response.message,
      });
    }
    return {
      status: response.status,
      numPatches: response.num_patches || 0,
      message: response.message,
      level: response.level,
    };
  }
  
  // If started as background task, poll for completion
  if (response.status === 'started' || response.status === 'in_progress') {
    if (!response.task_id) {
      throw new AtlasApiError({
        code: 'INVALID_RESPONSE',
        message: 'Background task started but no task_id returned',
      });
    }
    
    if (onProgress) {
      onProgress({
        phase: 'embedding',
        progress: response.progress || 0,
        message: response.message,
      });
    }
    
    // Poll until complete
    const finalStatus = await pollEmbeddingTask(
      response.task_id,
      (status) => {
        if (onProgress) {
          onProgress({
            phase: status.status === 'completed' ? 'complete' : 'embedding',
            progress: status.progress,
            message: status.message,
          });
        }
      }
    );
    
    if (finalStatus.status === 'failed') {
      throw new AtlasApiError({
        code: 'EMBEDDING_FAILED',
        message: finalStatus.error || 'Embedding failed',
      });
    }
    
    return {
      status: finalStatus.status,
      numPatches: finalStatus.num_patches,
      message: finalStatus.message,
      level: level,
    };
  }
  
  // Unexpected status
  throw new AtlasApiError({
    code: 'INVALID_RESPONSE',
    message: `Unexpected embedding status: ${response.status}`,
  });
}

// ====== Async Report Generation ======

export interface ReportTaskStatus {
  task_id: string;
  slide_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  message: string;
  stage: string;
  error?: string;
  elapsed_seconds: number;
  result?: {
    slide_id: string;
    report_json: Record<string, unknown>;
    summary_text: string;
  };
}

export interface AsyncReportResponse {
  task_id: string;
  slide_id: string;
  status: string;
  message: string;
  estimated_time_seconds: number;
}

/**
 * Start async report generation
 * Returns a task_id for polling
 */
export async function startReportGeneration(slideId: string): Promise<AsyncReportResponse> {
  return fetchApi<AsyncReportResponse>(
    '/api/report/async',
    {
      method: 'POST',
      body: JSON.stringify({ slide_id: slideId }),
    },
    { timeoutMs: 10000 }
  );
}

/**
 * Poll report generation status
 */
export async function getReportStatus(taskId: string): Promise<ReportTaskStatus> {
  return fetchApi<ReportTaskStatus>(
    `/api/report/status/${encodeURIComponent(taskId)}`,
    { method: 'GET' },
    { timeoutMs: 10000 }
  );
}

/**
 * Generate report with async polling (improved UX)
 * Returns progress updates via callback, then final report
 */
export async function generateReportWithProgress(
  request: ReportRequest,
  onProgress?: (progress: number, message: string) => void
): Promise<StructuredReport> {
  // Start async generation
  const asyncResponse = await startReportGeneration(request.slideId);
  const taskId = asyncResponse.task_id;
  
  // Poll for completion
  const maxWaitMs = 120000; // 2 minute max wait
  const pollIntervalMs = 1000; // Poll every second
  const startTime = Date.now();
  
  while (Date.now() - startTime < maxWaitMs) {
    const status = await getReportStatus(taskId);
    
    // Report progress
    if (onProgress) {
      onProgress(status.progress, status.message);
    }
    
    if (status.status === 'completed' && status.result) {
      // Transform backend response to frontend format
      return transformBackendReport(status.result);
    }
    
    if (status.status === 'failed') {
      throw new AtlasApiError({
        code: 'REPORT_GENERATION_FAILED',
        message: status.error || 'Report generation failed',
      });
    }
    
    // Wait before next poll
    await new Promise(resolve => setTimeout(resolve, pollIntervalMs));
  }
  
  throw new AtlasApiError({
    code: 'REPORT_TIMEOUT',
    message: 'Report generation timed out after 2 minutes',
    isTimeout: true,
  });
}

/**
 * Transform backend report to frontend StructuredReport format
 */
function transformBackendReport(backend: {
  slide_id: string;
  report_json: Record<string, unknown>;
  summary_text: string;
}): StructuredReport {
  const reportJson = backend.report_json as {
    case_id?: string;
    task?: string;
    patient_context?: PatientContext;
    model_output?: {
      label: string;
      probability: number;
      calibration_note?: string;
    };
    evidence?: Array<{
      patch_id?: string;
      coordinates?: number[];
      morphology_description?: string;
      significance?: string;
      attention_weight?: number;
    }>;
    similar_examples?: Array<{
      example_id?: string;
      slide_id?: string;
      label?: string;
      distance?: number;
      similarity_score?: number;
    }>;
    limitations?: string[];
    suggested_next_steps?: string[];
    safety_statement?: string;
    decision_support?: {
      risk_level: string;
      confidence_level: string;
      confidence_score: number;
      primary_recommendation: string;
      supporting_rationale?: string[];
      alternative_considerations?: string[];
      guideline_references?: Array<{
        source: string;
        section: string;
        recommendation: string;
        url?: string;
      }>;
      uncertainty_statement?: string;
      quality_warnings?: string[];
      suggested_workup?: string[];
      interpretation_note?: string;
      caveat?: string;
    };
  };

  const modelOutput = reportJson.model_output || { label: 'unknown', probability: 0.5 };
  
  const evidence = (reportJson.evidence || []).map((e, idx) => ({
    patchId: e.patch_id || `patch_${idx}`,
    coordsLevel0: (e.coordinates || [0, 0]) as [number, number],
    morphologyDescription: e.morphology_description || 
      `High-attention region with attention weight ${(e.attention_weight || 0).toFixed(3)}`,
    whyThisPatchMatters: e.significance ||
      'This region shows morphological patterns associated with the predicted classification.',
  }));

  const similarExamples = (reportJson.similar_examples || []).map((s) => ({
    exampleId: s.example_id || s.slide_id || 'unknown',
    label: s.label || 'unknown',
    distance: s.distance ?? (1 - (s.similarity_score || 0)),
  }));

  const decisionSupport = reportJson.decision_support ? {
    risk_level: reportJson.decision_support.risk_level as import('@/types').RiskLevel,
    confidence_level: reportJson.decision_support.confidence_level as import('@/types').ConfidenceLevel,
    confidence_score: reportJson.decision_support.confidence_score,
    primary_recommendation: reportJson.decision_support.primary_recommendation,
    supporting_rationale: reportJson.decision_support.supporting_rationale || [],
    alternative_considerations: reportJson.decision_support.alternative_considerations || [],
    guideline_references: reportJson.decision_support.guideline_references || [],
    uncertainty_statement: reportJson.decision_support.uncertainty_statement || 
      'Prediction confidence should be interpreted in the context of slide quality.',
    quality_warnings: reportJson.decision_support.quality_warnings || [],
    suggested_workup: reportJson.decision_support.suggested_workup || [],
    interpretation_note: reportJson.decision_support.interpretation_note ||
      'This is an AI-generated interpretation. Clinical correlation is essential.',
    caveat: reportJson.decision_support.caveat ||
      'This tool is for research purposes only.',
  } : undefined;

  const defaultLimitations = [
    'This is an uncalibrated research model',
    'Prediction is based on morphological patterns only',
    'Model trained on limited dataset',
  ];

  const defaultNextSteps = [
    'Correlate findings with clinical history',
    'Review high-attention regions',
    'Consider molecular profiling',
  ];

  return {
    caseId: reportJson.case_id || backend.slide_id,
    task: reportJson.task || 'Bevacizumab treatment response prediction',
    generatedAt: new Date().toISOString(),
    patientContext: reportJson.patient_context,
    modelOutput: {
      label: modelOutput.label.toUpperCase(),
      score: modelOutput.probability,
      confidence: Math.abs(modelOutput.probability - 0.5) * 2,
      calibrationNote: modelOutput.calibration_note || 
        'Model probability requires external validation.',
    },
    evidence,
    similarExamples,
    limitations: reportJson.limitations?.length ? reportJson.limitations : defaultLimitations,
    suggestedNextSteps: reportJson.suggested_next_steps?.length ? reportJson.suggested_next_steps : defaultNextSteps,
    safetyStatement: reportJson.safety_statement || 
      'This is a research tool. All findings must be validated by qualified clinicians.',
    summary: backend.summary_text,
    decisionSupport,
  };
}


// ====== Async Batch Analysis API ======

/**
 * Async batch task response types
 */
export interface AsyncBatchTaskStatus {
  task_id: string;
  status: "pending" | "running" | "completed" | "cancelled" | "failed";
  progress: number;
  current_slide_index: number;
  current_slide_id: string;
  total_slides: number;
  completed_slides: number;
  message: string;
  error?: string;
  elapsed_seconds: number;
  cancel_requested: boolean;
  results_count: number;
  // Full results only when completed
  results?: Array<{
    slide_id: string;
    prediction: string;
    score: number;
    confidence: number;
    patches_analyzed: number;
    requires_review: boolean;
    uncertainty_level: string;
    error?: string;
  }>;
  summary?: {
    total: number;
    completed: number;
    failed: number;
    responders: number;
    non_responders: number;
    uncertain: number;
    avg_confidence: number;
    requires_review_count: number;
  };
  processing_time_ms?: number;
}

/**
 * Start async batch analysis with progress tracking
 */
export async function startBatchAnalysisAsync(
  slideIds: string[],
  concurrency: number = 4
): Promise<{ task_id: string; status: string; total_slides: number; message: string }> {
  return fetchApi<{ task_id: string; status: string; total_slides: number; message: string }>(
    "/api/analyze-batch/async",
    {
      method: "POST",
      body: JSON.stringify({
        slide_ids: slideIds,
        concurrency,
      }),
    }
  );
}

/**
 * Get status of an async batch analysis task
 */
export async function getBatchAnalysisStatus(
  taskId: string
): Promise<AsyncBatchTaskStatus> {
  return fetchApi<AsyncBatchTaskStatus>(
    `/api/analyze-batch/status/${taskId}`
  );
}

/**
 * Cancel a running batch analysis task
 */
export async function cancelBatchAnalysis(
  taskId: string
): Promise<{ success: boolean; message: string; completed_slides?: number }> {
  return fetchApi<{ success: boolean; message: string; completed_slides?: number }>(
    `/api/analyze-batch/cancel/${taskId}`,
    { method: "POST" }
  );
}

/**
 * List all batch analysis tasks
 */
export async function listBatchTasks(
  status?: string
): Promise<{ tasks: AsyncBatchTaskStatus[]; total: number }> {
  const params = status ? `?status=${status}` : "";
  return fetchApi<{ tasks: AsyncBatchTaskStatus[]; total: number }>(
    `/api/analyze-batch/tasks${params}`
  );
}

/**
 * Poll batch analysis until completion or cancellation
 * @param taskId - Task ID to poll
 * @param onProgress - Callback for progress updates
 * @param pollIntervalMs - Poll interval in milliseconds (default 1000)
 * @returns Final task status with results
 */
export async function pollBatchAnalysis(
  taskId: string,
  onProgress?: (status: AsyncBatchTaskStatus) => void,
  pollIntervalMs: number = 1000
): Promise<AsyncBatchTaskStatus> {
  let status = await getBatchAnalysisStatus(taskId);
  
  while (status.status === "pending" || status.status === "running") {
    onProgress?.(status);
    await new Promise((resolve) => setTimeout(resolve, pollIntervalMs));
    status = await getBatchAnalysisStatus(taskId);
  }
  
  onProgress?.(status);
  return status;
}

/**
 * Convert async batch results to the standard BatchAnalyzeResponse format
 */
export function convertAsyncBatchResults(
  asyncStatus: AsyncBatchTaskStatus
): BatchAnalyzeResponse | null {
  if (!asyncStatus.results || !asyncStatus.summary) {
    return null;
  }
  
  return {
    results: asyncStatus.results.map((r) => ({
      slideId: r.slide_id,
      prediction: r.prediction,
      score: r.score,
      confidence: r.confidence,
      patchesAnalyzed: r.patches_analyzed,
      requiresReview: r.requires_review,
      uncertaintyLevel: r.uncertainty_level as BatchAnalysisResult["uncertaintyLevel"],
      error: r.error,
    })),
    summary: {
      total: asyncStatus.summary.total,
      completed: asyncStatus.summary.completed,
      failed: asyncStatus.summary.failed,
      responders: asyncStatus.summary.responders,
      nonResponders: asyncStatus.summary.non_responders,
      uncertain: asyncStatus.summary.uncertain,
      avgConfidence: asyncStatus.summary.avg_confidence,
      requiresReviewCount: asyncStatus.summary.requires_review_count,
    },
    processingTimeMs: asyncStatus.processing_time_ms || asyncStatus.elapsed_seconds * 1000,
  };
}
