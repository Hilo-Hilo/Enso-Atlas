// Enso Atlas API client.
// Backend communication utilities with robust error handling and optional project_id scoping.

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
  DecisionSupport,
  RiskLevel,
  ConfidenceLevel,
  ModelPrediction,
  AvailableModel,
  MultiModelResponse,
  AvailableModelsResponse,
  SimilarCase,
} from "@/types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "";

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
          const rawError = await response.json() as Record<string, unknown>;
          const detail = rawError?.detail;

          // FastAPI commonly returns { detail: "..." } or { detail: { error, message, ... } }
          if (detail && typeof detail === "object") {
            const detailObj = detail as Record<string, unknown>;
            errorData = {
              code: String(rawError?.code ?? detailObj.error ?? detailObj.code ?? `HTTP_${response.status}`),
              message:
                (typeof detailObj.message === "string" && detailObj.message) ||
                (typeof detailObj.error === "string" && detailObj.error) ||
                `HTTP ${response.status}: ${response.statusText}`,
              details: detailObj,
            };
          } else if (typeof detail === "string") {
            errorData = {
              code: String(rawError?.code ?? `HTTP_${response.status}`),
              message: detail,
            };
          } else {
            errorData = {
              code: String(rawError?.code ?? `HTTP_${response.status}`),
              message:
                (typeof rawError?.message === "string" && rawError.message) ||
                `HTTP ${response.status}: ${response.statusText}`,
              details: (rawError?.details as Record<string, unknown> | undefined),
            };
          }
        } catch (err) {
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
  display_name?: string | null;
  has_wsi?: boolean;
  has_embeddings: boolean;
  has_level0_embeddings?: boolean;  // Whether level 0 (full res) embeddings exist
  label?: string | null;
  num_patches?: number | null;
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

function normalizePatchCount(value: number | null | undefined): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function normalizeLabel(value: string | null | undefined): string | undefined {
  return typeof value === "string" ? value : undefined;
}

function normalizeProjectId(projectId?: string | null): string | null {
  const value = typeof projectId === "string" ? projectId.trim() : "";
  if (!value || value === "default") {
    return null;
  }
  return value;
}

/**
 * Fetch list of available slides
 */
export async function getSlides(params: { page?: number; perPage?: number; projectId?: string } = {}): Promise<SlidesListResponse> {
  if (params.projectId?.trim() === "default") {
    return {
      slides: [],
      total: 0,
    };
  }

  const query = new URLSearchParams();
  if (params.page !== undefined) query.set("page", String(params.page));
  if (params.perPage !== undefined) query.set("per_page", String(params.perPage));
  const projectId = normalizeProjectId(params.projectId);
  if (projectId) query.set("project_id", projectId);
  const endpoint = query.toString() ? `/api/slides?${query.toString()}` : "/api/slides";

  // Backend may return either an array or a paginated object.
  const backend = await fetchApi<BackendSlideInfo[] | BackendSlidesListResponse>(endpoint);

  const mapSlides = (items: BackendSlideInfo[]) =>
    items.map((s) => ({
      id: s.slide_id,
      filename: `${s.slide_id}.svs`,
      displayName: s.display_name ?? null,
      dimensions: s.dimensions ?? { width: 0, height: 0 },
      magnification: s.magnification ? parseInt(s.magnification.replace("x", ""), 10) : 40,
      mpp: s.mpp ?? 0.25,
      createdAt: new Date().toISOString(),
      // Extended fields from backend
      label: normalizeLabel(s.label),
      hasWsi: s.has_wsi,
      hasEmbeddings: s.has_embeddings,
      hasLevel0Embeddings: s.has_level0_embeddings ?? false, // Level 0 embedding status
      numPatches: normalizePatchCount(s.num_patches),
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
      const perPage = params.perPage ?? Math.max(mapped.length, 1);
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
          if (projectId) pageParams.set("project_id", projectId);
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
  const projectId = normalizeProjectId(request.projectId);
  const backend = await fetchApi<BackendAnalysisResponse>(
    "/api/analyze",
    {
      method: "POST",
      body: JSON.stringify({ 
        slide_id: request.slideId,
        project_id: projectId,
      }),
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
    evidencePatches: (() => {
      // Normalize attention weights: raw softmax values are tiny (e.g. 0.00025)
      // because they sum to 1 over all patches (often thousands).
      // Rescale so the highest-attention patch in top_evidence maps to ~1.0.
      const maxAttn = Math.max(...backend.top_evidence.map(e => e.attention_weight), 1e-12);
      return backend.top_evidence.map(e => {
        const patchId = `patch_${e.patch_index}`;
        const x = e.coordinates?.[0];
        const y = e.coordinates?.[1];
        return ({
          id: patchId,
          patchId,
          coordinates: {
            x: x ?? 0,
            y: y ?? 0,
            width: 224,
            height: 224,
            level: 0,
          },
          attentionWeight: e.attention_weight / maxAttn,
          thumbnailUrl: getPatchUrl(backend.slide_id, patchId, {
            projectId,
            size: 224,
            x,
            y,
            patchSize: 224,
          }),
          rank: e.rank,
          tissueType: e.tissue_type as import("@/types").TissueType | undefined,
          tissueConfidence: e.tissue_confidence,
        });
      });
    })(),
    similarCases: backend.similar_cases.map(s => ({
      slideId: s.slide_id,
      similarity: s.similarity_score,
      distance: s.distance,
      label: s.label || undefined,
      thumbnailUrl: getThumbnailUrl(s.slide_id, projectId, 128),
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
 * Fetch similar cases for a slide using the dedicated FAISS similarity endpoint.
 * Uses slide-level mean embeddings with cosine similarity (fast, <100ms).
 */
export async function fetchSimilarCases(
  slideId: string,
  k: number = 5,
  projectId?: string
): Promise<SimilarCase[]> {
  interface BackendSimilarResponse {
    slide_id: string;
    similar_cases: Array<{
      slide_id: string;
      similarity_score: number;
      distance: number;
      label?: string | null;
      n_patches?: number;
    }>;
    num_queries: number;
  }

  try {
    const params = new URLSearchParams();
    params.set('slide_id', slideId);
    params.set('k', String(k));
    const scopedProjectId = normalizeProjectId(projectId);
    if (scopedProjectId) params.set('project_id', scopedProjectId);
    
    const backend = await fetchApi<BackendSimilarResponse>(
      `/api/similar?${params.toString()}`,
      {},
      { timeoutMs: 10000 }
    );

    return backend.similar_cases.map((s) => ({
      slideId: s.slide_id,
      similarity: s.similarity_score,
      distance: s.distance,
      label: s.label || undefined,
      thumbnailUrl: getThumbnailUrl(s.slide_id, scopedProjectId, 128),
    }));
  } catch (error) {
    console.warn("[API] Failed to fetch similar cases:", error);
    return [];
  }
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

function toFrontendDecisionSupport(raw: unknown): DecisionSupport | undefined {
  if (!raw || typeof raw !== "object") return undefined;
  const obj = raw as Record<string, unknown>;
  const primary = String(obj.primary_recommendation ?? "").trim();
  if (!primary) return undefined;

  const asStringArray = (value: unknown): string[] =>
    Array.isArray(value) ? value.map((v) => String(v).trim()).filter(Boolean) : [];

  const rawScore = Number(obj.confidence_score);
  const confidenceScore = Number.isFinite(rawScore)
    ? Math.max(0, Math.min(1, rawScore))
    : 0.5;

  const confidenceLevel: ConfidenceLevel =
    obj.confidence_level === "high" || obj.confidence_level === "moderate" || obj.confidence_level === "low"
      ? (obj.confidence_level as ConfidenceLevel)
      : confidenceScore >= 0.7
      ? "high"
      : confidenceScore >= 0.35
      ? "moderate"
      : "low";

  const riskLevel: RiskLevel =
    obj.risk_level === "high_confidence" ||
    obj.risk_level === "moderate_confidence" ||
    obj.risk_level === "low_confidence" ||
    obj.risk_level === "inconclusive"
      ? (obj.risk_level as RiskLevel)
      : confidenceLevel === "high"
      ? "high_confidence"
      : confidenceLevel === "moderate"
      ? "moderate_confidence"
      : "low_confidence";

  const guidelineRaw = Array.isArray(obj.guideline_references) ? obj.guideline_references : [];
  const guidelineRefs = guidelineRaw
    .filter((g): g is Record<string, unknown> => !!g && typeof g === "object")
    .map((g) => ({
      source: String(g.source ?? "").trim() || "Clinical guideline",
      section: String(g.section ?? "").trim() || "General",
      recommendation: String(g.recommendation ?? "").trim() || "Correlate with multidisciplinary review.",
      ...(String(g.url ?? "").trim() ? { url: String(g.url).trim() } : {}),
    }));

  return {
    risk_level: riskLevel,
    confidence_level: confidenceLevel,
    confidence_score: confidenceScore,
    primary_recommendation: primary,
    supporting_rationale: asStringArray(obj.supporting_rationale),
    alternative_considerations: asStringArray(obj.alternative_considerations),
    guideline_references: guidelineRefs,
    uncertainty_statement: String(obj.uncertainty_statement ?? "").trim(),
    quality_warnings: asStringArray(obj.quality_warnings),
    suggested_workup: asStringArray(obj.suggested_workup),
    interpretation_note: String(obj.interpretation_note ?? "").trim(),
    caveat: String(obj.caveat ?? "").trim(),
  };
}

/**
 * Generate a structured report for a slide
 * Uses longer timeout for MedGemma inference
 */
export async function generateReport(
  request: ReportRequest
): Promise<StructuredReport> {
  const projectId = normalizeProjectId(request.projectId);
  const backend = await fetchApi<BackendReportResponse>(
    "/api/report",
    {
      method: "POST",
      body: JSON.stringify({ 
        slide_id: request.slideId,
        project_id: projectId,
      }),
    },
    { timeoutMs: 420000 } // Allow slow CPU MedGemma runs
  );

  // Transform backend response to frontend StructuredReport format
  const reportJson = backend.report_json;
  const modelOutput = reportJson.model_output;
  
  // Transform evidence array
  const evidence = (reportJson.evidence || []).map((e, idx) => ({
    patchId: e.patch_id || `patch_${idx}`,
    coordsLevel0: (e.coordinates || [0, 0]) as [number, number],
    morphologyDescription: e.morphology_description || 
      `Tissue region (${e.coordinates?.[0] || 0}, ${e.coordinates?.[1] || 0})`,
    whyThisPatchMatters: e.significance ||
      "Regional morphology contributes to the model's slide-level prediction signal.",
  }));

  // Transform similar examples
  const similarExamples = (reportJson.similar_examples || []).map((s) => ({
    exampleId: s.example_id || s.slide_id || "unknown",
    label: s.label || "unknown",
    distance: s.distance ?? (1 - (s.similarity_score || 0)),
  }));

  // Decision support is only rendered when supplied by MedGemma.
  const decisionSupport = toFrontendDecisionSupport(reportJson.decision_support);

  return {
    caseId: reportJson.case_id || backend.slide_id,
    task: reportJson.task || "Treatment response prediction",
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
    limitations: reportJson.limitations?.length ? reportJson.limitations : [],
    suggestedNextSteps: reportJson.suggested_next_steps?.length ? reportJson.suggested_next_steps : [],
    safetyStatement: reportJson.safety_statement || 
      "This prediction is for decision support and to enhance interpretability by the physician. Clinical decisions should integrate multiple factors including patient history, other biomarkers, and clinician expertise.",
    summary: backend.summary_text,
    decisionSupport,
  };
}

/**
 * Get Deep Zoom Image (DZI) metadata for OpenSeadragon
 * 
 * Uses local Next.js API proxy to avoid CORS issues.
 * The proxy routes through /api/slides/{id}/dzi and /api/slides/{id}/dzi_files/...
 * which forward to the backend while keeping same-origin.
 */
export function getDziUrl(slideId: string, projectId?: string): string {
  // Use local proxy to avoid CORS - OpenSeadragon will fetch tiles from
  // /api/slides/{id}/dzi_files/{level}/{col}_{row}.jpeg (relative to DZI URL)
  const params = new URLSearchParams();
  const scopedProjectId = normalizeProjectId(projectId);
  if (scopedProjectId) params.set("project_id", scopedProjectId);
  const qs = params.toString() ? `?${params.toString()}` : "";
  return `/api/slides/${encodeURIComponent(slideId)}/dzi${qs}`;
}

/**
 * Get heatmap overlay image URL
 * @param slideId - Slide identifier
 * @param modelId - Optional model ID for multi-model heatmaps
 * @param level - Downsample level (0-4): 0=2048px highest detail, 2=512px default, 4=128px fastest
 * 
 * Uses local Next.js API proxy to avoid CORS issues.
 */
export function getHeatmapUrl(
  slideId: string,
  modelId?: string,
  level?: number,
  alphaPower?: number,
  projectId?: string,
  smooth?: boolean,
): string {
  const params = new URLSearchParams();
  if (level !== undefined) params.set('level', String(level));
  if (alphaPower !== undefined && Math.abs(alphaPower - 0.7) > 0.01) {
    params.set('alpha_power', alphaPower.toFixed(2));
  }
  const scopedProjectId = normalizeProjectId(projectId);
  if (scopedProjectId) params.set('project_id', scopedProjectId);
  if (smooth) params.set('smooth', 'true');
  const qs = params.toString() ? `?${params.toString()}` : '';
  if (modelId) {
    return `/api/heatmap/${encodeURIComponent(slideId)}/${encodeURIComponent(modelId)}${qs}`;
  }
  return `/api/heatmap/${encodeURIComponent(slideId)}${qs}`;
}

/**
 * Get thumbnail URL for a slide
 * 
 * Uses local Next.js API proxy to avoid CORS issues.
 */
export function getThumbnailUrl(slideId: string, projectId?: string | null, size?: number): string {
  const params = new URLSearchParams();
  const scopedProjectId = normalizeProjectId(projectId);
  if (scopedProjectId) params.set("project_id", scopedProjectId);
  if (typeof size === "number" && Number.isFinite(size) && size > 0) {
    params.set("size", String(Math.round(size)));
  }
  const qs = params.toString() ? `?${params.toString()}` : "";
  return `/api/slides/${encodeURIComponent(slideId)}/thumbnail${qs}`;
}

/**
 * Get patch image URL
 * 
 * Uses local Next.js API proxy to avoid CORS issues.
 */
export function getPatchUrl(
  slideId: string,
  patchId: string,
  opts?: {
    projectId?: string | null;
    size?: number;
    x?: number;
    y?: number;
    patchSize?: number;
  }
): string {
  const params = new URLSearchParams();
  const scopedProjectId = normalizeProjectId(opts?.projectId);
  if (scopedProjectId) params.set("project_id", scopedProjectId);
  if (typeof opts?.size === "number" && Number.isFinite(opts.size) && opts.size > 0) {
    params.set("size", String(Math.round(opts.size)));
  }
  if (typeof opts?.x === "number" && Number.isFinite(opts.x)) {
    params.set("x", String(Math.round(opts.x)));
  }
  if (typeof opts?.y === "number" && Number.isFinite(opts.y)) {
    params.set("y", String(Math.round(opts.y)));
  }
  if (typeof opts?.patchSize === "number" && Number.isFinite(opts.patchSize) && opts.patchSize > 0) {
    params.set("patch_size", String(Math.round(opts.patchSize)));
  }
  const qs = params.toString() ? `?${params.toString()}` : "";
  return `/api/slides/${encodeURIComponent(slideId)}/patches/${encodeURIComponent(patchId)}${qs}`;
}

/**
 * Export report as PDF.
 *
 * NOTE: legacy slide-scoped endpoints (/api/slides/{id}/report/*) were removed
 * on the backend. This helper now uses the supported flow:
 * 1) POST /api/report
 * 2) POST /api/report/pdf
 */
export async function exportReportPdf(
  slideId: string,
  projectId?: string
): Promise<Blob> {
  const { controller, timeoutId } = createTimeoutController(120000);

  try {
    const scopedProjectId = normalizeProjectId(projectId);

    const reportResponse = await fetchApi<BackendReportResponse>(
      "/api/report",
      {
        method: "POST",
        body: JSON.stringify({
          slide_id: slideId,
          project_id: scopedProjectId,
        }),
      },
      { timeoutMs: 420000 }
    );

    const response = await fetch(`${API_BASE_URL}/api/report/pdf`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        report: reportResponse.report_json,
        case_id: reportResponse.slide_id,
      }),
      signal: controller.signal,
    });

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
 * Export report as structured JSON via the current /api/report route.
 */
export async function exportReportJson(
  slideId: string,
  projectId?: string
): Promise<StructuredReport> {
  return generateReport({
    slideId,
    evidencePatchIds: [],
    projectId,
  });
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
  topK: number = 5,
  projectId?: string
): Promise<SemanticSearchResponse> {
  const scopedProjectId = normalizeProjectId(projectId);
  const backend = await fetchApi<BackendSemanticSearchResponse>(
    "/api/semantic-search",
    {
      method: "POST",
      body: JSON.stringify({
        slide_id: slideId,
        query,
        top_k: topK,
        project_id: scopedProjectId,
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
    topEvidence: (() => {
      const maxAttn = Math.max(...backend.top_evidence.map(e => e.attention_weight), 1e-12);
      return backend.top_evidence.map((e) => ({
        rank: e.rank,
        patchIndex: e.patch_index,
        attentionWeight: e.attention_weight / maxAttn,
        attentionUncertainty: e.attention_uncertainty,
        coordinates: e.coordinates,
      }));
    })(),
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
    ["Positive Class", summary.responders.toString()],
    ["Negative Class", summary.nonResponders.toString()],
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
  decision_threshold?: number | null;
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
  by_category: Record<string, BackendModelPrediction[]>;
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
    category: backend.category,
    score: backend.score,
    decisionThreshold: backend.decision_threshold ?? undefined,
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
      category: m.category,
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
  level: number = 0,
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
  level: number = 0,  // fixed to full-resolution policy
  force: boolean = false,  // bypass cache
  projectId?: string  // scope models to project's classification_models
): Promise<MultiModelResponse> {
  const scopedProjectId = normalizeProjectId(projectId);
  const backend = await fetchApi<BackendMultiModelResponse>(
    "/api/analyze-multi",
    {
      method: "POST",
      body: JSON.stringify({
        slide_id: slideId,
        models: (models === undefined ? null : models),
        project_id: scopedProjectId,
        return_attention: returnAttention,
        level: level,
        force: force,
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
      cancerSpecific: Object.entries(backend.by_category)
        .filter(([key]) => key !== "general_pathology")
        .flatMap(([, preds]) => preds.map(transformModelPrediction)),
      generalPathology: (backend.by_category.general_pathology ?? []).map(transformModelPrediction),
    },
    nPatches: backend.n_patches,
    processingTimeMs: backend.processing_time_ms,
  };
}


// ====== Cached Results API ======

interface CachedResultEntry {
  model_id: string;
  score: number;
  label: string;
  confidence: number;
  threshold: number | null;
  created_at: string | null;
}

interface CachedResultsResponse {
  slide_id: string;
  results: CachedResultEntry[];
  count: number;
  cached: boolean;
}

/**
 * Fetch cached analysis results for a slide.
 * Returns empty array if no cached results exist.
 */
export async function getSlideCachedResults(
  slideId: string,
  projectId?: string
): Promise<CachedResultsResponse> {
  const scopedProjectId = normalizeProjectId(projectId);
  const qs = new URLSearchParams();
  if (scopedProjectId) qs.set("project_id", scopedProjectId);
  const endpoint =
    `/api/slides/${encodeURIComponent(slideId)}/cached-results` +
    (qs.toString() ? `?${qs.toString()}` : "");

  try {
    return await fetchApi<CachedResultsResponse>(
      endpoint
    );
  } catch (err) {
    // Cache is optional - return empty on failure
    return { slide_id: slideId, results: [], count: 0, cached: false };
  }
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
 * Falls back to empty array if endpoint not available
 */
export async function getAllTags(): Promise<Tag[]> {
  try {
    const backend = await fetchApi<BackendTag[]>("/api/tags", {}, { skipRetry: true });
    return backend.map(t => ({
      name: t.name,
      color: t.color,
      count: t.count,
    }));
  } catch (error) {
    // Endpoint not implemented - return empty array
    if (error instanceof AtlasApiError && error.statusCode === 404) {
      console.warn("[API] Tags endpoint not available, returning empty array");
      return [];
    }
    throw error;
  }
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
 * Falls back to empty array if endpoint not available
 */
export async function getGroups(): Promise<Group[]> {
  try {
    const backend = await fetchApi<BackendGroup[]>("/api/groups", {}, { skipRetry: true });
    return backend.map(g => ({
      id: g.id,
      name: g.name,
      description: g.description,
      slideIds: g.slide_ids,
      createdAt: g.created_at,
      updatedAt: g.updated_at,
    }));
  } catch (error) {
    // Endpoint not implemented - return empty array
    if (error instanceof AtlasApiError && error.statusCode === 404) {
      console.warn("[API] Groups endpoint not available, returning empty array");
      return [];
    }
    throw error;
  }
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
 * Apply client-side filtering when search endpoint is not available
 */
function applyClientSideFilters(
  slides: SlideInfo[],
  filters: SlideFilters
): { slides: SlideInfo[]; total: number } {
  let filtered = [...slides];

  // Search filter (filename/id)
  if (filters.search) {
    const searchLower = filters.search.toLowerCase();
    filtered = filtered.filter(
      (s) =>
        s.id.toLowerCase().includes(searchLower) ||
        s.filename.toLowerCase().includes(searchLower)
    );
  }

  // Label filter - map UI values to data values (project-aware)
  if (filters.label) {
    // Generic label mapping for common patterns; also supports raw label matching
    const labelMap: Record<string, string> = {
      "platinum_sensitive": "1",
      "Sensitive": "1",
      "platinum_resistant": "0", 
      "Resistant": "0",
      "Advanced": "1",
      "Early": "0",
      "ADVANCED": "1",
      "EARLY": "0",
    };
    const dataLabel = labelMap[filters.label] ?? filters.label;
    // Try both mapped value and original label for case-insensitive matching
    filtered = filtered.filter((s) => 
      s.label === dataLabel || 
      s.label?.toLowerCase() === filters.label?.toLowerCase()
    );
  }

  // Embeddings filter
  if (filters.hasEmbeddings !== undefined) {
    filtered = filtered.filter((s) => s.hasEmbeddings === filters.hasEmbeddings);
  }

  // Patch count filters
  if (filters.minPatches !== undefined) {
    filtered = filtered.filter((s) => (s.numPatches ?? 0) >= filters.minPatches!);
  }
  if (filters.maxPatches !== undefined) {
    filtered = filtered.filter((s) => (s.numPatches ?? 0) <= filters.maxPatches!);
  }

  // Starred filter (starred may be on ExtendedSlideInfo)
  if (filters.starred) {
    filtered = filtered.filter((s) => (s as { starred?: boolean }).starred === true);
  }

  // Sort
  if (filters.sortBy) {
    filtered.sort((a, b) => {
      let aVal: unknown, bVal: unknown;
      switch (filters.sortBy) {
        case "name":
          aVal = a.filename;
          bVal = b.filename;
          break;
        case "date":
          aVal = a.createdAt;
          bVal = b.createdAt;
          break;
        case "patches":
          aVal = a.numPatches ?? 0;
          bVal = b.numPatches ?? 0;
          break;
        default:
          aVal = a.id;
          bVal = b.id;
      }
      if (aVal === bVal) return 0;
      const cmp = aVal! < bVal! ? -1 : 1;
      return filters.sortOrder === "desc" ? -cmp : cmp;
    });
  }

  const total = filtered.length;

  // Pagination
  const page = filters.page ?? 1;
  const perPage = filters.perPage ?? 20;
  const start = (page - 1) * perPage;
  const paginated = filtered.slice(start, start + perPage);

  return { slides: paginated, total };
}

/**
 * Search and filter slides with pagination
 * Falls back to client-side filtering if search endpoint not available
 */
export async function searchSlides(
  filters: SlideFilters,
  projectId?: string
): Promise<SlideSearchResult> {
  const params = new URLSearchParams();
  const scopedProjectId = normalizeProjectId(projectId);
  
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
  if (scopedProjectId) params.set("project_id", scopedProjectId);

  try {
    const backend = await fetchApi<BackendSlideSearchResult>(
      `/api/slides/search?${params.toString()}`,
      {},
      { skipRetry: true }
    );

    return {
      slides: backend.slides.map(s => ({
        id: s.slide_id,
        filename: `${s.slide_id}.svs`,
        dimensions: s.dimensions ?? { width: 0, height: 0 },
        magnification: s.magnification ? parseInt(s.magnification.replace('x', ''), 10) : 40,
        mpp: s.mpp ?? 0.25,
        createdAt: new Date().toISOString(),
        label: normalizeLabel(s.label),
        hasEmbeddings: s.has_embeddings,
        numPatches: normalizePatchCount(s.num_patches),
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
  } catch (error) {
    // Endpoint not implemented - fall back to client-side filtering
    if (error instanceof AtlasApiError && error.statusCode === 404) {
      console.warn("[API] Search endpoint not available, using client-side filtering");
      
      // Fetch all slides and filter client-side
      const allSlides = await getSlides({
        projectId: scopedProjectId || undefined,
      });
      const { slides: filtered, total } = applyClientSideFilters(allSlides.slides, filters);
      
      return {
        slides: filtered,
        total,
        page: filters.page ?? 1,
        perPage: filters.perPage ?? 20,
        filters,
      };
    }
    throw error;
  }
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
  level: number = 0,
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
  level: number = 0,
  force: boolean = false,
  onProgress?: (status: { phase: "embedding" | "complete"; progress: number; message: string }) => void
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
export async function startReportGeneration(request: ReportRequest): Promise<AsyncReportResponse> {
  const projectId = normalizeProjectId(request.projectId);
  const payload: Record<string, unknown> = {
    slide_id: request.slideId,
  };

  if (projectId) {
    payload.project_id = projectId;
  }
  if (typeof request.includeDetails === "boolean") {
    payload.include_evidence = request.includeDetails;
    payload.include_similar = request.includeDetails;
  }

  return fetchApi<AsyncReportResponse>(
    '/api/report/async',
    {
      method: 'POST',
      body: JSON.stringify(payload),
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
  const asyncResponse = await startReportGeneration(request);
  const taskId = asyncResponse.task_id;
  
  // Poll for completion
  const maxWaitMs = 420000; // Keep polling long enough for slow CPU MedGemma runs
  const pollIntervalMs = 2000; // Poll every 2 seconds
  const startTime = Date.now();
  let lastProgress = 0;
  let progressStalledSince = 0;
  
  while (Date.now() - startTime < maxWaitMs) {
    let status: ReportTaskStatus;
    try {
      status = await getReportStatus(taskId);
    } catch (pollError) {
      // Transient poll failure — wait and retry
      console.warn('[API] Report status poll failed:', pollError);
      await new Promise(resolve => setTimeout(resolve, pollIntervalMs));
      continue;
    }
    
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
    
    // Detect stalled progress (same value for >60s = likely stuck)
    if (status.progress !== lastProgress) {
      lastProgress = status.progress;
      progressStalledSince = Date.now();
    } else if (progressStalledSince > 0 && Date.now() - progressStalledSince > 60000) {
      console.warn(`[API] Report progress stalled at ${status.progress}% for >60s`);
      // Don't throw yet — backend auto-expiry should handle it
    }
    
    // Wait before next poll
    await new Promise(resolve => setTimeout(resolve, pollIntervalMs));
  }
  
  throw new AtlasApiError({
    code: 'REPORT_TIMEOUT',
    message: 'Report generation timed out. The backend may still be processing — please retry.',
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
      `Tissue region (${e.coordinates?.[0] || 0}, ${e.coordinates?.[1] || 0})`,
    whyThisPatchMatters: e.significance ||
      "Regional morphology contributes to the model's slide-level prediction signal.",
  }));

  const similarExamples = (reportJson.similar_examples || []).map((s) => ({
    exampleId: s.example_id || s.slide_id || 'unknown',
    label: s.label || 'unknown',
    distance: s.distance ?? (1 - (s.similarity_score || 0)),
  }));

  const decisionSupport = toFrontendDecisionSupport(reportJson.decision_support);

  return {
    caseId: reportJson.case_id || backend.slide_id,
    task: reportJson.task || 'Treatment response prediction',
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
    limitations: reportJson.limitations?.length ? reportJson.limitations : [],
    suggestedNextSteps: reportJson.suggested_next_steps?.length ? reportJson.suggested_next_steps : [],
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
    model_results?: Array<{
      model_id: string;
      model_name: string;
      prediction: string;
      score: number;
      confidence: number;
      positive_label: string;
      negative_label: string;
      error?: string;
    }> | null;
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
  concurrency: number = 4,
  options?: {
    modelIds?: string[];
    level?: number;
    forceReembed?: boolean;
  }
): Promise<{ task_id: string; status: string; total_slides: number; message: string }> {
  return fetchApi<{ task_id: string; status: string; total_slides: number; message: string }>(
    "/api/analyze-batch/async",
    {
      method: "POST",
      body: JSON.stringify({
        slide_ids: slideIds,
        concurrency,
        model_ids: options?.modelIds ?? null,
        level: options?.level ?? 0,
        force_reembed: options?.forceReembed ?? false,
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
      modelResults: r.model_results?.map((mr) => ({
        modelId: mr.model_id,
        modelName: mr.model_name,
        prediction: mr.prediction,
        score: mr.score,
        confidence: mr.confidence,
        positiveLabel: mr.positive_label,
        negativeLabel: mr.negative_label,
        error: mr.error,
      })),
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

// ====== Visual Search (Image-to-Image Similarity) ======

// Backend response type (snake_case)
interface BackendVisualSearchResultPatch {
  slide_id: string;
  patch_index: number;
  coordinates?: [number, number];
  distance: number;
  similarity: number;
  label?: string;
  thumbnail_url?: string;
}

interface BackendVisualSearchResponse {
  query_slide_id?: string;
  query_patch_index?: number;
  query_coordinates?: [number, number];
  results: BackendVisualSearchResultPatch[];
  total_patches_searched: number;
  search_time_ms: number;
}

/**
 * Search for visually similar patches across the database using FAISS.
 * 
 * Finds histologically similar regions by comparing patch embeddings.
 * Can search by:
 * - Patch index within a slide
 * - Coordinates within a slide
 * - Direct embedding vector
 */
export async function visualSearch(
  request: import("@/types").VisualSearchRequest
): Promise<import("@/types").VisualSearchResponse> {
  const body: Record<string, unknown> = {
    top_k: request.topK ?? 10,
    exclude_same_slide: request.excludeSameSlide ?? true,
  };
  
  if (request.slideId) body.slide_id = request.slideId;
  if (request.patchIndex !== undefined) body.patch_index = request.patchIndex;
  if (request.coordinates) body.coordinates = request.coordinates;
  if (request.patchEmbedding) body.patch_embedding = request.patchEmbedding;
  
  const backend = await fetchApi<BackendVisualSearchResponse>(
    "/api/search/visual",
    {
      method: "POST",
      body: JSON.stringify(body),
    },
    { timeoutMs: 30000 }
  );
  
  return {
    querySlideId: backend.query_slide_id,
    queryPatchIndex: backend.query_patch_index,
    queryCoordinates: backend.query_coordinates,
    results: backend.results.map((r) => ({
      slideId: r.slide_id,
      patchIndex: r.patch_index,
      coordinates: r.coordinates,
      distance: r.distance,
      similarity: r.similarity,
      label: r.label,
      thumbnailUrl: r.thumbnail_url,
    })),
    totalPatchesSearched: backend.total_patches_searched,
    searchTimeMs: backend.search_time_ms,
  };
}

/**
 * Get the status of the visual search FAISS index.
 */
export async function getVisualSearchStatus(): Promise<import("@/types").VisualSearchStatus> {
  const backend = await fetchApi<{
    index_loaded: boolean;
    total_patches: number;
    total_slides: number;
    embedding_dim: number;
  }>("/api/search/visual/status");
  
  return {
    indexLoaded: backend.index_loaded,
    totalPatches: backend.total_patches,
    totalSlides: backend.total_slides,
    embeddingDim: backend.embedding_dim,
  };
}

// ====== Batch Re-Embed API ======

export interface BatchEmbedProgress {
  task_id: string;
  status: "pending" | "running" | "completed" | "cancelled" | "failed";
  level: number;
  force: boolean;
  concurrency: number;
  progress: number;
  current_slide_index: number;
  current_slide_id: string;
  total_slides: number;
  completed_slides: number;
  message: string;
  error?: string;
  elapsed_seconds: number;
  cancel_requested: boolean;
  // Present when completed
  results?: Array<{
    slide_id: string;
    status: string;
    num_patches: number;
    processing_time_seconds: number;
    error?: string;
  }>;
  summary?: {
    total: number;
    succeeded: number;
    failed: number;
    skipped: number;
    total_patches: number;
    elapsed_seconds: number;
  };
}

/**
 * Start batch re-embedding of all (or specific) slides.
 * Returns a batch_task_id to poll for progress.
 */
export async function startBatchEmbed(params: {
  level?: number;
  force?: boolean;
  slideIds?: string[];
  concurrency?: number;
} = {}): Promise<{
  batch_task_id: string;
  status: string;
  total: number;
  message: string;
}> {
  return fetchApi<{
    batch_task_id: string;
    status: string;
    total: number;
    message: string;
  }>(
    "/api/embed-slides/batch",
    {
      method: "POST",
      body: JSON.stringify({
        level: params.level ?? 0,
        force: params.force ?? true,
        slide_ids: params.slideIds ?? null,
        concurrency: params.concurrency ?? 1,
      }),
    },
    { timeoutMs: 30000 }
  );
}

/**
 * Get progress of a batch embedding task.
 */
export async function getBatchEmbedStatus(batchTaskId: string): Promise<BatchEmbedProgress> {
  return fetchApi<BatchEmbedProgress>(
    `/api/embed-slides/batch/status/${encodeURIComponent(batchTaskId)}`,
    {},
    { timeoutMs: 10000, skipRetry: true }
  );
}

/**
 * Cancel a running batch embedding task.
 */
export async function cancelBatchEmbed(batchTaskId: string): Promise<{ success: boolean; message: string }> {
  return fetchApi<{ success: boolean; message: string }>(
    `/api/embed-slides/batch/cancel/${encodeURIComponent(batchTaskId)}`,
    { method: "POST" }
  );
}

/**
 * Get currently active batch embed task (if any).
 */
export async function getActiveBatchEmbed(): Promise<BatchEmbedProgress | { status: "idle"; message: string }> {
  return fetchApi<BatchEmbedProgress | { status: "idle"; message: string }>(
    "/api/embed-slides/batch/active",
    {},
    { timeoutMs: 10000, skipRetry: true }
  );
}

/**
 * Poll batch embedding until done. Calls onProgress for each update.
 */
export async function pollBatchEmbed(
  batchTaskId: string,
  onProgress?: (status: BatchEmbedProgress) => void,
  pollIntervalMs: number = 3000,
  maxWaitMs: number = 86400000 // 24 hours for overnight runs
): Promise<BatchEmbedProgress> {
  const startTime = Date.now();

  while (Date.now() - startTime < maxWaitMs) {
    try {
      const status = await getBatchEmbedStatus(batchTaskId);
      if (onProgress) onProgress(status);

      if (
        status.status === "completed" ||
        status.status === "failed" ||
        status.status === "cancelled"
      ) {
        return status;
      }

      await new Promise((resolve) => setTimeout(resolve, pollIntervalMs));
    } catch (error) {
      console.warn("Batch embed polling error, retrying...", error);
      await new Promise((resolve) => setTimeout(resolve, pollIntervalMs * 2));
    }
  }

  throw new AtlasApiError({
    code: "TIMEOUT",
    message: `Batch embedding did not complete within ${maxWaitMs / 3600000} hours`,
    isTimeout: true,
  });
}

// ====== Project-Scoped Resources API ======

/**
 * Get model IDs assigned to a project
 */
export async function getProjectModels(projectId: string): Promise<string[]> {
  const resp = await fetchApi<{ model_ids: string[] } | string[]>(
    `/api/projects/${encodeURIComponent(projectId)}/models`
  );
  // Backend returns {project_id, model_ids, count} -- extract the array
  if (Array.isArray(resp)) return resp;
  if (resp && typeof resp === "object" && "model_ids" in resp) return resp.model_ids;
  return [];
}

/**
 * Assign slides to a project
 */
export async function assignSlidesToProject(projectId: string, slideIds: string[]): Promise<unknown> {
  return fetchApi(
    `/api/projects/${encodeURIComponent(projectId)}/slides`,
    {
      method: "POST",
      body: JSON.stringify({ slide_ids: slideIds }),
    }
  );
}

/**
 * Unassign slides from a project
 */
export async function unassignSlidesFromProject(projectId: string, slideIds: string[]): Promise<unknown> {
  return fetchApi(
    `/api/projects/${encodeURIComponent(projectId)}/slides`,
    {
      method: "DELETE",
      body: JSON.stringify({ slide_ids: slideIds }),
    }
  );
}

/**
 * Assign models to a project
 */
export async function assignModelsToProject(projectId: string, modelIds: string[]): Promise<unknown> {
  return fetchApi(
    `/api/projects/${encodeURIComponent(projectId)}/models`,
    {
      method: "POST",
      body: JSON.stringify({ model_ids: modelIds }),
    }
  );
}

/**
 * Unassign models from a project
 */
export async function unassignModelsFromProject(projectId: string, modelIds: string[]): Promise<unknown> {
  return fetchApi(
    `/api/projects/${encodeURIComponent(projectId)}/models`,
    {
      method: "DELETE",
      body: JSON.stringify({ model_ids: modelIds }),
    }
  );
}

// -- Modularity overhaul API additions --

export interface AvailableModelDetail {
  id: string;
  displayName: string;
  description: string;
  auc: number;
  category: string;
  positiveLabel: string;
  negativeLabel: string;
}

/**
 * Fetch available classification models for a project from the API.
 * Returns an empty list when no projectId is provided to avoid accidental global-model leakage.
 */
export async function getProjectAvailableModels(
  projectId: string
): Promise<AvailableModelDetail[]> {
  if (!projectId || projectId === "default") {
    return [];
  }

  const resp = await fetchApi<{ models: BackendAvailableModel[] }>(
    `/api/models?project_id=${encodeURIComponent(projectId)}`
  );
  if (resp && typeof resp === "object" && "models" in resp) {
    return resp.models.map((m) => ({
      id: m.id,
      displayName: m.name,
      description: m.description,
      auc: m.auc,
      category: m.category,
      positiveLabel: m.positive_label,
      negativeLabel: m.negative_label,
    }));
  }
  return [];
}

/**
 * Rename a slide (set display_name alias).
 * Pass null to clear the alias.
 */
export async function renameSlide(
  slideId: string,
  displayName: string | null
): Promise<{ slide_id: string; display_name: string | null }> {
  return fetchApi(`/api/slides/${encodeURIComponent(slideId)}`, {
    method: "PATCH",
    body: JSON.stringify({ display_name: displayName }),
  });
}

export interface SlideEmbeddingStatus {
  slide_id: string;
  has_level0_embeddings: boolean;
  has_level1_embeddings: boolean;
  num_patches: number | null;
  embedding_date: string | null;
  cached_model_ids: string[];
}

/**
 * Get embedding and analysis status for a slide.
 */
export async function getSlideEmbeddingStatus(
  slideId: string
): Promise<SlideEmbeddingStatus> {
  return fetchApi(`/api/slides/${encodeURIComponent(slideId)}/embedding-status`);
}

// ====== Patch Coordinates ======

/**
 * Fetch patch (x,y) coordinates for a slide.
 * Used for spatial selection on the WSI viewer.
 */
export async function getPatchCoords(
  slideId: string
): Promise<{ slideId: string; count: number; coords: Array<[number, number]> }> {
  const backend = await fetchApi<{
    slide_id: string;
    count: number;
    coords: Array<[number, number]>;
  }>(
    `/api/slides/${encodeURIComponent(slideId)}/patch-coords`,
    {},
    { timeoutMs: 15000 }
  );

  return {
    slideId: backend.slide_id,
    count: backend.count,
    coords: backend.coords,
  };
}

// ====== Outlier Tissue Detection ======

// Backend response (snake_case)
interface BackendOutlierPatch {
  patch_idx: number;
  x: number;
  y: number;
  distance: number;
  z_score: number;
}

interface BackendOutlierDetectionResponse {
  slide_id: string;
  outlier_patches: BackendOutlierPatch[];
  total_patches: number;
  outlier_count: number;
  mean_distance: number;
  std_distance: number;
  threshold: number;
  heatmap_data: Array<{ x: number; y: number; score: number }>;
}

/**
 * Detect outlier tissue patches using embedding distance from centroid.
 */
export async function detectOutliers(
  slideId: string,
  threshold?: number
): Promise<import("@/types").OutlierDetectionResult> {
  const params = threshold !== undefined ? `?threshold=${threshold}` : "";
  const backend = await fetchApi<BackendOutlierDetectionResponse>(
    `/api/slides/${encodeURIComponent(slideId)}/outlier-detection${params}`,
    { method: "POST" },
    { timeoutMs: 30000 }
  );

  return {
    slideId: backend.slide_id,
    outlierPatches: backend.outlier_patches.map((p) => ({
      patchIdx: p.patch_idx,
      x: p.x,
      y: p.y,
      distance: p.distance,
      zScore: p.z_score,
    })),
    totalPatches: backend.total_patches,
    outlierCount: backend.outlier_count,
    meanDistance: backend.mean_distance,
    stdDistance: backend.std_distance,
    threshold: backend.threshold,
    heatmapData: backend.heatmap_data,
  };
}

// ====== Few-Shot Patch Classification ======

// Backend response (snake_case)
interface BackendPatchClassificationItem {
  patch_idx: number;
  x: number;
  y: number;
  predicted_class: string;
  confidence: number;
  probabilities: Record<string, number>;
}

interface BackendPatchClassifyResponse {
  slide_id: string;
  classes: string[];
  total_patches: number;
  predictions: BackendPatchClassificationItem[];
  class_counts: Record<string, number>;
  accuracy_estimate: number | null;
  heatmap_data: Array<{ x: number; y: number; class_idx: number; confidence: number }>;
}

/**
 * Few-shot patch classification: train logistic regression on user-selected
 * example patches and classify all patches in the slide.
 */
export async function classifyPatches(
  slideId: string,
  classes: Record<string, number[]>
): Promise<import("@/types").PatchClassifyResult> {
  const backend = await fetchApi<BackendPatchClassifyResponse>(
    `/api/slides/${encodeURIComponent(slideId)}/patch-classify`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ classes }),
    },
    { timeoutMs: 30000 }
  );

  return {
    slideId: backend.slide_id,
    classes: backend.classes,
    totalPatches: backend.total_patches,
    predictions: backend.predictions.map((p) => ({
      patchIdx: p.patch_idx,
      x: p.x,
      y: p.y,
      predictedClass: p.predicted_class,
      confidence: p.confidence,
      probabilities: p.probabilities,
    })),
    classCounts: backend.class_counts,
    accuracyEstimate: backend.accuracy_estimate,
    heatmapData: backend.heatmap_data.map((h) => ({
      x: h.x,
      y: h.y,
      classIdx: h.class_idx,
      confidence: h.confidence,
    })),
  };
}
