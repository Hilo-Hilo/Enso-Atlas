// Enso Atlas - Type Definitions
// Professional pathology evidence engine types

// Patient demographic and clinical context
export interface PatientContext {
  age?: number;
  sex?: string;
  stage?: string;
  grade?: string;
  prior_lines?: number;
  histology?: string;
}

// Slide metadata from the backend
export interface SlideInfo {
  id: string;
  filename: string;
  dimensions: {
    width: number;
    height: number;
  };
  magnification: number;
  mpp: number; // microns per pixel
  thumbnailUrl?: string;
  createdAt: string;
  // Extended fields from backend
  label?: string;
  hasEmbeddings?: boolean;
  hasLevel0Embeddings?: boolean; // Whether full-resolution (level 0) embeddings exist
  numPatches?: number;
  patient?: PatientContext;
}

// Patch coordinates and metadata
export interface PatchCoordinates {
  x: number;
  y: number;
  level: number;
  width: number;
  height: number;
}

// Tissue type classification
export type TissueType = "tumor" | "stroma" | "necrosis" | "inflammatory" | "normal" | "artifact" | "unknown";

// Evidence patch with attention weight
export interface EvidencePatch {
  id: string;
  patchId: string;
  coordinates: PatchCoordinates;
  attentionWeight: number;
  thumbnailUrl: string;
  morphologyDescription?: string;
  tissueType?: TissueType;
  tissueConfidence?: number;
}

// Similar case from FAISS retrieval
export interface SimilarCase {
  caseId?: string;
  slideId: string;
  patchId?: string;
  similarity: number;
  distance?: number;
  label?: string;
  thumbnailUrl?: string;
  coordinates?: PatchCoordinates;
}

// Prediction result from MIL model
export interface PredictionResult {
  label: string;
  score: number;
  confidence: number;
  calibrationNote?: string;
}

// Heatmap data for visualization
export interface HeatmapData {
  imageUrl: string;
  minValue?: number;
  maxValue?: number;
  colorScale?: string;
  opacity?: number;
  bounds?: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
}

// Structured report from MedGemma
export interface StructuredReport {
  caseId: string;
  task: string;
  generatedAt: string;
  patientContext?: PatientContext;
  modelOutput: PredictionResult;
  evidence: Array<{
    patchId: string;
    coordsLevel0: [number, number];
    morphologyDescription: string;
    whyThisPatchMatters: string;
  }>;
  similarExamples: Array<{
    exampleId: string;
    label: string;
    distance: number;
  }>;
  limitations: string[];
  suggestedNextSteps: string[];
  safetyStatement: string;
  summary: string;
  // Clinical decision support section
  decisionSupport?: DecisionSupport;
}

// Analysis request payload
export interface AnalysisRequest {
  slideId: string;
  taskType?: string;
  patchBudget?: number;
  magnification?: number;
}

// Analysis response from backend
export interface AnalysisResponse {
  slideInfo: SlideInfo;
  prediction: PredictionResult;
  evidencePatches: EvidencePatch[];
  similarCases: SimilarCase[];
  heatmap: HeatmapData;
  report?: StructuredReport;
  processingTimeMs: number;
}

// Report generation request
export interface ReportRequest {
  slideId: string;
  evidencePatchIds: string[];
  includeDetails?: boolean;
}

// API error response
export interface ApiError {
  code: string;
  message: string;
  details?: Record<string, unknown>;
}

// Slide list response
export interface SlidesListResponse {
  slides: SlideInfo[];
  total: number;
}

// Viewer state for OpenSeadragon
export interface ViewerState {
  zoom: number;
  center: { x: number; y: number };
  rotation: number;
  showHeatmap: boolean;
  heatmapOpacity: number;
}

// UI panel visibility state
export interface PanelVisibility {
  prediction: boolean;
  evidence: boolean;
  similarCases: boolean;
  report: boolean;
}

// Semantic search result from MedSigLIP
export interface SemanticSearchResult {
  patch_index: number;
  similarity: number;
  coordinates?: [number, number];
}

// Semantic search response
export interface SemanticSearchResponse {
  slide_id: string;
  query: string;
  results: SemanticSearchResult[];
}

// Annotation for pathologist review
export interface Annotation {
  id: string;
  slideId: string;
  type: "circle" | "rectangle" | "freehand" | "marker" | "note" | "measurement";
  coordinates: {
    x: number;
    y: number;
    width: number;
    height: number;
    points?: Array<{ x: number; y: number }>; // For freehand annotations
  };
  text?: string;
  color?: string;
  category?: string;
  createdAt: string;
  createdBy?: string;
}

// Annotation request/response types
export interface AnnotationRequest {
  slideId: string;
  type: Annotation["type"];
  coordinates: Annotation["coordinates"];
  text?: string;
  color?: string;
  category?: string;
}

export interface AnnotationsResponse {
  slideId: string;
  annotations: Annotation[];
  total: number;
}

// Slide quality control metrics
export interface SlideQCMetrics {
  slideId: string;
  tissueCoverage: number;  // 0-1, percentage of slide with tissue
  blurScore: number;       // 0-1, 0=sharp, 1=blurry
  stainUniformity: number; // 0-1, 0=poor, 1=excellent
  artifactDetected: boolean;
  penMarks: boolean;
  foldDetected: boolean;
  overallQuality: "poor" | "acceptable" | "good";
}

// Clinical guideline reference
export interface GuidelineReference {
  source: string;
  section: string;
  recommendation: string;
  url?: string;
}

// Uncertainty quantification result from MC Dropout
export interface UncertaintyResult {
  slideId: string;
  prediction: string;
  probability: number;
  uncertainty: number;
  confidenceInterval: [number, number];
  isUncertain: boolean;
  requiresReview: boolean;
  uncertaintyLevel: "low" | "moderate" | "high";
  clinicalRecommendation: string;
  patchesAnalyzed: number;
  nSamples: number;
  samples: number[];
  topEvidence: Array<{
    rank: number;
    patchIndex: number;
    attentionWeight: number;
    attentionUncertainty: number;
    coordinates: [number, number];
  }>;
}

// Risk stratification levels
export type RiskLevel = "high_confidence" | "moderate_confidence" | "low_confidence" | "inconclusive";
export type ConfidenceLevel = "high" | "moderate" | "low";

// Clinical decision support output
export interface DecisionSupport {
  risk_level: RiskLevel;
  confidence_level: ConfidenceLevel;
  confidence_score: number;
  primary_recommendation: string;
  supporting_rationale: string[];
  alternative_considerations: string[];
  guideline_references: GuidelineReference[];
  uncertainty_statement: string;
  quality_warnings: string[];
  suggested_workup: string[];
  interpretation_note: string;
  caveat: string;
}

// Batch analysis types for clinical workflow
export type UncertaintyLevel = "low" | "moderate" | "high" | "unknown";

export interface BatchAnalysisResult {
  slideId: string;
  prediction: string;
  score: number;
  confidence: number;
  patchesAnalyzed: number;
  requiresReview: boolean;
  uncertaintyLevel: UncertaintyLevel;
  error?: string;
}

export interface BatchAnalysisSummary {
  total: number;
  completed: number;
  failed: number;
  responders: number;
  nonResponders: number;
  uncertain: number;
  avgConfidence: number;
  requiresReviewCount: number;
}

export interface BatchAnalyzeRequest {
  slideIds: string[];
}

export interface BatchAnalyzeResponse {
  results: BatchAnalysisResult[];
  summary: BatchAnalysisSummary;
  processingTimeMs: number;
}

// ====== Multi-Model Prediction Types ======

// Single model prediction result
export interface ModelPrediction {
  modelId: string;
  modelName: string;
  category: 'ovarian_cancer' | 'general_pathology';
  score: number;
  label: string;
  positiveLabel: string;
  negativeLabel: string;
  confidence: number;
  auc: number;
  nTrainingSlides: number;
  description: string;
  confidenceInterval?: { lower: number; upper: number };
}

// Available model info
export interface AvailableModel {
  id: string;
  name: string;
  description: string;
  confidenceInterval?: { lower: number; upper: number };
  auc: number;
  nSlides: number;
  category: 'ovarian_cancer' | 'general_pathology';
  positiveLabel: string;
  negativeLabel: string;
  available: boolean;
}

// Multi-model request
export interface MultiModelRequest {
  slideId: string;
  models?: string[];  // null = run all models
  returnAttention?: boolean;
}

// Multi-model response
export interface MultiModelResponse {
  slideId: string;
  predictions: Record<string, ModelPrediction>;
  byCategory: {
    ovarianCancer: ModelPrediction[];
    generalPathology: ModelPrediction[];
  };
  nPatches: number;
  processingTimeMs: number;
}

// Available models response
export interface AvailableModelsResponse {
  models: AvailableModel[];
}

// ====== Slide Manager Types ======

// Tag for organizing slides
export interface Tag {
  name: string;
  color?: string;
  count: number;
}

// Group for collections of slides
export interface Group {
  id: string;
  name: string;
  description?: string;
  slideIds: string[];
  createdAt: string;
  updatedAt: string;
}

// Filters for searching/filtering slides
export interface SlideFilters {
  search?: string;
  tags?: string[];
  groupId?: string;
  hasEmbeddings?: boolean;
  label?: string;
  minPatches?: number;
  maxPatches?: number;
  starred?: boolean;
  dateFrom?: string;
  dateTo?: string;
  sortBy?: string;
  sortOrder?: 'asc' | 'desc';
  page?: number;
  perPage?: number;
}

// Search result with pagination
export interface SlideSearchResult {
  slides: SlideInfo[];
  total: number;
  page: number;
  perPage: number;
  filters: SlideFilters;
}

// Extended SlideInfo fields (add to existing interface via intersection or extend)
export interface ExtendedSlideInfo extends SlideInfo {
  tags?: string[];
  groups?: string[];
  starred?: boolean;
  customMetadata?: Record<string, unknown>;
  notes?: string;
}

// ====== Visual Search (Image-to-Image) Types ======

// Request for visual similarity search
export interface VisualSearchRequest {
  slideId?: string;
  patchIndex?: number;
  coordinates?: [number, number];
  patchEmbedding?: number[];
  topK?: number;
  excludeSameSlide?: boolean;
}

// Single result from visual search
export interface VisualSearchResultPatch {
  slideId: string;
  patchIndex: number;
  coordinates?: [number, number];
  distance: number;
  similarity: number;
  label?: string;
  thumbnailUrl?: string;
}

// Response from visual search
export interface VisualSearchResponse {
  querySlideId?: string;
  queryPatchIndex?: number;
  queryCoordinates?: [number, number];
  results: VisualSearchResultPatch[];
  totalPatchesSearched: number;
  searchTimeMs: number;
}

// Visual search status
export interface VisualSearchStatus {
  indexLoaded: boolean;
  totalPatches: number;
  totalSlides: number;
  embeddingDim: number;
}
