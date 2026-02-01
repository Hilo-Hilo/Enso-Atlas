// Enso Atlas - Type Definitions
// Professional pathology evidence engine types

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
  numPatches?: number;
}

// Patch coordinates and metadata
export interface PatchCoordinates {
  x: number;
  y: number;
  level: number;
  width: number;
  height: number;
}

// Evidence patch with attention weight
export interface EvidencePatch {
  id: string;
  patchId: string;
  coordinates: PatchCoordinates;
  attentionWeight: number;
  thumbnailUrl: string;
  morphologyDescription?: string;
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
