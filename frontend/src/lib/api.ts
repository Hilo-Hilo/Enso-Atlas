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

/**
 * Fetch list of available slides
 */
export async function getSlides(): Promise<SlidesListResponse> {
  return fetchApi<SlidesListResponse>("/api/slides");
}

/**
 * Get details for a specific slide
 */
export async function getSlide(slideId: string): Promise<SlideInfo> {
  return fetchApi<SlideInfo>(`/api/slides/${slideId}`);
}

/**
 * Analyze a slide with the pathology model
 */
export async function analyzeSlide(
  request: AnalysisRequest
): Promise<AnalysisResponse> {
  return fetchApi<AnalysisResponse>("/api/analyze", {
    method: "POST",
    body: JSON.stringify(request),
  });
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
