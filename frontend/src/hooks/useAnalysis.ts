// Enso Atlas - Analysis Hook
// State management for slide analysis workflow

import { useState, useCallback, useRef } from "react";
import type {
  AnalysisResponse,
  AnalysisRequest,
  StructuredReport,
  ReportRequest,
  UncertaintyResult,
} from "@/types";
import { analyzeSlide, generateReportWithProgress, analyzeWithUncertainty, generateReport } from "@/lib/api";

// Analysis progress steps
export const ANALYSIS_STEPS = [
  { id: "embeddings", label: "Loading slide embeddings...", description: "Retrieving pre-computed feature vectors" },
  { id: "prediction", label: "Running TransMIL prediction...", description: "Multiple instance learning inference" },
  { id: "similar", label: "Finding similar cases...", description: "FAISS vector similarity search" },
  { id: "evidence", label: "Generating evidence...", description: "Extracting top attention regions" },
] as const;

export type AnalysisStepId = typeof ANALYSIS_STEPS[number]["id"];

interface UseAnalysisState {
  isAnalyzing: boolean;
  isGeneratingReport: boolean;
  isAnalyzingUncertainty: boolean;
  analysisResult: AnalysisResponse | null;
  uncertaintyResult: UncertaintyResult | null;
  report: StructuredReport | null;
  error: string | null;
  // Progress tracking
  analysisStep: number; // -1 = not started, 0-3 = step index
  analysisStepId: AnalysisStepId | null;
  // Report generation progress
  reportProgress: number; // 0-100
  reportProgressMessage: string;
}

interface UseAnalysisReturn extends UseAnalysisState {
  analyze: (request: AnalysisRequest) => Promise<AnalysisResponse | null>;
  analyzeUncertainty: (slideId: string, nSamples?: number) => Promise<UncertaintyResult | null>;
  generateSlideReport: (request: ReportRequest) => Promise<void>;
  clearResults: () => void;
  clearError: () => void;
  retryAnalysis: () => Promise<AnalysisResponse | null>;
  retryReport: () => Promise<void>;
}

export function useAnalysis(): UseAnalysisReturn {
  const [state, setState] = useState<UseAnalysisState>({
    isAnalyzing: false,
    isGeneratingReport: false,
    isAnalyzingUncertainty: false,
    analysisResult: null,
    uncertaintyResult: null,
    report: null,
    error: null,
    analysisStep: -1,
    analysisStepId: null,
    reportProgress: 0,
    reportProgressMessage: "Preparing report generation...",
  });

  // Store last requests for retry functionality
  const lastAnalysisRequest = useRef<AnalysisRequest | null>(null);
  const lastReportRequest = useRef<ReportRequest | null>(null);
  const lastUncertaintySlideId = useRef<string | null>(null);

  // Simulate progress steps during analysis
  // In a real implementation, the backend would send progress updates via SSE/WebSocket
  const simulateProgress = useCallback((onComplete: () => void) => {
    const stepDurations = [400, 800, 600, 500]; // ms per step
    let currentStep = 0;

    const advanceStep = () => {
      if (currentStep < ANALYSIS_STEPS.length) {
        const step = ANALYSIS_STEPS[currentStep];
        if (step) {
          setState((prev) => ({
            ...prev,
            analysisStep: currentStep,
            analysisStepId: step.id,
          }));
        }
        currentStep++;
        setTimeout(advanceStep, stepDurations[currentStep - 1] || 500);
      } else {
        onComplete();
      }
    };

    advanceStep();
  }, []);

  const analyze = useCallback(async (request: AnalysisRequest): Promise<AnalysisResponse | null> => {
    lastAnalysisRequest.current = request;

    setState((prev) => ({
      ...prev,
      isAnalyzing: true,
      error: null,
      analysisStep: 0,
      analysisStepId: ANALYSIS_STEPS[0].id,
    }));

    // Start progress simulation
    let progressComplete = false;
    const progressPromise = new Promise<void>((resolve) => {
      simulateProgress(() => {
        progressComplete = true;
        resolve();
      });
    });

    try {
      // Run actual API call in parallel with progress simulation
      const [result] = await Promise.all([
        analyzeSlide(request),
        progressPromise,
      ]);

      // If API finished before progress, wait for progress to complete
      if (!progressComplete) {
        await progressPromise;
      }

      setState((prev) => ({
        ...prev,
        isAnalyzing: false,
        analysisResult: result,
        report: result.report || null,
        analysisStep: ANALYSIS_STEPS.length, // Completed
        analysisStepId: null,
      }));
      return result;
    } catch (err) {
      const message = err instanceof Error ? err.message : "Analysis failed";
      setState((prev) => ({
        ...prev,
        isAnalyzing: false,
        error: message,
        analysisStep: -1,
        analysisStepId: null,
      }));
      return null;
    }
  }, [simulateProgress]);

  const retryAnalysis = useCallback(async (): Promise<AnalysisResponse | null> => {
    if (lastAnalysisRequest.current) {
      return analyze(lastAnalysisRequest.current);
    }
    return null;
  }, [analyze]);

  const generateSlideReport = useCallback(async (request: ReportRequest) => {
    lastReportRequest.current = request;

    setState((prev) => ({
      ...prev,
      isGeneratingReport: true,
      error: null,
      reportProgress: 0,
      reportProgressMessage: "Starting report generation...",
    }));

    try {
      // Use async report generation with progress tracking
      const report = await generateReportWithProgress(
        request,
        (progress: number, message: string) => {
          setState((prev) => ({
            ...prev,
            reportProgress: progress,
            reportProgressMessage: message,
          }));
        }
      );
      
      setState((prev) => ({
        ...prev,
        isGeneratingReport: false,
        report,
        reportProgress: 100,
        reportProgressMessage: "Report generated successfully!",
      }));
    } catch (err) {
      // Fallback to synchronous if async fails
      console.warn("Async report generation failed, trying synchronous:", err);
      try {
        const report = await generateReport(request);
        setState((prev) => ({
          ...prev,
          isGeneratingReport: false,
          report,
          reportProgress: 100,
          reportProgressMessage: "Report generated successfully!",
        }));
      } catch (fallbackErr) {
        const message = fallbackErr instanceof Error ? fallbackErr.message : "Report generation failed";
        setState((prev) => ({
          ...prev,
          isGeneratingReport: false,
          error: message,
          reportProgress: 0,
          reportProgressMessage: "",
        }));
      }
    }
  }, []);

  const retryReport = useCallback(async () => {
    if (lastReportRequest.current) {
      return generateSlideReport(lastReportRequest.current);
    }
  }, [generateSlideReport]);

  // Uncertainty analysis with MC Dropout
  const analyzeUncertainty = useCallback(async (
    slideId: string,
    nSamples: number = 20
  ): Promise<UncertaintyResult | null> => {
    lastUncertaintySlideId.current = slideId;

    setState((prev) => ({
      ...prev,
      isAnalyzingUncertainty: true,
      error: null,
    }));

    try {
      const result = await analyzeWithUncertainty(slideId, nSamples);
      
      setState((prev) => ({
        ...prev,
        isAnalyzingUncertainty: false,
        uncertaintyResult: result,
      }));
      
      return result;
    } catch (err) {
      const message = err instanceof Error ? err.message : "Uncertainty analysis failed";
      setState((prev) => ({
        ...prev,
        isAnalyzingUncertainty: false,
        error: message,
      }));
      return null;
    }
  }, []);

  const clearResults = useCallback(() => {
    setState({
      isAnalyzing: false,
      isGeneratingReport: false,
      isAnalyzingUncertainty: false,
      analysisResult: null,
      uncertaintyResult: null,
      report: null,
      error: null,
      analysisStep: -1,
      analysisStepId: null,
      reportProgress: 0,
      reportProgressMessage: "",
    });
    lastAnalysisRequest.current = null;
    lastReportRequest.current = null;
    lastUncertaintySlideId.current = null;
  }, []);

  const clearError = useCallback(() => {
    setState((prev) => ({
      ...prev,
      error: null,
    }));
  }, []);

  return {
    ...state,
    analyze,
    analyzeUncertainty,
    generateSlideReport,
    clearResults,
    clearError,
    retryAnalysis,
    retryReport,
  };
}
