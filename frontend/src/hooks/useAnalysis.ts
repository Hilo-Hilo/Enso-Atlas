// Enso Atlas - Analysis Hook
// State management for slide analysis workflow

import { useState, useCallback, useRef, useEffect } from "react";
import type {
  AnalysisResponse,
  AnalysisRequest,
  StructuredReport,
  ReportRequest,
  UncertaintyResult,
} from "@/types";
import {
  analyzeSlide,
  generateReportWithProgress,
  analyzeWithUncertainty,
  generateReport,
} from "@/lib/api";

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

  // Track currently active analysis run to avoid stale async state updates.
  const analysisRunIdRef = useRef(0);
  const analysisProgressTimersRef = useRef<number[]>([]);

  const clearAnalysisProgressTimers = useCallback(() => {
    for (const timer of analysisProgressTimersRef.current) {
      clearTimeout(timer);
    }
    analysisProgressTimersRef.current = [];
  }, []);

  const startAnalysisProgressSimulation = useCallback(
    (runId: number) => {
      clearAnalysisProgressTimers();

      const stepDurations = [400, 800, 600, 500]; // ms per step

      const scheduleStep = (stepIndex: number, delayMs: number) => {
        const timerId = window.setTimeout(() => {
          if (analysisRunIdRef.current !== runId) return;
          const step = ANALYSIS_STEPS[stepIndex];
          if (!step) return;

          setState((prev) => ({
            ...prev,
            analysisStep: stepIndex,
            analysisStepId: step.id,
          }));

          if (stepIndex + 1 < ANALYSIS_STEPS.length) {
            scheduleStep(stepIndex + 1, stepDurations[stepIndex + 1] || 500);
          }
        }, delayMs);

        analysisProgressTimersRef.current.push(timerId);
      };

      // Step 0 is set immediately when analysis starts; advance through later steps on timers.
      if (ANALYSIS_STEPS.length > 1) {
        scheduleStep(1, stepDurations[0]);
      }
    },
    [clearAnalysisProgressTimers]
  );

  useEffect(() => {
    return () => {
      clearAnalysisProgressTimers();
    };
  }, [clearAnalysisProgressTimers]);

  const analyze = useCallback(async (request: AnalysisRequest): Promise<AnalysisResponse | null> => {
    const runId = ++analysisRunIdRef.current;
    lastAnalysisRequest.current = request;

    setState((prev) => ({
      ...prev,
      isAnalyzing: true,
      error: null,
      analysisStep: 0,
      analysisStepId: ANALYSIS_STEPS[0].id,
    }));

    startAnalysisProgressSimulation(runId);

    try {
      const result = await analyzeSlide(request);
      if (analysisRunIdRef.current !== runId) {
        return null;
      }

      clearAnalysisProgressTimers();

      setState((prev) => ({
        ...prev,
        isAnalyzing: false,
        analysisResult: result,
        report: result.report || null,
        analysisStep: Math.max(ANALYSIS_STEPS.length - 1, 0),
        analysisStepId: null,
      }));
      return result;
    } catch (err) {
      if (analysisRunIdRef.current !== runId) {
        return null;
      }

      clearAnalysisProgressTimers();

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
  }, [startAnalysisProgressSimulation, clearAnalysisProgressTimers]);

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
    analysisRunIdRef.current += 1;
    clearAnalysisProgressTimers();

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
  }, [clearAnalysisProgressTimers]);

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
