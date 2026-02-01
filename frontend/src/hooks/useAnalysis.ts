// Enso Atlas - Analysis Hook
// State management for slide analysis workflow

import { useState, useCallback } from "react";
import type {
  AnalysisResponse,
  AnalysisRequest,
  StructuredReport,
  ReportRequest,
} from "@/types";
import { analyzeSlide, generateReport } from "@/lib/api";

interface UseAnalysisState {
  isAnalyzing: boolean;
  isGeneratingReport: boolean;
  analysisResult: AnalysisResponse | null;
  report: StructuredReport | null;
  error: string | null;
}

interface UseAnalysisReturn extends UseAnalysisState {
  analyze: (request: AnalysisRequest) => Promise<AnalysisResponse | null>;
  generateSlideReport: (request: ReportRequest) => Promise<void>;
  clearResults: () => void;
  clearError: () => void;
}

export function useAnalysis(): UseAnalysisReturn {
  const [state, setState] = useState<UseAnalysisState>({
    isAnalyzing: false,
    isGeneratingReport: false,
    analysisResult: null,
    report: null,
    error: null,
  });

  const analyze = useCallback(async (request: AnalysisRequest): Promise<AnalysisResponse | null> => {
    setState((prev) => ({
      ...prev,
      isAnalyzing: true,
      error: null,
    }));

    try {
      const result = await analyzeSlide(request);
      setState((prev) => ({
        ...prev,
        isAnalyzing: false,
        analysisResult: result,
        report: result.report || null,
      }));
      return result;
    } catch (err) {
      const message = err instanceof Error ? err.message : "Analysis failed";
      setState((prev) => ({
        ...prev,
        isAnalyzing: false,
        error: message,
      }));
      return null;
    }
  }, []);

  const generateSlideReport = useCallback(async (request: ReportRequest) => {
    setState((prev) => ({
      ...prev,
      isGeneratingReport: true,
      error: null,
    }));

    try {
      const report = await generateReport(request);
      setState((prev) => ({
        ...prev,
        isGeneratingReport: false,
        report,
      }));
    } catch (err) {
      const message = err instanceof Error ? err.message : "Report generation failed";
      setState((prev) => ({
        ...prev,
        isGeneratingReport: false,
        error: message,
      }));
    }
  }, []);

  const clearResults = useCallback(() => {
    setState({
      isAnalyzing: false,
      isGeneratingReport: false,
      analysisResult: null,
      report: null,
      error: null,
    });
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
    generateSlideReport,
    clearResults,
    clearError,
  };
}
