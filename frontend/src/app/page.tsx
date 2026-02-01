"use client";

import React, { useState, useCallback, useEffect } from "react";
import dynamic from "next/dynamic";
import { Header } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";
import {
  SlideSelector,
  PredictionPanel,
  EvidencePanel,
  SimilarCasesPanel,
  ReportPanel,
} from "@/components/panels";
import { useAnalysis } from "@/hooks/useAnalysis";
import { getDziUrl, healthCheck, exportReportPdf } from "@/lib/api";
import type { SlideInfo, PatchCoordinates } from "@/types";

// Dynamically import WSIViewer to prevent SSR issues with OpenSeadragon
const WSIViewer = dynamic(
  () => import("@/components/viewer/WSIViewer").then((mod) => mod.WSIViewer),
  { ssr: false, loading: () => <div className="h-full flex items-center justify-center bg-gray-100 rounded-lg">Loading viewer...</div> }
);

export default function HomePage() {
  // State
  const [selectedSlide, setSelectedSlide] = useState<SlideInfo | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [selectedPatchId, setSelectedPatchId] = useState<string | undefined>();

  // Analysis hook
  const {
    isAnalyzing,
    isGeneratingReport,
    analysisResult,
    report,
    error,
    analyze,
    generateSlideReport,
    clearResults,
    clearError,
  } = useAnalysis();

  // Check backend connection
  useEffect(() => {
    const checkConnection = async () => {
      try {
        await healthCheck();
        setIsConnected(true);
      } catch {
        setIsConnected(false);
      }
    };

    checkConnection();
    const interval = setInterval(checkConnection, 30000);
    return () => clearInterval(interval);
  }, []);

  // Handle slide selection
  const handleSlideSelect = useCallback(
    (slide: SlideInfo) => {
      setSelectedSlide(slide);
      clearResults();
      setSelectedPatchId(undefined);
    },
    [clearResults]
  );

  // Handle analyze button
  const handleAnalyze = useCallback(async () => {
    if (!selectedSlide) return;

    await analyze({
      slideId: selectedSlide.id,
      patchBudget: 8000,
      magnification: 20,
    });
  }, [selectedSlide, analyze]);

  // Handle patch click - navigate viewer
  const handlePatchClick = useCallback((coords: PatchCoordinates) => {
    // The WSIViewer will handle navigation internally via state/props
    setSelectedPatchId(`${coords.x}_${coords.y}`);
  }, []);

  // Handle similar case click
  const handleCaseClick = useCallback((caseId: string) => {
    // Open similar case in new tab (placeholder - would navigate to case detail view)
    window.open(`/case/${caseId}`, "_blank");
  }, []);

  // Handle report generation
  const handleGenerateReport = useCallback(async () => {
    if (!selectedSlide || !analysisResult) return;

    await generateSlideReport({
      slideId: selectedSlide.id,
      evidencePatchIds: analysisResult.evidencePatches.map((p) => p.id),
      includeDetails: true,
    });
  }, [selectedSlide, analysisResult, generateSlideReport]);

  // Handle PDF export
  const handleExportPdf = useCallback(async () => {
    if (!selectedSlide) return;
    try {
      const blob = await exportReportPdf(selectedSlide.id);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `atlas-report-${selectedSlide.id}.pdf`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error("PDF export failed:", err);
    }
  }, [selectedSlide]);

  // Handle JSON export
  const handleExportJson = useCallback(async () => {
    if (!selectedSlide || !report) return;
    const blob = new Blob([JSON.stringify(report, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `atlas-report-${selectedSlide.id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [selectedSlide, report]);

  // Get DZI and heatmap URLs
  const dziUrl = selectedSlide ? getDziUrl(selectedSlide.id) : undefined;
  const heatmapData = analysisResult?.heatmap;

  return (
    <div className="flex flex-col h-screen bg-surface-secondary">
      {/* Header */}
      <Header
        isConnected={isConnected}
        isProcessing={isAnalyzing || isGeneratingReport}
        version="0.1.0"
        institutionName="Enso Labs"
        userName="Clinician"
      />

      {/* Main Content */}
      <main className="flex-1 flex overflow-hidden">
        {/* Left Sidebar - Slide Selection */}
        <aside className="w-80 border-r border-surface-border bg-white p-4 overflow-y-auto shrink-0">
          <SlideSelector
            selectedSlideId={selectedSlide?.id ?? null}
            onSlideSelect={handleSlideSelect}
            onAnalyze={handleAnalyze}
            isAnalyzing={isAnalyzing}
          />

          {/* Error Display */}
          {error && (
            <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-sm text-red-700">{error}</p>
              <button
                onClick={clearError}
                className="mt-2 text-xs text-red-600 underline"
              >
                Dismiss
              </button>
            </div>
          )}
        </aside>

        {/* Center - WSI Viewer */}
        <section className="flex-1 p-4 overflow-hidden">
          {selectedSlide && dziUrl ? (
            <WSIViewer
              slideId={selectedSlide.id}
              dziUrl={dziUrl}
              heatmap={heatmapData}
              mpp={selectedSlide.mpp}
              onRegionClick={handlePatchClick}
              className="h-full"
            />
          ) : (
            <div className="h-full flex items-center justify-center bg-gray-100 rounded-lg border-2 border-dashed border-gray-300">
              <div className="text-center">
                <div className="w-16 h-16 bg-gray-200 rounded-full flex items-center justify-center mx-auto mb-4">
                  <svg
                    className="w-8 h-8 text-gray-400"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                    />
                  </svg>
                </div>
                <h3 className="text-lg font-medium text-gray-900 mb-1">
                  No Slide Selected
                </h3>
                <p className="text-sm text-gray-500 max-w-sm">
                  Select a whole-slide image from the sidebar to begin analysis.
                  The slide will be displayed here with interactive viewing and
                  heatmap overlay capabilities.
                </p>
              </div>
            </div>
          )}
        </section>

        {/* Right Sidebar - Results */}
        <aside className="w-96 border-l border-surface-border bg-white p-4 overflow-y-auto shrink-0 space-y-4">
          {/* Prediction Results */}
          <PredictionPanel
            prediction={analysisResult?.prediction ?? null}
            isLoading={isAnalyzing}
            processingTime={analysisResult?.processingTimeMs}
          />

          {/* Evidence Patches */}
          <EvidencePanel
            patches={analysisResult?.evidencePatches ?? []}
            isLoading={isAnalyzing}
            onPatchClick={handlePatchClick}
            selectedPatchId={selectedPatchId}
          />

          {/* Similar Cases */}
          <SimilarCasesPanel
            cases={analysisResult?.similarCases ?? []}
            isLoading={isAnalyzing}
            onCaseClick={handleCaseClick}
          />

          {/* Clinical Report */}
          <ReportPanel
            report={report}
            isLoading={isGeneratingReport}
            onGenerateReport={
              analysisResult && !report ? handleGenerateReport : undefined
            }
            onExportPdf={report ? handleExportPdf : undefined}
            onExportJson={report ? handleExportJson : undefined}
          />
        </aside>
      </main>

      {/* Footer */}
      <Footer version="0.1.0" />
    </div>
  );
}
