"use client";

import React, { useState, useCallback, useEffect, useRef, useMemo } from "react";
import nextDynamic from "next/dynamic";
import { Header } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";
import {
  SlideSelector,
  PredictionPanel,
  EvidencePanel,
  SimilarCasesPanel,
  ReportPanel,
  SemanticSearchPanel,
  CaseNotesPanel,
  QuickStatsPanel,
  OncologistSummaryView,
  recordAnalysis,
  getCaseNotes,
} from "@/components/panels";
import { PatchZoomModal, KeyboardShortcutsModal } from "@/components/modals";
import { useAnalysis } from "@/hooks/useAnalysis";
import { useKeyboardShortcuts, type KeyboardShortcut } from "@/hooks/useKeyboardShortcuts";
import { getDziUrl, healthCheck, semanticSearch, getSlideQC } from "@/lib/api";
import { generatePdfReport, downloadPdf } from "@/lib/pdfExport";
import type { SlideInfo, PatchCoordinates, SemanticSearchResult, EvidencePatch, SlideQCMetrics } from "@/types";

// Dynamically import WSIViewer to prevent SSR issues with OpenSeadragon
const WSIViewer = nextDynamic(
  () => import("@/components/viewer/WSIViewer").then((mod) => mod.WSIViewer),
  { ssr: false, loading: () => <div className="h-full flex items-center justify-center bg-gray-100 rounded-lg">Loading viewer...</div> }
);


export default function HomePage() {
  // State
  const [selectedSlide, setSelectedSlide] = useState<SlideInfo | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [selectedPatchId, setSelectedPatchId] = useState<string | undefined>();

  // View mode: "wsi" for full WSI viewer, "summary" for oncologist summary
  const [viewMode, setViewMode] = useState<"wsi" | "summary">("wsi");

  // Patch zoom modal state
  const [zoomModalOpen, setZoomModalOpen] = useState(false);
  const [zoomedPatch, setZoomedPatch] = useState<EvidencePatch | null>(null);

  // Keyboard shortcuts modal state
  const [shortcutsModalOpen, setShortcutsModalOpen] = useState(false);

  // Semantic search state
  const [semanticResults, setSemanticResults] = useState<SemanticSearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);

  // Slide QC metrics state
  const [slideQCMetrics, setSlideQCMetrics] = useState<SlideQCMetrics | null>(null);

  // Slide list state for keyboard navigation
  const [slideList, setSlideList] = useState<SlideInfo[]>([]);
  const [slideIndex, setSlideIndex] = useState<number>(-1);

  // Refs for panel focusing
  const slideSelectorRef = useRef<HTMLElement>(null);
  const viewerRef = useRef<HTMLElement>(null);
  const predictionPanelRef = useRef<HTMLDivElement>(null);
  const evidencePanelRef = useRef<HTMLDivElement>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);

  // Viewer control refs
  const viewerControlsRef = useRef<{
    zoomIn: () => void;
    zoomOut: () => void;
    resetZoom: () => void;
    toggleHeatmap: () => void;
    toggleFullscreen: () => void;
  } | null>(null);

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
    retryAnalysis,
    retryReport,
    analysisStep,
  } = useAnalysis();

  // Keyboard shortcut handlers
  const handleNavigateSlides = useCallback((direction: "up" | "down") => {
    if (slideList.length === 0) return;
    
    let newIndex = slideIndex;
    if (direction === "up") {
      newIndex = slideIndex <= 0 ? slideList.length - 1 : slideIndex - 1;
    } else {
      newIndex = slideIndex >= slideList.length - 1 ? 0 : slideIndex + 1;
    }
    
    setSlideIndex(newIndex);
    const slide = slideList[newIndex];
    if (slide) {
      setSelectedSlide(slide);
      clearResults();
      setSelectedPatchId(undefined);
      setSemanticResults([]);
      setSearchError(null);
    }
  }, [slideList, slideIndex, clearResults]);

  const handleClearSelection = useCallback(() => {
    if (zoomModalOpen) {
      setZoomModalOpen(false);
    } else if (shortcutsModalOpen) {
      setShortcutsModalOpen(false);
    } else {
      setSelectedSlide(null);
      setSlideIndex(-1);
      clearResults();
      setSelectedPatchId(undefined);
      setSemanticResults([]);
    }
  }, [zoomModalOpen, shortcutsModalOpen, clearResults]);

  const handleFocusPanel = useCallback((panel: number) => {
    switch (panel) {
      case 1:
        slideSelectorRef.current?.focus();
        break;
      case 2:
        viewerRef.current?.focus();
        break;
      case 3:
        predictionPanelRef.current?.focus();
        predictionPanelRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
        break;
      case 4:
        evidencePanelRef.current?.focus();
        evidencePanelRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
        break;
    }
  }, []);

  const handleFocusSearch = useCallback(() => {
    searchInputRef.current?.focus();
    searchInputRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });
  }, []);

  const handleViewerZoomIn = useCallback(() => {
    viewerControlsRef.current?.zoomIn();
  }, []);

  const handleViewerZoomOut = useCallback(() => {
    viewerControlsRef.current?.zoomOut();
  }, []);

  const handleViewerResetZoom = useCallback(() => {
    viewerControlsRef.current?.resetZoom();
  }, []);

  const handleToggleHeatmap = useCallback(() => {
    viewerControlsRef.current?.toggleHeatmap();
  }, []);

  const handleToggleFullscreen = useCallback(() => {
    viewerControlsRef.current?.toggleFullscreen();
  }, []);

  const handlePrintReport = useCallback(() => {
    if (report) {
      window.print();
    }
  }, [report]);

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
    async (slide: SlideInfo) => {
      setSelectedSlide(slide);
      clearResults();
      setSelectedPatchId(undefined);
      setSemanticResults([]);
      setSearchError(null);
      setSlideQCMetrics(null);

      // Fetch QC metrics for the selected slide
      try {
        const qcMetrics = await getSlideQC(slide.id);
        setSlideQCMetrics(qcMetrics);
      } catch (err) {
        console.error("Failed to fetch QC metrics:", err);
        // QC metrics are optional, don't block on failure
      }
    },
    [clearResults]
  );

  // Handle semantic search
  const handleSemanticSearch = useCallback(
    async (query: string, topK: number) => {
      if (!selectedSlide) return;

      setIsSearching(true);
      setSearchError(null);

      try {
        const response = await semanticSearch(selectedSlide.id, query, topK);
        setSemanticResults(response.results);
      } catch (err) {
        console.error("Semantic search failed:", err);
        setSearchError(
          err instanceof Error ? err.message : "Search failed. Please try again."
        );
        setSemanticResults([]);
      } finally {
        setIsSearching(false);
      }
    },
    [selectedSlide]
  );

  // Handle analyze button
  const handleAnalyze = useCallback(async () => {
    if (!selectedSlide) return;

    const startTime = Date.now();
    const result = await analyze({
      slideId: selectedSlide.id,
      patchBudget: 8000,
      magnification: 20,
    });

    // Record stats for the quick stats panel
    if (result) {
      const processingTime = result.processingTimeMs || (Date.now() - startTime);
      recordAnalysis(
        selectedSlide.id,
        result.prediction.label,
        result.prediction.confidence,
        processingTime
      );
    }
  }, [selectedSlide, analyze]);

  // Handle patch click - navigate viewer
  const handlePatchClick = useCallback((coords: PatchCoordinates) => {
    // The WSIViewer will handle navigation internally via state/props
    setSelectedPatchId(`${coords.x}_${coords.y}`);
  }, []);

  // Handle patch zoom - open modal with enlarged view
  const handlePatchZoom = useCallback((patch: EvidencePatch) => {
    setZoomedPatch(patch);
    setZoomModalOpen(true);
  }, []);

  // Handle patch modal navigation
  const handlePatchModalNavigate = useCallback(
    (direction: "prev" | "next") => {
      if (!zoomedPatch || !analysisResult) return;

      const patches = analysisResult.evidencePatches;
      const currentIndex = patches.findIndex((p) => p.id === zoomedPatch.id);

      if (direction === "prev" && currentIndex > 0) {
        setZoomedPatch(patches[currentIndex - 1]);
      } else if (direction === "next" && currentIndex < patches.length - 1) {
        setZoomedPatch(patches[currentIndex + 1]);
      }
    },
    [zoomedPatch, analysisResult]
  );

  // Handle similar case click
  const handleCaseClick = useCallback((caseId: string) => {
    if (typeof window === "undefined") return;
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

  // Define keyboard shortcuts
  const keyboardShortcuts = useMemo<KeyboardShortcut[]>(() => [
    // Navigation shortcuts
    {
      key: "ArrowUp",
      description: "Previous slide",
      category: "Navigation",
      handler: () => handleNavigateSlides("up"),
    },
    {
      key: "ArrowDown",
      description: "Next slide",
      category: "Navigation",
      handler: () => handleNavigateSlides("down"),
    },
    {
      key: "Enter",
      description: "Analyze selected slide",
      category: "Navigation",
      handler: () => {
        if (selectedSlide && !isAnalyzing) {
          handleAnalyze();
        }
      },
    },
    {
      key: "Escape",
      description: "Clear selection / Close modal",
      category: "Navigation",
      handler: handleClearSelection,
    },
    // Viewer controls
    {
      key: "=",
      description: "Zoom in",
      category: "Viewer",
      handler: handleViewerZoomIn,
    },
    {
      key: "+",
      description: "Zoom in",
      category: "Viewer",
      handler: handleViewerZoomIn,
    },
    {
      key: "-",
      description: "Zoom out",
      category: "Viewer",
      handler: handleViewerZoomOut,
    },
    {
      key: "0",
      description: "Reset zoom",
      category: "Viewer",
      handler: handleViewerResetZoom,
    },
    {
      key: "h",
      description: "Toggle heatmap overlay",
      category: "Viewer",
      handler: handleToggleHeatmap,
    },
    {
      key: "f",
      description: "Fullscreen viewer",
      category: "Viewer",
      handler: handleToggleFullscreen,
    },
    // Panel shortcuts
    {
      key: "1",
      description: "Focus slide selector",
      category: "Panels",
      handler: () => handleFocusPanel(1),
    },
    {
      key: "2",
      description: "Focus viewer",
      category: "Panels",
      handler: () => handleFocusPanel(2),
    },
    {
      key: "3",
      description: "Focus prediction panel",
      category: "Panels",
      handler: () => handleFocusPanel(3),
    },
    {
      key: "4",
      description: "Focus evidence panel",
      category: "Panels",
      handler: () => handleFocusPanel(4),
    },
    {
      key: "s",
      description: "Focus semantic search",
      category: "Panels",
      handler: handleFocusSearch,
    },
    // Action shortcuts
    {
      key: "a",
      description: "Analyze selected slide",
      category: "Actions",
      handler: () => {
        if (selectedSlide && !isAnalyzing) {
          handleAnalyze();
        }
      },
    },
    {
      key: "r",
      description: "Generate report",
      category: "Actions",
      handler: () => {
        if (selectedSlide && analysisResult && !report && !isGeneratingReport) {
          handleGenerateReport();
        }
      },
    },
    {
      key: "p",
      description: "Print report",
      category: "Actions",
      handler: handlePrintReport,
    },
    {
      key: "?",
      description: "Show keyboard shortcuts",
      category: "Actions",
      handler: () => setShortcutsModalOpen(true),
    },
  ], [
    handleNavigateSlides,
    handleClearSelection,
    handleViewerZoomIn,
    handleViewerZoomOut,
    handleViewerResetZoom,
    handleToggleHeatmap,
    handleToggleFullscreen,
    handleFocusPanel,
    handleFocusSearch,
    handlePrintReport,
    handleAnalyze,
    handleGenerateReport,
    selectedSlide,
    isAnalyzing,
    analysisResult,
    report,
    isGeneratingReport,
  ]);

  // Use the keyboard shortcuts hook
  const { shortcuts } = useKeyboardShortcuts({
    enabled: !zoomModalOpen, // Disable when patch modal is open (it has its own shortcuts)
    shortcuts: keyboardShortcuts,
  });

  // Handle PDF export (client-side generation)
  const handleExportPdf = useCallback(async () => {
    if (!selectedSlide || !report) return;
    if (typeof document === "undefined") return;
    try {
      // Get case notes for this slide
      const caseNotes = getCaseNotes(selectedSlide.id);
      
      // Generate PDF client-side
      const blob = await generatePdfReport({
        report,
        slideId: selectedSlide.id,
        caseNotes,
        institutionName: "Enso Labs",
      });
      
      // Download the PDF
      downloadPdf(blob, `atlas-report-${selectedSlide.id}.pdf`);
    } catch (err) {
      console.error("PDF export failed:", err);
    }
  }, [selectedSlide, report]);

  // Handle JSON export
  const handleExportJson = useCallback(async () => {
    if (!selectedSlide || !report) return;
    if (typeof document === "undefined") return;
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
        onOpenShortcuts={() => setShortcutsModalOpen(true)}
      />

      {/* Main Content */}
      <main className="flex-1 flex overflow-hidden">
        {/* Left Sidebar - Slide Selection */}
        <aside
          ref={slideSelectorRef as React.RefObject<HTMLElement>}
          tabIndex={-1}
          className="w-80 border-r border-surface-border bg-white p-4 overflow-y-auto shrink-0 space-y-4 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-clinical-500"
        >
          <SlideSelector
            selectedSlideId={selectedSlide?.id ?? null}
            onSlideSelect={handleSlideSelect}
            onAnalyze={handleAnalyze}
            isAnalyzing={isAnalyzing}
            analysisStep={analysisStep}
          />

          {/* Error Display */}
          {error && (
            <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
              <div className="flex items-start gap-2">
                <svg
                  className="h-4 w-4 text-red-500 mt-0.5 shrink-0"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
                <div className="flex-1">
                  <p className="text-sm font-medium text-red-800">
                    Operation Failed
                  </p>
                  <p className="text-xs text-red-700 mt-0.5">{error}</p>
                  <div className="flex gap-2 mt-2">
                    <button
                      onClick={retryAnalysis}
                      className="text-xs text-red-700 font-medium hover:text-red-900 underline"
                    >
                      Retry
                    </button>
                    <button
                      onClick={clearError}
                      className="text-xs text-red-600 hover:text-red-800"
                    >
                      Dismiss
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Quick Stats Dashboard */}
          <QuickStatsPanel />

          {/* Case Notes */}
          <CaseNotesPanel slideId={selectedSlide?.id ?? null} />
        </aside>

        {/* Center - WSI Viewer */}
        <section
          ref={viewerRef as React.RefObject<HTMLElement>}
          tabIndex={-1}
          className="flex-1 p-4 overflow-hidden focus:outline-none focus:ring-2 focus:ring-inset focus:ring-clinical-500"
        >
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
          <div ref={predictionPanelRef} tabIndex={-1} className="focus:outline-none focus:ring-2 focus:ring-clinical-500 focus:ring-offset-2 rounded-lg">
            <PredictionPanel
              prediction={analysisResult?.prediction ?? null}
              isLoading={isAnalyzing}
              processingTime={analysisResult?.processingTimeMs}
              analysisStep={analysisStep}
              error={!isAnalyzing && !analysisResult ? error : null}
              onRetry={retryAnalysis}
              qcMetrics={slideQCMetrics}
            />
          </div>

          {/* Evidence Patches */}
          <div ref={evidencePanelRef} tabIndex={-1} className="focus:outline-none focus:ring-2 focus:ring-clinical-500 focus:ring-offset-2 rounded-lg">
            <EvidencePanel
              patches={analysisResult?.evidencePatches ?? []}
              isLoading={isAnalyzing}
              onPatchClick={handlePatchClick}
              onPatchZoom={handlePatchZoom}
              selectedPatchId={selectedPatchId}
              error={!isAnalyzing && error && !analysisResult ? error : null}
              onRetry={retryAnalysis}
            />
          </div>

          {/* Semantic Search */}
          <SemanticSearchPanel
            slideId={selectedSlide?.id ?? null}
            isAnalyzed={!!analysisResult}
            onSearch={handleSemanticSearch}
            results={semanticResults}
            isSearching={isSearching}
            error={searchError}
            onPatchClick={handlePatchClick}
            selectedPatchId={selectedPatchId}
          />

          {/* Similar Cases */}
          <SimilarCasesPanel
            cases={analysisResult?.similarCases ?? []}
            isLoading={isAnalyzing}
            onCaseClick={handleCaseClick}
            error={!isAnalyzing && error && !analysisResult ? error : null}
            onRetry={retryAnalysis}
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
            error={!isGeneratingReport && !report && error ? error : null}
            onRetry={retryReport}
          />
        </aside>
      </main>

      {/* Footer */}
      <Footer version="0.1.0" />

      {/* Patch Zoom Modal */}
      <PatchZoomModal
        isOpen={zoomModalOpen}
        onClose={() => setZoomModalOpen(false)}
        patch={zoomedPatch}
        allPatches={analysisResult?.evidencePatches ?? []}
        onNavigate={handlePatchModalNavigate}
        slideId={selectedSlide?.id}
      />

      {/* Keyboard Shortcuts Modal */}
      <KeyboardShortcutsModal
        isOpen={shortcutsModalOpen}
        onClose={() => setShortcutsModalOpen(false)}
        shortcuts={shortcuts}
      />
    </div>
  );
}
