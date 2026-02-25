"use client";

// Main app page with project-scoped state resets, model hydration, and data fetches.
import React, { useState, useCallback, useEffect, useRef, useMemo, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import nextDynamic from "next/dynamic";
import { Header } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";
import { DemoMode, WelcomeModal, DEMO_SLIDE, DEMO_ANALYSIS_RESULT, DEMO_MULTI_MODEL, DEMO_REPORT, DEMO_QC_METRICS } from "@/components/demo";
import {
  SlideSelector,
  PredictionPanel,
  MultiModelPredictionPanel,
  EvidencePanel,
  SimilarCasesPanel,
  ReportPanel,
  SemanticSearchPanel,
  CaseNotesPanel,
  OncologistSummaryView,
  PathologistView,
  AIAssistantPanel,
  AnalysisControls,
  OutlierDetectorPanel,
  PatchClassifierPanel,
  recordAnalysis,
  getCaseNotes,
} from "@/components/panels";
import type { UserViewMode } from "@/components/layout/Header";
import { PatchZoomModal, KeyboardShortcutsModal } from "@/components/modals";
import { useAnalysis } from "@/hooks/useAnalysis";
import { useKeyboardShortcuts, type KeyboardShortcut } from "@/hooks/useKeyboardShortcuts";
import { getDziUrl, getHeatmapUrl, healthCheck, semanticSearch, getSlideQC, getAnnotations, saveAnnotation, deleteAnnotation, getSlides, analyzeSlideMultiModel, embedSlideWithPolling, visualSearch, getSlideCachedResults, getPatchCoords, getProjectAvailableModels, type AvailableModelDetail } from "@/lib/api";
import { deduplicateSlides } from "@/lib/slideUtils";
import { useProject } from "@/contexts/ProjectContext";
import { Panel, Group as PanelGroup, Separator as PanelResizeHandle } from "react-resizable-panels";
import type { PanelImperativeHandle } from "react-resizable-panels";
import { generatePdfReport, downloadPdf } from "@/lib/pdfExport";
import type { SlideInfo, PatchCoordinates, SemanticSearchResult, EvidencePatch, SlideQCMetrics, Annotation, MultiModelResponse, VisualSearchResponse, SimilarCase, StructuredReport, PatchOverlay } from "@/types";
import { cn } from "@/lib/utils";
import { useToast } from "@/components/ui";
import { Layers, BarChart3, Sparkles, Search, Grid3X3, Users, Microscope, AlertTriangle, TrendingUp } from "lucide-react";

// ResizableLayout removed - using react-resizable-panels directly

// Dynamically import WSIViewer to prevent SSR issues with OpenSeadragon
const WSIViewer = nextDynamic(
  () => import("@/components/viewer/WSIViewer").then((mod) => mod.WSIViewer),
  { ssr: false, loading: () => <div className="h-full flex items-center justify-center bg-gray-100 rounded-lg">Loading viewer...</div> }
);

// Fallback heatmap models -- derived from shared model config

// Mobile Panel Tab Component
type MobilePanelTab = "slides" | "results";
type RightSidebarPanelKey =
  | "pathologist-workspace"
  | "medgemma"
  | "evidence"
  | "prediction"
  | "multi-model"
  | "semantic-search"
  | "similar-cases"
  | "outlier-detector";

function MobilePanelTabs({
  activeTab,
  onTabChange,
  hasResults,
}: {
  activeTab: MobilePanelTab;
  onTabChange: (tab: MobilePanelTab) => void;
  hasResults: boolean;
}) {
  return (
    <div className="flex items-center bg-white border-b border-gray-200 lg:hidden">
      <button
        onClick={() => onTabChange("slides")}
        className={cn(
          "flex-1 flex items-center justify-center gap-2 px-4 py-3 text-sm font-medium transition-colors border-b-2",
          activeTab === "slides"
            ? "text-clinical-600 border-clinical-500 bg-clinical-50/50"
            : "text-gray-500 border-transparent hover:text-gray-700 hover:bg-gray-50"
        )}
      >
        <Layers className="h-4 w-4" />
        <span>Slides</span>
      </button>
      <button
        onClick={() => onTabChange("results")}
        className={cn(
          "flex-1 flex items-center justify-center gap-2 px-4 py-3 text-sm font-medium transition-colors border-b-2 relative",
          activeTab === "results"
            ? "text-clinical-600 border-clinical-500 bg-clinical-50/50"
            : "text-gray-500 border-transparent hover:text-gray-700 hover:bg-gray-50"
        )}
      >
        <BarChart3 className="h-4 w-4" />
        <span>Results</span>
        {hasResults && activeTab !== "results" && (
          <span className="absolute top-2 right-1/4 w-2 h-2 bg-clinical-500 rounded-full" />
        )}
      </button>
    </div>
  );
}

// Icon mapping for right sidebar panel tabs
const RIGHT_PANEL_ICONS: Record<RightSidebarPanelKey, React.ElementType> = {
  "pathologist-workspace": Microscope,
  medgemma: Sparkles,
  evidence: Grid3X3,
  prediction: TrendingUp,
  "multi-model": Layers,
  "semantic-search": Search,
  "similar-cases": Users,
  "outlier-detector": AlertTriangle,
};

function RightSidebarTabs({
  options,
  activePanel,
  onPanelChange,
}: {
  options: Array<{ value: RightSidebarPanelKey; label: string }>;
  activePanel: RightSidebarPanelKey;
  onPanelChange: (panel: RightSidebarPanelKey) => void;
}) {
  return (
    <div className="sticky top-0 z-10 bg-white border-b border-gray-200">
      <nav className="flex" role="tablist" aria-label="Analysis tools">
        {options.map((opt) => {
          const Icon = RIGHT_PANEL_ICONS[opt.value];
          const isActive = activePanel === opt.value;
          return (
            <button
              key={opt.value}
              role="tab"
              aria-selected={isActive}
              aria-label={opt.label}
              title={opt.label}
              onClick={() => onPanelChange(opt.value)}
              className={cn(
                "flex-1 flex items-center justify-center gap-1.5 px-1 py-2.5 text-xs font-medium transition-colors border-b-2 min-w-0",
                isActive
                  ? "text-clinical-700 border-clinical-500 bg-clinical-50/40"
                  : "text-gray-400 border-transparent hover:text-gray-600 hover:bg-gray-50"
              )}
            >
              <Icon className="h-4 w-4 shrink-0" />
              <span className="truncate hidden xl:inline">{opt.label}</span>
            </button>
          );
        })}
      </nav>
    </div>
  );
}

function humanizeModelId(modelId: string): string {
  return modelId
    .replace(/[\-_]+/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function buildProjectFallbackAvailableModels(currentProject: {
  prediction_target?: string;
  cancer_type?: string;
  positive_class?: string;
  classes?: string[];
}): import("@/types").AvailableModel[] {
  const primaryId = currentProject.prediction_target;
  if (!primaryId) return [];

  const positiveLabel =
    currentProject.positive_class ||
    currentProject.classes?.[1] ||
    "Positive";
  const negativeLabel =
    currentProject.classes?.find((c) => c !== positiveLabel) ||
    currentProject.classes?.[0] ||
    "Negative";

  return [
    {
      id: primaryId,
      name: humanizeModelId(primaryId),
      description: `Primary ${currentProject.cancer_type || "project"} model`,
      auc: 0,
      nSlides: 0,
      category: "project_specific",
      positiveLabel,
      negativeLabel,
      available: true,
    },
  ];
}

function mapDetailToAvailableModel(
  detail: AvailableModelDetail
): import("@/types").AvailableModel {
  return {
    id: detail.id,
    name: detail.displayName,
    description: detail.description,
    auc: detail.auc ?? 0,
    nSlides: 0,
    category: detail.category,
    positiveLabel: detail.positiveLabel ?? "Positive",
    negativeLabel: detail.negativeLabel ?? "Negative",
    available: true,
  };
}

export default function HomePageWrapper() {
  return (
    <Suspense fallback={<div className="h-screen flex items-center justify-center">Loading...</div>}>
      <HomePage />
    </Suspense>
  );
}

function HomePage() {
  const searchParams = useSearchParams();
  const { currentProject } = useProject();

  // State
  const [selectedSlide, setSelectedSlide] = useState<SlideInfo | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [selectedPatchId, setSelectedPatchId] = useState<string | undefined>();
  const [targetCoordinates, setTargetCoordinates] = useState<PatchCoordinates | null>(null);

  // User view mode: oncologist vs pathologist (affects entire UI layout)
  const [userViewMode, setUserViewMode] = useState<UserViewMode>("oncologist");
  const handleUserViewModeChange = useCallback((mode: UserViewMode) => {
    setUserViewMode(mode);
    setActiveRightPanel((prev) => {
      if (mode === "pathologist") return "pathologist-workspace";
      if (prev === "pathologist-workspace") return "medgemma";
      return prev;
    });
  }, []);

  // View mode: "wsi" for full WSI viewer, "summary" for oncologist summary
  const [viewMode, setViewMode] = useState<"wsi" | "summary">("wsi");

  // Annotations state (for pathologist mode)
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [activeAnnotationTool, setActiveAnnotationTool] = useState<"pointer" | "circle" | "rectangle" | "freehand" | "point">("pointer");
  const [selectedAnnotationId, setSelectedAnnotationId] = useState<string | null>(null);

  // Patch zoom modal state
  const [zoomModalOpen, setZoomModalOpen] = useState(false);
  const [zoomedPatch, setZoomedPatch] = useState<EvidencePatch | null>(null);

  // Keyboard shortcuts modal state
  const [shortcutsModalOpen, setShortcutsModalOpen] = useState(false);

  // Demo mode state
  const [demoMode, setDemoMode] = useState(false);
  const [showWelcomeModal, setShowWelcomeModal] = useState(false);

  // Mobile panel state
  const [mobilePanelTab, setMobilePanelTab] = useState<MobilePanelTab>("slides");
  const [activeRightPanel, setActiveRightPanel] = useState<RightSidebarPanelKey>("medgemma");

  // Desktop sidebar collapse state (left only; right panel is always visible)
  const [leftSidebarOpen, setLeftSidebarOpen] = useState(true);
  const leftPanelRef = useRef<PanelImperativeHandle>(null);

  // Check if this is a first visit (show welcome modal)
  useEffect(() => {
    const hasSeenWelcome = localStorage.getItem("medgemma-welcome-seen");
    if (!hasSeenWelcome) {
      // Small delay to ensure page is rendered
      const timer = setTimeout(() => setShowWelcomeModal(true), 500);
      return () => clearTimeout(timer);
    }
  }, []);

  const handleCloseWelcome = useCallback(() => {
    setShowWelcomeModal(false);
    localStorage.setItem("medgemma-welcome-seen", "true");
  }, []);

  const handleStartDemo = useCallback(() => {
    setDemoMode(true);
  }, []);

  const handleDemoModeToggle = useCallback(() => {
    setDemoMode((prev) => !prev);
  }, []);

  // Mock analysis result overlay for demo mode (analysisResult is owned by useAnalysis hook)
  const [demoAnalysisResult, setDemoAnalysisResult] = useState<import("@/types").AnalysisResponse | null>(null);

  // Reference to pre-demo state so we can restore on exit
  const preDemoState = useRef<{
    slide: SlideInfo | null;
    multiModel: MultiModelResponse | null;
    report: StructuredReport | null;
    qc: SlideQCMetrics | null;
  } | null>(null);

  // Inject / clear mock data when demo mode toggles
  useEffect(() => {
    if (demoMode) {
      // Save current state before overwriting
      preDemoState.current = {
        slide: selectedSlide,
        multiModel: multiModelResult,
        report: agentReport,
        qc: slideQCMetrics,
      };
      // Inject mock data so every tour target has content
      setSelectedSlide(DEMO_SLIDE);
      setDemoAnalysisResult(DEMO_ANALYSIS_RESULT);
      setMultiModelResult(DEMO_MULTI_MODEL);
      setAgentReport(DEMO_REPORT);
      setSlideQCMetrics(DEMO_QC_METRICS);
      setIsCachedResult(false);
      setCachedResultTimestamp(null);
    } else {
      // Clear demo overlay
      setDemoAnalysisResult(null);
      if (preDemoState.current) {
        // Restore previous state
        setSelectedSlide(preDemoState.current.slide);
        setMultiModelResult(preDemoState.current.multiModel);
        setAgentReport(preDemoState.current.report);
        setSlideQCMetrics(preDemoState.current.qc);
        preDemoState.current = null;
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [demoMode]);

  // Semantic search state
  const [semanticResults, setSemanticResults] = useState<SemanticSearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);

  // Outlier detection state
  const [outlierHeatmapData, setOutlierHeatmapData] = useState<Array<{ x: number; y: number; score: number }> | null>(null);
  const [showOutlierHeatmap, setShowOutlierHeatmap] = useState(false);

  // Patch classification state
  const [classifyResult, setClassifyResult] = useState<import("@/types").PatchClassifyResult | null>(null);
  const [showClassifyHeatmap, setShowClassifyHeatmap] = useState(false);

  // Spatial patch selection state (for few-shot classifier)
  const [patchSelectionMode, setPatchSelectionMode] = useState<{ activeClassIdx: number; classColor: string } | null>(null);
  const [patchCoordinates, setPatchCoordinates] = useState<Array<{ x: number; y: number }> | null>(null);

  // Visual search (image-to-image) state
  const [visualSearchResults, setVisualSearchResults] = useState<SimilarCase[]>([]);
  const [isSearchingVisual, setIsSearchingVisual] = useState(false);
  const [visualSearchError, setVisualSearchError] = useState<string | null>(null);
  const [visualSearchQuery, setVisualSearchQuery] = useState<{ slideId: string; patchIndex: number } | null>(null);

  // Slide QC metrics state
  const [slideQCMetrics, setSlideQCMetrics] = useState<SlideQCMetrics | null>(null);
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [resolutionLevel, setResolutionLevel] = useState<number>(0); // policy default: 0 = full res, dense
  const [forceReembed, setForceReembed] = useState(false);
  const [heatmapModel, setHeatmapModel] = useState<string | null>(null);
  const [heatmapLevel, setHeatmapLevel] = useState<number>(2); // 0-4, default 2 (512px)
  const [heatmapAlphaPower, setHeatmapAlphaPower] = useState<number>(0.7); // 0.1-1.5, controls low-attention visibility
  const [heatmapSmooth, setHeatmapSmooth] = useState<boolean>(false); // optional interpolated view (visual only)
  // Debounce alpha power so heatmap only re-fetches after user stops sliding
  const [debouncedAlphaPower, setDebouncedAlphaPower] = useState<number>(0.7);
  useEffect(() => {
    const timer = setTimeout(() => setDebouncedAlphaPower(heatmapAlphaPower), 400);
    return () => clearTimeout(timer);
  }, [heatmapAlphaPower]);

  // Multi-model analysis state
  const [multiModelResult, setMultiModelResult] = useState<MultiModelResponse | null>(null);
  const [isAnalyzingMultiModel, setIsAnalyzingMultiModel] = useState(false);
  const [multiModelError, setMultiModelError] = useState<string | null>(null);

  // Cached results state
  const [isCachedResult, setIsCachedResult] = useState(false);
  const [cachedResultTimestamp, setCachedResultTimestamp] = useState<string | null>(null);

  // Project-specific available models for preview + heatmap controls
  const [projectAvailableModels, setProjectAvailableModels] = useState<import("@/types").AvailableModel[]>([]);
  const [projectAvailableModelsScopeId, setProjectAvailableModelsScopeId] = useState<string | null>(null);
  const fallbackProjectModels = useMemo(
    () =>
      buildProjectFallbackAvailableModels({
        prediction_target: currentProject.prediction_target,
        cancer_type: currentProject.cancer_type,
        positive_class: currentProject.positive_class,
        classes: currentProject.classes,
      }),
    [
      currentProject.prediction_target,
      currentProject.cancer_type,
      currentProject.positive_class,
      currentProject.classes,
    ]
  );
  const hasCurrentProjectModels =
    projectAvailableModelsScopeId === currentProject.id && projectAvailableModels.length > 0;
  const scopedProjectModels = hasCurrentProjectModels
    ? projectAvailableModels
    : fallbackProjectModels;

  // Embedding progress state for better UX during long operations
  const [embeddingProgress, setEmbeddingProgress] = useState<{
    phase: "idle" | "embedding" | "analyzing" | "complete";
    progress: number;
    message: string;
    startTime?: number;
  } | null>(null);
  
  // State for tracking embedding generation separately from analysis
  const [isGeneratingEmbeddings, setIsGeneratingEmbeddings] = useState(false);

  // Slide list state for keyboard navigation
  const [slideList, setSlideList] = useState<SlideInfo[]>([]);
  const [slideListLoading, setSlideListLoading] = useState(true);
  const [slideListError, setSlideListError] = useState<string | null>(null);
  const [slideIndex, setSlideIndex] = useState<number>(-1);

  // Refs for panel focusing
  const slideSelectorRef = useRef<HTMLElement>(null);
  const viewerRef = useRef<HTMLElement>(null);
  const predictionPanelRef = useRef<HTMLDivElement>(null);
  const evidencePanelRef = useRef<HTMLDivElement>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);

  // Request guards to prevent stale async responses from clobbering current UI state
  const selectedSlideIdRef = useRef<string | null>(null);
  const slideSelectionRequestRef = useRef(0);
  const semanticSearchRequestRef = useRef(0);
  const visualSearchRequestRef = useRef(0);
  const embeddingRequestRef = useRef(0);
  const multiModelRequestRef = useRef(0);

  // Current viewer zoom level (synced from WSIViewer)
  const [viewerZoom, setViewerZoom] = useState(1);

  // Viewer control refs
  const viewerControlsRef = useRef<{
    zoomIn: () => void;
    zoomOut: () => void;
    resetZoom: () => void;
    zoomTo: (level: number) => void;
    getZoom: () => number;
    toggleHeatmap: () => void;
    toggleFullscreen: () => void;
    toggleHeatmapOnly: () => void;
    toggleGrid: () => void;
  } | null>(null);

  // Analysis hook
  const {
    isAnalyzing,
    isGeneratingReport,
    analysisResult: hookAnalysisResult,
    report: generatedReport,
    error,
    analyze,
    generateSlideReport,
    clearResults,
    clearError,
    retryAnalysis,
    retryReport,
    analysisStep,
    reportProgress,
    reportProgressMessage,
  } = useAnalysis();

  // Effective analysis result: demo overlay takes precedence when active
  const analysisResult = demoAnalysisResult ?? hookAnalysisResult;

  useEffect(() => {
    selectedSlideIdRef.current = selectedSlide?.id ?? null;
  }, [selectedSlide?.id]);

  // Invalidate in-flight search requests when slide changes to avoid stale results.
  useEffect(() => {
    semanticSearchRequestRef.current += 1;
    visualSearchRequestRef.current += 1;
    setIsSearching(false);
    setIsSearchingVisual(false);
  }, [selectedSlide?.id]);

  // UX gating: hide low-signal evidence by default
  const EVIDENCE_SIGNIFICANCE_THRESHOLD = 0.6;
  const MIN_SIGNIFICANT_EVIDENCE_PATCHES = 2;
  const significantEvidencePatches = useMemo(() => {
    const patches = analysisResult?.evidencePatches ?? [];
    const significant = patches.filter(
      (p) => (p.attentionWeight ?? 0) >= EVIDENCE_SIGNIFICANCE_THRESHOLD
    );
    return significant.length >= MIN_SIGNIFICANT_EVIDENCE_PATCHES ? significant : [];
  }, [analysisResult]);

  const showMultiModelPanel = scopedProjectModels.length > 1;
  const isTumorStageProject =
    currentProject.id === "lung-stage" ||
    currentProject.prediction_target === "tumor_stage" ||
    currentProject.prediction_target?.includes("stage");
  const primaryPredictionPanelLabel = isTumorStageProject
    ? "Tumor Stage"
    : "Resistance to Therapy";
  const rightSidebarPanelOptions = useMemo<
    Array<{ value: RightSidebarPanelKey; label: string }>
  >(
    () =>
      userViewMode === "pathologist"
        ? [
            { value: "pathologist-workspace" as const, label: "Pathologist Workspace" },
            { value: "semantic-search", label: "MedSigLIP Semantic Search" },
            { value: "evidence", label: "Evidence Patches" },
            { value: "outlier-detector", label: "Outlier Tissue Detector" },
          ]
        : [
            { value: "medgemma", label: "MedGemma" },
            { value: "prediction", label: primaryPredictionPanelLabel },
            ...(showMultiModelPanel
              ? [{ value: "multi-model" as const, label: "Survival AI Predictions" }]
              : []),
            { value: "similar-cases", label: "Similar Cases" },
          ],
    [showMultiModelPanel, userViewMode, primaryPredictionPanelLabel]
  );

  useEffect(() => {
    if (!rightSidebarPanelOptions.some((panel) => panel.value === activeRightPanel)) {
      setActiveRightPanel(rightSidebarPanelOptions[0]?.value ?? "medgemma");
    }
  }, [activeRightPanel, rightSidebarPanelOptions]);

  // Report state from agent workflow (AI Assistant). This hydrates the main Clinical Report panel.
  const [agentReport, setAgentReport] = useState<StructuredReport | null>(null);
  const report = agentReport ?? generatedReport;

  // Normalize agent workflow report (snake_case) to StructuredReport (camelCase)
  const normalizeAgentReport = useCallback((raw: Record<string, unknown>): StructuredReport | null => {
    if (!raw || typeof raw !== 'object') return null;
    
    // Extract predictions from snake_case format
    const predictions = raw.predictions as Record<string, { model_name?: string; label?: string; score?: number; confidence?: number }> | undefined;
    const primaryKey = predictions ? Object.keys(predictions)[0] : null;
    const primary = primaryKey && predictions ? predictions[primaryKey] : null;
    
    // Extract evidence from snake_case format
    const rawEvidence = raw.evidence as Array<{ patch_index?: number; attention_weight?: number; coordinates?: [number, number] }> | undefined;
    const maxEvAttn = Math.max(...(rawEvidence || []).map(e => e.attention_weight || 0), 1e-12);
    const evidence = (rawEvidence || []).map((e, i) => {
      const normalizedWeight = (e.attention_weight || 0) / maxEvAttn;
      return {
        patchId: `patch-${e.patch_index ?? i}`,
        coordsLevel0: (e.coordinates || [0, 0]) as [number, number],
        morphologyDescription: `Patch ${e.patch_index ?? i} with attention ${(normalizedWeight * 100).toFixed(1)}%`,
        whyThisPatchMatters: `High attention region (${(normalizedWeight * 100).toFixed(1)}% weight)`,
      };
    });
    
    // Extract similar cases from snake_case format
    const rawSimilar = raw.similar_cases as Array<{ slide_id?: string; similarity_score?: number; label?: string | null }> | undefined;
    const similarExamples = (rawSimilar || []).map((s, i) => ({
      exampleId: s.slide_id || `case-${i}`,
      label: s.label || 'Unknown',
      distance: 1 - (s.similarity_score || 0),
    }));
    
    return {
      caseId: (raw.case_id as string) || 'Unknown',
      task: (raw.task as string) || 'Multi-model slide analysis',
      generatedAt: (raw.generated_at as string) || new Date().toISOString(),
      modelOutput: {
        label: primary?.label || 'Unknown',
        score: primary?.score || 0,
        confidence: primary?.confidence || 0,
      },
      evidence,
      similarExamples,
      limitations: (raw.limitations as string[]) || ['Research model - not for clinical use'],
      suggestedNextSteps: [],
      safetyStatement: (raw.safety_statement as string) || 'This analysis is for research and decision support only.',
      summary: (raw.reasoning_summary as string) || 'Analysis complete.',
    };
  }, []);

  // Toast notifications for user feedback
  const toast = useToast();

  // Annotation handlers
  const handleAddAnnotation = useCallback(
    async (annotation: Omit<Annotation, "id" | "createdAt">) => {
      if (!selectedSlide) return;

      try {
        const newAnnotation = await saveAnnotation(selectedSlide.id, {
          slideId: selectedSlide.id,
          type: annotation.type,
          coordinates: annotation.coordinates,
          text: annotation.text,
          color: annotation.color,
          category: annotation.category,
        });
        setAnnotations((prev) => [...prev, newAnnotation]);
        setSelectedAnnotationId(newAnnotation.id);
      } catch (err) {
        console.error("Failed to save annotation:", err);
        // Fall back to local-only annotation for demo
        const localAnnotation: Annotation = {
          ...annotation,
          id: `local_${Date.now()}`,
          createdAt: new Date().toISOString(),
        };
        setAnnotations((prev) => [...prev, localAnnotation]);
        setSelectedAnnotationId(localAnnotation.id);
      }
    },
    [selectedSlide]
  );

  const handleDeleteAnnotation = useCallback(
    async (annotationId: string) => {
      if (!selectedSlide) return;

      try {
        await deleteAnnotation(selectedSlide.id, annotationId);
      } catch (err) {
        console.error("Failed to delete annotation:", err);
        // Continue with local deletion anyway
      }
      setAnnotations((prev) => prev.filter((a) => a.id !== annotationId));
      setSelectedAnnotationId((prev) => (prev === annotationId ? null : prev));
    },
    [selectedSlide]
  );

  // Load annotations when slide is selected
  useEffect(() => {
    if (!selectedSlide) {
      setAnnotations([]);
      setSelectedAnnotationId(null);
      setActiveAnnotationTool("pointer");
      return;
    }

    const loadAnnotations = async () => {
      try {
        const response = await getAnnotations(selectedSlide.id);
        setAnnotations(response.annotations);
        setSelectedAnnotationId(null);
      } catch (err) {
        console.error("Failed to load annotations:", err);
        // Annotations are optional, don't block on failure
        setAnnotations([]);
      }
    };

    loadAnnotations();
  }, [selectedSlide, selectedModels]);

  // Load slide list for keyboard navigation and batch mode
  const fetchSlideList = useCallback(async () => {
    setSlideListLoading(true);
    setSlideListError(null);
    try {
      const response = await getSlides({ projectId: currentProject.id });
      // Deduplicate slides (remove non-UUID duplicates) and filter test files
      setSlideList(deduplicateSlides(response.slides));
    } catch (err) {
      console.error("Failed to load slide list:", err);
      const isNetworkError = err instanceof TypeError || (err instanceof Error && (err.message.includes("fetch") || err.message.includes("network") || err.message.includes("timeout")));
      const message = err instanceof Error ? err.message : "Failed to load slides";
      setSlideListError(isNetworkError ? `${message} -- Backend may be restarting (~3.5 min warmup)` : message);
    } finally {
      setSlideListLoading(false);
    }
  }, [currentProject.id]);

  useEffect(() => {
    fetchSlideList();
  }, [fetchSlideList]);

  // Reset model-dependent state and rehydrate project-scoped models on project change
  useEffect(() => {
    let cancelled = false;

    // Reset selected models to empty (ModelPicker will repopulate from project-scoped models)
    setSelectedModels([]);
    // Clear any cached analysis results from previous project
    setMultiModelResult(null);
    setIsCachedResult(false);
    setCachedResultTimestamp(null);
    // Clear single-model analysis results to avoid stale data from previous project
    clearResults();
    // Clear selected slide
    selectedSlideIdRef.current = null;
    setSelectedSlide(null);
    setSlideIndex(0);
    // Clear async/search state to avoid cross-project stale UI
    semanticSearchRequestRef.current += 1;
    visualSearchRequestRef.current += 1;
    setIsSearching(false);
    setIsSearchingVisual(false);
    setSemanticResults([]);
    setSearchError(null);
    setVisualSearchResults([]);
    setVisualSearchQuery(null);
    setVisualSearchError(null);
    // Clear current project model cache while refetching
    setProjectAvailableModels([]);
    setProjectAvailableModelsScopeId(null);
    setHeatmapModel(fallbackProjectModels[0]?.id ?? currentProject.prediction_target ?? null);

    const fetchProjectModels = async () => {
      if (!currentProject.id || currentProject.id === "default") {
        return;
      }

      try {
        const details = await getProjectAvailableModels(currentProject.id);
        if (cancelled) return;

        if (details.length > 0) {
          const mapped = details.map(mapDetailToAvailableModel);
          setProjectAvailableModels(mapped);
          setProjectAvailableModelsScopeId(currentProject.id);
          setHeatmapModel(mapped[0]?.id ?? fallbackProjectModels[0]?.id ?? currentProject.prediction_target ?? null);
          return;
        }
      } catch (err) {
        if (!cancelled) {
          console.warn("Failed to fetch project models for preview:", err);
        }
      }

      if (!cancelled) {
        // Keep safe project-derived fallback, never global hardcoded catalog
        setProjectAvailableModels([]);
        setProjectAvailableModelsScopeId(currentProject.id);
        setHeatmapModel(fallbackProjectModels[0]?.id ?? currentProject.prediction_target ?? null);
      }
    };

    fetchProjectModels();

    return () => {
      cancelled = true;
    };
  }, [
    currentProject.id,
    currentProject.prediction_target,
    fallbackProjectModels,
    clearResults,
  ]);

  // Auto-select slide from URL query params (e.g. /?slide=TCGA-... from Slide Manager)
  useEffect(() => {
    const slideId = searchParams.get("slide");
    const analyze = searchParams.get("analyze");
    if (slideId && slideList.length > 0 && !selectedSlide) {
      const match = slideList.find((s) => s.id === slideId);
      if (match) {
        setSelectedSlide(match);
        const idx = slideList.indexOf(match);
        if (idx >= 0) setSlideIndex(idx);
        // If analyze=true, trigger analysis automatically
        if (analyze === "true") {
          // Small delay to let the slide load first
          setTimeout(() => {
            const runBtn = document.querySelector("[data-action='run-analysis']") as HTMLButtonElement;
            if (runBtn) runBtn.click();
          }, 1000);
        }
      }
    }
  }, [searchParams, slideList, selectedSlide]);

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
      selectedSlideIdRef.current = slide.id;
      setSelectedSlide(slide);
      clearResults();
      setSelectedPatchId(undefined);
      setSemanticResults([]);
      setSearchError(null);
      setVisualSearchResults([]);
      setVisualSearchQuery(null);
      setVisualSearchError(null);
      // Clear multi-model results and cache
      setMultiModelResult(null);
      setMultiModelError(null);
      setIsCachedResult(false);
      setCachedResultTimestamp(null);
    }
  }, [slideList, slideIndex, clearResults]);

  const handleClearSelection = useCallback(() => {
    if (zoomModalOpen) {
      setZoomModalOpen(false);
    } else if (shortcutsModalOpen) {
      setShortcutsModalOpen(false);
    } else {
      selectedSlideIdRef.current = null;
      setSelectedSlide(null);
      setSlideIndex(-1);
      clearResults();
      setSelectedPatchId(undefined);
      setSemanticResults([]);
      setSearchError(null);
      setVisualSearchResults([]);
      setVisualSearchQuery(null);
      setVisualSearchError(null);
      // Clear multi-model results and cache
      setMultiModelResult(null);
      setMultiModelError(null);
      setIsCachedResult(false);
      setCachedResultTimestamp(null);
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
        setActiveRightPanel("prediction");
        requestAnimationFrame(() => {
          predictionPanelRef.current?.focus();
          predictionPanelRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
        });
        break;
      case 4:
        setActiveRightPanel("evidence");
        requestAnimationFrame(() => {
          evidencePanelRef.current?.focus();
          evidencePanelRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
        });
        break;
    }
  }, []);

  const handleFocusSearch = useCallback(() => {
    setActiveRightPanel("semantic-search");
    requestAnimationFrame(() => {
      searchInputRef.current?.focus();
      searchInputRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });
    });
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

  const handleToggleHeatmapOnly = useCallback(() => {
    viewerControlsRef.current?.toggleHeatmapOnly();
  }, []);

  const handleToggleGrid = useCallback(() => {
    viewerControlsRef.current?.toggleGrid();
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
      } catch (err) {
        console.warn("Health check failed:", err);
        setIsConnected(false);
      }
    };

    checkConnection();
    const interval = setInterval(checkConnection, 30000);
    return () => clearInterval(interval);
  }, []);

  // Handle manual reconnect attempt
  const handleReconnect = useCallback(async () => {
    try {
      await healthCheck();
      setIsConnected(true);
    } catch (err) {
      console.warn("Reconnect attempt failed:", err);
      setIsConnected(false);
    }
  }, []);

  // Handle slide selection
  const handleSlideSelect = useCallback(
    async (slide: SlideInfo) => {
      const requestId = ++slideSelectionRequestRef.current;
      const slideId = slide.id;
      selectedSlideIdRef.current = slideId;

      setSelectedSlide(slide);
      clearResults();
      setSelectedPatchId(undefined);

      semanticSearchRequestRef.current += 1;
      visualSearchRequestRef.current += 1;
      setIsSearching(false);
      setIsSearchingVisual(false);
      setSemanticResults([]);
      setSearchError(null);
      setVisualSearchResults([]);
      setVisualSearchQuery(null);
      setVisualSearchError(null);

      setSlideQCMetrics(null);
      // Clear multi-model results and cache state
      setMultiModelResult(null);
      setMultiModelError(null);
      setIsCachedResult(false);
      setCachedResultTimestamp(null);

      // Fetch QC metrics and cached results in parallel
      const qcPromise = getSlideQC(slideId).catch((err) => {
        console.error("Failed to fetch QC metrics:", err);
        return null;
      });

      const cachedPromise = getSlideCachedResults(slideId, currentProject.id).catch((err) => {
        console.error("Failed to fetch cached results:", err);
        return null;
      });

      const [qcResult, cachedResult] = await Promise.all([qcPromise, cachedPromise]);

      if (
        slideSelectionRequestRef.current !== requestId ||
        selectedSlideIdRef.current !== slideId
      ) {
        return;
      }

      if (qcResult) {
        setSlideQCMetrics(qcResult);
      }

      // If cached results exist, build a MultiModelResponse from them
      if (cachedResult && cachedResult.count > 0) {
        // Build model metadata lookup from scoped project models only
        const MODEL_META: Record<string, {
          name: string; category: string;
          auc: number; posLabel: string; negLabel: string; desc: string;
        }> = {};

        for (const model of scopedProjectModels) {
          MODEL_META[model.id] = {
            name: model.name,
            category: model.category,
            auc: model.auc ?? 0,
            posLabel: model.positiveLabel ?? "Positive",
            negLabel: model.negativeLabel ?? "Negative",
            desc: model.description ?? "",
          };
        }

        // Try to refresh with latest project-scoped metadata if available
        if (currentProject.id && currentProject.id !== "default") {
          try {
            const projectModels = await getProjectAvailableModels(currentProject.id);
            if (
              slideSelectionRequestRef.current !== requestId ||
              selectedSlideIdRef.current !== slideId
            ) {
              return;
            }
            for (const model of projectModels) {
              MODEL_META[model.id] = {
                name: model.displayName,
                category: model.category,
                auc: model.auc ?? 0,
                posLabel: model.positiveLabel ?? "Positive",
                negLabel: model.negativeLabel ?? "Negative",
                desc: model.description ?? "",
              };
            }
          } catch (err) {
            console.warn("Failed to refresh project model metadata:", err);
          }
        }

        const projectPositiveLabel =
          currentProject.positive_class ||
          currentProject.classes?.[1] ||
          "Positive";
        const projectNegativeLabel =
          currentProject.classes?.find((c) => c !== projectPositiveLabel) ||
          currentProject.classes?.[0] ||
          "Negative";

        const allowedModelIds = new Set<string>([
          ...scopedProjectModels.map((m) => m.id),
          ...(currentProject.prediction_target ? [currentProject.prediction_target] : []),
        ]);

        const predictions: Record<string, import("@/types").ModelPrediction> = {};
        const cancerSpecific: import("@/types").ModelPrediction[] = [];
        const generalPathology: import("@/types").ModelPrediction[] = [];
        let latestTimestamp: string | null = null;

        for (const r of cachedResult.results) {
          // Never render cross-project model IDs from stale cache payloads.
          if (!allowedModelIds.has(r.model_id)) {
            continue;
          }
          const meta = MODEL_META[r.model_id];
          const category =
            meta?.category ||
            (r.model_id === currentProject.prediction_target ? "project_specific" : "general_pathology");

          const pred: import("@/types").ModelPrediction = {
            modelId: r.model_id,
            modelName: meta?.name ?? humanizeModelId(r.model_id),
            category,
            score: r.score,
            decisionThreshold: r.threshold ?? undefined,
            label: r.label,
            positiveLabel: meta?.posLabel ?? projectPositiveLabel,
            negativeLabel: meta?.negLabel ?? projectNegativeLabel,
            confidence: r.confidence,
            auc: meta?.auc ?? 0,
            nTrainingSlides: 0,
            description: meta?.desc ?? "",
          };

          predictions[r.model_id] = pred;
          if (category !== "general_pathology") {
            cancerSpecific.push(pred);
          } else {
            generalPathology.push(pred);
          }

          if (r.created_at && (!latestTimestamp || r.created_at > latestTimestamp)) {
            latestTimestamp = r.created_at;
          }
        }

        if (
          slideSelectionRequestRef.current !== requestId ||
          selectedSlideIdRef.current !== slideId
        ) {
          return;
        }

        if (Object.keys(predictions).length > 0) {
          setMultiModelResult({
            slideId,
            predictions,
            byCategory: { cancerSpecific, generalPathology },
            nPatches: 0,
            processingTimeMs: 0,
          });
          setIsCachedResult(true);
          setCachedResultTimestamp(latestTimestamp);
        }

        // Auto-run single-model analysis to populate evidence patches + semantic search
        analyze({
          slideId,
          patchBudget: 8000,
          magnification: 20,
          projectId: currentProject.id,
        }).catch(() => {}); // Silent -- cached multi-model results already shown
      }
    },
    [
      clearResults,
      analyze,
      scopedProjectModels,
      currentProject.id,
      currentProject.prediction_target,
      currentProject.positive_class,
      currentProject.classes,
    ]
  );

  // Handle semantic search
  const handleSemanticSearch = useCallback(
    async (query: string, topK: number) => {
      if (!selectedSlide) return;

      const slideId = selectedSlide.id;
      const requestId = ++semanticSearchRequestRef.current;

      setIsSearching(true);
      setSearchError(null);

      try {
        const response = await semanticSearch(slideId, query, topK, currentProject?.id);
        if (
          semanticSearchRequestRef.current !== requestId ||
          selectedSlideIdRef.current !== slideId
        ) {
          return;
        }
        setSemanticResults(response.results);
      } catch (err) {
        if (
          semanticSearchRequestRef.current !== requestId ||
          selectedSlideIdRef.current !== slideId
        ) {
          return;
        }
        console.error("Semantic search failed:", err);
        setSearchError(
          err instanceof Error ? err.message : "Search failed. Please try again."
        );
        setSemanticResults([]);
      } finally {
        if (semanticSearchRequestRef.current === requestId) {
          setIsSearching(false);
        }
      }
    },
    [selectedSlide, currentProject.id]
  );

  // Handle visual search (find similar patches)
  const handleVisualSearch = useCallback(
    async (patch: EvidencePatch) => {
      if (!selectedSlide) return;

      const slideId = selectedSlide.id;
      const requestId = ++visualSearchRequestRef.current;

      // Extract patch index from patchId (format: "patch_{index}")
      const patchIndex = parseInt(patch.patchId.replace(/\D/g, ""), 10) || 0;

      setIsSearchingVisual(true);
      setVisualSearchError(null);
      setVisualSearchQuery({ slideId, patchIndex });

      try {
        const response = await visualSearch({
          slideId,
          patchIndex,
          topK: 10,
          excludeSameSlide: true,
        });

        if (
          visualSearchRequestRef.current !== requestId ||
          selectedSlideIdRef.current !== slideId
        ) {
          return;
        }

        // Convert VisualSearchResultPatch to SimilarCase for the panel
        const similarCases: SimilarCase[] = response.results.map((r) => ({
          slideId: r.slideId,
          patchId: `patch_${r.patchIndex}`,
          similarity: r.similarity,
          distance: r.distance,
          label: r.label,
          thumbnailUrl: r.thumbnailUrl || undefined,
          coordinates: r.coordinates ? {
            x: r.coordinates[0],
            y: r.coordinates[1],
            level: 0,
            width: 224,
            height: 224,
          } : undefined,
        }));

        setVisualSearchResults(similarCases);

        // Show toast notification
        toast.success(
          "Visual Search Complete",
          `Found ${similarCases.length} similar patches in ${response.searchTimeMs.toFixed(0)}ms`
        );
      } catch (err) {
        if (
          visualSearchRequestRef.current !== requestId ||
          selectedSlideIdRef.current !== slideId
        ) {
          return;
        }
        console.error("Visual search failed:", err);
        setVisualSearchError(
          err instanceof Error ? err.message : "Visual search failed. Please try again."
        );
        setVisualSearchResults([]);
        toast.error("Visual Search Failed", err instanceof Error ? err.message : "Unknown error");
      } finally {
        if (visualSearchRequestRef.current === requestId) {
          setIsSearchingVisual(false);
        }
      }
    },
    [selectedSlide, toast]
  );

  // Clear visual search results when slide changes
  useEffect(() => {
    setVisualSearchResults([]);
    setVisualSearchQuery(null);
    setVisualSearchError(null);
  }, [selectedSlide?.id]);

  // Handle generating level 0 embeddings (separate from analysis)
  const handleGenerateEmbeddings = useCallback(async () => {
    if (!selectedSlide) return;

    const slideId = selectedSlide.id;
    const requestId = ++embeddingRequestRef.current;

    setIsGeneratingEmbeddings(true);
    setEmbeddingProgress({
      phase: "embedding",
      progress: 0,
      message: "Starting embedding generation...",
      startTime: Date.now(),
    });

    const toastId = toast.loading(
      "Generating Embeddings",
      `Starting level ${resolutionLevel} embedding for ${slideId}...`
    );

    try {
      const embedResult = await embedSlideWithPolling(
        slideId,
        resolutionLevel,
        forceReembed,
        (progress) => {
          if (embeddingRequestRef.current !== requestId) return;

          const nextPhase: "embedding" | "complete" =
            progress.phase === "complete" ? "complete" : "embedding";
          const nextProgress = {
            phase: nextPhase,
            progress: progress.progress,
            message: progress.message,
          };

          setEmbeddingProgress((prev) => ({
            ...nextProgress,
            startTime: prev?.startTime ?? Date.now(),
          }));

          toast.updateToast(toastId, {
            message: `${progress.message} (${Math.round(progress.progress)}%)`,
          });
        }
      );

      toast.removeToast(toastId);
      toast.success(
        "Embeddings Ready",
        `Generated embeddings for ${embedResult.numPatches} patches. You can now run analysis.`
      );

      // Refresh slide list to update hasLevel0Embeddings status
      const response = await getSlides({ projectId: currentProject.id });
      const dedupedSlides = deduplicateSlides(response.slides);
      if (embeddingRequestRef.current !== requestId) return;

      setSlideList(dedupedSlides);

      // Update selected slide with new embedding status only if the same slide is still selected
      const updatedSlide = dedupedSlides.find((s) => s.id === slideId);
      if (updatedSlide && selectedSlideIdRef.current === slideId) {
        setSelectedSlide(updatedSlide);
      }
    } catch (err) {
      if (embeddingRequestRef.current !== requestId) return;

      console.error("Embedding generation failed:", err);
      toast.removeToast(toastId);

      const errorMessage = err instanceof Error ? err.message : "Embedding failed";
      toast.error("Embedding Failed", errorMessage);
    } finally {
      if (embeddingRequestRef.current === requestId) {
        setIsGeneratingEmbeddings(false);
        setEmbeddingProgress(null);
      }
    }
  }, [selectedSlide, resolutionLevel, forceReembed, toast, currentProject.id]);

  // Handle multi-model analysis with improved error handling and progress
  const handleMultiModelAnalyze = useCallback(async (forceRefresh: boolean = false) => {
    if (!selectedSlide) return;

    const slideId = selectedSlide.id;
    const requestId = ++multiModelRequestRef.current;

    // Safety gate: single-model projects should never execute multi-model inference.
    if (scopedProjectModels.length <= 1) {
      setMultiModelResult(null);
      setMultiModelError(null);
      return;
    }

    if (selectedModels.length === 0) {
      setMultiModelResult(null);
      setMultiModelError("Select at least one model before running multi-model analysis.");
      toast.warning("No Models Selected", "Pick one or more models to run multi-model analysis.");
      return;
    }

    // Clear previous results and cache state
    setMultiModelResult(null);
    setIsCachedResult(false);
    setCachedResultTimestamp(null);

    // Check if level 0 is selected but embeddings don't exist
    if (resolutionLevel === 0 && !selectedSlide.hasLevel0Embeddings) {
      toast.error(
        "Embeddings Required",
        "Please generate Level 0 embeddings first before running analysis."
      );
      return;
    }

    setIsAnalyzingMultiModel(true);
    setMultiModelError(null);
    setEmbeddingProgress({
      phase: "embedding",
      progress: 0,
      message: "Preparing tissue patches for analysis...",
      startTime: Date.now(),
    });

    const toastId = toast.loading(
      "Starting Analysis",
      "Checking if slide embeddings exist..."
    );

    try {
      // First, ensure embeddings exist (uses background task with polling for level 0)
      toast.updateToast(toastId, {
        message: `Generating Path Foundation embeddings at level ${resolutionLevel} (level 0 may take 5-20 min)...`,
      });

      // Use the new polling-based embed function
      const embedResult = await embedSlideWithPolling(
        slideId,
        resolutionLevel,
        forceReembed,
        (progress) => {
          if (multiModelRequestRef.current !== requestId) return;

          // Update progress UI with real-time status from backend
          const nextPhase: "embedding" | "analyzing" =
            progress.phase === "complete" ? "analyzing" : "embedding";
          const nextProgress = {
            phase: nextPhase,
            progress: progress.progress,
            message: progress.message,
          };

          setEmbeddingProgress((prev) => ({
            ...nextProgress,
            startTime: prev?.startTime ?? Date.now(),
          }));

          toast.updateToast(toastId, {
            message: `${progress.message} (${Math.round(progress.progress)}%)`,
          });
        }
      );

      if (
        multiModelRequestRef.current !== requestId ||
        selectedSlideIdRef.current !== slideId
      ) {
        return;
      }

      const embeddingMsg =
        embedResult.status === "exists"
          ? "Using cached embeddings"
          : `Generated embeddings for ${embedResult.numPatches} patches`;

      toast.updateToast(toastId, {
        title: "Embeddings Ready",
        message: embeddingMsg + ". Running model inference...",
      });

      setEmbeddingProgress((prev) => ({
        phase: "analyzing",
        progress: 100,
        message: "Running multi-model inference...",
        startTime: prev?.startTime ?? Date.now(),
      }));

      // Then run multi-model analysis with explicit model selection only
      const modelIdsForAnalysis = selectedModels;

      const result = await analyzeSlideMultiModel(
        slideId,
        modelIdsForAnalysis,
        false,
        resolutionLevel,
        forceRefresh,
        currentProject.id
      );

      if (
        multiModelRequestRef.current !== requestId ||
        selectedSlideIdRef.current !== slideId
      ) {
        return;
      }

      const allowedModelIds = new Set<string>(modelIdsForAnalysis);

      const filteredPredictions = Object.fromEntries(
        Object.entries(result.predictions).filter(([modelId]) => allowedModelIds.has(modelId))
      );
      const filteredCancerSpecific = result.byCategory.cancerSpecific.filter((p) =>
        allowedModelIds.has(p.modelId)
      );
      const filteredGeneralPathology = result.byCategory.generalPathology.filter((p) =>
        allowedModelIds.has(p.modelId)
      );

      const scopedResult: MultiModelResponse = {
        ...result,
        predictions: filteredPredictions,
        byCategory: {
          cancerSpecific: filteredCancerSpecific,
          generalPathology: filteredGeneralPathology,
        },
      };

      setMultiModelResult(scopedResult);

      // Success toast
      toast.removeToast(toastId);
      toast.success(
        "Analysis Complete",
        `Analyzed ${scopedResult.nPatches} patches with ${(scopedResult.byCategory.cancerSpecific?.length ?? 0) + (scopedResult.byCategory.generalPathology?.length ?? 0)} models`
      );
    } catch (err) {
      if (multiModelRequestRef.current !== requestId) {
        return;
      }

      console.error("Multi-model analysis failed:", err);

      toast.removeToast(toastId);

      // Determine if it was an embedding error or analysis error
      const errorMessage = err instanceof Error ? err.message : "Analysis failed";
      const isEmbeddingError =
        errorMessage.toLowerCase().includes("embed") ||
        errorMessage.toLowerCase().includes("dino") ||
        errorMessage.toLowerCase().includes("patch") ||
        errorMessage.toLowerCase().includes("timeout");

      const lowerError = errorMessage.toLowerCase();

      // Busy overload contract from backend while batch embedding is active
      const isServerBusy =
        lowerError.includes("server_busy") ||
        lowerError.includes("batch embedding") ||
        lowerError.includes("temporarily unavailable to avoid gpu contention");

      // Check if it's a level 0 embeddings required error
      const needsLevel0 =
        lowerError.includes("level0_embeddings_required") ||
        lowerError.includes("level 0");

      if (isServerBusy) {
        const busyMsg =
          "Level 0 batch embedding is currently running. Multi-model analysis is temporarily paused to keep the server stable.\n\n" +
          "Please wait for embedding to finish (or cancel the batch job) and try again.";
        setMultiModelError(busyMsg);
        toast.warning(
          "Server Busy",
          "Batch embedding is in progress. Please retry analysis in a moment."
        );
      } else if (needsLevel0) {
        setMultiModelError(
          "Level 0 embeddings are required for full-resolution analysis. Please click 'Generate Level 0 Embeddings' first."
        );
        toast.error(
          "Embeddings Required",
          "Generate Level 0 embeddings before running analysis."
        );
      } else if (isEmbeddingError) {
        const userFriendlyMsg =
          "Failed to generate slide embeddings. This could be due to:\n" +
          "• Slide file not found or corrupted\n" +
          "• GPU memory issues on the server\n" +
          "• Network timeout (level 0 embedding can take 5-20 minutes)\n\n" +
          "Please try again or contact support if the issue persists.";

        setMultiModelError(userFriendlyMsg);
        toast.error(
          "Embedding Failed",
          "Could not generate embeddings for this slide. See details in panel."
        );
      } else {
        setMultiModelError(
          `Model analysis failed: ${errorMessage}.\n\nThis may be due to server load or a temporary issue. Please try again.`
        );
        toast.error(
          "Analysis Failed",
          errorMessage.length > 100 ? errorMessage.substring(0, 100) + "..." : errorMessage
        );
      }
    } finally {
      if (multiModelRequestRef.current === requestId) {
        setIsAnalyzingMultiModel(false);
        setEmbeddingProgress(null);
      }
    }
  }, [selectedSlide, selectedModels, scopedProjectModels.length, resolutionLevel, forceReembed, toast, currentProject.id]);
  // Handle analyze button
  const handleAnalyze = useCallback(async () => {
    if (!selectedSlide) return;

    // On mobile, switch to results tab when analysis starts
    setMobilePanelTab("results");
    
    toast.info("Starting Analysis", "Running TransMIL prediction...");

    const startTime = Date.now();
    const result = await analyze({
      slideId: selectedSlide.id,
      patchBudget: 8000,
      magnification: 20,
      projectId: currentProject.id,
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
      toast.success(
        "Analysis Complete",
        `Prediction: ${result.prediction.label} (${Math.round(result.prediction.confidence * 100)}% confidence)`
      );
    }

    // Only run multi-model analysis when this project actually has multiple scoped models.
    if (scopedProjectModels.length > 1) {
      // force=true to bypass cache when user explicitly clicks
      handleMultiModelAnalyze(true);
    } else {
      setMultiModelResult(null);
      setMultiModelError(null);
    }
  }, [selectedSlide, analyze, toast, handleMultiModelAnalyze, currentProject.id, scopedProjectModels.length]);

  // Retry multi-model analysis (always force)
  const handleRetryMultiModel = useCallback(() => {
    handleMultiModelAnalyze(true);
  }, [handleMultiModelAnalyze]);

  // Re-analyze: force a fresh analysis bypassing cache
  const handleReanalyze = useCallback(() => {
    handleMultiModelAnalyze(true);
  }, [handleMultiModelAnalyze]);

  // Handle patch click - navigate viewer
  const handlePatchClick = useCallback((coords: PatchCoordinates) => {
    // Set selected patch ID for highlighting
    setSelectedPatchId(`${coords.x}_${coords.y}`);
    // Clone coordinates so repeated clicks on the same patch still trigger navigation.
    setTargetCoordinates({ ...coords });
  }, []);

  const handleReportEvidenceClick = useCallback((coords: PatchCoordinates) => {
    // Ensure the clicked evidence is visible in the full WSI viewer.
    if (userViewMode === "oncologist") {
      setViewMode("wsi");
    }
    handlePatchClick(coords);
  }, [handlePatchClick, userViewMode]);

  // Handle patch zoom - open modal with enlarged view
  const handlePatchZoom = useCallback((patch: EvidencePatch) => {
    setZoomedPatch(patch);
    setZoomModalOpen(true);
  }, []);

  // Handle patch modal navigation
  const handlePatchModalNavigate = useCallback(
    (direction: "prev" | "next") => {
      if (!zoomedPatch || !analysisResult) return;

      const patches = significantEvidencePatches;
      const currentIndex = patches.findIndex((p) => p.id === zoomedPatch.id);

      if (direction === "prev" && currentIndex > 0) {
        setZoomedPatch(patches[currentIndex - 1]);
      } else if (direction === "next" && currentIndex < patches.length - 1) {
        setZoomedPatch(patches[currentIndex + 1]);
      }
    },
    [zoomedPatch, analysisResult, significantEvidencePatches]
  );

  // Handle similar case click - switch to viewing that slide
  const handleCaseClick = useCallback((caseId: string) => {
    // Find the slide in the slide list
    const slide = slideList.find((s) => s.id === caseId);
    if (slide) {
      // Update slide index for keyboard navigation
      const newIndex = slideList.findIndex((s) => s.id === caseId);
      setSlideIndex(newIndex);
      
      // Select the slide and clear previous results
      selectedSlideIdRef.current = slide.id;
      setSelectedSlide(slide);
      clearResults();
      setSelectedPatchId(undefined);
      setSemanticResults([]);
      setSearchError(null);
      setVisualSearchResults([]);
      setVisualSearchQuery(null);
      setVisualSearchError(null);
      setSlideQCMetrics(null);
      // Clear multi-model results and cache
      setMultiModelResult(null);
      setMultiModelError(null);
      setIsCachedResult(false);
      setCachedResultTimestamp(null);
      
      // Switch to slides tab on mobile
      setMobilePanelTab("slides");
    } else {
      // Slide not found - show a message or try to load it directly
      console.warn(`Similar case ${caseId} not found in slide list`);
    }
  }, [slideList, clearResults]);

  // Handle report generation (works with single-model or cached multi-model results)
  const handleGenerateReport = useCallback(async () => {
    if (!selectedSlide || (!analysisResult && !multiModelResult)) return;

    await generateSlideReport({
      slideId: selectedSlide.id,
      evidencePatchIds: analysisResult?.evidencePatches?.map((p) => p.id) ?? [],
      includeDetails: true,
      projectId: currentProject.id,
    });
  }, [selectedSlide, analysisResult, multiModelResult, generateSlideReport, currentProject.id]);

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
      modifiers: { shift: true },
      description: "Zoom in",
      category: "Viewer",
      handler: handleViewerZoomIn,
      hidden: true, // Duplicate of "=" — same physical key; hidden from shortcuts modal
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
      key: "j",
      description: "Toggle heatmap-only mode",
      category: "Viewer",
      handler: handleToggleHeatmapOnly,
    },
    {
      key: "g",
      description: "Toggle patch grid overlay",
      category: "Viewer",
      handler: handleToggleGrid,
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
        if (selectedSlide && (analysisResult || multiModelResult) && !report && !isGeneratingReport) {
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
      modifiers: { shift: true },
      description: "Toggle keyboard shortcuts",
      category: "Actions",
      handler: () => setShortcutsModalOpen((prev) => !prev),
    },
  ], [
    handleNavigateSlides,
    handleClearSelection,
    handleViewerZoomIn,
    handleViewerZoomOut,
    handleViewerResetZoom,
    handleToggleHeatmap,
    handleToggleFullscreen,
    handleToggleHeatmapOnly,
    handleToggleGrid,
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
    enabled: !zoomModalOpen && !shortcutsModalOpen,
    shortcuts: keyboardShortcuts,
  });

  // Handle PDF export (client-side generation)
  const handleExportPdf = useCallback(async () => {
    if (!selectedSlide || !report) return;
    if (typeof document === "undefined") return;
    
    try {
      toast.info("Generating PDF", "Creating professional report with heatmap...");
      
      // Prepare prediction data
      const predictionData = analysisResult ? {
        prediction: analysisResult.prediction?.label ?? "Unknown",
        score: analysisResult.prediction?.score ?? 0,
        confidence: analysisResult.prediction?.confidence ?? 0,
        patchesAnalyzed: analysisResult.evidencePatches?.length ?? 0,
      } : {
        prediction: report.modelOutput?.label || "Unknown",
        score: report.modelOutput?.score || 0,
        confidence: report.modelOutput?.confidence || 0,
      };
      
      // Call backend PDF export endpoint
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || ""}/api/export/pdf`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          slide_id: selectedSlide.id,
          report_data: report,
          prediction_data: predictionData,
          include_heatmap: true,
          include_evidence_patches: true,
          patient_context: selectedSlide.patient || report.patientContext || null,
        }),
      });
      
      if (!response.ok) {
        // Try the lightweight /api/report/pdf endpoint as a second attempt
        console.warn("Server PDF export failed, trying lightweight endpoint...");
        try {
          const lightResponse = await fetch(
            `${process.env.NEXT_PUBLIC_API_URL || ""}/api/report/pdf`,
            {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ report, case_id: selectedSlide.id }),
            }
          );
          if (lightResponse.ok) {
            const lightBlob = await lightResponse.blob();
            const lightFilename =
              lightResponse.headers
                .get("Content-Disposition")
                ?.split("filename=")[1]
                ?.replace(/"/g, "") ||
              `atlas-report-${selectedSlide.id}.pdf`;
            downloadPdf(lightBlob, lightFilename);
            toast.success("PDF Exported", "Report downloaded");
            return;
          }
        } catch (lightErr) {
          console.warn("Lightweight PDF endpoint also failed:", lightErr);
        }

        // Final fallback to client-side PDF generation
        const caseNotes = getCaseNotes(selectedSlide.id);
        const blob = await generatePdfReport({
          report,
          slideId: selectedSlide.id,
          caseNotes,
          institutionName: "Enso Labs",
          slideInfo: selectedSlide,
        });
        downloadPdf(blob, `atlas-report-${selectedSlide.id}.pdf`);
        toast.success("PDF Exported", "Report downloaded (client-side fallback)");
        return;
      }
      
      // Download the PDF from server response
      const blob = await response.blob();
      const filename = response.headers.get("Content-Disposition")
        ?.split("filename=")[1]
        ?.replace(/"/g, "") || `atlas-report-${selectedSlide.id}.pdf`;
      
      downloadPdf(blob, filename);
      toast.success("PDF Exported", "Professional report with heatmap downloaded");
    } catch (err) {
      console.error("PDF export failed:", err);
      
      // Fallback to client-side PDF
      try {
        const caseNotes = getCaseNotes(selectedSlide.id);
        const blob = await generatePdfReport({
          report,
          slideId: selectedSlide.id,
          caseNotes,
          institutionName: "Enso Labs",
          slideInfo: selectedSlide,
        });
        downloadPdf(blob, `atlas-report-${selectedSlide.id}.pdf`);
        toast.success("PDF Exported", "Report downloaded (client-side)");
      } catch (fallbackErr) {
        const message = fallbackErr instanceof Error ? fallbackErr.message : "PDF export failed";
        toast.error("PDF Export Failed", message);
      }
    }
  }, [selectedSlide, report, analysisResult, toast]);

  // Handle JSON export
  const handleExportJson = useCallback(async () => {
    if (!selectedSlide || !report) return;
    if (typeof document === "undefined") return;
    try {
      const caseNotes = getCaseNotes(selectedSlide.id);
      const exportPayload = {
        exportedAt: new Date().toISOString(),
        slide: selectedSlide,
        analysis: analysisResult,
        report,
        qcMetrics: slideQCMetrics,
        semanticResults,
        multiModelResult,
        caseNotes,
      };
      const blob = new Blob([JSON.stringify(exportPayload, null, 2)], {
        type: "application/json",
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `atlas-report-${selectedSlide.id}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error("JSON export failed:", err);
      const message = err instanceof Error ? err.message : "JSON export failed";
      toast.error("JSON Export Failed", message);
    }
  }, [
    selectedSlide,
    report,
    analysisResult,
    slideQCMetrics,
    semanticResults,
    multiModelResult,
    toast,
  ]);

  // Compute embedding status from selected slide info
  const embeddingStatus = useMemo(() => {
    if (!selectedSlide) return undefined;
    return {
      hasLevel0: selectedSlide.hasLevel0Embeddings ?? false,
      hasLevel1: selectedSlide.hasEmbeddings ?? false,
    };
  }, [selectedSlide]);

  // Compute patch overlay for WSI viewer (outlier or classifier heatmap)
  const patchOverlayData = useMemo<PatchOverlay | null>(() => {
    if (showOutlierHeatmap && outlierHeatmapData && outlierHeatmapData.length > 0) {
      return {
        type: 'outlier',
        data: outlierHeatmapData.map(d => ({ x: d.x, y: d.y, score: d.score })),
      };
    }
    if (showClassifyHeatmap && classifyResult && classifyResult.heatmapData.length > 0) {
      return {
        type: 'classifier',
        data: classifyResult.heatmapData.map(d => ({
          x: d.x,
          y: d.y,
          classIdx: d.classIdx,
          confidence: d.confidence,
        })),
        classes: classifyResult.classes,
      };
    }
    return null;
  }, [showOutlierHeatmap, outlierHeatmapData, showClassifyHeatmap, classifyResult]);

  // Load patch coordinates when slide changes (for spatial selection)
  useEffect(() => {
    if (!selectedSlide) {
      setPatchCoordinates(null);
      return;
    }
    getPatchCoords(selectedSlide.id)
      .then((result) => {
        setPatchCoordinates(result.coords.map(([x, y]) => ({ x, y })));
      })
      .catch(() => {
        // Coordinates not available -- spatial selection will be disabled
        setPatchCoordinates(null);
      });
  }, [selectedSlide?.id]);

  // Handler for spatial patch selection on the WSI viewer
  const handlePatchSelectedOnSlide = useCallback((patchIdx: number, x: number, y: number) => {
    if (!patchSelectionMode) return;
    // Dispatch event to PatchClassifierPanel via a custom callback
    // The panel will handle adding the patch index to the appropriate class
    if ((window as any).__patchClassifierAddPatch) {
      (window as any).__patchClassifierAddPatch(patchSelectionMode.activeClassIdx, patchIdx);
    }
  }, [patchSelectionMode]);

  // Get DZI and heatmap URLs
  const dziUrl = selectedSlide ? getDziUrl(selectedSlide.id, currentProject?.id) : undefined;
  
  // Build heatmap data with selected model
  const scopedProjectModelIds = useMemo(
    () => new Set(scopedProjectModels.map((m) => m.id)),
    [scopedProjectModels]
  );
  const normalizedHeatmapModel =
    heatmapModel && scopedProjectModelIds.size > 0 && !scopedProjectModelIds.has(heatmapModel)
      ? null
      : heatmapModel;
  const effectiveHeatmapModel =
    normalizedHeatmapModel ?? scopedProjectModels[0]?.id ?? currentProject?.prediction_target ?? null;

  const heatmapData = selectedSlide && effectiveHeatmapModel ? {
    imageUrl: getHeatmapUrl(
      selectedSlide.id,
      effectiveHeatmapModel,
      heatmapLevel,
      debouncedAlphaPower,
      currentProject?.id,
      heatmapSmooth,
    ),
    minValue: 0,
    maxValue: 1,
    colorScale: "viridis" as const,
  } : undefined;

  const primaryMultiModelPrediction =
    multiModelResult?.predictions?.[currentProject.prediction_target];
  const sidebarPrediction = primaryMultiModelPrediction
    ? {
        score: primaryMultiModelPrediction.score,
        label: primaryMultiModelPrediction.label,
        confidence: primaryMultiModelPrediction.confidence,
      }
    : analysisResult?.prediction ?? null;
  const sidebarPredictionProcessingTime = primaryMultiModelPrediction
    ? multiModelResult?.processingTimeMs
    : analysisResult?.processingTimeMs;
  const sidebarPredictionIsCached =
    isCachedResult && (!!primaryMultiModelPrediction || !analysisResult);

  // Determine if we have results to show (cached multi-model counts too)
  const hasResults = !!analysisResult || !!multiModelResult || isAnalyzing;

  // Render left sidebar content
  const renderLeftSidebarContent = () => (
    <>
      <SlideSelector
        selectedSlideId={selectedSlide?.id ?? null}
        onSlideSelect={handleSlideSelect}
        onAnalyze={handleAnalyze}
        onGenerateEmbeddings={handleGenerateEmbeddings}
        isAnalyzing={isAnalyzing}
        analysisStep={analysisStep}
        selectedModels={selectedModels}
        onModelsChange={setSelectedModels}
        resolutionLevel={resolutionLevel}
        onResolutionChange={setResolutionLevel}
        forceReembed={forceReembed}
        onForceReembedChange={setForceReembed}
        isGeneratingEmbeddings={isGeneratingEmbeddings}
        embeddingProgress={embeddingProgress}
        embeddingStatus={embeddingStatus}
      />

      {/* Analysis Controls - separate from case selection */}
      <AnalysisControls
        selectedSlideId={selectedSlide?.id ?? null}
        selectedSlideHasLevel0={selectedSlide?.hasLevel0Embeddings ?? false}
        selectedModels={selectedModels}
        onModelsChange={setSelectedModels}
        resolutionLevel={resolutionLevel}
        onResolutionChange={setResolutionLevel}
        forceReembed={forceReembed}
        onForceReembedChange={setForceReembed}
        onAnalyze={handleAnalyze}
        onGenerateEmbeddings={handleGenerateEmbeddings}
        isAnalyzing={isAnalyzing}
        analysisStep={analysisStep}
        isGeneratingEmbeddings={isGeneratingEmbeddings}
        embeddingProgress={embeddingProgress}
        embeddingStatus={embeddingStatus}
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

      {/* Case Notes */}
      <CaseNotesPanel slideId={selectedSlide?.id ?? null} />
    </>
  );

  const renderSelectedRightSidebarPanel = () => {
    if (activeRightPanel === "pathologist-workspace") {
      if (!selectedSlide) {
        return (
          <div className="rounded-lg border border-violet-200 bg-violet-50 p-4 text-sm text-violet-800">
            Select a slide to open Pathologist Workspace.
          </div>
        );
      }

      return (
        <PathologistView
          analysisResult={analysisResult}
          annotations={annotations}
          onAddAnnotation={handleAddAnnotation}
          onDeleteAnnotation={handleDeleteAnnotation}
          onPatchClick={(patchId) => setSelectedPatchId(patchId)}
          onSwitchToOncologistView={() => handleUserViewModeChange("oncologist")}
          selectedPatchId={selectedPatchId}
          slideId={selectedSlide.id}
          viewerZoom={viewerZoom}
          onZoomTo={(level) => viewerControlsRef.current?.zoomTo(level)}
          onAnnotationToolChange={setActiveAnnotationTool}
          selectedAnnotationId={selectedAnnotationId}
          onSelectAnnotation={setSelectedAnnotationId}
          onExportPdf={handleExportPdf}
          report={report}
          mpp={selectedSlide.mpp}
        />
      );
    }

    if (activeRightPanel === "medgemma") {
      return (
        <div data-demo="report-panel">
          <ReportPanel
            report={report}
            isLoading={isGeneratingReport}
            progress={reportProgress}
            progressMessage={reportProgressMessage}
            onGenerateReport={
              (analysisResult || multiModelResult) && !report ? handleGenerateReport : undefined
            }
            onExportPdf={report ? handleExportPdf : undefined}
            onExportJson={report ? handleExportJson : undefined}
            error={!isGeneratingReport && !report && error ? error : null}
            onRetry={retryReport}
            onEvidenceClick={handleReportEvidenceClick}
          />
        </div>
      );
    }

    if (activeRightPanel === "evidence") {
      return (
        <div
          ref={evidencePanelRef}
          tabIndex={-1}
          className="focus:outline-none focus:ring-2 focus:ring-clinical-500 focus:ring-offset-2 rounded-lg"
          data-demo="evidence-panel"
        >
          <EvidencePanel
            patches={significantEvidencePatches}
            isLoading={isAnalyzing}
            onPatchClick={handlePatchClick}
            onPatchZoom={handlePatchZoom}
            onFindSimilar={handleVisualSearch}
            selectedPatchId={selectedPatchId}
            error={!isAnalyzing && error && !analysisResult ? error : null}
            onRetry={retryAnalysis}
            isSearchingVisual={isSearchingVisual}
          />
        </div>
      );
    }

    if (activeRightPanel === "prediction") {
      return (
        <div
          ref={predictionPanelRef}
          tabIndex={-1}
          className="focus:outline-none focus:ring-2 focus:ring-clinical-500 focus:ring-offset-2 rounded-lg"
          data-demo="prediction-panel"
        >
          <PredictionPanel
            prediction={sidebarPrediction}
            isLoading={isAnalyzing}
            processingTime={sidebarPredictionProcessingTime}
            analysisStep={analysisStep}
            error={!isAnalyzing && !analysisResult && !multiModelResult ? error : null}
            onRetry={retryAnalysis}
            qcMetrics={slideQCMetrics}
            isCached={sidebarPredictionIsCached}
            cachedAt={cachedResultTimestamp}
            onReanalyze={selectedSlide ? handleAnalyze : undefined}
          />
        </div>
      );
    }

    if (activeRightPanel === "multi-model" && showMultiModelPanel) {
      return (
        <div data-demo="multi-model-panel">
          <MultiModelPredictionPanel
            multiModelResult={multiModelResult}
            isLoading={isAnalyzingMultiModel}
            processingTime={multiModelResult?.processingTimeMs}
            error={multiModelError}
            onRetry={handleRetryMultiModel}
            embeddingProgress={embeddingProgress}
            isCached={isCachedResult}
            cachedAt={cachedResultTimestamp}
            onReanalyze={selectedSlide ? handleReanalyze : undefined}
            availableModels={scopedProjectModels.length > 0 ? scopedProjectModels : undefined}
          />
        </div>
      );
    }

    if (activeRightPanel === "semantic-search") {
      return (
        <SemanticSearchPanel
          slideId={selectedSlide?.id ?? null}
          projectId={currentProject.id}
          isAnalyzed={!!analysisResult}
          onSearch={handleSemanticSearch}
          results={semanticResults}
          isSearching={isSearching}
          error={searchError}
          onPatchClick={handlePatchClick}
          selectedPatchId={selectedPatchId}
          inputRef={searchInputRef}
          onClearResults={() => {
            semanticSearchRequestRef.current += 1;
            setSemanticResults([]);
            setSearchError(null);
            setIsSearching(false);
          }}
        />
      );
    }

    if (activeRightPanel === "similar-cases") {
      return (
        <div data-demo="similar-cases">
          <SimilarCasesPanel
            cases={visualSearchResults.length > 0 ? visualSearchResults : (analysisResult?.similarCases ?? [])}
            isLoading={isAnalyzing || isSearchingVisual}
            onCaseClick={handleCaseClick}
            error={visualSearchError || (!isAnalyzing && error && !analysisResult ? error : null)}
            onRetry={visualSearchResults.length > 0 ? () => {
              setVisualSearchResults([]);
              setVisualSearchError(null);
            } : retryAnalysis}
          />
          {visualSearchResults.length > 0 && (
            <div className="mt-2 flex items-center justify-between px-2">
              <p className="text-xs text-gray-500">
                Showing patches similar to query from slide {visualSearchQuery?.slideId?.slice(0, 12)}...
              </p>
              <button
                onClick={() => {
                  setVisualSearchResults([]);
                  setVisualSearchQuery(null);
                }}
                className="text-xs text-clinical-600 hover:text-clinical-700 font-medium"
              >
                Clear visual search
              </button>
            </div>
          )}
        </div>
      );
    }

    if (activeRightPanel === "outlier-detector") {
      return (
        <OutlierDetectorPanel
          slideId={selectedSlide?.id ?? null}
          onHeatmapToggle={(enabled, data) => {
            setShowOutlierHeatmap(enabled);
            setOutlierHeatmapData(data);
          }}
          onPatchClick={handlePatchClick}
        />
      );
    }

    return null;
  };

  // Render right sidebar content — icon tab bar replaces the old "Right panel" dropdown
  const renderRightSidebarContent = () => (
    <>
      <RightSidebarTabs
        options={rightSidebarPanelOptions}
        activePanel={activeRightPanel}
        onPanelChange={setActiveRightPanel}
      />
      <div className="p-4">{renderSelectedRightSidebarPanel()}</div>
    </>
  );

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
        viewMode={userViewMode}
        onViewModeChange={handleUserViewModeChange}
        demoMode={demoMode}
        onDemoModeToggle={handleDemoModeToggle}
        onReconnect={handleReconnect}
      />

      <MobilePanelTabs
        activeTab={mobilePanelTab}
        onTabChange={setMobilePanelTab}
        hasResults={hasResults}
      />

      {/* Main Content */}
      <main className="flex-1 flex flex-col lg:flex-row overflow-hidden relative">
        {/* Left Sidebar - Slide Selection */}
        <aside
          ref={slideSelectorRef as React.RefObject<HTMLElement>}
          tabIndex={-1}
          className={cn(
            "bg-white border-b lg:border-b-0 lg:border-r border-surface-border overflow-y-auto overflow-x-hidden shrink-0 space-y-4 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-clinical-500 relative transition-all duration-300",
            // Mobile: Full width, show/hide based on tab
            "lg:hidden",
            mobilePanelTab === "slides" ? "flex-1 p-3 sm:p-4" : "hidden"
          )}
        >
          {renderLeftSidebarContent()}
        </aside>

        {/* Desktop Resizable Layout */}
        <PanelGroup
          orientation="horizontal"
          id="enso-atlas-layout-v2"
          className="hidden lg:flex flex-1"
        >
        {/* Left Sidebar - Desktop (Resizable) */}
        <Panel
          panelRef={leftPanelRef}
          defaultSize="22%"
          minSize="10%"
          maxSize="35%"
          collapsible
          collapsedSize="0%"
          onResize={(size) => {
            if (size.asPercentage === 0 && leftSidebarOpen) setLeftSidebarOpen(false);
            if (size.asPercentage > 0 && !leftSidebarOpen) setLeftSidebarOpen(true);
          }}
        >
        <aside
          ref={slideSelectorRef as React.RefObject<HTMLElement>}
          tabIndex={-1}
          className={cn(
            "h-full bg-white border-r border-surface-border space-y-4 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-clinical-500 relative overflow-x-hidden",
            leftSidebarOpen ? "p-4 overflow-y-auto overflow-x-hidden" : "overflow-hidden"
          )}
          data-demo="slide-selector"
        >
          {leftSidebarOpen && renderLeftSidebarContent()}
        </aside>
        </Panel>
        <PanelResizeHandle className="relative w-1.5 bg-gray-100 hover:bg-clinical-200 active:bg-clinical-300 transition-colors cursor-col-resize flex items-center justify-center group">
          <div className="w-0.5 h-8 bg-gray-300 group-hover:bg-clinical-400 rounded-full transition-colors" />
        </PanelResizeHandle>

        {/* Center - WSI Viewer or Oncologist Summary */}
        <Panel defaultSize="50%" minSize="30%">
        <section
          ref={viewerRef as React.RefObject<HTMLElement>}
          tabIndex={-1}
          className="h-full flex flex-col overflow-hidden focus:outline-none focus:ring-2 focus:ring-inset focus:ring-clinical-500"
        >
          {/* View Mode Toggle - Only show in oncologist mode */}
          {userViewMode === "oncologist" && selectedSlide && analysisResult && (
            <div className="flex items-center justify-center gap-2 p-2 bg-white border-b border-gray-200">
              <span className="text-xs text-gray-500 mr-2 hidden sm:inline">View Mode:</span>
              <div className="flex items-center bg-gray-100 rounded-lg p-1">
                <button
                  onClick={() => setViewMode("wsi")}
                  className={`px-2 sm:px-3 py-1.5 rounded-md text-xs sm:text-sm font-medium transition-all ${
                    viewMode === "wsi"
                      ? "bg-white text-clinical-700 shadow-sm"
                      : "text-gray-600 hover:text-gray-900"
                  }`}
                >
                  WSI Viewer
                </button>
                <button
                  onClick={() => setViewMode("summary")}
                  className={`px-2 sm:px-3 py-1.5 rounded-md text-xs sm:text-sm font-medium transition-all ${
                    viewMode === "summary"
                      ? "bg-white text-clinical-700 shadow-sm"
                      : "text-gray-600 hover:text-gray-900"
                  }`}
                >
                  Summary
                </button>
              </div>
            </div>
          )}

          {/* Pathologist mode header */}
          {userViewMode === "pathologist" && selectedSlide && (
            <div className="flex items-center justify-between px-3 sm:px-4 py-2 bg-violet-50 border-b border-violet-200">
              <span className="text-xs sm:text-sm font-medium text-violet-700">
                WSI Viewer
              </span>
              <span className="text-2xs sm:text-xs text-violet-500 hidden sm:inline">
                Pan, zoom, and annotate
              </span>
            </div>
          )}

          {/* Content Area */}
          <div className="flex-1 p-2 sm:p-3 lg:p-4 overflow-hidden" data-demo="wsi-viewer">
            {userViewMode === "oncologist" && viewMode === "summary" && analysisResult ? (
              <OncologistSummaryView
                analysisResult={analysisResult}
                report={report}
                onPatchZoom={handlePatchZoom}
                onSwitchToFullView={() => setViewMode("wsi")}
              />
            ) : selectedSlide && dziUrl ? (
              <WSIViewer
                slideId={selectedSlide.id}
                dziUrl={dziUrl}
                hasWsi={selectedSlide.hasWsi}
                heatmap={heatmapData}
                mpp={selectedSlide.mpp}
                onRegionClick={handlePatchClick}
                targetCoordinates={targetCoordinates}
                className="h-full"
                heatmapModel={normalizedHeatmapModel}
                heatmapLevel={heatmapLevel}
                onHeatmapLevelChange={setHeatmapLevel}
                heatmapAlphaPower={heatmapAlphaPower}
                onHeatmapAlphaPowerChange={setHeatmapAlphaPower}
                heatmapSmooth={heatmapSmooth}
                onHeatmapSmoothChange={setHeatmapSmooth}
                onHeatmapModelChange={setHeatmapModel}
                availableModels={scopedProjectModels.map((m) => ({ id: m.id, name: m.name }))}
                onControlsReady={(controls) => { viewerControlsRef.current = controls; }}
                onZoomChange={setViewerZoom}
                annotations={annotations}
                activeAnnotationTool={userViewMode === "pathologist" ? activeAnnotationTool : "pointer"}
                onAnnotationCreate={userViewMode === "pathologist" ? (ann) => {
                  handleAddAnnotation({
                    slideId: selectedSlide.id,
                    type: (ann.type === "point" ? "marker" : ann.type) as Annotation["type"],
                    coordinates: ann.coordinates,
                    text: ann.type === "point" ? "Mitotic marker" : `${ann.type} annotation`,
                    color: ann.type === "point" ? "#ef4444" : "#8b5cf6",
                    category: ann.type,
                  });
                } : undefined}
                onAnnotationSelect={userViewMode === "pathologist" ? setSelectedAnnotationId : undefined}
                onAnnotationDelete={userViewMode === "pathologist" ? handleDeleteAnnotation : undefined}
                selectedAnnotationId={selectedAnnotationId}
                patchOverlay={patchOverlayData}
                patchSelectionMode={patchSelectionMode}
                patchCoordinates={patchCoordinates}
                onPatchSelected={handlePatchSelectedOnSlide}
              />
            ) : (
              <div className="h-full flex items-center justify-center bg-gray-100 rounded-lg border-2 border-dashed border-gray-300 p-4">
                <div className="text-center max-w-sm">
                  <div className="w-12 h-12 sm:w-16 sm:h-16 bg-gray-200 rounded-full flex items-center justify-center mx-auto mb-3 sm:mb-4">
                    <svg
                      className="w-6 h-6 sm:w-8 sm:h-8 text-gray-400"
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
                  <h3 className="text-base sm:text-lg font-medium text-gray-900 mb-1">
                    No Slide Selected
                  </h3>
                  <p className="text-xs sm:text-sm text-gray-500">
                    Select a whole-slide image from the{" "}
                    <button
                      onClick={() => setMobilePanelTab("slides")}
                      className="text-clinical-600 font-medium underline lg:no-underline lg:cursor-default"
                    >
                      sidebar
                    </button>{" "}
                    to begin analysis.
                  </p>
                </div>
              </div>
            )}
          </div>
        </section>
        </Panel>
        <PanelResizeHandle className="relative w-1.5 bg-gray-100 hover:bg-clinical-200 active:bg-clinical-300 transition-colors cursor-col-resize flex items-center justify-center group">
          <div className="w-0.5 h-8 bg-gray-300 group-hover:bg-clinical-400 rounded-full transition-colors" />
        </PanelResizeHandle>

        {/* Right Sidebar - Desktop (Resizable, always visible) */}
        <Panel
          defaultSize="28%"
          minSize="15%"
          maxSize="45%"
        >
        <aside
          className="h-full bg-white overflow-y-auto overflow-x-hidden relative"
        >
          {renderRightSidebarContent()}
        </aside>
        </Panel>
        </PanelGroup>

        {/* Right Sidebar - Mobile Version */}
        <aside
          className={cn(
            "lg:hidden bg-white overflow-y-auto space-y-4",
            mobilePanelTab === "results" ? "flex-1 p-3 sm:p-4" : "hidden"
          )}
        >
          {renderRightSidebarContent()}
        </aside>
      </main>

      {/* Footer - Hidden on mobile when viewing panels */}
      <div className="hidden sm:block">
        <Footer version="0.1.0" />
      </div>

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

      {/* Demo Mode */}
      <DemoMode
        isActive={demoMode}
        onClose={() => setDemoMode(false)}
      />

      {/* Welcome Modal */}
      <WelcomeModal
        isOpen={showWelcomeModal}
        onClose={handleCloseWelcome}
        onStartDemo={handleStartDemo}
      />
    </div>
  );
}
