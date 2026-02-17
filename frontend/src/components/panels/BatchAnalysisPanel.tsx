"use client";

import React, { useState, useEffect, useMemo, useCallback, useRef } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { Spinner } from "@/components/ui/Spinner";
import { cn } from "@/lib/utils";
import {
  Layers,
  Play,
  Download,
  AlertTriangle,
  CheckCircle,
  XCircle,
  ArrowUpDown,
  ArrowUp,
  ArrowDown,
  BarChart3,
  Users,
  Clock,
  Square,
  CheckSquare,
  MinusSquare,
  StopCircle,
  Zap,
  FlaskConical,
  ChevronDown,
  ChevronUp,
  ChevronRight,
  Activity,
  RefreshCw,
} from "lucide-react";
import {
  getSlides,
  downloadBatchCsv,
  startBatchAnalysisAsync,
  getBatchAnalysisStatus,
  cancelBatchAnalysis,
  convertAsyncBatchResults,
  getProjectAvailableModels,
  type AsyncBatchTaskStatus,
  type AvailableModelDetail,
} from "@/lib/api";
import { cleanSlideName, cleanSlideId, deduplicateSlides } from "@/lib/slideUtils";
import type {
  SlideInfo,
  BatchAnalysisResult,
  BatchAnalysisSummary,
  BatchModelResult,
} from "@/types";
import { useProject } from "@/contexts/ProjectContext";
import { AVAILABLE_MODELS, type ModelConfig } from "./ModelPicker";

interface BatchAnalysisPanelProps {
  onSlideSelect?: (slideId: string) => void;
  className?: string;
}

type SortField = "slideId" | "prediction" | "confidence" | "uncertaintyLevel";
type SortOrder = "asc" | "desc";
type FilterMode = "all" | "uncertain" | "responders" | "non-responders" | "errors";

export function BatchAnalysisPanel({
  onSlideSelect,
  className,
}: BatchAnalysisPanelProps) {
  // Project-aware labels
  const { currentProject } = useProject();
  const positiveLabel = currentProject.positive_class || currentProject.classes?.[1] || "Positive";
  const negativeLabel = currentProject.classes?.find(c => c !== currentProject.positive_class) || currentProject.classes?.[0] || "Negative";

  // Slide selection state
  const [slides, setSlides] = useState<SlideInfo[]>([]);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [isLoadingSlides, setIsLoadingSlides] = useState(true);

  // Model selection state
  const [selectedModelIds, setSelectedModelIds] = useState<string[]>([]);
  const [resolutionLevel, setResolutionLevel] = useState(1);
  const [forceReembed, setForceReembed] = useState(false);
  const [modelConfigExpanded, setModelConfigExpanded] = useState(true);
  const [apiModelDetails, setApiModelDetails] = useState<AvailableModelDetail[]>([]);

  // Analysis state
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [progress, setProgress] = useState({ current: 0, total: 0, currentSlide: "" });
  const [results, setResults] = useState<BatchAnalysisResult[]>([]);
  const [summary, setSummary] = useState<BatchAnalysisSummary | null>(null);
  const [processingTime, setProcessingTime] = useState<number>(0);
  const [error, setError] = useState<string | null>(null);
  const [isCancelling, setIsCancelling] = useState(false);

  // Concurrency setting
  const [concurrency, setConcurrency] = useState(4);

  // UI state
  const [sortField, setSortField] = useState<SortField>("confidence");
  const [sortOrder, setSortOrder] = useState<SortOrder>("asc");
  const [filterMode, setFilterMode] = useState<FilterMode>("all");
  const [showResults, setShowResults] = useState(false);
  const [expandedRows, setExpandedRows] = useState<Set<string>>(new Set());

  // Polling interval ref
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Build model list from API or fallback
  const models = useMemo(() => {
    if (apiModelDetails.length > 0) {
      return apiModelDetails.map((d) => ({
        id: d.id,
        displayName: d.displayName,
        description: d.description,
        auc: d.auc,
        category: d.category,
        positiveLabel: d.positiveLabel,
        negativeLabel: d.negativeLabel,
      }));
    }
    return AVAILABLE_MODELS;
  }, [apiModelDetails]);

  // Fetch available models from project
  useEffect(() => {
    const fetchModels = async () => {
      // Try API first (unless using default project)
      if (currentProject.id && currentProject.id !== "default") {
        try {
          const details = await getProjectAvailableModels(currentProject.id);
          if (details.length > 0) {
            setApiModelDetails(details);
            // Auto-select primary model
            const primaryId = currentProject.prediction_target;
            if (primaryId && details.some((d) => d.id === primaryId)) {
              setSelectedModelIds([primaryId]);
            } else {
              setSelectedModelIds([details[0].id]);
            }
            return;
          }
        } catch {
          // ignore API error, fall through to defaults
        }
      }
      // Fallback: select first hardcoded model
      if (AVAILABLE_MODELS.length > 0 && selectedModelIds.length === 0) {
        setSelectedModelIds([AVAILABLE_MODELS[0].id]);
      }
    };
    fetchModels();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentProject.id, currentProject.prediction_target]);

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, []);

  // Load slides on mount
  useEffect(() => {
    let cancelled = false;
    const loadSlides = async () => {
      setIsLoadingSlides(true);
      setError(null);
      try {
        const response = await getSlides({ projectId: currentProject.id });
        if (!cancelled) {
          // Deduplicate slides and filter out test files
          setSlides(deduplicateSlides(response.slides));
        }
      } catch (err) {
        console.error("Failed to load slides:", err);
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load slides. Check backend connection.");
        }
      } finally {
        if (!cancelled) {
          setIsLoadingSlides(false);
        }
      }
    };
    loadSlides();
    return () => { cancelled = true; };
  }, [currentProject.id]);

  // Handle select all / deselect all
  const handleSelectAll = useCallback(() => {
    if (selectedIds.size === slides.length) {
      setSelectedIds(new Set());
    } else {
      setSelectedIds(new Set(slides.map((s) => s.id)));
    }
  }, [slides, selectedIds.size]);

  // Handle individual selection
  const handleToggleSelect = useCallback((slideId: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(slideId)) {
        next.delete(slideId);
      } else {
        next.add(slideId);
      }
      return next;
    });
  }, []);

  // Model toggle
  const toggleModel = useCallback((modelId: string) => {
    setSelectedModelIds((prev) =>
      prev.includes(modelId)
        ? prev.filter((id) => id !== modelId)
        : [...prev, modelId]
    );
  }, []);

  const selectAllModels = useCallback(() => {
    setSelectedModelIds(models.map((m) => m.id));
  }, [models]);

  const selectNoModels = useCallback(() => {
    setSelectedModelIds([]);
  }, []);

  // Poll for status updates
  const pollStatus = useCallback(async (currentTaskId: string) => {
    try {
      const status = await getBatchAnalysisStatus(currentTaskId);
      
      setProgress({
        current: status.completed_slides,
        total: status.total_slides,
        currentSlide: status.current_slide_id,
      });

      if (status.status === "completed" || status.status === "cancelled" || status.status === "failed") {
        if (pollIntervalRef.current) {
          clearInterval(pollIntervalRef.current);
          pollIntervalRef.current = null;
        }

        setIsAnalyzing(false);
        setIsCancelling(false);

        if (status.status === "failed") {
          setError(status.error || "Batch analysis failed");
          return;
        }

        const converted = convertAsyncBatchResults(status);
        if (converted) {
          setResults(converted.results);
          setSummary(converted.summary);
          setProcessingTime(converted.processingTimeMs);
          setShowResults(true);
        }

        if (status.status === "cancelled") {
          setError(`Analysis cancelled after ${status.completed_slides} of ${status.total_slides} slides`);
        }
      }
    } catch (err) {
      console.error("Failed to poll status:", err);
    }
  }, []);

  // Run batch analysis with async API
  const handleAnalyze = useCallback(async () => {
    if (selectedIds.size === 0) return;

    setIsAnalyzing(true);
    setError(null);
    setProgress({ current: 0, total: selectedIds.size, currentSlide: "" });
    setShowResults(false);
    setResults([]);
    setSummary(null);
    setExpandedRows(new Set());

    try {
      const slideIds = Array.from(selectedIds);
      const response = await startBatchAnalysisAsync(slideIds, concurrency, {
        modelIds: selectedModelIds.length > 0 ? selectedModelIds : undefined,
        level: resolutionLevel,
        forceReembed,
      });
      
      setTaskId(response.task_id);
      
      pollIntervalRef.current = setInterval(() => {
        pollStatus(response.task_id);
      }, 1000);
      
      pollStatus(response.task_id);

    } catch (err) {
      console.error("Failed to start batch analysis:", err);
      setError(err instanceof Error ? err.message : "Failed to start batch analysis");
      setIsAnalyzing(false);
    }
  }, [selectedIds, concurrency, selectedModelIds, resolutionLevel, forceReembed, pollStatus]);

  // Cancel batch analysis
  const handleCancel = useCallback(async () => {
    if (!taskId) return;

    setIsCancelling(true);
    try {
      await cancelBatchAnalysis(taskId);
    } catch (err) {
      console.error("Failed to cancel:", err);
      setError(err instanceof Error ? err.message : "Failed to cancel");
      setIsCancelling(false);
    }
  }, [taskId]);

  // Handle export
  const handleExport = useCallback(() => {
    if (results.length === 0 || !summary) return;

    const timestamp = new Date().toISOString().split("T")[0];
    downloadBatchCsv(results, summary, `batch_analysis_${timestamp}.csv`);
  }, [results, summary]);

  // Handle sort
  const handleSort = useCallback((field: SortField) => {
    if (sortField === field) {
      setSortOrder((prev) => (prev === "asc" ? "desc" : "asc"));
    } else {
      setSortField(field);
      setSortOrder("asc");
    }
  }, [sortField]);

  // Toggle expanded row
  const toggleExpandedRow = useCallback((slideId: string) => {
    setExpandedRows((prev) => {
      const next = new Set(prev);
      if (next.has(slideId)) {
        next.delete(slideId);
      } else {
        next.add(slideId);
      }
      return next;
    });
  }, []);

  // Check if results have multi-model data
  const hasMultiModelResults = useMemo(() => {
    return results.some((r) => r.modelResults && r.modelResults.length > 1);
  }, [results]);

  // Filter and sort results
  const filteredResults = useMemo(() => {
    let filtered = [...results];

    switch (filterMode) {
      case "uncertain":
        filtered = filtered.filter((r) => r.requiresReview);
        break;
      case "responders":
        filtered = filtered.filter((r) => 
          r.prediction === "RESPONDER" || 
          r.prediction.toUpperCase() === positiveLabel.toUpperCase()
        );
        break;
      case "non-responders":
        filtered = filtered.filter((r) => 
          r.prediction === "NON-RESPONDER" || 
          r.prediction.toUpperCase() === negativeLabel.toUpperCase()
        );
        break;
      case "errors":
        filtered = filtered.filter((r) => r.error);
        break;
    }

    filtered.sort((a, b) => {
      let comparison = 0;
      switch (sortField) {
        case "slideId":
          comparison = a.slideId.localeCompare(b.slideId);
          break;
        case "prediction":
          comparison = a.prediction.localeCompare(b.prediction);
          break;
        case "confidence":
          comparison = a.confidence - b.confidence;
          break;
        case "uncertaintyLevel":
          const order = { high: 0, moderate: 1, low: 2, unknown: 3 };
          comparison = (order[a.uncertaintyLevel] || 3) - (order[b.uncertaintyLevel] || 3);
          break;
      }
      return sortOrder === "asc" ? comparison : -comparison;
    });

    return filtered;
  }, [results, filterMode, sortField, sortOrder, positiveLabel, negativeLabel]);

  // Selection state
  const selectionState = useMemo(() => {
    if (selectedIds.size === 0) return "none";
    if (selectedIds.size === slides.length) return "all";
    return "partial";
  }, [selectedIds.size, slides.length]);

  const cancerSpecificModels = models.filter((m) => m.category !== "general_pathology");
  const generalModels = models.filter((m) => m.category === "general_pathology");

  return (
    <Card className={cn("flex flex-col h-full", className)}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Layers className="h-4 w-4 text-clinical-600" />
            Batch Analysis
          </CardTitle>
          {showResults && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowResults(false)}
              className="text-xs"
            >
              Back to Selection
            </Button>
          )}
        </div>
      </CardHeader>

      <CardContent className="flex-1 flex flex-col min-h-0 space-y-4 pt-0">
        {/* Error State */}
        {error && (
          <div className="p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg flex items-center gap-2">
            <AlertTriangle className="h-4 w-4 text-red-500 shrink-0" />
            <span className="text-sm text-red-700 dark:text-red-400">{error}</span>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setError(null)}
              className="ml-auto"
            >
              Dismiss
            </Button>
          </div>
        )}

        {/* Loading State */}
        {isLoadingSlides && (
          <div className="flex-1 flex items-center justify-center py-8">
            <div className="text-center">
              <Spinner size="md" />
              <p className="text-sm text-gray-500 dark:text-gray-400 mt-2">Loading slides...</p>
            </div>
          </div>
        )}

        {/* Analysis Progress */}
        {isAnalyzing && (
          <div className="p-4 bg-clinical-50 dark:bg-clinical-900/20 border border-clinical-200 dark:border-clinical-800 rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <div className="flex-1">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-sm font-medium text-clinical-800 dark:text-clinical-200">
                    Analyzing slide {progress.current + 1}/{progress.total}
                  </span>
                  <span className="text-xs text-clinical-600 dark:text-clinical-400">
                    {Math.round((progress.current / progress.total) * 100)}%
                  </span>
                </div>
                {progress.currentSlide && (
                  <p className="text-xs text-clinical-600 dark:text-clinical-400 truncate" title={progress.currentSlide}>
                    Current: {cleanSlideId(progress.currentSlide)}
                  </p>
                )}
              </div>
              <Button
                variant="secondary"
                size="sm"
                onClick={handleCancel}
                disabled={isCancelling}
                className="ml-4 text-red-600 hover:text-red-700 hover:bg-red-50 dark:hover:bg-red-900/20"
              >
                {isCancelling ? (
                  <Spinner size="sm" className="mr-1" />
                ) : (
                  <StopCircle className="h-4 w-4 mr-1" />
                )}
                Cancel
              </Button>
            </div>
            <div className="w-full bg-clinical-200 dark:bg-clinical-800 rounded-full h-2">
              <div
                className="bg-clinical-600 h-2 rounded-full transition-all duration-300"
                style={{
                  width: `${progress.total > 0 ? (progress.current / progress.total) * 100 : 0}%`,
                }}
              />
            </div>
          </div>
        )}

        {/* Slide Selection View */}
        {!showResults && !isLoadingSlides && !isAnalyzing && (
          <>
            {/* Model Configuration Card */}
            <div className="rounded-lg border border-gray-200 dark:border-slate-600 bg-white dark:bg-slate-700">
              <button
                onClick={() => setModelConfigExpanded(!modelConfigExpanded)}
                className={cn(
                  "w-full flex items-center justify-between px-3 py-2.5 text-left",
                  "hover:bg-gray-50 dark:hover:bg-slate-600 transition-colors rounded-lg"
                )}
              >
                <div className="flex items-center gap-2">
                  <FlaskConical className="h-4 w-4 text-clinical-600" />
                  <span className="text-sm font-medium text-gray-900 dark:text-gray-100">
                    Model & Embedding Config
                  </span>
                  <Badge variant="default" size="sm">
                    {selectedModelIds.length}/{models.length} models
                  </Badge>
                  <Badge variant="default" size="sm">
                    L{resolutionLevel}
                  </Badge>
                </div>
                {modelConfigExpanded ? (
                  <ChevronUp className="h-4 w-4 text-gray-400" />
                ) : (
                  <ChevronDown className="h-4 w-4 text-gray-400" />
                )}
              </button>

              {modelConfigExpanded && (
                <div className="px-3 pb-3 space-y-3 border-t border-gray-100 dark:border-slate-600 pt-3">
                  {/* Resolution Level */}
                  <div className="pb-3 border-b border-gray-100 dark:border-slate-600">
                    <div className="flex items-center gap-1.5 mb-2">
                      <Layers className="h-3 w-3 text-purple-500" />
                      <span className="text-xs font-semibold text-gray-600 dark:text-gray-400 uppercase tracking-wide">
                        Resolution Level
                      </span>
                    </div>
                    <div className="flex gap-2">
                      <button
                        onClick={() => setResolutionLevel(1)}
                        className={cn(
                          "flex-1 px-3 py-2 rounded-md text-sm font-medium transition-colors",
                          resolutionLevel === 1
                            ? "bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 border-2 border-purple-300 dark:border-purple-600"
                            : "bg-gray-100 dark:bg-slate-600 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-slate-500 border-2 border-transparent"
                        )}
                      >
                        <div className="text-center">
                          <span>Level 1</span>
                          <div className="text-xs opacity-70">Fast (~100-500 patches)</div>
                        </div>
                      </button>
                      <button
                        onClick={() => setResolutionLevel(0)}
                        className={cn(
                          "flex-1 px-3 py-2 rounded-md text-sm font-medium transition-colors",
                          resolutionLevel === 0
                            ? "bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 border-2 border-purple-300 dark:border-purple-600"
                            : "bg-gray-100 dark:bg-slate-600 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-slate-500 border-2 border-transparent"
                        )}
                      >
                        <div className="text-center">
                          <span>Level 0</span>
                          <div className="text-xs opacity-70">Full res (~5K-30K patches)</div>
                        </div>
                      </button>
                    </div>

                    {/* Force Re-embed */}
                    <label className="mt-3 flex items-start gap-2 text-xs text-gray-600 dark:text-gray-400">
                      <input
                        type="checkbox"
                        checked={forceReembed}
                        onChange={(e) => setForceReembed(e.target.checked)}
                        className="mt-0.5 h-4 w-4 rounded border-gray-300 dark:border-slate-500 text-clinical-600 focus:ring-clinical-500"
                      />
                      <span className="flex-1">
                        <span className="font-medium text-gray-700 dark:text-gray-300">
                          <RefreshCw className="h-3 w-3 inline mr-1" />
                          Force Re-embed
                        </span>
                        <span className="block text-gray-500 dark:text-gray-500">
                          Regenerate embeddings even if cached
                        </span>
                      </span>
                    </label>
                  </div>

                  {/* Model Selection */}
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-semibold text-gray-600 dark:text-gray-400 uppercase tracking-wide">
                        Models
                      </span>
                      <div className="flex gap-1.5">
                        <button
                          onClick={selectAllModels}
                          className="text-xs px-2 py-0.5 rounded bg-gray-100 dark:bg-slate-600 hover:bg-gray-200 dark:hover:bg-slate-500 text-gray-700 dark:text-gray-300 transition-colors"
                        >
                          All
                        </button>
                        <button
                          onClick={selectNoModels}
                          className="text-xs px-2 py-0.5 rounded bg-gray-100 dark:bg-slate-600 hover:bg-gray-200 dark:hover:bg-slate-500 text-gray-700 dark:text-gray-300 transition-colors"
                        >
                          None
                        </button>
                      </div>
                    </div>

                    {/* Cancer-specific models */}
                    {cancerSpecificModels.length > 0 && (
                      <div className="mb-2">
                        <div className="flex items-center gap-1.5 mb-1">
                          <Activity className="h-3 w-3 text-pink-500" />
                          <span className="text-xs text-gray-500 dark:text-gray-400">
                            {currentProject.cancer_type || "Cancer Specific"}
                          </span>
                        </div>
                        <div className="space-y-1">
                          {cancerSpecificModels.map((model) => (
                            <BatchModelCheckbox
                              key={model.id}
                              model={model}
                              checked={selectedModelIds.includes(model.id)}
                              onChange={() => toggleModel(model.id)}
                              isPrimary={model.id === currentProject.prediction_target}
                            />
                          ))}
                        </div>
                      </div>
                    )}

                    {/* General pathology models */}
                    {generalModels.length > 0 && (
                      <div>
                        <div className="flex items-center gap-1.5 mb-1">
                          <FlaskConical className="h-3 w-3 text-blue-500" />
                          <span className="text-xs text-gray-500 dark:text-gray-400">General Pathology</span>
                        </div>
                        <div className="space-y-1">
                          {generalModels.map((model) => (
                            <BatchModelCheckbox
                              key={model.id}
                              model={model}
                              checked={selectedModelIds.includes(model.id)}
                              onChange={() => toggleModel(model.id)}
                            />
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>

            {/* Selection Header */}
            <div className="flex items-center justify-between">
              <button
                onClick={handleSelectAll}
                className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 hover:text-clinical-600"
              >
                {selectionState === "all" ? (
                  <CheckSquare className="h-4 w-4 text-clinical-600" />
                ) : selectionState === "partial" ? (
                  <MinusSquare className="h-4 w-4 text-clinical-400" />
                ) : (
                  <Square className="h-4 w-4" />
                )}
                <span>
                  {selectedIds.size === 0
                    ? "Select All"
                    : `${selectedIds.size} selected`}
                </span>
              </button>
              <span className="text-xs text-gray-400">
                {slides.length} slides available
              </span>
            </div>

            {/* Concurrency Setting */}
            <div className="flex items-center gap-2 px-2 py-1.5 bg-gray-50 dark:bg-slate-700 rounded-lg">
              <Zap className="h-3.5 w-3.5 text-amber-500" />
              <span className="text-xs text-gray-600 dark:text-gray-400">Parallel processing:</span>
              <select
                value={concurrency}
                onChange={(e) => setConcurrency(parseInt(e.target.value, 10))}
                className="text-xs bg-white dark:bg-slate-600 border border-gray-200 dark:border-slate-500 rounded px-2 py-1 text-gray-900 dark:text-gray-100"
              >
                {[1, 2, 4, 6, 8, 10].map((n) => (
                  <option key={n} value={n}>{n} slides</option>
                ))}
              </select>
            </div>

            {/* Slide List */}
            <div className="flex-1 overflow-y-auto space-y-1 scrollbar-hide">
              {slides.map((slide) => (
                <button
                  key={slide.id}
                  onClick={() => handleToggleSelect(slide.id)}
                  className={cn(
                    "w-full flex items-center gap-3 p-2 rounded-lg border transition-all text-left",
                    "hover:border-clinical-400 hover:bg-clinical-50/50 dark:hover:bg-clinical-900/20",
                    selectedIds.has(slide.id)
                      ? "border-clinical-500 bg-clinical-50 dark:bg-clinical-900/20 dark:border-clinical-400"
                      : "border-gray-200 dark:border-slate-600 bg-white dark:bg-slate-700"
                  )}
                >
                  {selectedIds.has(slide.id) ? (
                    <CheckSquare className="h-4 w-4 text-clinical-600 shrink-0" />
                  ) : (
                    <Square className="h-4 w-4 text-gray-400 shrink-0" />
                  )}
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-900 dark:text-gray-100 truncate" title={slide.id}>
                      {cleanSlideId(slide.id)}
                    </p>
                    <p className="text-xs text-gray-500 dark:text-gray-400">
                      {slide.numPatches?.toLocaleString() || "?"} patches
                    </p>
                  </div>
                  {slide.label && (
                    <Badge
                      variant={
                        slide.label === "1" ? "success" : "danger"
                      }
                      size="sm"
                    >
                      {slide.label === "1" ? "R" : "NR"}
                    </Badge>
                  )}
                </button>
              ))}
            </div>

            {/* Analyze Button */}
            <Button
              variant="primary"
              size="lg"
              onClick={handleAnalyze}
              disabled={selectedIds.size === 0 || selectedModelIds.length === 0 || isAnalyzing}
              isLoading={isAnalyzing}
              className="w-full"
            >
              <Play className="h-4 w-4 mr-2" />
              Analyze {selectedIds.size} Slide{selectedIds.size !== 1 ? "s" : ""} with {selectedModelIds.length} Model{selectedModelIds.length !== 1 ? "s" : ""}
            </Button>
          </>
        )}

        {/* Results View */}
        {showResults && summary && (
          <>
            {/* Summary Cards */}
            <div className="grid grid-cols-4 gap-2">
              <SummaryCard
                icon={Users}
                label="Total"
                value={summary.total}
                color="gray"
              />
              <SummaryCard
                icon={CheckCircle}
                label={positiveLabel}
                value={summary.responders}
                color="green"
              />
              <SummaryCard
                icon={XCircle}
                label={negativeLabel}
                value={summary.nonResponders}
                color="red"
              />
              <SummaryCard
                icon={AlertTriangle}
                label="Review"
                value={summary.requiresReviewCount}
                color="yellow"
              />
            </div>

            {/* Stats Bar */}
            <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400 px-1">
              <div className="flex items-center gap-4">
                <span className="flex items-center gap-1">
                  <BarChart3 className="h-3 w-3" />
                  Avg Conf: {(summary.avgConfidence * 100).toFixed(1)}%
                </span>
                <span className="flex items-center gap-1">
                  <Clock className="h-3 w-3" />
                  {(processingTime / 1000).toFixed(1)}s
                </span>
                {hasMultiModelResults && (
                  <span className="flex items-center gap-1">
                    <FlaskConical className="h-3 w-3" />
                    Multi-model
                  </span>
                )}
              </div>
              <Button
                variant="secondary"
                size="sm"
                onClick={handleExport}
                className="text-xs h-7"
              >
                <Download className="h-3 w-3 mr-1" />
                Export CSV
              </Button>
            </div>

            {/* Filter Tabs */}
            <div className="flex gap-1 p-1 bg-gray-100 dark:bg-slate-700 rounded-lg">
              {[
                { key: "all", label: "All", count: results.length },
                { key: "uncertain", label: "Uncertain", count: summary.uncertain },
                { key: "responders", label: positiveLabel, count: summary.responders },
                { key: "non-responders", label: negativeLabel, count: summary.nonResponders },
                { key: "errors", label: "Errors", count: summary.failed },
              ].map(({ key, label, count }) => (
                <button
                  key={key}
                  onClick={() => setFilterMode(key as FilterMode)}
                  className={cn(
                    "flex-1 px-2 py-1.5 text-xs font-medium rounded transition-colors",
                    filterMode === key
                      ? "bg-white dark:bg-slate-600 text-clinical-700 dark:text-clinical-300 shadow-sm"
                      : "text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200"
                  )}
                >
                  {label} ({count})
                </button>
              ))}
            </div>

            {/* Results Table */}
            <div className="flex-1 overflow-auto">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-gray-50 dark:bg-slate-700">
                  <tr>
                    {hasMultiModelResults && (
                      <th className="px-1 py-2 w-6" />
                    )}
                    <SortableHeader
                      label="Slide"
                      field="slideId"
                      currentField={sortField}
                      order={sortOrder}
                      onClick={handleSort}
                    />
                    <SortableHeader
                      label="Prediction"
                      field="prediction"
                      currentField={sortField}
                      order={sortOrder}
                      onClick={handleSort}
                    />
                    <SortableHeader
                      label="Confidence"
                      field="confidence"
                      currentField={sortField}
                      order={sortOrder}
                      onClick={handleSort}
                    />
                    <SortableHeader
                      label="Status"
                      field="uncertaintyLevel"
                      currentField={sortField}
                      order={sortOrder}
                      onClick={handleSort}
                    />
                  </tr>
                </thead>
                <tbody>
                  {filteredResults.map((result) => (
                    <React.Fragment key={result.slideId}>
                      <ResultRow
                        result={result}
                        onClick={() => onSlideSelect?.(result.slideId)}
                        positiveClass={currentProject.positive_class}
                        hasMultiModel={hasMultiModelResults}
                        isExpanded={expandedRows.has(result.slideId)}
                        onToggleExpand={() => toggleExpandedRow(result.slideId)}
                      />
                      {/* Expanded model results */}
                      {hasMultiModelResults && expandedRows.has(result.slideId) && result.modelResults && (
                        result.modelResults.map((mr) => (
                          <ModelResultRow key={`${result.slideId}-${mr.modelId}`} modelResult={mr} />
                        ))
                      )}
                    </React.Fragment>
                  ))}
                </tbody>
              </table>
              {filteredResults.length === 0 && (
                <div className="py-8 text-center text-gray-500 dark:text-gray-400 text-sm">
                  No results match the current filter
                </div>
              )}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}

// Batch model checkbox (compact version for batch panel)
function BatchModelCheckbox({
  model,
  checked,
  onChange,
  isPrimary,
}: {
  model: ModelConfig;
  checked: boolean;
  onChange: () => void;
  isPrimary?: boolean;
}) {
  return (
    <label
      className={cn(
        "flex items-center gap-2 p-1.5 rounded cursor-pointer",
        "hover:bg-gray-50 dark:hover:bg-slate-600 transition-colors",
        checked && "bg-clinical-50 dark:bg-clinical-900/20"
      )}
    >
      <input
        type="checkbox"
        checked={checked}
        onChange={onChange}
        className="h-3.5 w-3.5 rounded border-gray-300 dark:border-slate-500 text-clinical-600 focus:ring-clinical-500"
      />
      <div className="flex-1 min-w-0 flex items-center gap-1.5">
        <span className="text-xs font-medium text-gray-900 dark:text-gray-100 truncate">
          {model.displayName}
        </span>
        {isPrimary && (
          <Badge variant="info" size="sm" className="text-2xs">Primary</Badge>
        )}
        <span className="text-2xs text-gray-400 font-mono ml-auto shrink-0">
          {model.auc.toFixed(2)}
        </span>
      </div>
    </label>
  );
}

// Summary Card Component
function SummaryCard({
  icon: Icon,
  label,
  value,
  color,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: number;
  color: "gray" | "green" | "red" | "yellow";
}) {
  const colors = {
    gray: "bg-gray-50 dark:bg-gray-800 border-gray-200 dark:border-gray-700 text-gray-600 dark:text-gray-300",
    green: "bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800 text-green-600 dark:text-green-400",
    red: "bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800 text-red-600 dark:text-red-400",
    yellow: "bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800 text-yellow-600 dark:text-yellow-400",
  };

  return (
    <div className={cn("p-2 rounded-lg border", colors[color])}>
      <div className="flex items-center gap-1.5 mb-1">
        <Icon className="h-3 w-3" />
        <span className="text-xs font-medium">{label}</span>
      </div>
      <p className="text-lg font-bold">{value}</p>
    </div>
  );
}

// Sortable Header Component
function SortableHeader({
  label,
  field,
  currentField,
  order,
  onClick,
}: {
  label: string;
  field: SortField;
  currentField: SortField;
  order: SortOrder;
  onClick: (field: SortField) => void;
}) {
  const isActive = field === currentField;

  return (
    <th
      className="px-2 py-2 text-left font-medium text-gray-600 dark:text-gray-400 cursor-pointer hover:text-gray-900 dark:hover:text-gray-200"
      onClick={() => onClick(field)}
    >
      <div className="flex items-center gap-1">
        {label}
        {isActive ? (
          order === "asc" ? (
            <ArrowUp className="h-3 w-3" />
          ) : (
            <ArrowDown className="h-3 w-3" />
          )
        ) : (
          <ArrowUpDown className="h-3 w-3 opacity-30" />
        )}
      </div>
    </th>
  );
}

// Result Row Component
function ResultRow({
  result,
  onClick,
  positiveClass,
  hasMultiModel,
  isExpanded,
  onToggleExpand,
}: {
  result: BatchAnalysisResult;
  onClick: () => void;
  positiveClass?: string;
  hasMultiModel: boolean;
  isExpanded: boolean;
  onToggleExpand: () => void;
}) {
  const isPositive = result.prediction === "RESPONDER" || 
    (positiveClass && result.prediction.toUpperCase() === positiveClass.toUpperCase());
  const hasModelData = result.modelResults && result.modelResults.length > 1;

  const getUncertaintyBadge = () => {
    switch (result.uncertaintyLevel) {
      case "high":
        return <Badge variant="danger" size="sm">High Risk</Badge>;
      case "moderate":
        return <Badge variant="warning" size="sm">Review</Badge>;
      case "low":
        return <Badge variant="success" size="sm">OK</Badge>;
      default:
        return <Badge variant="default" size="sm">Unknown</Badge>;
    }
  };

  if (result.error) {
    return (
      <tr className="border-t border-gray-100 dark:border-slate-600 bg-red-50/50 dark:bg-red-900/10">
        {hasMultiModel && <td className="px-1 py-2" />}
        <td className="px-2 py-2">
          <button
            onClick={onClick}
            className="text-left hover:text-clinical-600 font-mono text-xs"
            title={result.slideId}
          >
            {cleanSlideId(result.slideId)}
          </button>
        </td>
        <td colSpan={3} className="px-2 py-2 text-red-600 dark:text-red-400 text-xs">
          Error: {result.error}
        </td>
      </tr>
    );
  }

  return (
    <tr
      className={cn(
        "border-t border-gray-100 dark:border-slate-600 hover:bg-gray-50 dark:hover:bg-slate-600 cursor-pointer",
        result.requiresReview && "bg-yellow-50/30 dark:bg-yellow-900/10"
      )}
      onClick={onClick}
    >
      {hasMultiModel && (
        <td className="px-1 py-2">
          {hasModelData && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                onToggleExpand();
              }}
              className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
            >
              {isExpanded ? (
                <ChevronDown className="h-3.5 w-3.5" />
              ) : (
                <ChevronRight className="h-3.5 w-3.5" />
              )}
            </button>
          )}
        </td>
      )}
      <td className="px-2 py-2">
        <span className="font-mono text-xs hover:text-clinical-600 dark:text-gray-300" title={result.slideId}>
          {cleanSlideId(result.slideId)}
        </span>
      </td>
      <td className="px-2 py-2">
        <Badge
          variant={isPositive ? "success" : "danger"}
          size="sm"
        >
          {isPositive ? "+" : "-"}
        </Badge>
      </td>
      <td className="px-2 py-2">
        <div className="flex items-center gap-2">
          <div className="w-16 bg-gray-200 dark:bg-slate-600 rounded-full h-1.5">
            <div
              className={cn(
                "h-1.5 rounded-full",
                result.confidence >= 0.6
                  ? "bg-green-500"
                  : result.confidence >= 0.3
                  ? "bg-yellow-500"
                  : "bg-red-500"
              )}
              style={{ width: `${result.confidence * 100}%` }}
            />
          </div>
          <span className="text-xs text-gray-600 dark:text-gray-400 w-10">
            {(result.confidence * 100).toFixed(0)}%
          </span>
        </div>
      </td>
      <td className="px-2 py-2">{getUncertaintyBadge()}</td>
    </tr>
  );
}

// Model result sub-row (shown when expanding a multi-model result)
function ModelResultRow({ modelResult }: { modelResult: BatchModelResult }) {
  if (modelResult.error) {
    return (
      <tr className="bg-gray-50/50 dark:bg-slate-800/50">
        <td className="px-1 py-1" />
        <td className="px-2 py-1 pl-6">
          <span className="text-xs text-gray-500 dark:text-gray-400">{modelResult.modelName}</span>
        </td>
        <td colSpan={3} className="px-2 py-1 text-red-500 text-xs">
          Error: {modelResult.error}
        </td>
      </tr>
    );
  }

  const isPositive = modelResult.score > 0.5;

  return (
    <tr className="bg-gray-50/50 dark:bg-slate-800/50 border-t border-gray-50 dark:border-slate-700">
      <td className="px-1 py-1" />
      <td className="px-2 py-1 pl-6">
        <span className="text-xs text-gray-600 dark:text-gray-400 font-medium">
          {modelResult.modelName}
        </span>
      </td>
      <td className="px-2 py-1">
        <span className={cn(
          "text-xs font-medium",
          isPositive ? "text-green-600 dark:text-green-400" : "text-red-600 dark:text-red-400"
        )}>
          {modelResult.prediction}
        </span>
      </td>
      <td className="px-2 py-1">
        <div className="flex items-center gap-2">
          <div className="w-12 bg-gray-200 dark:bg-slate-600 rounded-full h-1">
            <div
              className={cn(
                "h-1 rounded-full",
                modelResult.confidence >= 0.6
                  ? "bg-green-500"
                  : modelResult.confidence >= 0.3
                  ? "bg-yellow-500"
                  : "bg-red-500"
              )}
              style={{ width: `${modelResult.confidence * 100}%` }}
            />
          </div>
          <span className="text-2xs text-gray-500 dark:text-gray-400">
            {(modelResult.confidence * 100).toFixed(0)}%
          </span>
        </div>
      </td>
      <td className="px-2 py-1" />
    </tr>
  );
}
