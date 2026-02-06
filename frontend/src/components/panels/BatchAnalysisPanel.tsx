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
} from "lucide-react";
import {
  getSlides,
  downloadBatchCsv,
  startBatchAnalysisAsync,
  getBatchAnalysisStatus,
  cancelBatchAnalysis,
  convertAsyncBatchResults,
  type AsyncBatchTaskStatus,
} from "@/lib/api";
import type {
  SlideInfo,
  BatchAnalysisResult,
  BatchAnalysisSummary,
} from "@/types";
import { useProject } from "@/contexts/ProjectContext";

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
  const positiveLabel = currentProject.positive_class || currentProject.classes?.[1] || "Responder";
  const negativeLabel = currentProject.classes?.find(c => c !== currentProject.positive_class) || currentProject.classes?.[0] || "Non-Responder";

  // Slide selection state
  const [slides, setSlides] = useState<SlideInfo[]>([]);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [isLoadingSlides, setIsLoadingSlides] = useState(true);

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

  // Polling interval ref
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);

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
    const loadSlides = async () => {
      setIsLoadingSlides(true);
      try {
        const response = await getSlides();
        setSlides(response.slides);
      } catch (err) {
        console.error("Failed to load slides:", err);
        setError("Failed to load slides");
      } finally {
        setIsLoadingSlides(false);
      }
    };
    loadSlides();
  }, []);

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

  // Poll for status updates
  const pollStatus = useCallback(async (currentTaskId: string) => {
    try {
      const status = await getBatchAnalysisStatus(currentTaskId);
      
      setProgress({
        current: status.completed_slides,
        total: status.total_slides,
        currentSlide: status.current_slide_id,
      });

      // Check if completed/cancelled/failed
      if (status.status === "completed" || status.status === "cancelled" || status.status === "failed") {
        // Stop polling
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

        // Convert and set results
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
      // Don't stop polling on transient errors
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

    try {
      const slideIds = Array.from(selectedIds);
      const response = await startBatchAnalysisAsync(slideIds, concurrency);
      
      setTaskId(response.task_id);
      
      // Start polling for updates
      pollIntervalRef.current = setInterval(() => {
        pollStatus(response.task_id);
      }, 1000);
      
      // Initial poll
      pollStatus(response.task_id);

    } catch (err) {
      console.error("Failed to start batch analysis:", err);
      setError(err instanceof Error ? err.message : "Failed to start batch analysis");
      setIsAnalyzing(false);
    }
  }, [selectedIds, concurrency, pollStatus]);

  // Cancel batch analysis
  const handleCancel = useCallback(async () => {
    if (!taskId) return;

    setIsCancelling(true);
    try {
      await cancelBatchAnalysis(taskId);
      // Polling will detect the cancellation and update state
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

  // Filter and sort results
  const filteredResults = useMemo(() => {
    let filtered = [...results];

    // Apply filter
    switch (filterMode) {
      case "uncertain":
        filtered = filtered.filter((r) => r.requiresReview);
        break;
      case "responders":
        filtered = filtered.filter((r) => r.prediction === "RESPONDER");
        break;
      case "non-responders":
        filtered = filtered.filter((r) => r.prediction === "NON-RESPONDER");
        break;
      case "errors":
        filtered = filtered.filter((r) => r.error);
        break;
    }

    // Apply sort
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
  }, [results, filterMode, sortField, sortOrder]);

  // Selection state
  const selectionState = useMemo(() => {
    if (selectedIds.size === 0) return "none";
    if (selectedIds.size === slides.length) return "all";
    return "partial";
  }, [selectedIds.size, slides.length]);

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
          <div className="p-3 bg-red-50 border border-red-200 rounded-lg flex items-center gap-2">
            <AlertTriangle className="h-4 w-4 text-red-500 shrink-0" />
            <span className="text-sm text-red-700">{error}</span>
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
              <p className="text-sm text-gray-500 mt-2">Loading slides...</p>
            </div>
          </div>
        )}

        {/* Analysis Progress */}
        {isAnalyzing && (
          <div className="p-4 bg-clinical-50 border border-clinical-200 rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <div className="flex-1">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-sm font-medium text-clinical-800">
                    Analyzing slide {progress.current + 1}/{progress.total}
                  </span>
                  <span className="text-xs text-clinical-600">
                    {Math.round((progress.current / progress.total) * 100)}%
                  </span>
                </div>
                {progress.currentSlide && (
                  <p className="text-xs text-clinical-600 truncate">
                    Current: {progress.currentSlide.slice(0, 30)}...
                  </p>
                )}
              </div>
              <Button
                variant="secondary"
                size="sm"
                onClick={handleCancel}
                disabled={isCancelling}
                className="ml-4 text-red-600 hover:text-red-700 hover:bg-red-50"
              >
                {isCancelling ? (
                  <Spinner size="sm" className="mr-1" />
                ) : (
                  <StopCircle className="h-4 w-4 mr-1" />
                )}
                Cancel
              </Button>
            </div>
            <div className="w-full bg-clinical-200 rounded-full h-2">
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
            {/* Selection Header */}
            <div className="flex items-center justify-between">
              <button
                onClick={handleSelectAll}
                className="flex items-center gap-2 text-sm text-gray-600 hover:text-clinical-600"
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
            <div className="flex items-center gap-2 px-2 py-1.5 bg-gray-50 rounded-lg">
              <Zap className="h-3.5 w-3.5 text-amber-500" />
              <span className="text-xs text-gray-600">Parallel processing:</span>
              <select
                value={concurrency}
                onChange={(e) => setConcurrency(parseInt(e.target.value, 10))}
                className="text-xs bg-white border border-gray-200 rounded px-2 py-1"
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
                    "hover:border-clinical-400 hover:bg-clinical-50/50",
                    selectedIds.has(slide.id)
                      ? "border-clinical-500 bg-clinical-50"
                      : "border-gray-200 bg-white"
                  )}
                >
                  {selectedIds.has(slide.id) ? (
                    <CheckSquare className="h-4 w-4 text-clinical-600 shrink-0" />
                  ) : (
                    <Square className="h-4 w-4 text-gray-400 shrink-0" />
                  )}
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-900 truncate">
                      {slide.id}
                    </p>
                    <p className="text-xs text-gray-500">
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
              disabled={selectedIds.size === 0 || isAnalyzing}
              isLoading={isAnalyzing}
              className="w-full"
            >
              <Play className="h-4 w-4 mr-2" />
              Analyze {selectedIds.size} Slide{selectedIds.size !== 1 ? "s" : ""}
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
            <div className="flex items-center justify-between text-xs text-gray-500 px-1">
              <div className="flex items-center gap-4">
                <span className="flex items-center gap-1">
                  <BarChart3 className="h-3 w-3" />
                  Avg Conf: {(summary.avgConfidence * 100).toFixed(1)}%
                </span>
                <span className="flex items-center gap-1">
                  <Clock className="h-3 w-3" />
                  {(processingTime / 1000).toFixed(1)}s
                </span>
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
            <div className="flex gap-1 p-1 bg-gray-100 rounded-lg">
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
                      ? "bg-white text-clinical-700 shadow-sm"
                      : "text-gray-600 hover:text-gray-900"
                  )}
                >
                  {label} ({count})
                </button>
              ))}
            </div>

            {/* Results Table */}
            <div className="flex-1 overflow-auto">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-gray-50">
                  <tr>
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
                    <ResultRow
                      key={result.slideId}
                      result={result}
                      onClick={() => onSlideSelect?.(result.slideId)}
                    />
                  ))}
                </tbody>
              </table>
              {filteredResults.length === 0 && (
                <div className="py-8 text-center text-gray-500 text-sm">
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
    gray: "bg-gray-50 border-gray-200 text-gray-600",
    green: "bg-green-50 border-green-200 text-green-600",
    red: "bg-red-50 border-red-200 text-red-600",
    yellow: "bg-yellow-50 border-yellow-200 text-yellow-600",
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
      className="px-2 py-2 text-left font-medium text-gray-600 cursor-pointer hover:text-gray-900"
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
}: {
  result: BatchAnalysisResult;
  onClick: () => void;
}) {
  const getUncertaintyBadge = () => {
    switch (result.uncertaintyLevel) {
      case "high":
        return (
          <Badge variant="danger" size="sm">
            High Risk
          </Badge>
        );
      case "moderate":
        return (
          <Badge variant="warning" size="sm">
            Review
          </Badge>
        );
      case "low":
        return (
          <Badge variant="success" size="sm">
            OK
          </Badge>
        );
      default:
        return (
          <Badge variant="default" size="sm">
            Unknown
          </Badge>
        );
    }
  };

  if (result.error) {
    return (
      <tr className="border-t border-gray-100 bg-red-50/50">
        <td className="px-2 py-2">
          <button
            onClick={onClick}
            className="text-left hover:text-clinical-600 font-mono text-xs"
          >
            {result.slideId.slice(0, 12)}...
          </button>
        </td>
        <td colSpan={3} className="px-2 py-2 text-red-600 text-xs">
          Error: {result.error}
        </td>
      </tr>
    );
  }

  return (
    <tr
      className={cn(
        "border-t border-gray-100 hover:bg-gray-50 cursor-pointer",
        result.requiresReview && "bg-yellow-50/30"
      )}
      onClick={onClick}
    >
      <td className="px-2 py-2">
        <span className="font-mono text-xs hover:text-clinical-600">
          {result.slideId.slice(0, 12)}...
        </span>
      </td>
      <td className="px-2 py-2">
        <Badge
          variant={result.prediction === "RESPONDER" ? "success" : "danger"}
          size="sm"
        >
          {result.prediction === "RESPONDER" ? "R" : "NR"}
        </Badge>
      </td>
      <td className="px-2 py-2">
        <div className="flex items-center gap-2">
          <div className="w-16 bg-gray-200 rounded-full h-1.5">
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
          <span className="text-xs text-gray-600 w-10">
            {(result.confidence * 100).toFixed(0)}%
          </span>
        </div>
      </td>
      <td className="px-2 py-2">{getUncertaintyBadge()}</td>
    </tr>
  );
}
