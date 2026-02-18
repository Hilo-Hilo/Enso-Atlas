"use client";

import React, { useState, useCallback, useEffect } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { cn, formatProbability } from "@/lib/utils";
import {
  Activity,
  AlertCircle,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Clock,
  Target,
  RefreshCw,
  FlaskConical,
  ChevronDown,
  ChevronRight,
  ChevronUp,
  Microscope,
  HeartPulse,
  BarChart3,
  Info,
  Layers,
  Brain,
  Dna,
  Sparkles,
  TrendingUp,
  Zap,
} from "lucide-react";
import type { ModelPrediction, MultiModelResponse, AvailableModel } from "@/types";
import { useProject } from "@/contexts/ProjectContext";

function humanizeModelId(modelId: string): string {
  return modelId
    .replace(/[\-_]+/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function getProjectFallbackPreviewModel(currentProject: {
  prediction_target?: string;
  cancer_type?: string;
}): Array<{ id: string; displayName: string; description: string }> {
  if (!currentProject.prediction_target) return [];
  return [
    {
      id: currentProject.prediction_target,
      displayName: humanizeModelId(currentProject.prediction_target),
      description: `Primary ${currentProject.cancer_type || "project"} model`,
    },
  ];
}

// Skeleton Loading Component for Multi-Model Panel
function ModelCardSkeleton() {
  return (
    <div className="rounded-lg border-2 border-gray-200 bg-gray-50 p-3 animate-pulse">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="h-5 w-5 rounded-full bg-gray-300" />
          <div className="space-y-2">
            <div className="h-4 w-32 bg-gray-300 rounded" />
            <div className="h-3 w-20 bg-gray-200 rounded" />
          </div>
        </div>
        <div className="text-right space-y-1">
          <div className="h-5 w-12 bg-gray-300 rounded ml-auto" />
          <div className="h-3 w-16 bg-gray-200 rounded" />
        </div>
      </div>
    </div>
  );
}

/**
 * Detect contradictions in survival predictions.
 * E.g. if 1-year says "Deceased" but 5-year says "Survived", that's impossible.
 * Returns a list of human-readable contradiction strings.
 */
function detectSurvivalContradictions(predictions: ModelPrediction[]): string[] {
  const contradictions: string[] = [];

  const isSurvivalModel = (p: ModelPrediction): boolean => {
    const haystack = `${p.modelId} ${p.modelName}`.toLowerCase();
    return /survival|\d+\s*[-_ ]?(year|yr)\b/.test(haystack);
  };

  const parseYears = (name: string): number | null => {
    const m = name.match(/(\d+)\s*[-_ ]?(?:[Yy]ear|[Yy]r)\b/);
    return m ? parseInt(m[1], 10) : null;
  };

  const isDeceasedLike = (p: ModelPrediction): boolean => {
    const label = (p.label || "").toLowerCase();
    if (/(deceased|dead|poor|unfavorable|non-?surviv)/.test(label)) return true;
    if (/(surviv|alive|favorable|good)/.test(label)) return false;
    return p.score < 0.5;
  };

  const isSurvivedLike = (p: ModelPrediction): boolean => {
    const label = (p.label || "").toLowerCase();
    if (/(surviv|alive|favorable|good)/.test(label)) return true;
    if (/(deceased|dead|poor|unfavorable|non-?surviv)/.test(label)) return false;
    return p.score >= 0.5;
  };

  // Extract survival models and sort by timeframe
  const survivalModels = predictions
    .filter(isSurvivalModel)
    .map((p) => {
      const years = parseYears(p.modelName);
      return years !== null ? { years, prediction: p } : null;
    })
    .filter((x): x is { years: number; prediction: ModelPrediction } => x !== null)
    .sort((a, b) => a.years - b.years);

  // Logical consistency: if dead at year N, cannot be alive at year M > N
  for (let i = 0; i < survivalModels.length; i++) {
    for (let j = i + 1; j < survivalModels.length; j++) {
      const shorter = survivalModels[i];
      const longer = survivalModels[j];
      if (isDeceasedLike(shorter.prediction) && isSurvivedLike(longer.prediction)) {
        contradictions.push(
          `${shorter.prediction.modelName} predicts deceased, but ${longer.prediction.modelName} predicts survived`
        );
      }
    }
  }

  return contradictions;
}

// Example/Placeholder Prediction Card
function PlaceholderModelCard({ modelName, description }: { modelName: string; description: string }) {
  return (
    <div className="rounded-lg border-2 border-dashed border-gray-300 bg-gradient-to-br from-gray-50 to-white p-3 opacity-60">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="h-5 w-5 rounded-full bg-gray-200 flex items-center justify-center">
            <Sparkles className="h-3 w-3 text-gray-400" />
          </div>
          <div>
            <h4 className="font-semibold text-gray-400 text-sm">{modelName}</h4>
            <p className="text-xs text-gray-400">{description}</p>
          </div>
        </div>
        <div className="text-right">
          <span className="font-mono text-gray-400 text-lg">—%</span>
          <div className="text-2xs text-gray-400">AUC: —</div>
        </div>
      </div>
    </div>
  );
}

// AUC Badge Component with color coding
function AUCBadge({ auc }: { auc: number }) {
  const getAUCConfig = () => {
    if (auc >= 0.8) {
      return {
        color: "text-green-700",
        bg: "bg-green-100",
        border: "border-green-300",
        label: "High",
      };
    } else if (auc >= 0.6) {
      return {
        color: "text-yellow-700",
        bg: "bg-yellow-100",
        border: "border-yellow-300",
        label: "Moderate",
      };
    } else {
      return {
        color: "text-red-700",
        bg: "bg-red-100",
        border: "border-red-300",
        label: "Low",
      };
    }
  };

  const config = getAUCConfig();

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-2xs font-bold border",
        config.bg,
        config.color,
        config.border
      )}
      title={`Model reliability: ${config.label} (AUC ${auc.toFixed(2)})`}
    >
      <TrendingUp className="h-2.5 w-2.5" />
      AUC: {auc.toFixed(2)}
    </span>
  );
}

interface ModelCardProps {
  prediction: ModelPrediction;
  isExpanded?: boolean;
  onToggle?: () => void;
}

function ModelCard({ prediction, isExpanded, onToggle }: ModelCardProps) {
  const isPositive = prediction.score >= 0.5;
  const probabilityPercent = Math.round(prediction.score * 100);

  // Color scheme based on prediction
  const colorScheme = isPositive
    ? {
        bg: "bg-gradient-to-br from-green-50 to-emerald-50",
        border: "border-green-200",
        text: "text-green-700",
        icon: "text-green-600",
        bar: "bg-gradient-to-r from-green-500 to-emerald-500",
      }
    : {
        bg: "bg-gradient-to-br from-red-50 to-rose-50",
        border: "border-red-200",
        text: "text-red-700",
        icon: "text-red-600",
        bar: "bg-gradient-to-r from-red-500 to-rose-500",
      };

  // AUC-based reliability indicator
  const reliability =
    prediction.auc >= 0.8
      ? { label: "High", color: "text-green-600", bg: "bg-green-100" }
      : prediction.auc >= 0.7
      ? { label: "Moderate", color: "text-yellow-600", bg: "bg-yellow-100" }
      : { label: "Low", color: "text-orange-600", bg: "bg-orange-100" };

  return (
    <div
      className={cn(
        "rounded-lg border-2 transition-all overflow-hidden shadow-sm hover:shadow-md",
        colorScheme.bg,
        colorScheme.border
      )}
    >
      {/* Header - always visible */}
      <button
        className="w-full p-3 flex items-center justify-between hover:bg-white transition-colors"
        onClick={onToggle}
      >
        <div className="flex items-center gap-3">
          {isPositive ? (
            <CheckCircle className={cn("h-5 w-5", colorScheme.icon)} />
          ) : (
            <XCircle className={cn("h-5 w-5", colorScheme.icon)} />
          )}
          <div className="text-left">
            <h4 className="font-semibold text-gray-900 text-sm">
              {prediction.modelName}
            </h4>
            <p className={cn("text-xs font-medium", colorScheme.text)}>
              {prediction.label}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <div className="text-right">
            <span className="font-mono font-bold text-lg text-gray-900">
              {probabilityPercent}%
            </span>
            <div className="mt-0.5">
              <AUCBadge auc={prediction.auc} />
            </div>
          </div>
          {onToggle && (
            isExpanded ? (
              <ChevronUp className="h-4 w-4 text-gray-400" />
            ) : (
              <ChevronDown className="h-4 w-4 text-gray-400" />
            )
          )}
        </div>
      </button>

      {/* Expanded details */}
      {isExpanded && (
        <div className="px-3 pb-3 space-y-3 border-t border-gray-100 bg-white">
          {/* Probability Bar with confidence interval visualization */}
          <div className="pt-3">
            <div className="flex justify-between text-2xs text-gray-500 mb-1">
              <span>{prediction.negativeLabel}</span>
              <span>{prediction.positiveLabel}</span>
            </div>
            <div className="relative h-3 bg-gray-200 rounded-full overflow-hidden">
              {/* Confidence interval background (if available) */}
              {prediction.confidenceInterval && (
                <div
                  className="absolute h-full bg-gray-300/50 rounded-full"
                  style={{
                    left: `${prediction.confidenceInterval.lower * 100}%`,
                    width: `${(prediction.confidenceInterval.upper - prediction.confidenceInterval.lower) * 100}%`,
                  }}
                />
              )}
              {/* Main value bar */}
              <div
                className={cn("h-full rounded-full transition-all", colorScheme.bar)}
                style={{ width: `${probabilityPercent}%` }}
              />
            </div>
            {/* Show confidence interval values if available */}
            {prediction.confidenceInterval && (
              <div className="flex justify-between text-2xs text-gray-400 mt-1">
                <span>95% CI: [{Math.round(prediction.confidenceInterval.lower * 100)}%</span>
                <span>- {Math.round(prediction.confidenceInterval.upper * 100)}%]</span>
              </div>
            )}
          </div>

          {/* Description */}
          <p className="text-xs text-gray-600">{prediction.description}</p>

          {/* Stats row */}
          <div className="flex items-center gap-4 text-2xs text-gray-500">
            <div className="flex items-center gap-1">
              <BarChart3 className="h-3 w-3" />
              <span>Confidence: {(prediction.confidence * 100).toFixed(0)}%</span>
            </div>
          </div>

          {/* Reliability badge */}
          <div className="flex items-center gap-2">
            <Badge
              size="sm"
              className={cn("font-medium", reliability.bg, reliability.color)}
            >
              {reliability.label} Reliability (AUC {prediction.auc.toFixed(2)})
            </Badge>
          </div>
        </div>
      )}
    </div>
  );
}

interface CategorySectionProps {
  title: string;
  icon: React.ReactNode;
  predictions: ModelPrediction[];
  expandedModels: Set<string>;
  onToggleModel: (modelId: string) => void;
}

function CategorySection({
  title,
  icon,
  predictions,
  expandedModels,
  onToggleModel,
}: CategorySectionProps) {
  if (predictions.length === 0) return null;

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        {icon}
        <h3 className="font-semibold text-sm text-gray-700">{title}</h3>
        <Badge variant="default" size="sm">
          {predictions.length} models
        </Badge>
      </div>
      <div className="space-y-2">
        {predictions.map((pred) => (
          <ModelCard
            key={pred.modelId}
            prediction={pred}
            isExpanded={expandedModels.has(pred.modelId)}
            onToggle={() => onToggleModel(pred.modelId)}
          />
        ))}
      </div>
    </div>
  );
}

interface EmbeddingProgress {
  phase: "idle" | "embedding" | "analyzing" | "complete";
  progress: number;
  message: string;
  startTime?: number;
}

interface MultiModelPredictionPanelProps {
  multiModelResult: MultiModelResponse | null;
  isLoading?: boolean;
  processingTime?: number;
  error?: string | null;
  onRetry?: () => void;
  availableModels?: AvailableModel[];
  selectedModels?: string[];
  onModelToggle?: (modelId: string) => void;
  embeddingProgress?: EmbeddingProgress | null;
  isCached?: boolean;
  cachedAt?: string | null;
  onReanalyze?: () => void;
}

export function MultiModelPredictionPanel({
  multiModelResult,
  isLoading,
  processingTime,
  error,
  onRetry,
  availableModels,
  selectedModels,
  onModelToggle,
  embeddingProgress,
  isCached,
  cachedAt,
  onReanalyze,
}: MultiModelPredictionPanelProps) {
  const [expandedModels, setExpandedModels] = useState<Set<string>>(new Set());
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  // Project context for dynamic labels (must be before any returns per Rules of Hooks)
  const { currentProject } = useProject();
  const cancerTypeLabel = currentProject.cancer_type || "Cancer";

  // Prefer project-scoped API models; otherwise use a safe project-derived fallback
  const modelsToPreview = (availableModels && availableModels.length > 0)
    ? availableModels.map((m) => ({ id: m.id, displayName: m.name, description: m.description }))
    : getProjectFallbackPreviewModel(currentProject);

  // Update elapsed time during embedding
  useEffect(() => {
    if (!embeddingProgress?.startTime) {
      setElapsedSeconds(0);
      return;
    }
    
    const interval = setInterval(() => {
      setElapsedSeconds(Math.floor((Date.now() - embeddingProgress.startTime!) / 1000));
    }, 1000);
    
    return () => clearInterval(interval);
  }, [embeddingProgress?.startTime]);

  const toggleModelExpanded = useCallback((modelId: string) => {
    setExpandedModels((prev) => {
      const next = new Set(prev);
      if (next.has(modelId)) {
        next.delete(modelId);
      } else {
        next.add(modelId);
      }
      return next;
    });
  }, []);

  // Show error state with retry - improved for multi-line errors
  if (error && !isLoading) {
    // Parse error message for bullet points
    const errorLines = error.split("\n").filter(line => line.trim());
    const hasDetails = errorLines.length > 1;
    
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-4 w-4 text-red-500" />
            Multi-Model Analysis
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-4">
            <div className="w-14 h-14 mx-auto mb-3 rounded-full bg-red-100 flex items-center justify-center">
              <AlertCircle className="h-7 w-7 text-red-500" />
            </div>
            <p className="text-sm font-medium text-red-700 mb-2">
              Analysis Failed
            </p>
          </div>
          
          {/* Error details */}
          <div className="bg-red-50 border border-red-200 rounded-lg p-3 mb-4">
            {hasDetails ? (
              <div className="text-left text-xs text-red-700 space-y-1">
                {errorLines.map((line, i) => (
                  <p key={i} className={line.startsWith("•") ? "pl-2" : ""}>
                    {line}
                  </p>
                ))}
              </div>
            ) : (
              <p className="text-xs text-red-700 text-center">{error}</p>
            )}
          </div>
          
          {onRetry && (
            <div className="text-center">
              <Button
                variant="secondary"
                size="sm"
                onClick={onRetry}
                leftIcon={<RefreshCw className="h-3.5 w-3.5" />}
              >
                Retry Analysis
              </Button>
            </div>
          )}
        </CardContent>
      </Card>
    );
  }

  // Loading state with embedding progress
  if (isLoading) {
    const isEmbedding = embeddingProgress?.phase === "embedding";
    const elapsedTime = elapsedSeconds;
    
    const formatElapsed = (seconds: number) => {
      if (seconds < 60) return `${seconds}s`;
      const mins = Math.floor(seconds / 60);
      const secs = seconds % 60;
      return `${mins}m ${secs}s`;
    };

    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-4 w-4 text-clinical-600 animate-pulse" />
            Multi-Model Analysis
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="text-center py-4">
            <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gradient-to-br from-clinical-100 to-clinical-200 flex items-center justify-center">
              {isEmbedding ? (
                <Dna className="h-8 w-8 text-clinical-600 animate-pulse" />
              ) : (
                <Brain className="h-8 w-8 text-clinical-600 animate-pulse" />
              )}
            </div>
            <p className="text-sm font-medium text-gray-700 mb-1">
              {isEmbedding ? "Generating Embeddings..." : "Running Model Inference..."}
            </p>
            <p className="text-xs text-gray-500 max-w-[250px] mx-auto">
              {embeddingProgress?.message || (isEmbedding 
                ? "DINOv2 is processing tissue patches (may take 2-3 min for new slides)"
                : `Processing ${modelsToPreview.length} TransMIL models in parallel`
              )}
            </p>
            {isEmbedding && elapsedTime > 0 && (
              <p className="text-xs text-clinical-600 mt-2 font-medium">
                Elapsed: {formatElapsed(elapsedTime)}
              </p>
            )}
          </div>

          {/* Phase indicator */}
          <div className="flex items-center justify-center gap-3">
            <div className="flex items-center gap-2">
              <div className={`w-3 h-3 rounded-full ${isEmbedding ? "bg-clinical-500 animate-pulse" : "bg-green-500"}`} />
              <span className={`text-xs ${isEmbedding ? "text-clinical-700 font-medium" : "text-green-600"}`}>
                {isEmbedding ? "Embedding" : "✓ Embedded"}
              </span>
            </div>
            <ChevronRight className="h-3 w-3 text-gray-300" />
            <div className="flex items-center gap-2">
              <div className={`w-3 h-3 rounded-full ${!isEmbedding ? "bg-clinical-500 animate-pulse" : "bg-gray-300"}`} />
              <span className={`text-xs ${!isEmbedding ? "text-clinical-700 font-medium" : "text-gray-400"}`}>
                Inference
              </span>
            </div>
          </div>

          {/* Skeleton loading cards */}
          <div className="space-y-2">
            <ModelCardSkeleton />
            <ModelCardSkeleton />
            <ModelCardSkeleton />
          </div>

          {/* Progress indicator */}
          <div className="flex items-center justify-center gap-2 text-xs text-gray-500">
            <div className="flex gap-1">
              <div className="w-2 h-2 rounded-full bg-clinical-400 animate-bounce" style={{ animationDelay: "0ms" }} />
              <div className="w-2 h-2 rounded-full bg-clinical-400 animate-bounce" style={{ animationDelay: "150ms" }} />
              <div className="w-2 h-2 rounded-full bg-clinical-400 animate-bounce" style={{ animationDelay: "300ms" }} />
            </div>
            <span>{isEmbedding ? "This is a one-time process per slide" : "Estimated: 2-5 seconds"}</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Empty state - no results yet (enhanced)
  if (!multiModelResult) {
    return (
      <Card className="overflow-hidden">
        <CardHeader className="bg-gradient-to-r from-gray-50 to-slate-50">
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-4 w-4 text-gray-400" />
            Multi-Model Analysis
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 pt-4">
          {/* Hero section with icon */}
          <div className="text-center py-4">
            <div className="relative w-20 h-20 mx-auto mb-4">
              <div className="absolute inset-0 rounded-full bg-gradient-to-br from-clinical-100 via-violet-100 to-blue-100 animate-pulse" />
              <div className="absolute inset-2 rounded-full bg-white flex items-center justify-center">
                <Brain className="h-8 w-8 text-clinical-500" />
              </div>
              {/* Decorative orbiting dots */}
              <div className="absolute -top-1 left-1/2 w-2 h-2 rounded-full bg-clinical-400 animate-bounce" />
              <div className="absolute -bottom-1 left-1/2 w-2 h-2 rounded-full bg-violet-400 animate-bounce" style={{ animationDelay: "150ms" }} />
              <div className="absolute top-1/2 -right-1 w-2 h-2 rounded-full bg-blue-400 animate-bounce" style={{ animationDelay: "300ms" }} />
            </div>
            <p className="text-sm font-semibold text-gray-700 mb-1">
              Multi-Model Ensemble Ready
            </p>
            <p className="text-xs text-gray-500 max-w-[280px] mx-auto leading-relaxed">
              Run analysis to get predictions from {modelsToPreview.length} specialized TransMIL models{currentProject.dataset_source ? ` trained on ${currentProject.dataset_source} ${cancerTypeLabel.toLowerCase()} data` : ""}.
            </p>
          </div>

          {/* Available models from project configuration */}
          <div className="space-y-2">
            <p className="text-2xs font-semibold text-gray-400 uppercase tracking-wide px-1">
              Available Models Preview
            </p>
            {modelsToPreview.map((model) => (
              <PlaceholderModelCard
                key={model.id}
                modelName={model.displayName}
                description={model.description}
              />
            ))}
          </div>

          {/* Info footer - analysis is triggered from the left sidebar */}
          <div className="flex items-center gap-2 text-2xs text-gray-400 justify-center">
            <Info className="h-3 w-3" />
            <span>Click &quot;Run Analysis&quot; in the sidebar to analyze</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Results view
  const { byCategory, nPatches, processingTimeMs } = multiModelResult;

  // Collect all predictions for contradiction detection
  const allPredictions = [
    ...(byCategory.cancerSpecific || []),
    ...(byCategory.generalPathology || []),
  ];
  const survivalContradictions = detectSurvivalContradictions(allPredictions);

  // Format the cached timestamp into a relative string
  const formatCachedTime = (isoStr: string | null | undefined): string => {
    if (!isoStr) return "";
    try {
      const then = new Date(isoStr).getTime();
      const now = Date.now();
      const diffMs = now - then;
      const diffMin = Math.floor(diffMs / 60000);
      if (diffMin < 1) return "just now";
      if (diffMin < 60) return `${diffMin} min ago`;
      const diffHr = Math.floor(diffMin / 60);
      if (diffHr < 24) return `${diffHr}h ago`;
      const diffDay = Math.floor(diffHr / 24);
      return `${diffDay}d ago`;
    } catch (err) {
      return "";
    }
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-4 w-4 text-clinical-600" />
            Multi-Model Analysis
            {isCached && (
              <Badge variant="default" size="sm" className="bg-blue-100 text-blue-700 border-blue-200">
                Cached
              </Badge>
            )}
          </CardTitle>
          <div className="flex items-center gap-2">
            {isCached && cachedAt && (
              <span className="text-2xs text-gray-400">
                {formatCachedTime(cachedAt)}
              </span>
            )}
            {isCached && onReanalyze && (
              <button
                onClick={onReanalyze}
                className="text-2xs text-clinical-600 hover:text-clinical-700 font-medium underline"
              >
                Re-analyze
              </button>
            )}
            {processingTimeMs > 0 && !isCached && (
              <div className="flex items-center gap-1 text-xs text-gray-500">
                <Clock className="h-3 w-3" />
                {processingTimeMs < 1000
                  ? `${Math.round(processingTimeMs)}ms`
                  : `${(processingTimeMs / 1000).toFixed(1)}s`}
              </div>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-5">
        {/* Warning Banner */}
        <div className="flex items-center gap-2 px-3 py-2 bg-amber-100 border border-amber-300 rounded-lg">
          <FlaskConical className="h-4 w-4 text-amber-700 shrink-0" />
          <span className="text-xs font-bold text-amber-800 uppercase tracking-wide">
            Research Use Only - Models Not Clinically Validated
          </span>
        </div>

        {/* Patches analyzed info */}
        {nPatches > 0 && (
          <div className="flex items-center gap-2 text-xs text-gray-600">
            <Layers className="h-3.5 w-3.5" />
            <span>Analyzed {nPatches} tissue patches</span>
          </div>
        )}

        {/* Survival Contradiction Warning */}
        {survivalContradictions.length > 0 && (
          <div className="flex items-start gap-2 px-3 py-2.5 bg-orange-50 border border-orange-300 rounded-lg">
            <AlertTriangle className="h-4 w-4 text-orange-600 shrink-0 mt-0.5" />
            <div className="space-y-1">
              <span className="text-xs font-semibold text-orange-800">
                Contradictory Predictions Detected
              </span>
              {survivalContradictions.map((c, i) => (
                <p key={i} className="text-xs text-orange-700">{c}</p>
              ))}
              <p className="text-2xs text-orange-600 italic">
                These models are independent and may disagree. Interpret with caution.
              </p>
            </div>
          </div>
        )}

        {/* Cancer-Specific Models */}
        <CategorySection
          title={`${cancerTypeLabel} Specific`}
          icon={<HeartPulse className="h-4 w-4 text-pink-600" />}
          predictions={byCategory.cancerSpecific}
          expandedModels={expandedModels}
          onToggleModel={toggleModelExpanded}
        />

        {/* General Pathology Models */}
        <CategorySection
          title="General Pathology"
          icon={<Microscope className="h-4 w-4 text-blue-600" />}
          predictions={byCategory.generalPathology}
          expandedModels={expandedModels}
          onToggleModel={toggleModelExpanded}
        />

        {/* Clinical Disclaimer */}
        <div className="pt-3 border-t border-gray-100">
          <div className="flex items-start gap-2">
            <Info className="h-4 w-4 text-gray-400 mt-0.5 shrink-0" />
            <p className="text-xs text-gray-500 leading-relaxed">
              {currentProject.disclaimer || `These predictions are from research models. Model reliability varies by AUC score. Clinical decisions should integrate multiple factors including patient history, other biomarkers, and clinician expertise.`}
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
