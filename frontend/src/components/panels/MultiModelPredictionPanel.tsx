"use client";

import React, { useEffect, useMemo, useState } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { cn } from "@/lib/utils";
import {
  Activity,
  AlertCircle,
  AlertTriangle,
  Brain,
  Clock,
  Info,
  Layers,
  RefreshCw,
  Sparkles,
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

function detectSurvivalContradictions(predictions: ModelPrediction[]): string[] {
  const contradictions: string[] = [];

  const parseYears = (name: string): number | null => {
    const m = name.match(/(\d+)\s*[-_ ]?(?:[Yy]ear|[Yy]r)\b/);
    return m ? parseInt(m[1], 10) : null;
  };

  const isDeceasedLike = (p: ModelPrediction): boolean => {
    const label = (p.label || "").toLowerCase();
    if (/(deceased|dead|poor|unfavorable|non-?surviv)/.test(label)) return true;
    if (/(surviv|alive|favorable|good)/.test(label)) return false;
    return p.score < (p.decisionThreshold ?? 0.5);
  };

  const isSurvivedLike = (p: ModelPrediction): boolean => {
    const label = (p.label || "").toLowerCase();
    if (/(surviv|alive|favorable|good)/.test(label)) return true;
    if (/(deceased|dead|poor|unfavorable|non-?surviv)/.test(label)) return false;
    return p.score >= (p.decisionThreshold ?? 0.5);
  };

  const survivalModels = predictions
    .map((p) => {
      const years = parseYears(`${p.modelName} ${p.modelId}`);
      return years !== null ? { years, prediction: p } : null;
    })
    .filter((x): x is { years: number; prediction: ModelPrediction } => x !== null)
    .sort((a, b) => a.years - b.years);

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

function PlaceholderModelRow({ modelName, description }: { modelName: string; description: string }) {
  return (
    <div className="rounded-md border border-gray-200 dark:border-navy-700 bg-gray-50 dark:bg-navy-900 px-3 py-2">
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0">
          <p className="text-sm font-medium text-gray-500 dark:text-gray-300 truncate">{modelName}</p>
          <p className="text-xs text-gray-400 dark:text-gray-500 truncate">{description}</p>
        </div>
        <div className="text-lg text-gray-400 dark:text-gray-500">--</div>
      </div>
    </div>
  );
}

function PredictionBarRow({ prediction }: { prediction: ModelPrediction }) {
  const decisionThreshold = prediction.decisionThreshold ?? 0.5;
  const isPositive = prediction.score >= decisionThreshold;
  const scorePercent = Math.max(0, Math.min(100, Math.round(prediction.score * 100)));
  const thresholdPercent = Math.max(0, Math.min(100, Math.round(decisionThreshold * 100)));

  return (
    <div
      className={cn(
        "rounded-md border px-3 py-3",
        isPositive ? "border-sky-200 dark:border-sky-800 bg-sky-50 dark:bg-sky-900/20" : "border-orange-200 dark:border-orange-800 bg-orange-50 dark:bg-orange-900/20"
      )}
    >
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0">
          <p className="text-sm font-semibold text-gray-900 dark:text-gray-100 truncate">{prediction.modelName}</p>
          <p className={cn("text-xs font-medium", isPositive ? "text-sky-700 dark:text-sky-300" : "text-orange-700 dark:text-orange-300")}>
            {prediction.label}
          </p>
        </div>
        <div className="shrink-0 flex items-center gap-2">
          <span className="text-sm font-mono font-semibold text-gray-900 dark:text-gray-100">{scorePercent}%</span>
        </div>
      </div>

      <div className="mt-2">
        <div className="relative h-2.5 rounded-full border border-gray-200 dark:border-navy-600 overflow-hidden">
          <div
            className="absolute left-0 top-0 h-full bg-gradient-to-r from-orange-100 to-orange-200 dark:from-orange-900/40 dark:to-orange-800/40"
            style={{ width: `${thresholdPercent}%` }}
          />
          <div
            className="absolute right-0 top-0 h-full bg-gradient-to-r from-sky-200 to-sky-100 dark:from-sky-800/40 dark:to-sky-900/40"
            style={{ width: `${100 - thresholdPercent}%` }}
          />
          <div
            className={cn(
              "absolute top-0 h-full w-1.5 rounded-full transition-all",
              isPositive ? "bg-sky-500 dark:bg-sky-400" : "bg-orange-500 dark:bg-orange-400"
            )}
            style={{ left: `calc(${scorePercent}% - 3px)` }}
          />
          <div
            className="absolute top-0 h-full w-px bg-gray-500/70 dark:bg-gray-400/70"
            style={{ left: `${thresholdPercent}%` }}
          />
        </div>

        <div className="mt-1 flex justify-between text-xs">
          <span className="text-orange-700 dark:text-orange-300">{prediction.negativeLabel}</span>
          <span className="text-gray-500 dark:text-gray-400">{thresholdPercent}% threshold</span>
          <span className="text-sky-700 dark:text-sky-300">{prediction.positiveLabel}</span>
        </div>
      </div>
    </div>
  );
}

function PredictionSection({
  title,
  predictions,
}: {
  title: string;
  predictions: ModelPrediction[];
}) {
  if (predictions.length === 0) return null;

  return (
    <section className="space-y-2">
      <h3 className="text-sm font-semibold text-gray-800 dark:text-gray-200">{title}</h3>
      <div className="space-y-2">
        {predictions.map((prediction) => (
          <PredictionBarRow key={prediction.modelId} prediction={prediction} />
        ))}
      </div>
    </section>
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
  embeddingProgress,
  isCached,
  cachedAt,
  onReanalyze,
}: MultiModelPredictionPanelProps) {
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const { currentProject } = useProject();

  const hasTrustedAvailableModels =
    !!availableModels &&
    availableModels.length > 0 &&
    (!currentProject.prediction_target ||
      availableModels.some((m) => m.id === currentProject.prediction_target));

  const modelsToPreview = hasTrustedAvailableModels
    ? availableModels!.map((m) => ({ id: m.id, displayName: m.name, description: m.description }))
    : getProjectFallbackPreviewModel(currentProject);

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
      console.warn("Cached time parse failed:", err);
      return "";
    }
  };

  const panelTitle = "Survival AI Predictions";

  if (error && !isLoading) {
    const errorLines = error.split("\n").filter((line) => line.trim());
    const hasDetails = errorLines.length > 1;

    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-4 w-4 text-red-500" />
            {panelTitle}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-4">
            <div className="w-14 h-14 mx-auto mb-3 rounded-full bg-red-100 dark:bg-red-900/30 flex items-center justify-center">
              <AlertCircle className="h-7 w-7 text-red-500" />
            </div>
            <p className="text-sm font-medium text-red-700 dark:text-red-300 mb-2">Analysis Failed</p>
          </div>

          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-3 mb-4">
            {hasDetails ? (
              <div className="text-left text-xs text-red-700 dark:text-red-300 space-y-1">
                {errorLines.map((line, i) => (
                  <p key={i} className={line.startsWith("•") ? "pl-2" : ""}>
                    {line}
                  </p>
                ))}
              </div>
            ) : (
              <p className="text-xs text-red-700 dark:text-red-300 text-center">{error}</p>
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

  if (isLoading) {
    const isEmbedding = embeddingProgress?.phase === "embedding";

    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-4 w-4 text-clinical-600 animate-pulse" />
            {panelTitle}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="text-center py-3">
            <div className="w-14 h-14 mx-auto mb-3 rounded-full bg-clinical-100 dark:bg-clinical-900/30 flex items-center justify-center">
              <Brain className="h-7 w-7 text-clinical-600 animate-pulse" />
            </div>
            <p className="text-sm font-medium text-gray-700 dark:text-gray-300">
              {isEmbedding ? "Generating Embeddings..." : "Running Model Inference..."}
            </p>
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
              {embeddingProgress?.message || "Running project models"}
            </p>
            {elapsedSeconds > 0 && (
              <p className="text-xs text-clinical-600 dark:text-clinical-400 mt-1">Elapsed: {elapsedSeconds}s</p>
            )}
          </div>

          <PlaceholderModelRow modelName="Model A" description="Waiting for output..." />
          <PlaceholderModelRow modelName="Model B" description="Waiting for output..." />
          <PlaceholderModelRow modelName="Model C" description="Waiting for output..." />
        </CardContent>
      </Card>
    );
  }

  if (!multiModelResult) {
    return (
      <Card className="overflow-hidden">
        <CardHeader className="bg-gradient-to-r from-gray-50 to-slate-50 dark:from-navy-900 dark:to-navy-800">
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-4 w-4 text-gray-400 dark:text-gray-500" />
            {panelTitle}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 pt-4">
          <div className="text-center py-2">
            <div className="w-14 h-14 mx-auto mb-3 rounded-full bg-gray-100 dark:bg-navy-700 flex items-center justify-center">
              <Sparkles className="h-6 w-6 text-gray-400" />
            </div>
            <p className="text-sm font-semibold text-gray-700 dark:text-gray-300">Models ready</p>
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
              Run analysis to generate survival and grading predictions.
            </p>
          </div>

          <div className="space-y-2">
            {modelsToPreview.map((model) => (
              <PlaceholderModelRow
                key={model.id}
                modelName={model.displayName}
                description={model.description}
              />
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  const primaryModelId = currentProject.prediction_target || "";
  const survivalPredictions = (multiModelResult.byCategory.cancerSpecific || []).filter((prediction) => {
    const name = prediction.modelName.toLowerCase();
    return (
      prediction.modelId !== primaryModelId &&
      !name.includes("platinum sensitivity")
    );
  });

  const gradingPredictions = multiModelResult.byCategory.generalPathology || [];
  const survivalContradictions = detectSurvivalContradictions(survivalPredictions);
  const hasAnyPredictions = survivalPredictions.length > 0 || gradingPredictions.length > 0;

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-4 w-4 text-clinical-600" />
            {panelTitle}
            {isCached && (
              <Badge variant="default" size="sm" className="bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 border-blue-200 dark:border-blue-800">
                Cached
              </Badge>
            )}
          </CardTitle>
          <div className="flex items-center gap-2">
            {isCached && cachedAt && (
              <span className="text-2xs text-gray-400 dark:text-gray-500">{formatCachedTime(cachedAt)}</span>
            )}
            {isCached && onReanalyze && (
              <button
                onClick={onReanalyze}
                className="text-2xs text-clinical-600 dark:text-clinical-400 hover:text-clinical-700 dark:hover:text-clinical-300 font-medium underline"
              >
                Re-analyze
              </button>
            )}
            {processingTime != null && processingTime > 0 && !isCached && (
              <div className="flex items-center gap-1 text-xs text-gray-500 dark:text-gray-400">
                <Clock className="h-3 w-3" />
                {processingTime < 1000
                  ? `${Math.round(processingTime)}ms`
                  : `${(processingTime / 1000).toFixed(1)}s`}
              </div>
            )}
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {multiModelResult.nPatches > 0 && (
          <div className="flex items-center gap-2 text-xs text-gray-600 dark:text-gray-300">
            <Layers className="h-3.5 w-3.5" />
            <span>Analyzed {multiModelResult.nPatches} tissue patches</span>
          </div>
        )}

        {survivalContradictions.length > 0 && (
          <div className="flex items-start gap-2 px-3 py-2 bg-orange-50 dark:bg-orange-900/20 border border-orange-200 dark:border-orange-800 rounded-md">
            <AlertTriangle className="h-4 w-4 text-orange-600 dark:text-orange-300 shrink-0 mt-0.5" />
            <div>
              <p className="text-xs font-semibold text-orange-800 dark:text-orange-200">Contradictory predictions detected</p>
              {survivalContradictions.map((line, idx) => (
                <p key={idx} className="text-xs text-orange-700 dark:text-orange-300">
                  {line}
                </p>
              ))}
            </div>
          </div>
        )}

        {hasAnyPredictions ? (
          <>
            <PredictionSection title="Survival AI Predictions" predictions={survivalPredictions} />
            <PredictionSection title="AI Tumor Grading" predictions={gradingPredictions} />
          </>
        ) : (
          <div className="text-center py-4 text-gray-500 dark:text-gray-400">
            <p className="text-sm font-medium">No secondary models configured</p>
            <p className="text-xs mt-1">Platinum sensitivity is shown in the top prediction panel.</p>
          </div>
        )}

        {currentProject.disclaimer && (
          <div className="pt-2 border-t border-gray-100 dark:border-navy-700 flex items-start gap-2">
            <Info className="h-4 w-4 text-gray-400 dark:text-gray-500 mt-0.5 shrink-0" />
            <p className="text-xs text-gray-500 dark:text-gray-400 leading-relaxed">{currentProject.disclaimer}</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
