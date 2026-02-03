"use client";

import React, { useState, useCallback } from "react";
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
        className="w-full p-3 flex items-center justify-between hover:bg-white/50 transition-colors"
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
        <div className="px-3 pb-3 space-y-3 border-t border-gray-100 bg-white/30">
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
            <div className="flex items-center gap-1">
              <Layers className="h-3 w-3" />
              <span>Trained on {prediction.nTrainingSlides} slides</span>
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

interface MultiModelPredictionPanelProps {
  multiModelResult: MultiModelResponse | null;
  isLoading?: boolean;
  processingTime?: number;
  error?: string | null;
  onRunAnalysis?: () => void;
  onRetry?: () => void;
  availableModels?: AvailableModel[];
  selectedModels?: string[];
  onModelToggle?: (modelId: string) => void;
}

export function MultiModelPredictionPanel({
  multiModelResult,
  isLoading,
  processingTime,
  error,
  onRunAnalysis,
  onRetry,
  availableModels,
  selectedModels,
  onModelToggle,
}: MultiModelPredictionPanelProps) {
  const [expandedModels, setExpandedModels] = useState<Set<string>>(new Set());

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

  // Show error state with retry
  if (error && !isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-4 w-4 text-red-500" />
            Multi-Model Analysis
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-6">
            <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-red-100 flex items-center justify-center">
              <AlertCircle className="h-8 w-8 text-red-500" />
            </div>
            <p className="text-sm font-medium text-red-700 mb-2">
              Analysis Failed
            </p>
            <p className="text-xs text-red-600 mb-4 max-w-[220px] mx-auto">
              {error}
            </p>
            {onRetry && (
              <Button
                variant="secondary"
                size="sm"
                onClick={onRetry}
                leftIcon={<RefreshCw className="h-3.5 w-3.5" />}
              >
                Retry Analysis
              </Button>
            )}
          </div>
        </CardContent>
      </Card>
    );
  }

  // Loading state with skeleton
  if (isLoading) {
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
              <Brain className="h-8 w-8 text-clinical-600 animate-pulse" />
            </div>
            <p className="text-sm font-medium text-gray-700 mb-1">
              Running Multi-Model Analysis...
            </p>
            <p className="text-xs text-gray-500">
              Processing 5 TransMIL models in parallel
            </p>
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
            <span>Estimated: 2-5 seconds</span>
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
              Run analysis to get predictions from 5 specialized TransMIL models trained on TCGA ovarian cancer data.
            </p>
          </div>

          {/* Example placeholder predictions */}
          <div className="space-y-2">
            <p className="text-2xs font-semibold text-gray-400 uppercase tracking-wide px-1">
              Available Models Preview
            </p>
            <PlaceholderModelCard
              modelName="Bevacizumab Response"
              description="Treatment response prediction"
            />
            <PlaceholderModelCard
              modelName="Platinum Sensitivity"
              description="Chemotherapy sensitivity"
            />
            <PlaceholderModelCard
              modelName="BRCA Status"
              description="Genomic biomarker prediction"
            />
          </div>

          {/* CTA Button */}
          {onRunAnalysis && (
            <Button
              variant="primary"
              size="lg"
              onClick={onRunAnalysis}
              leftIcon={<Zap className="h-4 w-4" />}
              className="w-full bg-gradient-to-r from-clinical-600 to-violet-600 hover:from-clinical-700 hover:to-violet-700"
            >
              Run Multi-Model Analysis
            </Button>
          )}

          {/* Info footer */}
          <div className="flex items-center gap-2 text-2xs text-gray-400 justify-center">
            <Info className="h-3 w-3" />
            <span>Models run in parallel for faster results</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Results view
  const { byCategory, nPatches, processingTimeMs } = multiModelResult;

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-4 w-4 text-clinical-600" />
            Multi-Model Analysis
          </CardTitle>
          {processingTimeMs > 0 && (
            <div className="flex items-center gap-1 text-xs text-gray-500">
              <Clock className="h-3 w-3" />
              {processingTimeMs < 1000
                ? `${Math.round(processingTimeMs)}ms`
                : `${(processingTimeMs / 1000).toFixed(1)}s`}
            </div>
          )}
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
        <div className="flex items-center gap-2 text-xs text-gray-600">
          <Layers className="h-3.5 w-3.5" />
          <span>Analyzed {nPatches} tissue patches</span>
        </div>

        {/* Ovarian Cancer Specific Models */}
        <CategorySection
          title="Ovarian Cancer Specific"
          icon={<HeartPulse className="h-4 w-4 text-pink-600" />}
          predictions={byCategory.ovarianCancer}
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
              These predictions are from research models trained on TCGA ovarian cancer data.
              Model reliability varies by AUC score. Clinical decisions should integrate
              multiple factors including patient history, other biomarkers, and clinician expertise.
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
