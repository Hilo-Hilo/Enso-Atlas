"use client";

import React from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { ProgressStepper, InlineProgress } from "@/components/ui/ProgressStepper";
import { SkeletonPrediction } from "@/components/ui/Skeleton";
import { PredictionGauge, UncertaintyDisplay } from "@/components/ui/PredictionGauge";
import { cn, formatProbability, humanizeIdentifier } from "@/lib/utils";
import {
  Activity,
  AlertCircle,
  Info,
  Clock,
  Target,
  RefreshCw,
  ShieldAlert,
  BarChart3,
} from "lucide-react";
import type { PredictionResult, SlideQCMetrics, UncertaintyResult } from "@/types";
import { ANALYSIS_STEPS } from "@/hooks/useAnalysis";
import { useProject } from "@/contexts/ProjectContext";

interface PredictionPanelProps {
  prediction: PredictionResult | null;
  isLoading?: boolean;
  processingTime?: number;
  analysisStep?: number;
  error?: string | null;
  onRetry?: () => void;
  qcMetrics?: SlideQCMetrics | null;
  // Uncertainty quantification
  uncertaintyResult?: UncertaintyResult | null;
  isAnalyzingUncertainty?: boolean;
  onRunUncertaintyAnalysis?: () => void;
  // Cache indicators
  isCached?: boolean;
  cachedAt?: string | null;
  onReanalyze?: () => void;
}

export function PredictionPanel({
  prediction,
  isLoading,
  processingTime,
  analysisStep = -1,
  error,
  onRetry,
  qcMetrics,
  uncertaintyResult,
  isAnalyzingUncertainty,
  onRunUncertaintyAnalysis,
  isCached,
  cachedAt,
  onReanalyze,
}: PredictionPanelProps) {
  // Project-aware labels (must be before any returns per Rules of Hooks)
  const { currentProject } = useProject();
  const positiveLabel = currentProject.positive_class || currentProject.classes?.[1] || "Positive";
  const negativeLabel = currentProject.classes?.find(c => c !== currentProject.positive_class) || currentProject.classes?.[0] || "Negative";
  const predictionTargetLabel = humanizeIdentifier(currentProject.prediction_target);
  const predictsTumorStage =
    currentProject.id === "lung-stage" ||
    currentProject.prediction_target === "tumor_stage" ||
    currentProject.prediction_target?.includes("stage");
  const panelTitle = predictsTumorStage
    ? "Tumor Stage - AI Prediction"
    : "Resistance to Therapy - AI Prediction";
  const predictionSummaryText = predictsTumorStage
    ? "Model predicts tumor stage."
    : "Model predicts likelihood of sensitivity to platinum therapy.";
  const projectDisclaimer =
    "This prediction is for decision support and to enhance interpretability by the physician. Clinical decisions should integrate multiple factors including patient history, other biomarkers, and clinician expertise.";

  const isEarlyStageLabel = (label: string): boolean => {
    const lower = String(label || "").toLowerCase();
    return /(early|stage\s*i\b|stage\s*ia\b|stage\s*ib\b|stage\s*ic\b|stage\s*1\b|stage\s*1a\b|stage\s*1b\b|stage\s*1c\b|\bi\b|\bia\b|\bib\b|\bic\b)/.test(lower);
  };

  const isLateStageLabel = (label: string): boolean => {
    const lower = String(label || "").toLowerCase();
    return /(late|stage\s*ii\b|stage\s*iii\b|stage\s*iv\b|stage\s*2\b|stage\s*3\b|stage\s*4\b|\bii\b|\biii\b|\biv\b)/.test(lower);
  };

  // Show error state with retry
  if (error && !isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-4 w-4 text-red-500" />
            {panelTitle}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-6">
            <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-red-100 dark:bg-red-900/30 flex items-center justify-center">
              <AlertCircle className="h-8 w-8 text-red-500" />
            </div>
            <p className="text-sm font-medium text-red-700 dark:text-red-300 mb-2">
              Analysis Failed
            </p>
            <p className="text-xs text-red-600 dark:text-red-400 mb-4 max-w-[220px] mx-auto">
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

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-4 w-4 text-clinical-600 animate-pulse" />
            {panelTitle}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="py-4 space-y-6">
            {/* Progress stepper */}
            <ProgressStepper
              steps={ANALYSIS_STEPS.map((s) => ({
                id: s.id,
                label: s.label,
                description: s.description,
              }))}
              currentStep={analysisStep}
            />

            {/* Skeleton preview while waiting */}
            <div className="pt-4 border-t border-gray-100 dark:border-navy-700">
              <SkeletonPrediction />
            </div>

            {/* Estimated time */}
            <div className="text-center">
              <p className="text-xs text-gray-500 dark:text-gray-400">Estimated time: ~2-4 seconds</p>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!prediction) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-4 w-4 text-gray-400 dark:text-gray-500" />
            {panelTitle}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-gray-500 dark:text-gray-400">
            <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gray-100 dark:bg-navy-700 flex items-center justify-center">
              <Target className="h-8 w-8 text-gray-400" />
            </div>
            <p className="text-sm font-medium text-gray-600 dark:text-gray-300">
              No analysis results yet
            </p>
            <p className="text-xs mt-1.5 text-gray-500 dark:text-gray-400 max-w-[220px] mx-auto">
              Select a slide and run analysis to see {predictionTargetLabel.toLowerCase()} predictions.
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Determine predicted class side (threshold-based binary decision)
  const isPositivePrediction = prediction.score >= 0.5;
  const predictedClassLabel = isPositivePrediction ? positiveLabel : negativeLabel;
  const predictedIsEarlyStage = isEarlyStageLabel(predictedClassLabel);
  const predictedIsLateStage = isLateStageLabel(predictedClassLabel);
  const useFavorableStyling = predictsTumorStage
    ? (predictedIsEarlyStage || prediction.score < 0.5)
    : isPositivePrediction;
  const leftLabelIsEarly = isEarlyStageLabel(negativeLabel);
  const leftLabelIsLate = isLateStageLabel(negativeLabel);
  const rightLabelIsEarly = isEarlyStageLabel(positiveLabel);
  const rightLabelIsLate = isLateStageLabel(positiveLabel);
  const labelsAmbiguousForStage =
    predictsTumorStage &&
    !leftLabelIsEarly &&
    !leftLabelIsLate &&
    !rightLabelIsEarly &&
    !rightLabelIsLate;
  const leftSideFavorable = predictsTumorStage
    ? (leftLabelIsEarly || labelsAmbiguousForStage)
    : false;
  const rightSideFavorable = predictsTumorStage
    ? rightLabelIsEarly
    : true;

  const probabilityPercent = Math.round(prediction.score * 100);
  const confidencePercent = Math.max(0, Math.min(100, Math.round(prediction.confidence * 100)));

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
              <span className="text-2xs text-gray-400 dark:text-gray-500">
                {(() => {
                  try {
                    const diffMs = Date.now() - new Date(cachedAt).getTime();
                    const min = Math.floor(diffMs / 60000);
                    if (min < 1) return "just now";
                    if (min < 60) return `${min} min ago`;
                    const hr = Math.floor(min / 60);
                    if (hr < 24) return `${hr}h ago`;
                    return `${Math.floor(hr / 24)}d ago`;
                  } catch (err) { console.warn("Date parse error:", err); return ""; }
                })()}
              </span>
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
                  ? `${processingTime}ms`
                  : `${(processingTime / 1000).toFixed(1)}s`}
              </div>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-5">
        {/* Primary Prediction Display */}
        <div
          className={cn(
            "p-4 rounded-xl border-2 transition-all",
            useFavorableStyling
              ? "bg-sky-50 dark:bg-sky-900/20 border-sky-200 dark:border-sky-800"
              : "bg-orange-50 dark:bg-orange-900/20 border-orange-200 dark:border-orange-800"
          )}
        >
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <div
                className={cn(
                  "h-6 w-6 rounded-full flex items-center justify-center border shadow-sm",
                  useFavorableStyling
                    ? "bg-white dark:bg-navy-800 border-sky-400 dark:border-sky-600 shadow-sky-200/60 dark:shadow-sky-900/40"
                    : "bg-white dark:bg-navy-800 border-orange-400 dark:border-orange-600 shadow-orange-200/60 dark:shadow-orange-900/40"
                )}
                aria-hidden="true"
              >
                <span
                  className={cn(
                    "h-2.5 w-2.5 rounded-full border border-white/80 shadow-inner",
                    useFavorableStyling
                      ? "bg-gradient-to-br from-sky-400 to-sky-600"
                      : "bg-gradient-to-br from-orange-300 to-orange-500"
                  )}
                />
              </div>
              <span
                className={cn(
                  "text-[17.5px] font-bold tracking-tight",
                  useFavorableStyling
                    ? "text-sky-700 dark:text-sky-300"
                    : "text-orange-700 dark:text-orange-300"
                )}
              >
                {predictedClassLabel.toUpperCase()}
              </span>
            </div>
          </div>

          {/* Visual Gauge + Score Display */}
          <div className="flex items-center gap-4">
              <PredictionGauge 
                value={prediction.score} 
                size="md" 
                showLabel={true}
                isFavorable={useFavorableStyling}
              />
              <div className="flex-1 space-y-2">
                <p className="text-xs text-gray-600 dark:text-gray-300 leading-relaxed">
                  {predictionSummaryText}
                </p>
              </div>
            </div>
          </div>

        {/* Probability Bar with Threshold */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-600 dark:text-gray-300 font-medium">{predictionTargetLabel} Score</span>
            <span className="font-mono font-semibold text-gray-900 dark:text-gray-100">
              {formatProbability(prediction.score)}
            </span>
          </div>

          {/* Visual Probability Bar */}
          <div className="probability-bar">
            {/* Lower-score zone */}
            <div
              className={cn(
                "absolute left-0 top-0 h-full",
                leftSideFavorable
                  ? "bg-gradient-to-r from-sky-100 to-sky-200 dark:from-sky-900/40 dark:to-sky-800/40"
                  : "bg-gradient-to-r from-orange-100 to-orange-200 dark:from-orange-900/40 dark:to-orange-800/40"
              )}
              style={{ width: "50%" }}
            />
            {/* Higher-score zone */}
            <div
              className={cn(
                "absolute right-0 top-0 h-full",
                rightSideFavorable
                  ? "bg-gradient-to-r from-sky-200 to-sky-100 dark:from-sky-800/40 dark:to-sky-900/40"
                  : "bg-gradient-to-r from-orange-200 to-orange-100 dark:from-orange-800/40 dark:to-orange-900/40"
              )}
              style={{ width: "50%" }}
            />
            {/* Actual value indicator */}
            <div
              className={cn(
                "absolute top-0 h-full w-1.5 rounded-full shadow-md transition-all duration-700 ease-out",
                useFavorableStyling ? "bg-sky-500 dark:bg-sky-400" : "bg-orange-500 dark:bg-orange-400"
              )}
              style={{ left: `calc(${probabilityPercent}% - 3px)` }}
            />
            {/* Threshold marker */}
            <div className="probability-bar-threshold left-1/2 z-10" />
          </div>

          {/* Scale Labels */}
          <div className="flex justify-between text-xs">
            <span className={cn("font-medium", leftSideFavorable ? "text-sky-600 dark:text-sky-300" : "text-orange-600 dark:text-orange-300")}>{negativeLabel}</span>
            <span className="text-gray-400 dark:text-gray-500">50% threshold</span>
            <span className={cn("font-medium", rightSideFavorable ? "text-sky-600 dark:text-sky-300" : "text-orange-600 dark:text-orange-300")}>{positiveLabel}</span>
          </div>
        </div>

        {/* Confidence Bar */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-600 dark:text-gray-300 font-medium">Model Confidence</span>
            <span className="font-mono font-semibold text-gray-900 dark:text-gray-100">
              {confidencePercent}%
            </span>
          </div>

          <div className="probability-bar h-3 bg-emerald-50 dark:bg-emerald-900/20">
            <div
              className="absolute left-0 top-0 h-full bg-gradient-to-r from-emerald-100 to-emerald-200 dark:from-emerald-900/40 dark:to-emerald-800/40 transition-all duration-700 ease-out"
              style={{ width: `${confidencePercent}%` }}
            />
            <div className="probability-bar-threshold left-1/3 z-10 bg-emerald-300" />
            <div className="probability-bar-threshold left-2/3 z-10 bg-emerald-300" />
            <div
              className="absolute top-0 h-full w-1.5 rounded-full shadow-md transition-all duration-700 ease-out bg-emerald-500"
              style={{ left: `calc(${confidencePercent}% - 3px)` }}
            />
          </div>

          <div className="flex justify-between text-xs">
            <span className="text-emerald-600 dark:text-emerald-300 font-medium">low</span>
            <span className="text-gray-400">moderate</span>
            <span className="text-emerald-600 dark:text-emerald-300 font-medium">high</span>
          </div>
        </div>

        {/* Uncertainty Quantification Section */}
        {uncertaintyResult ? (
          <UncertaintyDisplay
            uncertainty={uncertaintyResult.uncertainty}
            confidenceInterval={uncertaintyResult.confidenceInterval}
            samples={uncertaintyResult.samples}
          />
        ) : onRunUncertaintyAnalysis ? (
          <div className="p-3 bg-gray-50 dark:bg-navy-900 rounded-lg border border-gray-200 dark:border-navy-700">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <BarChart3 className="h-4 w-4 text-gray-500 dark:text-gray-400" />
                <div>
                  <p className="text-xs font-medium text-gray-700 dark:text-gray-200">
                    Uncertainty Analysis
                  </p>
                  <p className="text-2xs text-gray-500 dark:text-gray-400">
                    Run MC Dropout for confidence intervals
                  </p>
                </div>
              </div>
              <Button
                variant="secondary"
                size="sm"
                onClick={onRunUncertaintyAnalysis}
                disabled={isAnalyzingUncertainty}
                leftIcon={
                  isAnalyzingUncertainty ? (
                    <RefreshCw className="h-3 w-3 animate-spin" />
                  ) : (
                    <BarChart3 className="h-3 w-3" />
                  )
                }
              >
                {isAnalyzingUncertainty ? "Analyzing..." : "Analyze"}
              </Button>
            </div>
          </div>
        ) : null}

        {/* Slide Quality Warning */}
        {qcMetrics && qcMetrics.overallQuality === "poor" && (
          <div className="p-3 bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-800 rounded-lg animate-fade-in">
            <div className="flex items-start gap-2">
              <ShieldAlert className="h-4 w-4 text-red-600 dark:text-red-300 mt-0.5 shrink-0" />
              <div>
                <p className="text-xs font-semibold text-red-800 dark:text-red-200">
                  Slide Quality Warning
                </p>
                <p className="text-xs text-red-700 dark:text-red-300 mt-1 leading-relaxed">
                  Slide quality may affect prediction accuracy. Issues detected:{" "}
                  {[
                    qcMetrics.blurScore > 0.2 && "blur",
                    qcMetrics.tissueCoverage < 0.5 && "low tissue coverage",
                    qcMetrics.stainUniformity < 0.6 && "uneven staining",
                    qcMetrics.artifactDetected && "artifacts",
                    qcMetrics.penMarks && "pen marks",
                    qcMetrics.foldDetected && "tissue folds",
                  ]
                    .filter(Boolean)
                    .join(", ") || "general quality concerns"}.
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Acceptable Quality Note */}
        {qcMetrics && qcMetrics.overallQuality === "acceptable" && (
          <div className="p-3 bg-yellow-50 dark:bg-yellow-900/30 border border-yellow-200 dark:border-yellow-800 rounded-lg animate-fade-in">
            <div className="flex items-start gap-2">
              <ShieldAlert className="h-4 w-4 text-yellow-600 dark:text-yellow-300 mt-0.5 shrink-0" />
              <div>
                <p className="text-xs font-semibold text-yellow-800 dark:text-yellow-200">
                  Quality Note
                </p>
                <p className="text-xs text-yellow-700 dark:text-yellow-300 mt-1 leading-relaxed">
                  Slide quality is acceptable but not optimal. Review prediction with caution.
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Clinical Disclaimer */}
        <div className="pt-3 border-t border-gray-100 dark:border-navy-700">
          <div className="flex items-start gap-2">
            <Info className="h-4 w-4 text-gray-400 dark:text-gray-500 mt-0.5 shrink-0" />
            <p className="text-xs text-gray-500 dark:text-gray-400 leading-relaxed">
              {projectDisclaimer}
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
