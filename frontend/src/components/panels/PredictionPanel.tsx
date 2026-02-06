"use client";

import React from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { ProgressStepper, InlineProgress } from "@/components/ui/ProgressStepper";
import { SkeletonPrediction } from "@/components/ui/Skeleton";
import { PredictionGauge, ConfidenceGauge, UncertaintyDisplay } from "@/components/ui/PredictionGauge";
import { cn, formatProbability } from "@/lib/utils";
import {
  Activity,
  AlertCircle,
  AlertTriangle,
  CheckCircle,
  XCircle,
  TrendingUp,
  Info,
  Clock,
  Target,
  RefreshCw,
  ShieldAlert,
  FlaskConical,
  Gauge,
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
}: PredictionPanelProps) {
  // Project-aware labels (must be before any returns per Rules of Hooks)
  const { currentProject } = useProject();
  const positiveLabel = currentProject.positive_class || currentProject.classes?.[1] || "Responder";
  const negativeLabel = currentProject.classes?.find(c => c !== currentProject.positive_class) || currentProject.classes?.[0] || "Non-Responder";

  // Show error state with retry
  if (error && !isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-4 w-4 text-red-500" />
            Prediction Results
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

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-4 w-4 text-clinical-600 animate-pulse" />
            Prediction Results
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
            <div className="pt-4 border-t border-gray-100">
              <SkeletonPrediction />
            </div>

            {/* Estimated time */}
            <div className="text-center">
              <p className="text-xs text-gray-500">Estimated time: ~2-4 seconds</p>
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
            <Activity className="h-4 w-4 text-gray-400" />
            Prediction Results
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-gray-500">
            <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gray-100 flex items-center justify-center">
              <Target className="h-8 w-8 text-gray-400" />
            </div>
            <p className="text-sm font-medium text-gray-600">
              No analysis results yet
            </p>
            <p className="text-xs mt-1.5 text-gray-500 max-w-[200px] mx-auto">
              Select a slide and run analysis to see treatment response predictions.
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Determine responder status
  const isResponder = prediction.score >= 0.5;
  const probabilityPercent = Math.round(prediction.score * 100);

  // Risk stratification
  const riskLevel =
    prediction.confidence >= 0.7
      ? "high"
      : prediction.confidence >= 0.4
      ? "moderate"
      : "low";

  const riskLabels = {
    high: "High Confidence",
    moderate: "Moderate Confidence",
    low: "Low Confidence",
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-4 w-4 text-clinical-600" />
            Prediction Results
          </CardTitle>
          {processingTime && (
            <div className="flex items-center gap-1 text-xs text-gray-500">
              <Clock className="h-3 w-3" />
              {processingTime < 1000
                ? `${processingTime}ms`
                : `${(processingTime / 1000).toFixed(1)}s`}
            </div>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-5">
        {/* UNCALIBRATED WARNING BANNER */}
        <div className="flex items-center gap-2 px-3 py-2 bg-amber-100 border border-amber-300 rounded-lg">
          <FlaskConical className="h-4 w-4 text-amber-700 shrink-0" />
          <span className="text-xs font-bold text-amber-800 uppercase tracking-wide">
            Uncalibrated - Research Use Only
          </span>
        </div>

        {/* Primary Prediction Display */}
        <div
          className={cn(
            "p-4 rounded-xl border-2 transition-all",
            isResponder
              ? "bg-responder-positive-bg border-responder-positive-border"
              : "bg-responder-negative-bg border-responder-negative-border"
          )}
        >
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              {isResponder ? (
                <CheckCircle className="h-6 w-6 text-responder-positive" />
              ) : (
                <XCircle className="h-6 w-6 text-responder-negative" />
              )}
              <span
                className={cn(
                  "text-xl font-bold tracking-tight",
                  isResponder
                    ? "text-responder-positive"
                    : "text-responder-negative"
                )}
              >
                {prediction.label}
              </span>
            </div>
            <Badge
              variant={
                riskLevel === "high"
                  ? "success"
                  : riskLevel === "moderate"
                  ? "warning"
                  : "default"
              }
              size="sm"
              className="font-medium"
            >
              {riskLabels[riskLevel]}
            </Badge>
          </div>

          {/* Visual Gauge + Score Display */}
          <div className="flex items-center gap-4">
            <PredictionGauge 
              value={prediction.score} 
              size="md" 
              showLabel={true}
            />
            <div className="flex-1 space-y-2">
              <div className="flex items-center gap-2 text-xs text-amber-700 bg-amber-50 px-2 py-1 rounded">
                <AlertTriangle className="h-3 w-3 shrink-0" />
                <span>Uncalibrated - do not interpret as true probability</span>
              </div>
              <p className="text-xs text-gray-600 leading-relaxed">
                {isResponder
                  ? "Model predicts likely response to bevacizumab treatment based on histopathological features."
                  : "Model predicts unlikely response to bevacizumab treatment. Consider alternative therapies."}
              </p>
            </div>
          </div>
        </div>

        {/* Probability Bar with Threshold */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-600 font-medium">Response Probability</span>
            <span className="font-mono font-semibold text-gray-900">
              {formatProbability(prediction.score)}
            </span>
          </div>

          {/* Visual Probability Bar */}
          <div className="probability-bar">
            {/* Non-responder zone (red) */}
            <div
              className="absolute left-0 top-0 h-full bg-gradient-to-r from-red-100 to-red-200"
              style={{ width: "50%" }}
            />
            {/* Responder zone (green) */}
            <div
              className="absolute right-0 top-0 h-full bg-gradient-to-r from-green-200 to-green-100"
              style={{ width: "50%" }}
            />
            {/* Actual value indicator */}
            <div
              className={cn(
                "absolute top-0 h-full w-1.5 rounded-full shadow-md transition-all duration-700 ease-out",
                isResponder ? "bg-status-positive" : "bg-status-negative"
              )}
              style={{ left: `calc(${probabilityPercent}% - 3px)` }}
            />
            {/* Threshold marker */}
            <div className="probability-bar-threshold left-1/2 z-10" />
          </div>

          {/* Scale Labels */}
          <div className="flex justify-between text-xs">
            <span className="text-red-600 font-medium">{negativeLabel}</span>
            <span className="text-gray-400">50% threshold</span>
            <span className="text-green-600 font-medium">{positiveLabel}</span>
          </div>
        </div>

        {/* Confidence Visualization */}
        <div className="p-3 bg-surface-secondary rounded-lg border border-surface-border">
          <ConfidenceGauge
            value={prediction.confidence}
            level={riskLevel}
          />
        </div>

        {/* Uncertainty Quantification Section */}
        {uncertaintyResult ? (
          <UncertaintyDisplay
            uncertainty={uncertaintyResult.uncertainty}
            confidenceInterval={uncertaintyResult.confidenceInterval}
            samples={uncertaintyResult.samples}
          />
        ) : onRunUncertaintyAnalysis ? (
          <div className="p-3 bg-gray-50 rounded-lg border border-gray-200">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <BarChart3 className="h-4 w-4 text-gray-500" />
                <div>
                  <p className="text-xs font-medium text-gray-700">
                    Uncertainty Analysis
                  </p>
                  <p className="text-2xs text-gray-500">
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
          <div className="p-3 bg-red-50 border border-red-200 rounded-lg animate-fade-in">
            <div className="flex items-start gap-2">
              <ShieldAlert className="h-4 w-4 text-red-600 mt-0.5 shrink-0" />
              <div>
                <p className="text-xs font-semibold text-red-800">
                  Slide Quality Warning
                </p>
                <p className="text-xs text-red-700 mt-1 leading-relaxed">
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
          <div className="p-3 bg-yellow-50 border border-yellow-200 rounded-lg animate-fade-in">
            <div className="flex items-start gap-2">
              <ShieldAlert className="h-4 w-4 text-yellow-600 mt-0.5 shrink-0" />
              <div>
                <p className="text-xs font-semibold text-yellow-800">
                  Quality Note
                </p>
                <p className="text-xs text-yellow-700 mt-1 leading-relaxed">
                  Slide quality is acceptable but not optimal. Review prediction with caution.
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Calibration Note */}
        {prediction.calibrationNote && (
          <div className="p-3 bg-amber-50 border border-amber-200 rounded-lg animate-fade-in">
            <div className="flex items-start gap-2">
              <TrendingUp className="h-4 w-4 text-amber-600 mt-0.5 shrink-0" />
              <div>
                <p className="text-xs font-semibold text-amber-800">
                  Calibration Note
                </p>
                <p className="text-xs text-amber-700 mt-1 leading-relaxed">
                  {prediction.calibrationNote}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Clinical Disclaimer */}
        <div className="pt-3 border-t border-gray-100">
          <div className="flex items-start gap-2">
            <Info className="h-4 w-4 text-gray-400 mt-0.5 shrink-0" />
            <p className="text-xs text-gray-500 leading-relaxed">
              This prediction is for research and decision support only. Clinical
              decisions should integrate multiple factors including patient history,
              other biomarkers, and clinician expertise.
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
