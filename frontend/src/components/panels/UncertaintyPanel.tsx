"use client";

import React, { useState, useCallback } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { cn } from "@/lib/utils";
import {
  AlertTriangle,
  Activity,
  CheckCircle,
  XCircle,
  HelpCircle,
  TrendingUp,
  BarChart3,
  RefreshCw,
  ShieldAlert,
  Stethoscope,
  Info,
} from "lucide-react";
import type { UncertaintyResult } from "@/types";
import { analyzeWithUncertainty } from "@/lib/api";

interface UncertaintyPanelProps {
  slideId: string | null;
  onUncertaintyResult?: (result: UncertaintyResult) => void;
}

export function UncertaintyPanel({
  slideId,
  onUncertaintyResult,
}: UncertaintyPanelProps) {
  const [result, setResult] = useState<UncertaintyResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runUncertaintyAnalysis = useCallback(async () => {
    if (!slideId) return;

    setIsLoading(true);
    setError(null);

    try {
      const uncertaintyResult = await analyzeWithUncertainty(slideId, 20);
      setResult(uncertaintyResult);
      onUncertaintyResult?.(uncertaintyResult);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Uncertainty analysis failed";
      setError(message);
    } finally {
      setIsLoading(false);
    }
  }, [slideId, onUncertaintyResult]);

  // No slide selected state
  if (!slideId) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-4 w-4 text-gray-400" />
            Uncertainty Quantification
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-gray-500">
            <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gray-100 flex items-center justify-center">
              <HelpCircle className="h-8 w-8 text-gray-400" />
            </div>
            <p className="text-sm font-medium text-gray-600">
              Select a slide to analyze
            </p>
            <p className="text-xs mt-1.5 text-gray-500 max-w-[220px] mx-auto">
              Run uncertainty analysis to see how confident the model is about
              its prediction.
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Loading state
  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-4 w-4 text-clinical-600 animate-pulse" />
            Uncertainty Quantification
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8">
            <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-clinical-100 flex items-center justify-center animate-pulse">
              <Activity className="h-8 w-8 text-clinical-600" />
            </div>
            <p className="text-sm font-medium text-gray-600">
              Running MC Dropout...
            </p>
            <p className="text-xs mt-1.5 text-gray-500">
              Performing 20 forward passes with dropout
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Error state
  if (error) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-4 w-4 text-red-500" />
            Uncertainty Quantification
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-6">
            <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-red-100 flex items-center justify-center">
              <AlertTriangle className="h-8 w-8 text-red-500" />
            </div>
            <p className="text-sm font-medium text-red-700 mb-2">
              Analysis Failed
            </p>
            <p className="text-xs text-red-600 mb-4 max-w-[220px] mx-auto">
              {error}
            </p>
            <Button
              variant="secondary"
              size="sm"
              onClick={runUncertaintyAnalysis}
              leftIcon={<RefreshCw className="h-3.5 w-3.5" />}
            >
              Retry
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  }

  // No result yet - show button to run analysis
  if (!result) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-4 w-4 text-clinical-600" />
            Uncertainty Quantification
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-6">
            <p className="text-sm text-gray-600 mb-4 max-w-[260px] mx-auto">
              Run Monte Carlo Dropout analysis to quantify model uncertainty and
              get confidence intervals.
            </p>
            <Button
              variant="primary"
              onClick={runUncertaintyAnalysis}
              leftIcon={<TrendingUp className="h-4 w-4" />}
            >
              Run Uncertainty Analysis
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Results display
  const isResponder = result.prediction === "RESPONDER";
  const probabilityPercent = Math.round(result.probability * 100);
  const uncertaintyPercent = Math.round(result.uncertainty * 100);
  const ciLower = Math.round(result.confidenceInterval[0] * 100);
  const ciUpper = Math.round(result.confidenceInterval[1] * 100);

  // Uncertainty color coding
  const uncertaintyColors = {
    low: {
      bg: "bg-green-50",
      border: "border-green-200",
      text: "text-green-700",
      badge: "success" as const,
    },
    moderate: {
      bg: "bg-amber-50",
      border: "border-amber-200",
      text: "text-amber-700",
      badge: "warning" as const,
    },
    high: {
      bg: "bg-red-50",
      border: "border-red-200",
      text: "text-red-700",
      badge: "danger" as const,
    },
  };

  const levelColors = uncertaintyColors[result.uncertaintyLevel];

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-4 w-4 text-clinical-600" />
            Uncertainty Quantification
          </CardTitle>
          <Badge variant={levelColors.badge} size="sm">
            {result.uncertaintyLevel.toUpperCase()} UNCERTAINTY
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-5">
        {/* Review Required Banner */}
        {result.requiresReview && (
          <div className="flex items-center gap-2 px-3 py-2 bg-amber-100 border border-amber-300 rounded-lg">
            <ShieldAlert className="h-4 w-4 text-amber-700 shrink-0" />
            <span className="text-xs font-bold text-amber-800 uppercase tracking-wide">
              Flagged for Human Review
            </span>
          </div>
        )}

        {/* Prediction with Confidence Interval */}
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
                <CheckCircle className="h-5 w-5 text-responder-positive" />
              ) : (
                <XCircle className="h-5 w-5 text-responder-negative" />
              )}
              <span
                className={cn(
                  "text-lg font-bold tracking-tight",
                  isResponder
                    ? "text-responder-positive"
                    : "text-responder-negative"
                )}
              >
                {result.prediction}
              </span>
            </div>
          </div>

          {/* Probability with CI */}
          <div className="space-y-1">
            <div className="flex items-baseline gap-2">
              <span className="text-2xl font-bold font-mono text-gray-900">
                {probabilityPercent}%
              </span>
              <span className="text-sm text-gray-500">
                ({ciLower}% - {ciUpper}% CI)
              </span>
            </div>
          </div>
        </div>

        {/* Confidence Interval Visualization */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-600 font-medium">95% Confidence Interval</span>
          </div>

          {/* CI Bar Visualization */}
          <div className="relative h-8 bg-gray-100 rounded-lg overflow-hidden">
            {/* Background gradient */}
            <div className="absolute inset-0 flex">
              <div className="w-1/2 bg-gradient-to-r from-red-200 to-red-100" />
              <div className="w-1/2 bg-gradient-to-r from-green-100 to-green-200" />
            </div>

            {/* Confidence interval range */}
            <div
              className={cn(
                "absolute top-1 bottom-1 rounded-md transition-all duration-500",
                levelColors.bg,
                "border-2",
                levelColors.border
              )}
              style={{
                left: `${ciLower}%`,
                width: `${ciUpper - ciLower}%`,
              }}
            />

            {/* Mean prediction marker */}
            <div
              className={cn(
                "absolute top-0 bottom-0 w-1 rounded-full shadow-md z-10",
                isResponder ? "bg-green-600" : "bg-red-600"
              )}
              style={{ left: `calc(${probabilityPercent}% - 2px)` }}
            />

            {/* 50% threshold line */}
            <div className="absolute top-0 bottom-0 left-1/2 w-0.5 bg-gray-400 z-5" />
          </div>

          {/* Scale Labels */}
          <div className="flex justify-between text-xs">
            <span className="text-red-600 font-medium">0% (Non-R)</span>
            <span className="text-gray-400">50%</span>
            <span className="text-green-600 font-medium">100% (Resp)</span>
          </div>
        </div>

        {/* Uncertainty Metrics */}
        <div className={cn("p-3 rounded-lg border", levelColors.bg, levelColors.border)}>
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-medium text-gray-600 uppercase tracking-wide">
              Model Uncertainty (Std Dev)
            </span>
            <span className={cn("text-sm font-mono font-semibold", levelColors.text)}>
              {uncertaintyPercent}%
            </span>
          </div>
          <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
            <div
              className={cn(
                "h-full rounded-full transition-all duration-500",
                result.uncertaintyLevel === "low"
                  ? "bg-green-500"
                  : result.uncertaintyLevel === "moderate"
                  ? "bg-amber-500"
                  : "bg-red-500"
              )}
              style={{ width: `${Math.min(uncertaintyPercent * 5, 100)}%` }}
            />
          </div>
          <div className="flex justify-between mt-1.5 text-2xs text-gray-400">
            <span>Confident</span>
            <span>Uncertain</span>
          </div>
        </div>

        {/* MC Dropout Sample Distribution */}
        <div className="p-3 bg-surface-secondary rounded-lg border border-surface-border">
          <div className="flex items-center gap-2 mb-2">
            <BarChart3 className="h-3.5 w-3.5 text-gray-500" />
            <span className="text-xs font-medium text-gray-600 uppercase tracking-wide">
              Sample Distribution ({result.nSamples} samples)
            </span>
          </div>
          {/* Simple histogram visualization */}
          <div className="flex items-end gap-0.5 h-12">
            {(() => {
              // Create histogram bins
              const bins = Array(10).fill(0);
              result.samples.forEach((s) => {
                const binIdx = Math.min(Math.floor(s * 10), 9);
                bins[binIdx]++;
              });
              const maxBin = Math.max(...bins, 1);

              return bins.map((count, i) => (
                <div
                  key={i}
                  className={cn(
                    "flex-1 rounded-t transition-all",
                    i < 5 ? "bg-red-300" : "bg-green-300"
                  )}
                  style={{ height: `${(count / maxBin) * 100}%`, minHeight: count > 0 ? "4px" : "0" }}
                  title={`${i * 10}-${(i + 1) * 10}%: ${count} samples`}
                />
              ));
            })()}
          </div>
          <div className="flex justify-between text-2xs text-gray-400 mt-1">
            <span>0%</span>
            <span>50%</span>
            <span>100%</span>
          </div>
        </div>

        {/* Clinical Recommendation */}
        <div
          className={cn(
            "p-3 rounded-lg border",
            result.requiresReview
              ? "bg-amber-50 border-amber-200"
              : "bg-blue-50 border-blue-200"
          )}
        >
          <div className="flex items-start gap-2">
            <Stethoscope
              className={cn(
                "h-4 w-4 mt-0.5 shrink-0",
                result.requiresReview ? "text-amber-600" : "text-blue-600"
              )}
            />
            <div>
              <p
                className={cn(
                  "text-xs font-semibold",
                  result.requiresReview ? "text-amber-800" : "text-blue-800"
                )}
              >
                Clinical Recommendation
              </p>
              <p
                className={cn(
                  "text-xs mt-1 leading-relaxed",
                  result.requiresReview ? "text-amber-700" : "text-blue-700"
                )}
              >
                {result.clinicalRecommendation}
              </p>
            </div>
          </div>
        </div>

        {/* Technical Details */}
        <div className="pt-3 border-t border-gray-100">
          <div className="flex items-start gap-2">
            <Info className="h-4 w-4 text-gray-400 mt-0.5 shrink-0" />
            <p className="text-xs text-gray-500 leading-relaxed">
              Uncertainty estimated via Monte Carlo Dropout with {result.nSamples}{" "}
              forward passes. CI width indicates prediction reliability.
              Wide intervals suggest the model needs additional evidence.
            </p>
          </div>
        </div>

        {/* Re-run button */}
        <div className="pt-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={runUncertaintyAnalysis}
            leftIcon={<RefreshCw className="h-3.5 w-3.5" />}
            className="w-full"
          >
            Re-run Analysis
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
