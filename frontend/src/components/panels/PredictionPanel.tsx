"use client";

import React from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { cn, formatProbability, getConfidenceClass } from "@/lib/utils";
import { Activity, AlertCircle, CheckCircle, TrendingUp } from "lucide-react";
import type { PredictionResult } from "@/types";

interface PredictionPanelProps {
  prediction: PredictionResult | null;
  isLoading?: boolean;
  processingTime?: number;
}

export function PredictionPanel({
  prediction,
  isLoading,
  processingTime,
}: PredictionPanelProps) {
  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-4 w-4" />
            Prediction Results
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-8">
            <div className="animate-pulse flex flex-col items-center">
              <div className="h-16 w-16 bg-gray-200 rounded-full mb-4" />
              <div className="h-4 w-32 bg-gray-200 rounded" />
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
            <Activity className="h-4 w-4" />
            Prediction Results
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-gray-500">
            <AlertCircle className="h-8 w-8 mx-auto mb-2 text-gray-400" />
            <p className="text-sm">No analysis results yet.</p>
            <p className="text-xs mt-1">Select a slide and run analysis to see predictions.</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Convert numeric confidence to label
  const confidenceLabel = 
    prediction.confidence >= 0.7 ? "High" :
    prediction.confidence >= 0.4 ? "Moderate" : "Low";
  
  const confidenceVariant =
    prediction.confidence >= 0.7
      ? "success"
      : prediction.confidence >= 0.4
      ? "warning"
      : "danger";

  // Calculate the probability bar width
  const probabilityPercent = Math.round(prediction.score * 100);

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-4 w-4" />
            Prediction Results
          </CardTitle>
          {processingTime && (
            <span className="text-xs text-gray-500">
              {processingTime < 1000
                ? `${processingTime}ms`
                : `${(processingTime / 1000).toFixed(1)}s`}
            </span>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Primary Prediction */}
        <div className="text-center py-2">
          <div className="flex items-center justify-center gap-2 mb-2">
            {prediction.score >= 0.5 ? (
              <CheckCircle className="h-5 w-5 text-status-positive" />
            ) : (
              <AlertCircle className="h-5 w-5 text-status-negative" />
            )}
            <span className="text-lg font-semibold text-gray-900">
              {prediction.label}
            </span>
          </div>
          <Badge variant={confidenceVariant} size="md">
            {confidenceLabel} Confidence
          </Badge>
        </div>

        {/* Probability Score */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-600">Probability Score</span>
            <span className="font-mono font-medium text-gray-900">
              {formatProbability(prediction.score)}
            </span>
          </div>
          <div className="relative h-3 bg-gray-100 rounded-full overflow-hidden">
            <div
              className={cn(
                "absolute left-0 top-0 h-full rounded-full transition-all duration-500",
                prediction.score >= 0.7
                  ? "bg-status-positive"
                  : prediction.score >= 0.4
                  ? "bg-status-warning"
                  : "bg-status-negative"
              )}
              style={{ width: `${probabilityPercent}%` }}
            />
            {/* Threshold marker at 50% */}
            <div className="absolute left-1/2 top-0 h-full w-0.5 bg-gray-400" />
          </div>
          <div className="flex justify-between text-xs text-gray-500">
            <span>0%</span>
            <span>50% threshold</span>
            <span>100%</span>
          </div>
        </div>

        {/* Calibration Note */}
        {prediction.calibrationNote && (
          <div className="mt-4 p-3 bg-amber-50 border border-amber-200 rounded-lg">
            <div className="flex items-start gap-2">
              <TrendingUp className="h-4 w-4 text-amber-600 mt-0.5 shrink-0" />
              <div>
                <p className="text-xs font-medium text-amber-800">
                  Calibration Note
                </p>
                <p className="text-xs text-amber-700 mt-1">
                  {prediction.calibrationNote}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Disclaimer */}
        <div className="mt-4 pt-4 border-t border-gray-100">
          <p className="text-xs text-gray-500 leading-relaxed">
            This prediction is for research and decision support only. It should
            not be used as a sole basis for clinical decisions. Always consult
            with qualified healthcare professionals.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
