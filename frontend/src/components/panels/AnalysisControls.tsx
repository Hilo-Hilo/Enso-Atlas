"use client";

import React from "react";
import { ModelPicker } from "./ModelPicker";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { InlineProgress } from "@/components/ui/ProgressStepper";
import { ANALYSIS_STEPS } from "@/hooks/useAnalysis";
import { AlertCircle, Clock, Cpu, Play } from "lucide-react";

interface AnalysisControlsProps {
  selectedSlideId: string | null;
  selectedSlideHasLevel0: boolean;
  selectedModels: string[];
  onModelsChange: (models: string[]) => void;
  resolutionLevel: number;
  onResolutionChange: (level: number) => void;
  forceReembed: boolean;
  onForceReembedChange: (force: boolean) => void;
  onAnalyze: () => void;
  onGenerateEmbeddings?: () => void;
  isAnalyzing?: boolean;
  analysisStep?: number;
  isGeneratingEmbeddings?: boolean;
  embeddingProgress?: {
    phase: string;
    progress: number;
    message: string;
  } | null;
  embeddingStatus?: {
    hasLevel0: boolean;
    hasLevel1: boolean;
  };
}

export function AnalysisControls({
  selectedSlideId,
  selectedSlideHasLevel0,
  selectedModels,
  onModelsChange,
  resolutionLevel,
  onResolutionChange,
  forceReembed,
  onForceReembedChange,
  onAnalyze,
  onGenerateEmbeddings,
  isAnalyzing = false,
  analysisStep = -1,
  isGeneratingEmbeddings = false,
  embeddingProgress,
  embeddingStatus,
}: AnalysisControlsProps) {
  const hasSlideSelection = Boolean(selectedSlideId);
  const hasSelectedModels = selectedModels.length > 0;

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-sm">
          <Play className="h-4 w-4 text-clinical-600" />
          Run Analysis
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <ModelPicker
          selectedModels={selectedModels}
          onSelectionChange={onModelsChange}
          resolutionLevel={resolutionLevel}
          onResolutionChange={onResolutionChange}
          forceReembed={forceReembed}
          onForceReembedChange={onForceReembedChange}
          disabled={!hasSlideSelection || isAnalyzing || isGeneratingEmbeddings}
          embeddingStatus={embeddingStatus}
          selectedSlideId={selectedSlideId}
          className="shadow-sm"
        />

        {!hasSlideSelection && (
          <p className="text-xs text-gray-500">Select a case to enable analysis.</p>
        )}

        {isGeneratingEmbeddings && embeddingProgress && (
          <div className="p-3 bg-violet-50 border border-violet-200 rounded-md">
            <div className="flex items-center gap-2 mb-2">
              <Cpu className="h-4 w-4 text-violet-600 animate-pulse" />
              <span className="text-sm font-medium text-violet-800">Generating Embeddings</span>
            </div>
            <p className="text-xs text-violet-700 mb-2">{embeddingProgress.message}</p>
            <div className="w-full h-2 bg-violet-200 rounded-full overflow-hidden">
              <div
                className="h-full bg-violet-600 transition-all duration-500"
                style={{ width: `${embeddingProgress.progress}%` }}
              />
            </div>
            <div className="flex items-center justify-between mt-2 text-2xs text-violet-700">
              <span>{Math.round(embeddingProgress.progress)}%</span>
              <span className="inline-flex items-center gap-1">
                <Clock className="h-3 w-3" />
                Full-resolution runs may take 5-20 min
              </span>
            </div>
          </div>
        )}

        {isAnalyzing && analysisStep >= 0 ? (
          <div className="p-3 bg-clinical-50 border border-clinical-200 rounded-md">
            <InlineProgress
              steps={ANALYSIS_STEPS.map((s) => s.label)}
              currentStep={analysisStep}
            />
          </div>
        ) : (
          <>
            {resolutionLevel === 0 &&
            hasSlideSelection &&
            !selectedSlideHasLevel0 &&
            !isGeneratingEmbeddings &&
            onGenerateEmbeddings ? (
              <div className="space-y-2">
                <div className="p-3 bg-amber-50 border border-amber-200 rounded-md">
                  <div className="flex items-start gap-2">
                    <AlertCircle className="h-4 w-4 text-amber-600 mt-0.5 shrink-0" />
                    <p className="text-xs text-amber-700">
                      Level 0 embeddings are not ready for this case. Generate them first, or switch
                      to Level 1 for faster startup.
                    </p>
                  </div>
                </div>
                <Button
                  variant="secondary"
                  size="lg"
                  onClick={onGenerateEmbeddings}
                  disabled={!hasSlideSelection || isAnalyzing || isGeneratingEmbeddings}
                  className="w-full border-violet-300 text-violet-700 hover:bg-violet-50"
                >
                  <Cpu className="h-4 w-4 mr-2" />
                  Generate Level 0 Embeddings
                </Button>
              </div>
            ) : (
              <>
                <Button
                  variant="secondary"
                  size="lg"
                  onClick={onAnalyze}
                  disabled={!hasSlideSelection || !hasSelectedModels || isAnalyzing || isGeneratingEmbeddings}
                  isLoading={isAnalyzing}
                  className="w-full border-sky-300 bg-sky-100 text-sky-900 hover:bg-sky-200"
                  data-action="run-analysis"
                  data-demo="analyze-button"
                >
                  {isGeneratingEmbeddings
                    ? "Generating Embeddings..."
                    : !hasSelectedModels
                    ? "Select at least one model"
                    : "Run Analysis"}
                </Button>
                {hasSlideSelection && !hasSelectedModels && (
                  <p className="text-2xs text-amber-700">
                    Select one or more models before running analysis.
                  </p>
                )}
              </>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}
