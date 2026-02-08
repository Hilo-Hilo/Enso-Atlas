"use client";

import React from "react";
import { ModelPicker } from "./ModelPicker";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { InlineProgress } from "@/components/ui/ProgressStepper";
import { ANALYSIS_STEPS } from "@/hooks/useAnalysis";
import {
  AlertCircle,
  Cpu,
  Clock,
  Play,
} from "lucide-react";

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
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-sm">
          <Play className="h-4 w-4 text-clinical-600" />
          Analysis
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {isAnalyzing && analysisStep >= 0 ? (
          <div className="p-4 bg-clinical-50 border border-clinical-200 rounded-lg">
            <InlineProgress
              steps={ANALYSIS_STEPS.map((s) => s.label)}
              currentStep={analysisStep}
            />
          </div>
        ) : (
          <>
            {/* Model Picker */}
            <ModelPicker
              selectedModels={selectedModels}
              onSelectionChange={onModelsChange}
              resolutionLevel={resolutionLevel}
              onResolutionChange={onResolutionChange}
              forceReembed={forceReembed}
              onForceReembedChange={onForceReembedChange}
              disabled={isAnalyzing || isGeneratingEmbeddings}
              embeddingStatus={embeddingStatus}
              selectedSlideId={selectedSlideId}
            />

            {/* Embedding Progress for Level 0 */}
            {isGeneratingEmbeddings && embeddingProgress && (
              <div className="p-4 bg-violet-50 border border-violet-200 rounded-lg animate-fade-in">
                <div className="flex items-center gap-2 mb-2">
                  <Cpu className="h-4 w-4 text-violet-600 animate-pulse" />
                  <span className="text-sm font-medium text-violet-800">
                    Generating Level 0 Embeddings
                  </span>
                </div>
                <p className="text-xs text-violet-700 mb-2">{embeddingProgress.message}</p>
                <div className="w-full h-2 bg-violet-200 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-violet-600 transition-all duration-500"
                    style={{ width: `${embeddingProgress.progress}%` }}
                  />
                </div>
                <div className="flex items-center justify-between mt-2">
                  <span className="text-2xs text-violet-600">{Math.round(embeddingProgress.progress)}%</span>
                  <div className="flex items-center gap-1 text-2xs text-violet-600">
                    <Clock className="h-3 w-3" />
                    <span>Est. 5-20 min for full resolution</span>
                  </div>
                </div>
              </div>
            )}

            {/* Show "Generate Embeddings" for Level 0 when embeddings don't exist */}
            {resolutionLevel === 0 && selectedSlideId && !selectedSlideHasLevel0 && !isGeneratingEmbeddings ? (
              <div className="space-y-2">
                <div className="p-3 bg-amber-50 border border-amber-200 rounded-lg">
                  <div className="flex items-start gap-2">
                    <AlertCircle className="h-4 w-4 text-amber-600 mt-0.5 shrink-0" />
                    <div>
                      <p className="text-sm font-medium text-amber-800">
                        Level 0 Embeddings Required
                      </p>
                      <p className="text-xs text-amber-700 mt-1">
                        Full-resolution analysis requires generating embeddings first.
                        This extracts ~5,000-30,000 patches and may take 5-20 minutes.
                      </p>
                    </div>
                  </div>
                </div>
                <Button
                  variant="secondary"
                  size="lg"
                  onClick={onGenerateEmbeddings}
                  disabled={!selectedSlideId || isAnalyzing}
                  className="w-full border-violet-300 text-violet-700 hover:bg-violet-50"
                >
                  <Cpu className="h-4 w-4 mr-2" />
                  Generate Level 0 Embeddings
                </Button>
                <p className="text-2xs text-gray-500 text-center">
                  Or select Level 1 for faster analysis with existing embeddings
                </p>
              </div>
            ) : (
              <Button
                variant="primary"
                size="lg"
                onClick={onAnalyze}
                disabled={!selectedSlideId || isAnalyzing || isGeneratingEmbeddings}
                isLoading={isAnalyzing}
                className="w-full"
                data-demo="analyze-button"
              >
                {isAnalyzing ? "Analyzing..." : isGeneratingEmbeddings ? "Generating Embeddings..." : "Run Analysis"}
              </Button>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}
