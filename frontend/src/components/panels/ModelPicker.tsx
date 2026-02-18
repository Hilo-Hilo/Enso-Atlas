"use client";

import React, { useState, useEffect } from "react";
import { cn } from "@/lib/utils";

import { Badge } from "@/components/ui/Badge";
import { ChevronDown, ChevronUp, FlaskConical, Activity, Layers, CheckCircle, Circle, History } from "lucide-react";
import { useProject } from "@/contexts/ProjectContext";
import { getProjectModels as getProjectModelsApi, getProjectAvailableModels, getSlideEmbeddingStatus } from "@/lib/api";
import type { AvailableModelDetail } from "@/lib/api";

export interface ModelConfig {
  id: string;
  displayName: string;
  description: string;
  auc: number;
  category: string;
  positiveLabel?: string;
  negativeLabel?: string;
}

function humanizeModelId(modelId: string): string {
  return modelId
    .replace(/[\-_]+/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function mapDetailToModelConfig(detail: AvailableModelDetail): ModelConfig {
  return {
    id: detail.id,
    displayName: detail.displayName,
    description: detail.description,
    auc: detail.auc,
    category: detail.category,
    positiveLabel: detail.positiveLabel,
    negativeLabel: detail.negativeLabel,
  };
}

function buildProjectFallbackModels(currentProject: {
  prediction_target?: string;
  cancer_type?: string;
  positive_class?: string;
  classes?: string[];
}): ModelConfig[] {
  const primaryId = currentProject.prediction_target;
  if (!primaryId) return [];

  const positiveLabel =
    currentProject.positive_class ||
    currentProject.classes?.[1] ||
    "Positive";
  const negativeLabel =
    currentProject.classes?.find((c) => c !== positiveLabel) ||
    currentProject.classes?.[0] ||
    "Negative";

  return [
    {
      id: primaryId,
      displayName: humanizeModelId(primaryId),
      description: `Primary ${currentProject.cancer_type || "project"} model`,
      auc: 0,
      category: "project_specific",
      positiveLabel,
      negativeLabel,
    },
  ];
}

function buildModelFromId(
  modelId: string,
  currentProject: {
    prediction_target?: string;
    cancer_type?: string;
    positive_class?: string;
    classes?: string[];
  }
): ModelConfig {
  const base = buildProjectFallbackModels(currentProject)[0];
  const isPrimary = modelId === currentProject.prediction_target;

  return {
    id: modelId,
    displayName: humanizeModelId(modelId),
    description: isPrimary
      ? base?.description || `Primary ${currentProject.cancer_type || "project"} model`
      : `${currentProject.cancer_type || "Project"} model`,
    auc: 0,
    category: "project_specific",
    positiveLabel: base?.positiveLabel || "Positive",
    negativeLabel: base?.negativeLabel || "Negative",
  };
}

function dedupeModels(models: ModelConfig[]): ModelConfig[] {
  const seen = new Set<string>();
  const unique: ModelConfig[] = [];
  for (const model of models) {
    if (seen.has(model.id)) continue;
    seen.add(model.id);
    unique.push(model);
  }
  return unique;
}

interface EmbeddingStatus {
  hasLevel0: boolean;
  hasLevel1: boolean;
}

interface ModelPickerProps {
  selectedModels: string[];
  onSelectionChange: (models: string[]) => void;
  resolutionLevel: number;
  onResolutionChange: (level: number) => void;
  forceReembed: boolean;
  onForceReembedChange: (force: boolean) => void;
  disabled?: boolean;
  className?: string;
  embeddingStatus?: EmbeddingStatus;
  /** Currently selected slide ID, used to fetch "Previously Ran" status */
  selectedSlideId?: string | null;
}

export function ModelPicker({
  selectedModels,
  onSelectionChange,
  resolutionLevel,
  onResolutionChange,
  forceReembed,
  onForceReembedChange,
  disabled = false,
  className,
  embeddingStatus,
  selectedSlideId,
}: ModelPickerProps) {
  const [isExpanded, setIsExpanded] = React.useState(false);
  const { currentProject } = useProject();
  const cancerTypeLabel = currentProject.cancer_type || "Cancer Specific";

  // Fetch full model configs from project-scoped APIs
  const [apiModelDetails, setApiModelDetails] = useState<AvailableModelDetail[]>([]);
  const [projectModelIds, setProjectModelIds] = useState<string[]>([]);
  const [usingFallbackModels, setUsingFallbackModels] = useState(false);

  useEffect(() => {
    let cancelled = false;

    // Clear old models first to prevent race condition with auto-select
    setApiModelDetails([]);
    setProjectModelIds([]);
    setUsingFallbackModels(false);

    const fetchModels = async () => {
      // For default project, use safe project-derived fallback only
      if (!currentProject.id || currentProject.id === "default") {
        if (!cancelled) {
          setUsingFallbackModels(true);
        }
        return;
      }

      try {
        // Try the config-driven endpoint first
        const details = await getProjectAvailableModels(currentProject.id);
        if (!cancelled && details.length > 0) {
          setApiModelDetails(details);
          return;
        }
      } catch (err) {
        console.warn("Failed to fetch available models config:", err);
      }

      try {
        // Fallback to older project model IDs endpoint (still project-scoped)
        const modelIds = await getProjectModelsApi(currentProject.id);
        if (!cancelled && modelIds.length > 0) {
          setProjectModelIds(modelIds);
          setUsingFallbackModels(true);
          return;
        }
      } catch (err) {
        console.warn("Failed to fetch project model IDs:", err);
      }

      if (!cancelled) {
        setUsingFallbackModels(true);
      }
    };

    fetchModels();

    return () => {
      cancelled = true;
    };
  }, [currentProject.id]);

  const fallbackModels = React.useMemo(
    () =>
      buildProjectFallbackModels({
        prediction_target: currentProject.prediction_target,
        cancer_type: currentProject.cancer_type,
        positive_class: currentProject.positive_class,
        classes: currentProject.classes,
      }),
    [
      currentProject.prediction_target,
      currentProject.cancer_type,
      currentProject.positive_class,
      currentProject.classes,
    ]
  );

  // Build ModelConfig list from API details or project-safe fallback
  const models = React.useMemo(() => {
    let result: ModelConfig[];

    if (apiModelDetails.length > 0) {
      result = apiModelDetails.map(mapDetailToModelConfig);
    } else if (projectModelIds.length > 0) {
      result = projectModelIds.map((id) => buildModelFromId(id, currentProject));
    } else {
      result = fallbackModels;
    }

    const unique = dedupeModels(result);

    // Reorder: primary target first
    const primary = unique.find((m) => m.id === currentProject.prediction_target);
    if (!primary) return unique;
    return [primary, ...unique.filter((m) => m.id !== primary.id)];
  }, [apiModelDetails, projectModelIds, fallbackModels, currentProject]);

  // Auto-select all models when models are loaded and selectedModels is empty
  useEffect(() => {
    if (models.length > 0 && selectedModels.length === 0) {
      onSelectionChange(models.map((m) => m.id));
    }
  }, [models, selectedModels.length, onSelectionChange]);

  // Track which models have been previously run on the selected slide
  const [previouslyRanModels, setPreviouslyRanModels] = useState<Set<string>>(new Set());

  useEffect(() => {
    if (!selectedSlideId) {
      setPreviouslyRanModels(new Set());
      return;
    }
    const fetchStatus = async () => {
      try {
        const status = await getSlideEmbeddingStatus(selectedSlideId);
        setPreviouslyRanModels(new Set(status.cached_model_ids));
      } catch (err) {
        console.warn("Failed to fetch slide embedding status:", err);
        setPreviouslyRanModels(new Set());
      }
    };
    fetchStatus();
  }, [selectedSlideId]);

  const toggleModel = (modelId: string) => {
    if (disabled) return;
    if (selectedModels.includes(modelId)) {
      onSelectionChange(selectedModels.filter((id) => id !== modelId));
    } else {
      onSelectionChange([...selectedModels, modelId]);
    }
  };

  const selectAll = () => {
    onSelectionChange(models.map((m) => m.id));
  };

  const selectNone = () => {
    onSelectionChange([]);
  };

  const selectCancerSpecific = () => {
    onSelectionChange(
      models.filter((m) => m.category !== "general_pathology").map((m) => m.id)
    );
  };

  const selectGeneral = () => {
    onSelectionChange(
      models.filter((m) => m.category === "general_pathology").map((m) => m.id)
    );
  };

  const cancerSpecificModels = models.filter((m) => m.category !== "general_pathology");
  const generalModels = models.filter((m) => m.category === "general_pathology");

  // Determine embedding readiness for each level
  const level1Ready = embeddingStatus?.hasLevel1 ?? false;
  const level0Ready = embeddingStatus?.hasLevel0 ?? false;

  return (
    <div className={cn("rounded-lg border border-gray-200 bg-white", className)}>
      {/* Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        disabled={disabled}
        className={cn(
          "w-full flex items-center justify-between px-3 py-2.5 text-left",
          "hover:bg-gray-50 transition-colors rounded-lg",
          disabled && "opacity-50 cursor-not-allowed"
        )}
      >
        <div className="flex items-center gap-2">
          <FlaskConical className="h-4 w-4 text-clinical-600" />
          <span className="text-sm font-medium text-gray-900">Model Selection</span>
          <Badge variant="default" size="sm">
            {selectedModels.length}/{models.length}
          </Badge>
          {usingFallbackModels && (
            <Badge variant="default" size="sm" className="text-amber-600 bg-amber-50 border-amber-200">
              fallback
            </Badge>
          )}
        </div>
        {isExpanded ? (
          <ChevronUp className="h-4 w-4 text-gray-400" />
        ) : (
          <ChevronDown className="h-4 w-4 text-gray-400" />
        )}
      </button>

      {/* Expanded Content */}
      {isExpanded && (
        <div className="px-3 pb-3 space-y-3 border-t border-gray-100 pt-3">
          {/* Resolution Level Selector with Embedding Status */}
          <div className="pb-3 border-b border-gray-100">
            <div className="flex items-center gap-1.5 mb-2">
              <Layers className="h-3 w-3 text-purple-500" />
              <span className="text-xs font-semibold text-gray-600 uppercase tracking-wide">
                Resolution Level
              </span>
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => onResolutionChange(1)}
                disabled={disabled}
                className={cn(
                  "flex-1 px-3 py-2 rounded-md text-sm font-medium transition-colors",
                  resolutionLevel === 1
                    ? "bg-purple-100 text-purple-700 border-2 border-purple-300"
                    : "bg-gray-100 text-gray-600 hover:bg-gray-200 border-2 border-transparent"
                )}
              >
                <div className="text-center">
                  <div className="flex items-center justify-center gap-1.5">
                    <span>Level 1</span>
                    {embeddingStatus && (
                      level1Ready ? (
                        <CheckCircle className="h-3.5 w-3.5 text-green-600" />
                      ) : (
                        <Circle className="h-3.5 w-3.5 text-gray-400" />
                      )
                    )}
                  </div>
                  <div className="text-xs opacity-70">Fast (~100-500 patches)</div>
                  {embeddingStatus && (
                    <div className={cn("text-2xs mt-0.5", level1Ready ? "text-green-600" : "text-gray-400")}>
                      {level1Ready ? "Ready" : "Not generated"}
                    </div>
                  )}
                </div>
              </button>
              <button
                onClick={() => onResolutionChange(0)}
                disabled={disabled}
                className={cn(
                  "flex-1 px-3 py-2 rounded-md text-sm font-medium transition-colors",
                  resolutionLevel === 0
                    ? "bg-purple-100 text-purple-700 border-2 border-purple-300"
                    : "bg-gray-100 text-gray-600 hover:bg-gray-200 border-2 border-transparent"
                )}
              >
                <div className="text-center">
                  <div className="flex items-center justify-center gap-1.5">
                    <span>Level 0</span>
                    {embeddingStatus && (
                      level0Ready ? (
                        <CheckCircle className="h-3.5 w-3.5 text-green-600" />
                      ) : (
                        <Circle className="h-3.5 w-3.5 text-gray-400" />
                      )
                    )}
                  </div>
                  <div className="text-xs opacity-70">Full res (~5K-30K patches)</div>
                  {embeddingStatus && (
                    <div className={cn("text-2xs mt-0.5", level0Ready ? "text-green-600" : "text-gray-400")}>
                      {level0Ready ? "Ready" : "Not generated"}
                    </div>
                  )}
                </div>
              </button>
            </div>
            {/* Contextual note for missing embeddings */}
            {embeddingStatus && resolutionLevel === 0 && !level0Ready && (
              <p className="mt-2 text-xs text-amber-600">
                Embeddings will be generated on first analysis (~5-20 min)
              </p>
            )}
            {embeddingStatus && resolutionLevel === 1 && !level1Ready && (
              <p className="mt-2 text-xs text-amber-600">
                Embeddings will be generated on first analysis (~2-5 min)
              </p>
            )}
            {resolutionLevel === 0 && level0Ready && (
              <p className="mt-2 text-xs text-green-600">
                Level 0 embeddings ready. Full-resolution analysis available.
              </p>
            )}
            <label
              className={cn(
                "mt-3 flex items-start gap-2 text-xs text-gray-600",
                disabled && "opacity-50 cursor-not-allowed"
              )}
            >
              <input
                type="checkbox"
                checked={forceReembed}
                onChange={(event) => onForceReembedChange(event.target.checked)}
                disabled={disabled}
                className="mt-0.5 h-4 w-4 rounded border-gray-300 text-clinical-600 focus:ring-clinical-500"
              />
              <span className="flex-1">
                <span className="font-medium text-gray-700">Force Re-embed</span>
                <span className="block text-2xs text-gray-500">
                  Regenerate embeddings even if cached.
                </span>
              </span>
            </label>
          </div>

          {/* Quick Actions */}
          <div className="flex flex-wrap gap-1.5">
            <button
              onClick={selectAll}
              disabled={disabled}
              className="text-xs px-2 py-1 rounded bg-gray-100 hover:bg-gray-200 text-gray-700 transition-colors"
            >
              All
            </button>
            <button
              onClick={selectNone}
              disabled={disabled}
              className="text-xs px-2 py-1 rounded bg-gray-100 hover:bg-gray-200 text-gray-700 transition-colors"
            >
              None
            </button>
            <button
              onClick={selectCancerSpecific}
              disabled={disabled}
              className="text-xs px-2 py-1 rounded bg-pink-100 hover:bg-pink-200 text-pink-700 transition-colors"
            >
              {cancerTypeLabel}
            </button>
            <button
              onClick={selectGeneral}
              disabled={disabled}
              className="text-xs px-2 py-1 rounded bg-blue-100 hover:bg-blue-200 text-blue-700 transition-colors"
            >
              General
            </button>
          </div>

          <div className="space-y-3">
            {/* Cancer-Specific Models */}
            {cancerSpecificModels.length > 0 && (
              <div>
                <div className="flex items-center gap-1.5 mb-2">
                  <Activity className="h-3 w-3 text-pink-500" />
                  <span className="text-xs font-semibold text-gray-600 uppercase tracking-wide">
                    {cancerTypeLabel}
                  </span>
                </div>
                <div className="space-y-1.5">
                  {cancerSpecificModels.map((model) => (
                    <ModelCheckbox
                      key={model.id}
                      model={model}
                      checked={selectedModels.includes(model.id)}
                      onChange={() => toggleModel(model.id)}
                      disabled={disabled}
                      isPrimary={model.id === currentProject.prediction_target}
                      previouslyRan={previouslyRanModels.has(model.id)}
                    />
                  ))}
                </div>
              </div>
            )}

            {/* General Pathology Models */}
            {generalModels.length > 0 && (
              <div>
                <div className="flex items-center gap-1.5 mb-2">
                  <FlaskConical className="h-3 w-3 text-blue-500" />
                  <span className="text-xs font-semibold text-gray-600 uppercase tracking-wide">
                    General Pathology
                  </span>
                </div>
                <div className="space-y-1.5">
                  {generalModels.map((model) => (
                    <ModelCheckbox
                      key={model.id}
                      model={model}
                      checked={selectedModels.includes(model.id)}
                      onChange={() => toggleModel(model.id)}
                      disabled={disabled}
                      previouslyRan={previouslyRanModels.has(model.id)}
                    />
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function ModelCheckbox({
  model,
  checked,
  onChange,
  disabled,
  isPrimary,
  previouslyRan,
}: {
  model: ModelConfig;
  checked: boolean;
  onChange: () => void;
  disabled?: boolean;
  isPrimary?: boolean;
  previouslyRan?: boolean;
}) {
  return (
    <label
      className={cn(
        "flex items-center gap-2.5 p-2 rounded-md cursor-pointer",
        "hover:bg-gray-50 transition-colors",
        checked && "bg-clinical-50",
        disabled && "opacity-50 cursor-not-allowed"
      )}
    >
      <input
        type="checkbox"
        checked={checked}
        onChange={onChange}
        disabled={disabled}
        className="h-4 w-4 rounded border-gray-300 text-clinical-600 focus:ring-clinical-500"
      />
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between gap-1">
          <div className="flex items-center gap-1.5 min-w-0">
            <span className="text-sm font-medium text-gray-900 truncate">{model.displayName}</span>
            {isPrimary && (
              <Badge variant="info" size="sm">Primary</Badge>
            )}
            {previouslyRan && (
              <Badge variant="default" size="sm" className="bg-green-100 text-green-700 border-green-200">
                <History className="h-3 w-3 mr-0.5 inline" />
                Cached
              </Badge>
            )}
          </div>
          <span className="text-xs text-gray-400 font-mono shrink-0">
            {model.auc.toFixed(2)}
          </span>
        </div>
        <p className="text-xs text-gray-500 truncate">{model.description}</p>
      </div>
    </label>
  );
}
