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
  const [isLoadingModelOptions, setIsLoadingModelOptions] = useState(true);
  const hasRecoveredPrimaryOnlySelection = React.useRef(false);

  useEffect(() => {
    hasRecoveredPrimaryOnlySelection.current = false;
  }, [currentProject.id]);

  useEffect(() => {
    let cancelled = false;

    // Clear old models first to prevent race condition with auto-select
    setIsLoadingModelOptions(true);
    setApiModelDetails([]);
    setProjectModelIds([]);
    setUsingFallbackModels(false);

    const fetchModels = async () => {
      // For default project, use safe project-derived fallback only
      if (!currentProject.id || currentProject.id === "default") {
        if (!cancelled) {
          setUsingFallbackModels(true);
          setIsLoadingModelOptions(false);
        }
        return;
      }

      try {
        // Try the config-driven endpoint first
        const details = await getProjectAvailableModels(currentProject.id);
        if (!cancelled && details.length > 0) {
          const primaryId = currentProject.prediction_target;
          const includesPrimary = !primaryId || details.some((d) => d.id === primaryId);

          if (includesPrimary) {
            setApiModelDetails(details);
            setIsLoadingModelOptions(false);
            return;
          }

          console.warn(
            "Project model details missing project primary target; falling back to project model IDs"
          );
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
          setIsLoadingModelOptions(false);
          return;
        }
      } catch (err) {
        console.warn("Failed to fetch project model IDs:", err);
      }

      if (!cancelled) {
        setUsingFallbackModels(true);
        setIsLoadingModelOptions(false);
      }
    };

    fetchModels();

    return () => {
      cancelled = true;
    };
  }, [currentProject.id, currentProject.prediction_target]);

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

  // Prune stale/duplicate selected model IDs whenever project-scoped model options refresh.
  // This prevents carrying invalid IDs across project switches (for example 4 selected out of 3 available).
  useEffect(() => {
    if (models.length === 0 || selectedModels.length === 0) return;

    const availableModelIds = new Set(models.map((m) => m.id));
    const seen = new Set<string>();
    const prunedSelection = selectedModels.filter((id) => {
      if (!availableModelIds.has(id)) return false;
      if (seen.has(id)) return false;
      seen.add(id);
      return true;
    });

    if (prunedSelection.length !== selectedModels.length) {
      onSelectionChange(prunedSelection);
    }
  }, [models, selectedModels, onSelectionChange]);

  // Auto-select all models only after project model loading is complete.
  // This avoids locking the selection to the temporary single-model fallback.
  useEffect(() => {
    if (!isLoadingModelOptions && models.length > 0 && selectedModels.length === 0) {
      onSelectionChange(models.map((m) => m.id));
    }
  }, [isLoadingModelOptions, models, selectedModels.length, onSelectionChange]);

  // One-time recovery: when fallback auto-selected only the primary model before
  // full project models arrived, expand selection to all project models.
  useEffect(() => {
    const primaryId = currentProject.prediction_target;
    if (!primaryId) return;
    if (isLoadingModelOptions) return;
    if (hasRecoveredPrimaryOnlySelection.current) return;
    if (models.length <= 1) return;
    if (selectedModels.length !== 1 || selectedModels[0] !== primaryId) return;

    onSelectionChange(models.map((m) => m.id));
    hasRecoveredPrimaryOnlySelection.current = true;
  }, [
    currentProject.prediction_target,
    isLoadingModelOptions,
    models,
    onSelectionChange,
    selectedModels,
  ]);

  // Track which models have been previously run on the selected slide
  const [previouslyRanModels, setPreviouslyRanModels] = useState<Set<string>>(new Set());
  const embeddingStatusRequestRef = React.useRef(0);

  useEffect(() => {
    const requestId = ++embeddingStatusRequestRef.current;

    if (!selectedSlideId) {
      setPreviouslyRanModels(new Set());
      return;
    }

    const fetchStatus = async () => {
      try {
        const status = await getSlideEmbeddingStatus(selectedSlideId, currentProject.id);
        if (embeddingStatusRequestRef.current !== requestId) return;
        setPreviouslyRanModels(new Set(status.cached_model_ids));
      } catch (err) {
        if (embeddingStatusRequestRef.current !== requestId) return;
        console.warn("Failed to fetch slide embedding status:", err);
        setPreviouslyRanModels(new Set());
      }
    };

    fetchStatus();
  }, [selectedSlideId, currentProject.id]);

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
    <div className={cn("rounded-lg border border-gray-200 dark:border-navy-600 bg-white dark:bg-navy-800", className)}>
      {/* Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        disabled={disabled}
        className={cn(
          "w-full flex items-center justify-between px-3 py-2.5 text-left",
          "hover:bg-gray-50 dark:hover:bg-navy-700 transition-colors rounded-lg",
          disabled && "opacity-50 cursor-not-allowed"
        )}
      >
        <div className="flex items-center gap-2">
          <FlaskConical className="h-4 w-4 text-clinical-600 dark:text-clinical-400" />
          <span className="text-sm font-medium text-gray-900 dark:text-gray-100">Model Selection</span>
          <Badge variant="default" size="sm">
            {selectedModels.length}/{models.length}
          </Badge>
          {usingFallbackModels && (
            <Badge variant="default" size="sm" className="text-amber-600 dark:text-amber-400 bg-amber-50 dark:bg-amber-900/30 border-amber-200 dark:border-amber-700">
              fallback
            </Badge>
          )}
        </div>
        {isExpanded ? (
          <ChevronUp className="h-4 w-4 text-gray-400 dark:text-gray-500" />
        ) : (
          <ChevronDown className="h-4 w-4 text-gray-400 dark:text-gray-500" />
        )}
      </button>

      {/* Expanded Content */}
      {isExpanded && (
        <div className="px-3 pb-3 space-y-3 border-t border-gray-100 dark:border-navy-700 pt-3">
          {/* Resolution Level Selector with Embedding Status */}
          <div className="pb-3 border-b border-gray-100 dark:border-navy-700">
            <div className="flex items-center gap-1.5 mb-2">
              <Layers className="h-3 w-3 text-purple-500 dark:text-purple-400" />
              <span className="text-xs font-semibold text-gray-600 dark:text-gray-400 uppercase tracking-wide">
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
                    ? "bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 border-2 border-purple-300 dark:border-purple-600"
                    : "bg-gray-100 dark:bg-navy-700 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-navy-600 border-2 border-transparent"
                )}
              >
                <div className="text-center">
                  <div className="flex items-center justify-center gap-1.5">
                    <span>Level 1</span>
                    {embeddingStatus && (
                      level1Ready ? (
                        <CheckCircle className="h-3.5 w-3.5 text-green-600 dark:text-green-400" />
                      ) : (
                        <Circle className="h-3.5 w-3.5 text-gray-400 dark:text-gray-500" />
                      )
                    )}
                  </div>
                  <div className="text-xs opacity-70">Fast (~100-500 patches)</div>
                  {embeddingStatus && (
                    <div className={cn("text-2xs mt-0.5", level1Ready ? "text-green-600 dark:text-green-400" : "text-gray-400 dark:text-gray-500")}>
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
                    ? "bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 border-2 border-purple-300 dark:border-purple-600"
                    : "bg-gray-100 dark:bg-navy-700 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-navy-600 border-2 border-transparent"
                )}
              >
                <div className="text-center">
                  <div className="flex items-center justify-center gap-1.5">
                    <span>Level 0</span>
                    {embeddingStatus && (
                      level0Ready ? (
                        <CheckCircle className="h-3.5 w-3.5 text-green-600 dark:text-green-400" />
                      ) : (
                        <Circle className="h-3.5 w-3.5 text-gray-400 dark:text-gray-500" />
                      )
                    )}
                  </div>
                  <div className="text-xs opacity-70">Full res (~5K-30K patches)</div>
                  {embeddingStatus && (
                    <div className={cn("text-2xs mt-0.5", level0Ready ? "text-green-600 dark:text-green-400" : "text-gray-400 dark:text-gray-500")}>
                      {level0Ready ? "Ready" : "Not generated"}
                    </div>
                  )}
                </div>
              </button>
            </div>
            {/* Contextual note for missing embeddings */}
            {embeddingStatus && resolutionLevel === 0 && !level0Ready && (
              <p className="mt-2 text-xs text-amber-600 dark:text-amber-400">
                Embeddings will be generated on first analysis (~5-20 min)
              </p>
            )}
            {embeddingStatus && resolutionLevel === 1 && !level1Ready && (
              <p className="mt-2 text-xs text-amber-600 dark:text-amber-400">
                Embeddings will be generated on first analysis (~2-5 min)
              </p>
            )}
            {resolutionLevel === 0 && level0Ready && (
              <p className="mt-2 text-xs text-green-600 dark:text-green-400">
                Level 0 embeddings ready. Full-resolution analysis available.
              </p>
            )}
            <label
              className={cn(
                "mt-3 flex items-start gap-2 text-xs text-gray-600 dark:text-gray-400",
                disabled && "opacity-50 cursor-not-allowed"
              )}
            >
              <input
                type="checkbox"
                checked={forceReembed}
                onChange={(event) => onForceReembedChange(event.target.checked)}
                disabled={disabled}
                className="mt-0.5 h-4 w-4 rounded border-gray-300 dark:border-navy-500 text-clinical-600 focus:ring-clinical-500 dark:bg-navy-700"
              />
              <span className="flex-1">
                <span className="font-medium text-gray-700 dark:text-gray-300">Force Re-embed</span>
                <span className="block text-2xs text-gray-500 dark:text-gray-400">
                  Regenerate embeddings even if cached.
                </span>
              </span>
            </label>
          </div>

          {/* Quick Actions */}
          <div className="flex flex-wrap gap-1.5">
            <button
              onClick={selectAll}
              disabled={disabled || models.length === 0}
              className="text-xs px-2 py-1 rounded bg-gray-100 dark:bg-navy-700 hover:bg-gray-200 dark:hover:bg-navy-600 text-gray-700 dark:text-gray-300 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              All
            </button>
            <button
              onClick={selectNone}
              disabled={disabled || selectedModels.length === 0}
              className="text-xs px-2 py-1 rounded bg-gray-100 dark:bg-navy-700 hover:bg-gray-200 dark:hover:bg-navy-600 text-gray-700 dark:text-gray-300 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              None
            </button>
            <button
              onClick={selectCancerSpecific}
              disabled={disabled || cancerSpecificModels.length === 0}
              className="text-xs px-2 py-1 rounded bg-pink-100 dark:bg-pink-900/30 hover:bg-pink-200 dark:hover:bg-pink-900/50 text-pink-700 dark:text-pink-300 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {cancerTypeLabel}
            </button>
            <button
              onClick={selectGeneral}
              disabled={disabled || generalModels.length === 0}
              className="text-xs px-2 py-1 rounded bg-blue-100 dark:bg-blue-900/30 hover:bg-blue-200 dark:hover:bg-blue-900/50 text-blue-700 dark:text-blue-300 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              General
            </button>
          </div>

          <div className="space-y-3">
            {/* Cancer-Specific Models */}
            {cancerSpecificModels.length > 0 && (
              <div>
                <div className="flex items-center gap-1.5 mb-2">
                  <Activity className="h-3 w-3 text-pink-500 dark:text-pink-400" />
                  <span className="text-xs font-semibold text-gray-600 dark:text-gray-400 uppercase tracking-wide">
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
                  <FlaskConical className="h-3 w-3 text-blue-500 dark:text-blue-400" />
                  <span className="text-xs font-semibold text-gray-600 dark:text-gray-400 uppercase tracking-wide">
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
        "hover:bg-gray-50 dark:hover:bg-navy-700/50 transition-colors",
        checked && "bg-clinical-50 dark:bg-clinical-900/30",
        disabled && "opacity-50 cursor-not-allowed"
      )}
    >
      <input
        type="checkbox"
        checked={checked}
        onChange={onChange}
        disabled={disabled}
        className="h-4 w-4 rounded border-gray-300 dark:border-navy-500 text-clinical-600 focus:ring-clinical-500 dark:bg-navy-700"
      />
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between gap-1">
          <div className="flex items-center gap-1.5 min-w-0">
            <span className="text-sm font-medium text-gray-900 dark:text-gray-100 truncate">{model.displayName}</span>
            {isPrimary && (
              <Badge variant="info" size="sm">Primary</Badge>
            )}
            {previouslyRan && (
              <Badge variant="default" size="sm" className="bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300 border-green-200 dark:border-green-700">
                <History className="h-3 w-3 mr-0.5 inline" />
                Cached
              </Badge>
            )}
          </div>
          <span className="text-xs text-gray-400 dark:text-gray-500 font-mono shrink-0">
            {model.auc.toFixed(2)}
          </span>
        </div>
        <p className="text-xs text-gray-500 dark:text-gray-400 truncate">{model.description}</p>
        {model.positiveLabel && model.negativeLabel && (
          <p className="text-2xs text-gray-500 dark:text-gray-400 truncate">
            Labels: {model.positiveLabel} vs {model.negativeLabel}
          </p>
        )}
      </div>
    </label>
  );
}
