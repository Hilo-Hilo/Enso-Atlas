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
}

// Default models -- used as fallback when project context is not available
export const AVAILABLE_MODELS: ModelConfig[] = [
  {
    id: "platinum_sensitivity",
    displayName: "Platinum Sensitivity",
    description: "Response to platinum-based chemotherapy",
    auc: 0.907,
    category: "ovarian_cancer",
  },
  {
    id: "tumor_grade",
    displayName: "Tumor Grade",
    description: "High vs low grade classification",
    auc: 0.752,
    category: "general_pathology",
  },
  {
    id: "survival_5y",
    displayName: "5-Year Survival",
    description: "5-year overall survival probability",
    auc: 0.697,
    category: "ovarian_cancer",
  },
  {
    id: "survival_3y",
    displayName: "3-Year Survival",
    description: "3-year overall survival probability",
    auc: 0.645,
    category: "ovarian_cancer",
  },
  {
    id: "survival_1y",
    displayName: "1-Year Survival",
    description: "1-year overall survival probability",
    auc: 0.639,
    category: "ovarian_cancer",
  },
];

// getProjectModels is no longer used as a local helper -- model filtering
// now happens via the API and the apiModels state inside ModelPicker.

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
  const cancerTypeLabel = currentProject.cancer_type || "Ovarian Cancer";

  // Fetch full model configs from the project available-models API
  const [apiModelDetails, setApiModelDetails] = useState<AvailableModelDetail[]>([]);
  const [apiModels, setApiModels] = useState<string[]>([]);

  useEffect(() => {
    // Skip fetch until a real project is loaded (avoids 404 on "default")
    if (!currentProject.id || currentProject.id === "default") return;
    const fetchModels = async () => {
      try {
        // Try the new config-driven endpoint first
        const details = await getProjectAvailableModels(currentProject.id);
        if (details.length > 0) {
          setApiModelDetails(details);
          setApiModels(details.map((m) => m.id));
          return;
        }
      } catch {
        // ignore, fall through
      }
      try {
        // Fallback to the older model IDs endpoint
        const modelIds = await getProjectModelsApi(currentProject.id);
        setApiModels(modelIds);
      } catch {
        setApiModels(AVAILABLE_MODELS.map((m) => m.id));
      }
    };
    fetchModels();
  }, [currentProject.id]);

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
      } catch {
        setPreviouslyRanModels(new Set());
      }
    };
    fetchStatus();
  }, [selectedSlideId]);

  // Build ModelConfig list from API details, falling back to hardcoded AVAILABLE_MODELS
  const models = React.useMemo(() => {
    let result: ModelConfig[];
    if (apiModelDetails.length > 0) {
      result = apiModelDetails.map((d) => ({
        id: d.id,
        displayName: d.displayName,
        description: d.description,
        auc: d.auc,
        category: d.category,
      }));
    } else if (apiModels.length > 0) {
      result = AVAILABLE_MODELS.filter((m) => apiModels.includes(m.id));
    } else {
      result = AVAILABLE_MODELS;
    }
    // Reorder: primary target first
    const primary = result.find((m) => m.id === currentProject.prediction_target);
    if (!primary) return result;
    return [primary, ...result.filter((m) => m.id !== primary.id)];
  }, [apiModelDetails, apiModels, currentProject.prediction_target]);

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

  const selectOvarian = () => {
    onSelectionChange(
      models.filter((m) => m.category !== "general_pathology").map((m) => m.id)
    );
  };

  const selectGeneral = () => {
    onSelectionChange(
      models.filter((m) => m.category === "general_pathology").map((m) => m.id)
    );
  };

  const ovarianModels = models.filter((m) => m.category !== "general_pathology");
  const generalModels = models.filter((m) => m.category === "general_pathology");

  // Determine embedding readiness for each level
  const level1Ready = embeddingStatus?.hasLevel1 ?? false;
  const level0Ready = embeddingStatus?.hasLevel0 ?? false;

  return (
    <div className={cn("rounded-lg border border-gray-200 dark:border-slate-600 bg-white dark:bg-slate-700", className)}>
      {/* Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        disabled={disabled}
        className={cn(
          "w-full flex items-center justify-between px-3 py-2.5 text-left",
          "hover:bg-gray-50 dark:hover:bg-slate-600 transition-colors rounded-lg",
          disabled && "opacity-50 cursor-not-allowed"
        )}
      >
        <div className="flex items-center gap-2">
          <FlaskConical className="h-4 w-4 text-clinical-600" />
          <span className="text-sm font-medium text-gray-900 dark:text-gray-100">Model Selection</span>
          <Badge variant="default" size="sm">
            {selectedModels.length}/{models.length}
          </Badge>
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
                <span className="font-medium text-gray-700 dark:text-gray-300">Force Re-embed</span>
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
              onClick={selectOvarian}
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
            <div>
              <div className="flex items-center gap-1.5 mb-2">
                <Activity className="h-3 w-3 text-pink-500" />
                <span className="text-xs font-semibold text-gray-600 dark:text-gray-400 uppercase tracking-wide">
                  {cancerTypeLabel}
                </span>
              </div>
              <div className="space-y-1.5">
                {ovarianModels.map((model) => (
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

            {/* General Pathology Models */}
            <div>
              <div className="flex items-center gap-1.5 mb-2">
                <FlaskConical className="h-3 w-3 text-blue-500" />
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
        "hover:bg-gray-50 dark:hover:bg-slate-600 transition-colors",
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
