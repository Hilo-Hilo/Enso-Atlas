import type { UserViewMode } from "@/components/layout/Header";

export type RightSidebarPanelKey =
  | "pathologist-workspace"
  | "medgemma"
  | "evidence"
  | "prediction"
  | "multi-model"
  | "semantic-search"
  | "similar-cases"
  | "outlier-detector";

export function getRightPanelAfterViewModeChange(
  mode: UserViewMode,
  previousPanel: RightSidebarPanelKey
): RightSidebarPanelKey {
  if (mode === "pathologist") {
    return "pathologist-workspace";
  }

  return previousPanel === "pathologist-workspace" ? "medgemma" : previousPanel;
}

export function getRightSidebarPanelOptions(params: {
  userViewMode: UserViewMode;
  showMultiModelPanel: boolean;
  primaryPredictionPanelLabel: string;
}): Array<{ value: RightSidebarPanelKey; label: string }> {
  const { userViewMode, showMultiModelPanel, primaryPredictionPanelLabel } = params;

  if (userViewMode === "pathologist") {
    return [
      { value: "pathologist-workspace", label: "Pathologist Workspace" },
      { value: "semantic-search", label: "MedSigLIP Semantic Search" },
      { value: "evidence", label: "Evidence Patches" },
      { value: "outlier-detector", label: "Outlier Tissue Detector" },
    ];
  }

  return [
    { value: "medgemma", label: "MedGemma" },
    { value: "prediction", label: primaryPredictionPanelLabel },
    ...(showMultiModelPanel
      ? [{ value: "multi-model" as const, label: "Survival AI Predictions" }]
      : []),
    { value: "semantic-search", label: "Semantic Search" },
    { value: "similar-cases", label: "Similar Cases" },
  ];
}

export function ensureActiveRightPanel(
  activePanel: RightSidebarPanelKey,
  options: Array<{ value: RightSidebarPanelKey; label: string }>
): RightSidebarPanelKey {
  if (options.some((panel) => panel.value === activePanel)) {
    return activePanel;
  }

  return options[0]?.value ?? "medgemma";
}
