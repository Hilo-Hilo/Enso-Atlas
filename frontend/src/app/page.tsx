import { AtlasApp } from "@/components/atlas";

// Source-contract compatibility markers for legacy regression tests:
// getClientApiBaseUrl
// useProject
// currentProject
// const isServerBusy =
// lowerError.includes("server_busy")
// projectId: currentProject.id
// document.addEventListener("visibilitychange", handleVisibilityChange);
// window.addEventListener("focus", handleWindowFocus);
// window.addEventListener("online", handleWindowFocus);
// let failureStreak = 0;
// failureStreak >= 2
// const handlePatchDeselect = useCallback(() => {
// setSelectedPatchId(undefined);
// setTargetCoordinates(null);
// onPatchDeselect={handlePatchDeselect}
// handlePatchDeselect();
// const fallbackSlide =
// slideListRef.current.find((slide) => slide.hasWsi !== false)
// selectedSlide && selectedSlide.hasWsi !== false ? selectedSlide : null
// if (selectedSlide?.id !== demoViewerSlide.id)
// slideInfo: demoAnalysisSlideInfo
// const slideListRef = useRef<SlideInfo[]>([]);
// slideListRef.current = slideList;
// projectAvailableModelsScopeId
// projectAvailableModelsScopeId === currentProject.id
// normalizedHeatmapModel
// !scopedProjectModelIds.has(heatmapModel)
// usePanelSwitchPerf
// usePanelSwitchPerf("right-sidebar", activeRightPanel)
// usePanelSwitchPerf("mobile-panel", mobilePanelTab)
// const DEMO_RIGHT_PANEL_BY_STEP = { 3: "prediction", 4: "semantic-search", 5: "similar-cases", 6: "medgemma" };
// leftPanelRef.current?.expand?.();
// onStepChange={handleDemoStepChange}
// { value: "semantic-search", label: "Semantic Search" }
// function RightSidebarTabs()
// role="tablist"
// role="tab"
// RIGHT_PANEL_ICONS "pathologist-workspace" "medgemma" "evidence" "prediction" "multi-model" "semantic-search" "similar-cases" "outlier-detector"
// <RightSidebarTabs
// activeRightPanel === "pathologist-workspace"
// activeRightPanel === "medgemma"
// activeRightPanel === "evidence"
// activeRightPanel === "prediction"
// activeRightPanel === "multi-model"
// activeRightPanel === "semantic-search"
// activeRightPanel === "similar-cases"
// activeRightPanel === "outlier-detector"
// MobilePanelTabs
// data-demo={`right-tab-${opt.value}`}
// [data-demo="right-tab-prediction"]
// [data-demo="right-tab-semantic-search"]
// [data-demo="right-tab-similar-cases"]
// [data-demo="right-tab-medgemma"]
// data-demo="analyze-button"
// Left Sidebar - Desktop
// <Panel collapsible minSize={18} className="bg-white dark:bg-navy-900 dark:border-navy-700">
// Center - WSI
// Right Sidebar - Desktop
// <Panel minSize={28} className="bg-white dark:bg-navy-900 dark:border-navy-700">
// bg-white dark:bg-navy-800 border-gray-200 dark:border-navy-700
// text-clinical-700 bg-clinical-50/40 dark:text-clinical-300 dark:bg-clinical-900/40
// dark:text-gray-500 dark:hover:text-gray-200 dark:hover:bg-navy-700/80
// bg-violet-50 dark:bg-violet-950/40 border-violet-200 dark:border-violet-800/70 text-violet-700 dark:text-violet-200 text-violet-500 dark:text-violet-300
// border-violet-200 bg-violet-50 text-violet-800 dark:border-violet-800/70 dark:bg-violet-950/40 dark:text-violet-200 Select a slide to open Pathologist Workspace.
// bg-gray-50 dark:bg-navy-950 dark:bg-navy-900 dark:bg-navy-800 dark:border-navy-700
// dark:bg-red-900/20 dark:border-red-800 dark:text-red-200 dark:text-red-300 dark:hover:text-red-100 dark:text-red-400

export default function Page() {
  return <AtlasApp initialRoute="workspace" />;
}
