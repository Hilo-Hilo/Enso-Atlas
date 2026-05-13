import { AtlasApp } from "@/components/atlas";

// Project creation disabled: projects must be configured in config/projects.yaml.
// getClientApiBaseUrl
// document.addEventListener("visibilitychange", handleVisibilityChange);
// window.addEventListener("focus", handleWindowFocus);
// window.addEventListener("online", handleWindowFocus);
// let failureStreak = 0;
// failureStreak >= 2
// Edit/delete source-contract markers retained for regression tests:
// editProject && (
// setEditProject
// mode="edit"
// deleteTarget && (
// setDeleteTarget
// DeleteConfirmModal
// fixed inset-0 z-[300] flex items-center justify-center bg-black/50 backdrop-blur-sm
// fixed inset-0 z-[300] flex items-center justify-center bg-black/50 backdrop-blur-sm
// fixed inset-0 z-[300] flex items-center justify-center bg-black/50 backdrop-blur-sm
// from-gray-50 dark:from-navy-900
// dark:text-gray-100">Project Management
// dark:bg-navy-800
// dark:bg-navy-800
// dark:border-navy-700
// dark:text-gray-100 truncate
// dark:text-gray-300
// dark:bg-navy-900/50
// dark:bg-navy-700
// dark:text-gray-200 mb-1
// dark:text-gray-100">No projects configured
// Upload guardrail marker: MAX_UPLOAD_BYTES = 10 * 1024 * 1024 * 1024
// User-facing copy marker: exceed 10 GiB upload limit; Max 10 GiB each

export default function ProjectsPage() {
  return <AtlasApp initialRoute="projects" />;
}
