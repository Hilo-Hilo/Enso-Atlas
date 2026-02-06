"use client";

import React, { useEffect, useCallback, useState, useRef } from "react";
import { Button } from "@/components/ui/Button";
import { Toggle } from "@/components/ui/Toggle";
import {
  X,
  Settings,
  Moon,
  Sun,
  Monitor,
  Server,
  Eye,
  RefreshCw,
  Cpu,
  Play,
  Square,
  CheckCircle,
  AlertTriangle,
} from "lucide-react";
import { cn } from "@/lib/utils";
import {
  startBatchEmbed,
  getBatchEmbedStatus,
  cancelBatchEmbed,
  getActiveBatchEmbed,
  type BatchEmbedProgress,
} from "@/lib/api";

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

type ThemeMode = "light" | "dark" | "system";

interface DisplaySettings {
  showHeatmapByDefault: boolean;
  showConfidenceScores: boolean;
  compactMode: boolean;
  animationsEnabled: boolean;
}

interface ApiSettings {
  endpoint: string;
  timeout: number;
}

const DEFAULT_DISPLAY_SETTINGS: DisplaySettings = {
  showHeatmapByDefault: true,
  showConfidenceScores: true,
  compactMode: false,
  animationsEnabled: true,
};

const DEFAULT_API_SETTINGS: ApiSettings = {
  endpoint: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000",
  timeout: 30000,
};

export function SettingsModal({ isOpen, onClose }: SettingsModalProps) {
  const [activeTab, setActiveTab] = useState<"appearance" | "display" | "api" | "embedding">("appearance");
  const [theme, setTheme] = useState<ThemeMode>("system");
  const [displaySettings, setDisplaySettings] = useState<DisplaySettings>(DEFAULT_DISPLAY_SETTINGS);
  const [apiSettings, setApiSettings] = useState<ApiSettings>(DEFAULT_API_SETTINGS);
  const [isDirty, setIsDirty] = useState(false);

  // Batch re-embed state
  const [batchEmbedStatus, setBatchEmbedStatus] = useState<BatchEmbedProgress | null>(null);
  const [batchEmbedTaskId, setBatchEmbedTaskId] = useState<string | null>(null);
  const [batchEmbedError, setBatchEmbedError] = useState<string | null>(null);
  const [isStartingBatchEmbed, setIsStartingBatchEmbed] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Load settings from localStorage on mount
  useEffect(() => {
    if (typeof window === "undefined") return;
    
    const savedTheme = localStorage.getItem("atlas-theme") as ThemeMode | null;
    if (savedTheme) setTheme(savedTheme);

    const savedDisplay = localStorage.getItem("atlas-display-settings");
    if (savedDisplay) {
      try {
        setDisplaySettings(JSON.parse(savedDisplay));
      } catch {
        // Ignore parse errors
      }
    }

    const savedApi = localStorage.getItem("atlas-api-settings");
    if (savedApi) {
      try {
        setApiSettings(JSON.parse(savedApi));
      } catch {
        // Ignore parse errors
      }
    }
  }, []);

  // Handle escape key
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === "Escape" && isOpen) {
        onClose();
      }
    },
    [isOpen, onClose]
  );

  useEffect(() => {
    if (typeof window === "undefined") return;
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [handleKeyDown]);

  // Prevent scroll when modal is open
  useEffect(() => {
    if (typeof window === "undefined") return;
    if (isOpen) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "";
    }
    return () => {
      document.body.style.overflow = "";
    };
  }, [isOpen]);

  const handleThemeChange = (newTheme: ThemeMode) => {
    setTheme(newTheme);
    setIsDirty(true);
    if (typeof window !== "undefined") {
      localStorage.setItem("atlas-theme", newTheme);
      // Apply theme to document
      const root = document.documentElement;
      if (newTheme === "dark") {
        root.classList.add("dark");
      } else if (newTheme === "light") {
        root.classList.remove("dark");
      } else {
        // System preference
        const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
        if (prefersDark) {
          root.classList.add("dark");
        } else {
          root.classList.remove("dark");
        }
      }
    }
  };

  const handleDisplayChange = (key: keyof DisplaySettings, value: boolean) => {
    const newSettings = { ...displaySettings, [key]: value };
    setDisplaySettings(newSettings);
    setIsDirty(true);
    if (typeof window !== "undefined") {
      localStorage.setItem("atlas-display-settings", JSON.stringify(newSettings));
    }
  };

  const handleApiChange = (key: keyof ApiSettings, value: string | number) => {
    const newSettings = { ...apiSettings, [key]: value };
    setApiSettings(newSettings);
    setIsDirty(true);
    if (typeof window !== "undefined") {
      localStorage.setItem("atlas-api-settings", JSON.stringify(newSettings));
    }
  };

  // Check for active batch embed on open
  useEffect(() => {
    if (!isOpen) return;
    getActiveBatchEmbed().then((result) => {
      if ("task_id" in result && (result as BatchEmbedProgress).task_id) {
        const progress = result as BatchEmbedProgress;
        setBatchEmbedTaskId(progress.task_id);
        setBatchEmbedStatus(progress);
      }
    }).catch(() => {});
  }, [isOpen]);

  // Poll batch embed progress
  useEffect(() => {
    if (!batchEmbedTaskId) return;
    if (pollRef.current) clearInterval(pollRef.current);

    const poll = async () => {
      try {
        const status = await getBatchEmbedStatus(batchEmbedTaskId);
        setBatchEmbedStatus(status);
        if (status.status === "completed" || status.status === "failed" || status.status === "cancelled") {
          if (pollRef.current) clearInterval(pollRef.current);
          pollRef.current = null;
        }
      } catch {
        // ignore polling errors
      }
    };

    poll();
    pollRef.current = setInterval(poll, 3000);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [batchEmbedTaskId]);

  const handleStartBatchEmbed = async () => {
    setIsStartingBatchEmbed(true);
    setBatchEmbedError(null);
    try {
      const result = await startBatchEmbed({ level: 0, force: true, concurrency: 1 });
      setBatchEmbedTaskId(result.batch_task_id);
    } catch (err) {
      setBatchEmbedError(err instanceof Error ? err.message : "Failed to start batch embedding");
    } finally {
      setIsStartingBatchEmbed(false);
    }
  };

  const handleCancelBatchEmbed = async () => {
    if (!batchEmbedTaskId) return;
    try {
      await cancelBatchEmbed(batchEmbedTaskId);
    } catch {
      // ignore
    }
  };

  const handleReset = () => {
    setTheme("system");
    setDisplaySettings(DEFAULT_DISPLAY_SETTINGS);
    setApiSettings(DEFAULT_API_SETTINGS);
    if (typeof window !== "undefined") {
      localStorage.removeItem("atlas-theme");
      localStorage.removeItem("atlas-display-settings");
      localStorage.removeItem("atlas-api-settings");
      document.documentElement.classList.remove("dark");
    }
    setIsDirty(false);
  };

  if (!isOpen) return null;

  const tabs = [
    { id: "appearance" as const, label: "Appearance", icon: Sun },
    { id: "display" as const, label: "Display", icon: Eye },
    { id: "api" as const, label: "API", icon: Server },
    { id: "embedding" as const, label: "Embedding", icon: Cpu },
  ];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm animate-fade-in"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative z-10 w-full max-w-lg mx-4 bg-white rounded-xl shadow-2xl overflow-hidden animate-scale-in">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 bg-gray-50">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-clinical-600 flex items-center justify-center">
              <Settings className="h-5 w-5 text-white" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-gray-900">Settings</h2>
              <p className="text-sm text-gray-500">Configure your preferences</p>
            </div>
          </div>
          <Button variant="ghost" size="sm" onClick={onClose} className="p-2">
            <X className="h-5 w-5" />
          </Button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-gray-200">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={cn(
                "flex-1 flex items-center justify-center gap-2 px-4 py-3 text-sm font-medium transition-colors",
                activeTab === tab.id
                  ? "text-clinical-600 border-b-2 border-clinical-600 bg-clinical-50/50"
                  : "text-gray-500 hover:text-gray-700 hover:bg-gray-50"
              )}
            >
              <tab.icon className="h-4 w-4" />
              {tab.label}
            </button>
          ))}
        </div>

        {/* Content */}
        <div className="p-6 max-h-[60vh] overflow-y-auto">
          {activeTab === "appearance" && (
            <div className="space-y-6">
              <div>
                <h3 className="text-sm font-semibold text-gray-900 mb-3">Theme</h3>
                <div className="grid grid-cols-3 gap-3">
                  {[
                    { id: "light" as const, label: "Light", icon: Sun },
                    { id: "dark" as const, label: "Dark", icon: Moon },
                    { id: "system" as const, label: "System", icon: Monitor },
                  ].map((option) => (
                    <button
                      key={option.id}
                      onClick={() => handleThemeChange(option.id)}
                      className={cn(
                        "flex flex-col items-center gap-2 p-4 rounded-lg border-2 transition-all",
                        theme === option.id
                          ? "border-clinical-500 bg-clinical-50 text-clinical-700"
                          : "border-gray-200 hover:border-gray-300 text-gray-600"
                      )}
                    >
                      <option.icon className="h-6 w-6" />
                      <span className="text-sm font-medium">{option.label}</span>
                    </button>
                  ))}
                </div>
              </div>
              <p className="text-xs text-gray-500">
                Note: Dark mode support is limited in this version.
              </p>
            </div>
          )}

          {activeTab === "display" && (
            <div className="space-y-4">
              <div className="flex items-center justify-between py-2">
                <div>
                  <p className="text-sm font-medium text-gray-900">Show heatmap by default</p>
                  <p className="text-xs text-gray-500">Display probability heatmap overlay on slides</p>
                </div>
                <Toggle
                  checked={displaySettings.showHeatmapByDefault}
                  onChange={(v) => handleDisplayChange("showHeatmapByDefault", v)}
                />
              </div>
              <div className="flex items-center justify-between py-2">
                <div>
                  <p className="text-sm font-medium text-gray-900">Show confidence scores</p>
                  <p className="text-xs text-gray-500">Display confidence percentages on predictions</p>
                </div>
                <Toggle
                  checked={displaySettings.showConfidenceScores}
                  onChange={(v) => handleDisplayChange("showConfidenceScores", v)}
                />
              </div>
              <div className="flex items-center justify-between py-2">
                <div>
                  <p className="text-sm font-medium text-gray-900">Compact mode</p>
                  <p className="text-xs text-gray-500">Reduce spacing and panel sizes</p>
                </div>
                <Toggle
                  checked={displaySettings.compactMode}
                  onChange={(v) => handleDisplayChange("compactMode", v)}
                />
              </div>
              <div className="flex items-center justify-between py-2">
                <div>
                  <p className="text-sm font-medium text-gray-900">Enable animations</p>
                  <p className="text-xs text-gray-500">Smooth transitions and loading effects</p>
                </div>
                <Toggle
                  checked={displaySettings.animationsEnabled}
                  onChange={(v) => handleDisplayChange("animationsEnabled", v)}
                />
              </div>
            </div>
          )}

          {activeTab === "api" && (
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-900 mb-1">
                  Backend API Endpoint
                </label>
                <input
                  type="text"
                  value={apiSettings.endpoint}
                  onChange={(e) => handleApiChange("endpoint", e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-clinical-500 focus:border-transparent"
                  placeholder="http://localhost:8000"
                />
                <p className="mt-1 text-xs text-gray-500">
                  URL of the MedGemma backend service
                </p>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-900 mb-1">
                  Request Timeout (ms)
                </label>
                <input
                  type="number"
                  value={apiSettings.timeout}
                  onChange={(e) => handleApiChange("timeout", parseInt(e.target.value) || 30000)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-clinical-500 focus:border-transparent"
                  placeholder="30000"
                  min={5000}
                  max={120000}
                  step={1000}
                />
                <p className="mt-1 text-xs text-gray-500">
                  Maximum time to wait for API responses (5000-120000ms)
                </p>
              </div>
            </div>
          )}

          {activeTab === "embedding" && (
            <div className="space-y-4">
              <div>
                <h3 className="text-sm font-semibold text-gray-900 mb-2">Force Re-Embed All Slides</h3>
                <p className="text-xs text-gray-500 mb-4">
                  Re-generate Path Foundation embeddings for all slides at Level 0 (full resolution).
                  This uses the locally-cached model and does <strong>not</strong> download from HuggingFace.
                  Suitable for overnight batch runs (~5-20 min per slide).
                </p>

                {/* Progress display */}
                {batchEmbedStatus && batchEmbedStatus.status !== "idle" && (
                  <div className="mb-4 p-4 rounded-lg border bg-gray-50">
                    {/* Status header */}
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        {batchEmbedStatus.status === "running" && (
                          <Cpu className="h-4 w-4 text-violet-600 animate-pulse" />
                        )}
                        {batchEmbedStatus.status === "completed" && (
                          <CheckCircle className="h-4 w-4 text-green-600" />
                        )}
                        {(batchEmbedStatus.status === "failed" || batchEmbedStatus.status === "cancelled") && (
                          <AlertTriangle className="h-4 w-4 text-amber-600" />
                        )}
                        <span className="text-sm font-medium text-gray-800 capitalize">
                          {batchEmbedStatus.status}
                        </span>
                      </div>
                      <span className="text-xs text-gray-500">
                        {batchEmbedStatus.completed_slides}/{batchEmbedStatus.total_slides} slides
                      </span>
                    </div>

                    {/* Progress bar */}
                    <div className="w-full h-3 bg-gray-200 rounded-full overflow-hidden mb-2">
                      <div
                        className={cn(
                          "h-full transition-all duration-500 rounded-full",
                          batchEmbedStatus.status === "completed" ? "bg-green-500" :
                          batchEmbedStatus.status === "failed" ? "bg-red-500" :
                          batchEmbedStatus.status === "cancelled" ? "bg-amber-500" :
                          "bg-violet-600"
                        )}
                        style={{ width: `${Math.max(batchEmbedStatus.progress, 1)}%` }}
                      />
                    </div>

                    {/* Status message */}
                    <p className="text-xs text-gray-600">{batchEmbedStatus.message}</p>

                    {/* Elapsed time */}
                    {batchEmbedStatus.elapsed_seconds > 0 && (
                      <p className="text-xs text-gray-400 mt-1">
                        Elapsed: {Math.floor(batchEmbedStatus.elapsed_seconds / 60)}m {Math.floor(batchEmbedStatus.elapsed_seconds % 60)}s
                      </p>
                    )}

                    {/* Summary when completed */}
                    {batchEmbedStatus.status === "completed" && batchEmbedStatus.summary && (
                      <div className="mt-3 grid grid-cols-3 gap-2 text-xs">
                        <div className="text-center p-2 bg-green-50 rounded">
                          <div className="font-semibold text-green-700">{batchEmbedStatus.summary.succeeded}</div>
                          <div className="text-green-600">Succeeded</div>
                        </div>
                        <div className="text-center p-2 bg-red-50 rounded">
                          <div className="font-semibold text-red-700">{batchEmbedStatus.summary.failed}</div>
                          <div className="text-red-600">Failed</div>
                        </div>
                        <div className="text-center p-2 bg-blue-50 rounded">
                          <div className="font-semibold text-blue-700">{batchEmbedStatus.summary.total_patches?.toLocaleString()}</div>
                          <div className="text-blue-600">Patches</div>
                        </div>
                      </div>
                    )}

                    {/* Error display */}
                    {batchEmbedStatus.error && (
                      <p className="text-xs text-red-600 mt-2">{batchEmbedStatus.error}</p>
                    )}
                  </div>
                )}

                {/* Error from starting */}
                {batchEmbedError && (
                  <div className="mb-3 p-3 bg-red-50 border border-red-200 rounded-lg">
                    <p className="text-xs text-red-700">{batchEmbedError}</p>
                  </div>
                )}

                {/* Action buttons */}
                <div className="flex gap-2">
                  {(!batchEmbedStatus || batchEmbedStatus.status === "completed" || batchEmbedStatus.status === "failed" || batchEmbedStatus.status === "cancelled") ? (
                    <Button
                      variant="primary"
                      size="sm"
                      onClick={handleStartBatchEmbed}
                      disabled={isStartingBatchEmbed}
                      isLoading={isStartingBatchEmbed}
                      className="flex items-center gap-2"
                    >
                      <Play className="h-4 w-4" />
                      Force Re-Embed All Slides
                    </Button>
                  ) : (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={handleCancelBatchEmbed}
                      className="flex items-center gap-2 text-red-600 hover:text-red-700"
                    >
                      <Square className="h-4 w-4" />
                      Cancel
                    </Button>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-6 py-4 border-t border-gray-200 bg-gray-50">
          <Button
            variant="ghost"
            size="sm"
            onClick={handleReset}
            className="text-gray-500 hover:text-gray-700"
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            Reset to Defaults
          </Button>
          <Button variant="primary" size="sm" onClick={onClose}>
            Done
          </Button>
        </div>
      </div>
    </div>
  );
}
