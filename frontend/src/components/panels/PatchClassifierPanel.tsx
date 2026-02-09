"use client";

import React, { useState, useCallback, useEffect } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Spinner } from "@/components/ui/Spinner";
import { cn } from "@/lib/utils";
import {
  Layers,
  Plus,
  Trash2,
  Play,
  BarChart3,
  Map,
  AlertCircle,
  CheckCircle,
  X,
} from "lucide-react";
import type { PatchClassifyResult } from "@/types";
import { classifyPatches } from "@/lib/api";

// Predefined colors for classification classes (up to 8)
const CLASS_COLORS = [
  { bg: "bg-blue-500", text: "text-blue-700 dark:text-blue-300", light: "bg-blue-100 dark:bg-blue-900/30", hex: "#3b82f6" },
  { bg: "bg-red-500", text: "text-red-700 dark:text-red-300", light: "bg-red-100 dark:bg-red-900/30", hex: "#ef4444" },
  { bg: "bg-emerald-500", text: "text-emerald-700 dark:text-emerald-300", light: "bg-emerald-100 dark:bg-emerald-900/30", hex: "#10b981" },
  { bg: "bg-amber-500", text: "text-amber-700 dark:text-amber-300", light: "bg-amber-100 dark:bg-amber-900/30", hex: "#f59e0b" },
  { bg: "bg-purple-500", text: "text-purple-700 dark:text-purple-300", light: "bg-purple-100 dark:bg-purple-900/30", hex: "#8b5cf6" },
  { bg: "bg-pink-500", text: "text-pink-700 dark:text-pink-300", light: "bg-pink-100 dark:bg-pink-900/30", hex: "#ec4899" },
  { bg: "bg-cyan-500", text: "text-cyan-700 dark:text-cyan-300", light: "bg-cyan-100 dark:bg-cyan-900/30", hex: "#06b6d4" },
  { bg: "bg-orange-500", text: "text-orange-700 dark:text-orange-300", light: "bg-orange-100 dark:bg-orange-900/30", hex: "#f97316" },
];

interface ClassDefinition {
  name: string;
  patchIndicesText: string; // raw text input: "1,2,3,10-20"
}

/**
 * Parse a comma-separated list of patch indices, supporting ranges like "10-20".
 */
function parsePatchIndices(text: string): number[] {
  const indices: number[] = [];
  const parts = text.split(",").map((s) => s.trim()).filter(Boolean);
  for (const part of parts) {
    if (part.includes("-")) {
      const [startStr, endStr] = part.split("-").map((s) => s.trim());
      const start = parseInt(startStr, 10);
      const end = parseInt(endStr, 10);
      if (!isNaN(start) && !isNaN(end) && start <= end) {
        for (let i = start; i <= end; i++) {
          indices.push(i);
        }
      }
    } else {
      const num = parseInt(part, 10);
      if (!isNaN(num)) {
        indices.push(num);
      }
    }
  }
  return Array.from(new Set(indices)); // deduplicate
}

interface PatchClassifierPanelProps {
  slideId: string | null;
  isAnalyzed: boolean;
  totalPatches?: number;
  onClassifyResult?: (result: PatchClassifyResult | null) => void;
  onShowHeatmap?: (show: boolean) => void;
  showHeatmap?: boolean;
  // Spatial selection mode: when active, clicks on WSI add patches to a class
  onSelectionModeChange?: (mode: { activeClassIdx: number; classColor: string } | null) => void;
  hasPatchCoordinates?: boolean;
}

export function PatchClassifierPanel({
  slideId,
  isAnalyzed,
  totalPatches,
  onClassifyResult,
  onShowHeatmap,
  showHeatmap = false,
  onSelectionModeChange,
  hasPatchCoordinates = false,
}: PatchClassifierPanelProps) {
  const [classes, setClasses] = useState<ClassDefinition[]>([
    { name: "tumor", patchIndicesText: "" },
    { name: "stroma", patchIndicesText: "" },
  ]);
  const [isClassifying, setIsClassifying] = useState(false);
  const [result, setResult] = useState<PatchClassifyResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeSelectionClass, setActiveSelectionClass] = useState<number | null>(null);

  // Register global callback for spatial patch selection from WSI viewer
  useEffect(() => {
    (window as any).__patchClassifierAddPatch = (classIdx: number, patchIdx: number) => {
      setClasses((prev) =>
        prev.map((c, i) => {
          if (i !== classIdx) return c;
          const existing = parsePatchIndices(c.patchIndicesText);
          if (existing.includes(patchIdx)) return c; // already added
          const newText = c.patchIndicesText
            ? `${c.patchIndicesText},${patchIdx}`
            : String(patchIdx);
          return { ...c, patchIndicesText: newText };
        })
      );
    };
    return () => {
      delete (window as any).__patchClassifierAddPatch;
    };
  }, []);

  // Sync selection mode to parent when activeSelectionClass changes
  useEffect(() => {
    if (activeSelectionClass !== null) {
      const color = CLASS_COLORS[activeSelectionClass % CLASS_COLORS.length];
      onSelectionModeChange?.({ activeClassIdx: activeSelectionClass, classColor: color.hex });
    } else {
      onSelectionModeChange?.(null);
    }
  }, [activeSelectionClass, onSelectionModeChange]);

  // Clear selection mode when result arrives or component resets
  useEffect(() => {
    if (result) {
      setActiveSelectionClass(null);
    }
  }, [result]);

  const addClass = useCallback(() => {
    if (classes.length >= 8) return;
    setClasses((prev) => [...prev, { name: "", patchIndicesText: "" }]);
  }, [classes.length]);

  const removeClass = useCallback((index: number) => {
    if (classes.length <= 2) return;
    setClasses((prev) => prev.filter((_, i) => i !== index));
  }, [classes.length]);

  const updateClassName = useCallback((index: number, name: string) => {
    setClasses((prev) => prev.map((c, i) => (i === index ? { ...c, name } : c)));
  }, []);

  const updatePatchIndices = useCallback((index: number, text: string) => {
    setClasses((prev) => prev.map((c, i) => (i === index ? { ...c, patchIndicesText: text } : c)));
  }, []);

  const handleClassify = useCallback(async () => {
    if (!slideId) return;
    setError(null);

    // Validate class names
    const names = classes.map((c) => c.name.trim());
    if (names.some((n) => !n)) {
      setError("All classes must have a name");
      return;
    }
    if (new Set(names).size !== names.length) {
      setError("Class names must be unique");
      return;
    }

    // Parse indices
    const classMap: Record<string, number[]> = {};
    for (const cls of classes) {
      const indices = parsePatchIndices(cls.patchIndicesText);
      if (indices.length === 0) {
        setError(`Class "${cls.name.trim()}" needs at least 1 patch index`);
        return;
      }
      classMap[cls.name.trim()] = indices;
    }

    setIsClassifying(true);
    try {
      const res = await classifyPatches(slideId, classMap);
      setResult(res);
      onClassifyResult?.(res);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Classification failed";
      setError(msg);
      setResult(null);
      onClassifyResult?.(null);
    } finally {
      setIsClassifying(false);
    }
  }, [slideId, classes, onClassifyResult]);

  const handleReset = useCallback(() => {
    setResult(null);
    setError(null);
    setActiveSelectionClass(null);
    onClassifyResult?.(null);
    onShowHeatmap?.(false);
  }, [onClassifyResult, onShowHeatmap]);

  // Not ready state
  if (!slideId || !isAnalyzed) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Layers className="h-4 w-4 text-gray-400 dark:text-gray-500" />
            Patch Classification
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            Analyze a slide first to enable few-shot patch classification.
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Layers className="h-4 w-4 text-clinical-600 dark:text-clinical-400" />
            Patch Classification
          </CardTitle>
          {result && (
            <Button variant="ghost" size="sm" onClick={handleReset}>
              <X className="h-3 w-3 mr-1" />
              Reset
            </Button>
          )}
        </div>
        <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
          Define classes and provide example patch indices to train a classifier.
          {totalPatches ? ` (${totalPatches.toLocaleString()} patches available)` : ""}
        </p>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Class definitions */}
        {!result && (
          <>
            <div className="space-y-3">
              {classes.map((cls, idx) => {
                const color = CLASS_COLORS[idx % CLASS_COLORS.length];
                const parsedCount = parsePatchIndices(cls.patchIndicesText).length;
                return (
                  <div key={idx} className={cn("p-3 rounded-lg border", color.light, "border-gray-200 dark:border-gray-700")}>
                    <div className="flex items-center gap-2 mb-2">
                      <div className={cn("w-3 h-3 rounded-full", color.bg)} />
                      <input
                        type="text"
                        value={cls.name}
                        onChange={(e) => updateClassName(idx, e.target.value)}
                        placeholder="Class name"
                        className="flex-1 text-sm font-medium bg-transparent border-none outline-none text-gray-900 dark:text-gray-100 placeholder-gray-400 dark:placeholder-gray-500"
                      />
                      {parsedCount > 0 && (
                        <Badge variant="default" className="text-xs">
                          {parsedCount} patches
                        </Badge>
                      )}
                      {classes.length > 2 && (
                        <button
                          onClick={() => removeClass(idx)}
                          className="text-gray-400 hover:text-red-500 dark:text-gray-500 dark:hover:text-red-400 transition-colors"
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </button>
                      )}
                    </div>
                    <div className="flex items-center gap-1.5">
                      <input
                        type="text"
                        value={cls.patchIndicesText}
                        onChange={(e) => updatePatchIndices(idx, e.target.value)}
                        placeholder="Patch indices: 1,2,3 or 10-20"
                        className="flex-1 text-xs bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded px-2 py-1.5 text-gray-700 dark:text-gray-300 placeholder-gray-400 dark:placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-clinical-500"
                      />
                      {hasPatchCoordinates && (
                        <button
                          onClick={() => setActiveSelectionClass(activeSelectionClass === idx ? null : idx)}
                          className={cn(
                            "shrink-0 px-2 py-1.5 text-2xs rounded border transition-colors",
                            activeSelectionClass === idx
                              ? "bg-clinical-500 text-white border-clinical-500"
                              : "bg-white dark:bg-gray-800 text-gray-500 dark:text-gray-400 border-gray-200 dark:border-gray-600 hover:border-clinical-300"
                          )}
                          title={activeSelectionClass === idx ? "Stop selecting" : "Click patches on slide to add"}
                        >
                          {activeSelectionClass === idx ? "Selecting..." : "Select"}
                        </button>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>

            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={addClass}
                disabled={classes.length >= 8}
                className="text-xs"
              >
                <Plus className="h-3 w-3 mr-1" />
                Add Class
              </Button>
              <div className="flex-1" />
              <Button
                variant="primary"
                size="sm"
                onClick={handleClassify}
                disabled={isClassifying}
              >
                {isClassifying ? (
                  <>
                    <Spinner className="h-3 w-3 mr-1" />
                    Classifying...
                  </>
                ) : (
                  <>
                    <Play className="h-3 w-3 mr-1" />
                    Train & Classify
                  </>
                )}
              </Button>
            </div>
          </>
        )}

        {/* Error display */}
        {error && (
          <div className="flex items-start gap-2 p-3 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800">
            <AlertCircle className="h-4 w-4 text-red-500 mt-0.5 flex-shrink-0" />
            <p className="text-xs text-red-700 dark:text-red-300">{error}</p>
          </div>
        )}

        {/* Results */}
        {result && (
          <div className="space-y-3">
            {/* Accuracy estimate */}
            {result.accuracyEstimate !== null && (
              <div className="flex items-center gap-2 p-2 rounded-lg bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800">
                <CheckCircle className="h-4 w-4 text-green-600 dark:text-green-400" />
                <span className="text-xs text-green-700 dark:text-green-300">
                  Leave-one-out accuracy: <span className="font-semibold">{(result.accuracyEstimate * 100).toFixed(1)}%</span>
                </span>
              </div>
            )}

            {/* Class distribution */}
            <div className="space-y-2">
              <h4 className="text-xs font-medium text-gray-600 dark:text-gray-400 flex items-center gap-1">
                <BarChart3 className="h-3 w-3" />
                Class Distribution ({result.totalPatches.toLocaleString()} patches)
              </h4>
              {result.classes.map((cls, idx) => {
                const count = result.classCounts[cls] ?? 0;
                const pct = result.totalPatches > 0 ? (count / result.totalPatches) * 100 : 0;
                const color = CLASS_COLORS[idx % CLASS_COLORS.length];
                return (
                  <div key={cls} className="space-y-1">
                    <div className="flex items-center justify-between text-xs">
                      <div className="flex items-center gap-1.5">
                        <div className={cn("w-2.5 h-2.5 rounded-full", color.bg)} />
                        <span className={cn("font-medium", color.text)}>{cls}</span>
                      </div>
                      <span className="text-gray-500 dark:text-gray-400">
                        {count.toLocaleString()} ({pct.toFixed(1)}%)
                      </span>
                    </div>
                    <div className="w-full h-2 bg-gray-100 dark:bg-gray-700 rounded-full overflow-hidden">
                      <div
                        className={cn("h-full rounded-full transition-all", color.bg)}
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Heatmap toggle */}
            <div className="pt-2 border-t border-gray-200 dark:border-gray-700">
              <Button
                variant={showHeatmap ? "primary" : "secondary"}
                size="sm"
                className="w-full"
                onClick={() => onShowHeatmap?.(!showHeatmap)}
              >
                <Map className="h-3 w-3 mr-1" />
                {showHeatmap ? "Hide Classification Heatmap" : "Show as Heatmap"}
              </Button>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
