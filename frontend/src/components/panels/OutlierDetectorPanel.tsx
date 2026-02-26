"use client";

import React, { useState, useCallback } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { cn } from "@/lib/utils";
import { detectOutliers } from "@/lib/api";
import type { OutlierDetectionResult, OutlierPatch, PatchCoordinates } from "@/types";
import {
  AlertTriangle,
  Activity,
  Eye,
  EyeOff,
  BarChart3,
  Loader2,
} from "lucide-react";

interface OutlierDetectorPanelProps {
  slideId: string | null;
  onHeatmapToggle?: (
    enabled: boolean,
    heatmapData: Array<{ x: number; y: number; score: number }> | null
  ) => void;
  onPatchClick?: (coords: PatchCoordinates) => void;
  className?: string;
}

export function OutlierDetectorPanel({
  slideId,
  onHeatmapToggle,
  onPatchClick,
  className,
}: OutlierDetectorPanelProps) {
  const [result, setResult] = useState<OutlierDetectionResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [threshold, setThreshold] = useState(2.0);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [showAllOutliers, setShowAllOutliers] = useState(false);

  const handleDetect = useCallback(async () => {
    if (!slideId) return;

    setIsLoading(true);
    setError(null);

    try {
      const data = await detectOutliers(slideId, threshold);
      setResult(data);
    } catch (err) {
      console.error("Outlier detection failed:", err);
      setError(
        err instanceof Error ? err.message : "Outlier detection failed"
      );
      setResult(null);
    } finally {
      setIsLoading(false);
    }
  }, [slideId, threshold]);

  const handleHeatmapToggle = useCallback(() => {
    const newState = !showHeatmap;
    setShowHeatmap(newState);
    if (onHeatmapToggle) {
      if (newState && result) {
        // Only include actual outlier patches, not all patches
        const outlierCoords = new Set(
          result.outlierPatches.map((p) => `${p.x},${p.y}`)
        );
        const outlierData = result.heatmapData.filter((d) =>
          outlierCoords.has(`${d.x},${d.y}`)
        );
        onHeatmapToggle(true, outlierData);
      } else {
        onHeatmapToggle(false, null);
      }
    }
  }, [showHeatmap, result, onHeatmapToggle]);

  const handlePatchClick = useCallback(
    (patch: OutlierPatch) => {
      if (onPatchClick) {
        onPatchClick({
          x: patch.x,
          y: patch.y,
          width: 224,
          height: 224,
          level: 0,
        });
      }
    },
    [onPatchClick]
  );

  // Determine how many outliers to show
  const MAX_VISIBLE = 10;
  const visibleOutliers =
    result && !showAllOutliers
      ? result.outlierPatches.slice(0, MAX_VISIBLE)
      : result?.outlierPatches ?? [];
  const hasMore =
    result !== null && result.outlierPatches.length > MAX_VISIBLE;

  return (
    <Card className={cn("transition-all", className)}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-sm">
          <AlertTriangle className="h-4 w-4 text-amber-500 dark:text-amber-400" />
          Outlier Tissue Detector
        </CardTitle>
      </CardHeader>

      <CardContent className="space-y-3 pt-0">
          {/* Threshold slider */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <label className="text-xs font-medium text-gray-600 dark:text-gray-300">
                Z-Score Threshold
              </label>
              <span className="text-xs font-mono text-gray-500 dark:text-gray-400">
                {threshold.toFixed(1)} SD
              </span>
            </div>
            <input
              type="range"
              min="1.0"
              max="4.0"
              step="0.1"
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              className="w-full h-1.5 bg-gray-200 dark:bg-navy-700 rounded-lg appearance-none cursor-pointer accent-amber-500"
              disabled={isLoading}
            />
            <div className="flex justify-between text-2xs text-gray-400 dark:text-gray-500">
              <span>1.0 (more outliers)</span>
              <span>4.0 (fewer)</span>
            </div>
          </div>

          {/* Run button */}
          <Button
            variant="secondary"
            size="sm"
            onClick={handleDetect}
            disabled={!slideId || isLoading}
            className="w-full gap-2"
          >
            {isLoading ? (
              <>
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                Detecting...
              </>
            ) : (
              <>
                <Activity className="h-3.5 w-3.5" />
                Detect Outlier Tissue
              </>
            )}
          </Button>

          {/* Error display */}
          {error && (
            <div className="p-2.5 bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-800 rounded-lg">
              <p className="text-xs text-red-700 dark:text-red-300">{error}</p>
            </div>
          )}

          {/* Results */}
          {result && (
            <div className="space-y-3">
              {/* Stats summary */}
              <div className="grid grid-cols-2 gap-2">
                <div className="p-2 bg-gray-50 dark:bg-navy-900 rounded-lg">
                  <p className="text-2xs text-gray-500 dark:text-gray-400">
                    Total Patches
                  </p>
                  <p className="text-sm font-semibold text-gray-900 dark:text-gray-100">
                    {result.totalPatches.toLocaleString()}
                  </p>
                </div>
                <div className="p-2 bg-amber-50 dark:bg-amber-900/30 rounded-lg">
                  <p className="text-2xs text-amber-600 dark:text-amber-300">
                    Outliers Found
                  </p>
                  <p className="text-sm font-semibold text-amber-700 dark:text-amber-300">
                    {result.outlierCount}
                    <span className="text-2xs font-normal ml-1">
                      ({((result.outlierCount / result.totalPatches) * 100).toFixed(1)}%)
                    </span>
                  </p>
                </div>
                <div className="p-2 bg-gray-50 dark:bg-navy-900 rounded-lg">
                  <p className="text-2xs text-gray-500 dark:text-gray-400">
                    Mean Distance
                  </p>
                  <p className="text-sm font-mono text-gray-900 dark:text-gray-100">
                    {result.meanDistance.toFixed(2)}
                  </p>
                </div>
                <div className="p-2 bg-gray-50 dark:bg-navy-900 rounded-lg">
                  <p className="text-2xs text-gray-500 dark:text-gray-400">
                    Std Deviation
                  </p>
                  <p className="text-sm font-mono text-gray-900 dark:text-gray-100">
                    {result.stdDistance.toFixed(2)}
                  </p>
                </div>
              </div>

              {/* Heatmap toggle */}
              <button
                onClick={handleHeatmapToggle}
                className={cn(
                  "w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg text-xs font-medium transition-colors",
                  showHeatmap
                    ? "bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300 border border-amber-200 dark:border-amber-800"
                    : "bg-gray-100 dark:bg-navy-700 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-navy-600"
                )}
              >
                {showHeatmap ? (
                  <>
                    <EyeOff className="h-3.5 w-3.5" />
                    Hide Heatmap
                  </>
                ) : (
                  <>
                    <Eye className="h-3.5 w-3.5" />
                    Show as Heatmap
                  </>
                )}
              </button>

              {/* Outlier patch list */}
              {result.outlierPatches.length > 0 && (
                <div className="space-y-1.5">
                  <div className="flex items-center justify-between">
                    <p className="text-xs font-medium text-gray-600 dark:text-gray-300">
                      Top Outlier Patches
                    </p>
                    <Badge variant="warning" size="sm">
                      {result.outlierCount} flagged
                    </Badge>
                  </div>

                  <div className="max-h-48 overflow-y-auto space-y-1 pr-1">
                    {visibleOutliers.map((patch) => (
                      <button
                        key={patch.patchIdx}
                        onClick={() => handlePatchClick(patch)}
                        className="w-full flex items-center justify-between px-2.5 py-1.5 rounded-md text-left hover:bg-amber-50 dark:hover:bg-amber-900/20 transition-colors group"
                      >
                        <div className="flex items-center gap-2 min-w-0">
                          <BarChart3 className="h-3 w-3 text-amber-500 dark:text-amber-400 shrink-0" />
                          <span className="text-xs text-gray-700 dark:text-gray-200 truncate">
                            Patch {patch.patchIdx}
                          </span>
                          <span className="text-2xs text-gray-400 dark:text-gray-500">
                            ({patch.x}, {patch.y})
                          </span>
                        </div>
                        <div className="flex items-center gap-1.5 shrink-0">
                          <span className="text-xs font-mono text-amber-600 dark:text-amber-300">
                            {patch.zScore.toFixed(1)} SD
                          </span>
                        </div>
                      </button>
                    ))}
                  </div>

                  {/* Show more/less toggle */}
                  {hasMore && (
                    <button
                      onClick={() => setShowAllOutliers(!showAllOutliers)}
                      className="w-full text-center text-xs text-clinical-600 dark:text-clinical-400 hover:text-clinical-700 dark:hover:text-clinical-300 font-medium py-1"
                    >
                      {showAllOutliers
                        ? "Show less"
                        : `Show all ${result.outlierPatches.length} outliers`}
                    </button>
                  )}
                </div>
              )}

              {/* No outliers message */}
              {result.outlierPatches.length === 0 && (
                <div className="p-3 bg-green-50 dark:bg-green-900/30 border border-green-200 dark:border-green-800 rounded-lg text-center">
                  <p className="text-xs text-green-700 dark:text-green-300">
                    No outlier patches detected at {threshold.toFixed(1)} SD
                    threshold. Tissue appears homogeneous.
                  </p>
                </div>
              )}
            </div>
          )}

          {/* Disabled state */}
          {!slideId && (
            <p className="text-xs text-gray-400 dark:text-gray-500 text-center py-2">
              Select a slide to detect outlier tissue
            </p>
          )}
      </CardContent>
    </Card>
  );
}
