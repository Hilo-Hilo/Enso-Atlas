"use client";

import React, { useState } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { cn, formatProbability } from "@/lib/utils";
import {
  Grid3X3,
  List,
  ChevronLeft,
  ChevronRight,
  Eye,
  MapPin,
  Crosshair,
  ZoomIn,
  Layers,
  Info,
} from "lucide-react";
import type { EvidencePatch, PatchCoordinates } from "@/types";

interface EvidencePanelProps {
  patches: EvidencePatch[];
  isLoading?: boolean;
  onPatchClick?: (coords: PatchCoordinates) => void;
  selectedPatchId?: string;
}

export function EvidencePanel({
  patches,
  isLoading,
  onPatchClick,
  selectedPatchId,
}: EvidencePanelProps) {
  const [viewMode, setViewMode] = useState<"grid" | "list">("grid");
  const [currentPage, setCurrentPage] = useState(0);
  const patchesPerPage = viewMode === "grid" ? 6 : 4;

  const totalPages = Math.ceil(patches.length / patchesPerPage);
  const visiblePatches = patches.slice(
    currentPage * patchesPerPage,
    (currentPage + 1) * patchesPerPage
  );

  // Sort patches by attention weight for ranking
  const sortedPatches = [...patches].sort(
    (a, b) => b.attentionWeight - a.attentionWeight
  );
  const getRank = (patchId: string) =>
    sortedPatches.findIndex((p) => p.id === patchId) + 1;

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Layers className="h-4 w-4 text-clinical-600" />
            Evidence Patches
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-2">
            {[...Array(6)].map((_, i) => (
              <div key={i} className="relative aspect-square">
                <div className="absolute inset-0 bg-gray-100 rounded-lg animate-pulse" />
                <div className="absolute inset-0 bg-gradient-to-t from-gray-200 to-transparent rounded-lg" />
              </div>
            ))}
          </div>
          <p className="text-xs text-gray-500 text-center mt-3">
            Extracting attention regions...
          </p>
        </CardContent>
      </Card>
    );
  }

  if (!patches.length) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Layers className="h-4 w-4 text-gray-400" />
            Evidence Patches
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-gray-500">
            <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gray-100 flex items-center justify-center">
              <Grid3X3 className="h-8 w-8 text-gray-400" />
            </div>
            <p className="text-sm font-medium text-gray-600">
              No evidence patches
            </p>
            <p className="text-xs mt-1.5 text-gray-500 max-w-[200px] mx-auto">
              Run analysis to extract the most influential tissue regions.
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Layers className="h-4 w-4 text-clinical-600" />
            Evidence Patches
            <Badge variant="info" size="sm" className="font-mono">
              {patches.length}
            </Badge>
          </CardTitle>
          <div className="flex items-center gap-1 bg-surface-secondary rounded-lg p-0.5">
            <button
              onClick={() => setViewMode("grid")}
              className={cn(
                "p-1.5 rounded-md transition-all",
                viewMode === "grid"
                  ? "bg-white shadow-clinical text-clinical-700"
                  : "text-gray-500 hover:text-gray-700"
              )}
              title="Grid view"
            >
              <Grid3X3 className="h-4 w-4" />
            </button>
            <button
              onClick={() => setViewMode("list")}
              className={cn(
                "p-1.5 rounded-md transition-all",
                viewMode === "list"
                  ? "bg-white shadow-clinical text-clinical-700"
                  : "text-gray-500 hover:text-gray-700"
              )}
              title="List view"
            >
              <List className="h-4 w-4" />
            </button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-3 pt-0">
        {/* Patch Grid/List */}
        {viewMode === "grid" ? (
          <div className="grid grid-cols-3 gap-2">
            {visiblePatches.map((patch) => (
              <PatchThumbnail
                key={patch.id}
                patch={patch}
                rank={getRank(patch.id)}
                isSelected={selectedPatchId === patch.id}
                onClick={() => onPatchClick?.(patch.coordinates)}
              />
            ))}
          </div>
        ) : (
          <div className="space-y-2">
            {visiblePatches.map((patch) => (
              <PatchListItem
                key={patch.id}
                patch={patch}
                rank={getRank(patch.id)}
                isSelected={selectedPatchId === patch.id}
                onClick={() => onPatchClick?.(patch.coordinates)}
              />
            ))}
          </div>
        )}

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-between pt-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setCurrentPage((p) => Math.max(0, p - 1))}
              disabled={currentPage === 0}
              className="p-1.5"
            >
              <ChevronLeft className="h-4 w-4" />
            </Button>
            <div className="flex items-center gap-1.5">
              {[...Array(totalPages)].map((_, i) => (
                <button
                  key={i}
                  onClick={() => setCurrentPage(i)}
                  className={cn(
                    "w-2 h-2 rounded-full transition-all",
                    currentPage === i
                      ? "bg-clinical-600 w-4"
                      : "bg-gray-300 hover:bg-gray-400"
                  )}
                />
              ))}
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={() =>
                setCurrentPage((p) => Math.min(totalPages - 1, p + 1))
              }
              disabled={currentPage === totalPages - 1}
              className="p-1.5"
            >
              <ChevronRight className="h-4 w-4" />
            </Button>
          </div>
        )}

        {/* Legend */}
        <div className="pt-3 border-t border-gray-100">
          <div className="flex items-center justify-between text-xs text-gray-500 mb-2">
            <span className="font-medium">Attention Weight</span>
            <span>Click to navigate</span>
          </div>
          <div className="heatmap-legend" />
          <div className="flex justify-between text-2xs text-gray-400 mt-1">
            <span>Low</span>
            <span>High</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// Patch Thumbnail Component
interface PatchThumbnailProps {
  patch: EvidencePatch;
  rank: number;
  isSelected: boolean;
  onClick: () => void;
}

function PatchThumbnail({
  patch,
  rank,
  isSelected,
  onClick,
}: PatchThumbnailProps) {
  const attentionPercent = Math.round(patch.attentionWeight * 100);
  const attentionColor =
    attentionPercent >= 70
      ? "bg-red-500"
      : attentionPercent >= 40
      ? "bg-amber-500"
      : "bg-blue-500";

  return (
    <button
      onClick={onClick}
      className={cn(
        "relative aspect-square rounded-lg overflow-hidden border-2 transition-all group",
        "hover:border-clinical-500 hover:shadow-md hover:scale-[1.02]",
        "focus:outline-none focus:ring-2 focus:ring-clinical-500 focus:ring-offset-1",
        isSelected
          ? "border-clinical-600 ring-2 ring-clinical-200 shadow-md"
          : "border-gray-200"
      )}
    >
      {/* Patch Image */}
      <img
        src={patch.thumbnailUrl}
        alt={`Evidence patch ${rank}`}
        className="w-full h-full object-cover"
      />

      {/* Rank Badge */}
      <div className="absolute top-1 left-1 bg-navy-900/80 backdrop-blur-sm text-white text-xs font-bold px-1.5 py-0.5 rounded shadow">
        #{rank}
      </div>

      {/* Attention Score Indicator */}
      <div className="absolute top-1 right-1">
        <div
          className={cn(
            "w-6 h-6 rounded-full flex items-center justify-center text-2xs font-bold text-white shadow",
            attentionColor
          )}
        >
          {attentionPercent}
        </div>
      </div>

      {/* Hover overlay */}
      <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity">
        <div className="absolute bottom-0 left-0 right-0 p-2">
          <div className="flex items-center justify-between text-white">
            <div className="flex items-center gap-1">
              <Crosshair className="h-3 w-3" />
              <span className="text-2xs font-mono">
                ({patch.coordinates.x}, {patch.coordinates.y})
              </span>
            </div>
            <ZoomIn className="h-4 w-4" />
          </div>
        </div>
      </div>

      {/* Selected indicator */}
      {isSelected && (
        <div className="absolute inset-0 border-2 border-clinical-500 rounded-lg" />
      )}
    </button>
  );
}

// Patch List Item Component
interface PatchListItemProps {
  patch: EvidencePatch;
  rank: number;
  isSelected: boolean;
  onClick: () => void;
}

function PatchListItem({
  patch,
  rank,
  isSelected,
  onClick,
}: PatchListItemProps) {
  const attentionPercent = Math.round(patch.attentionWeight * 100);

  return (
    <button
      onClick={onClick}
      className={cn(
        "w-full flex items-center gap-3 p-3 rounded-lg border transition-all text-left group",
        "hover:border-clinical-500 hover:bg-clinical-50/50 hover:shadow-clinical",
        "focus:outline-none focus:ring-2 focus:ring-clinical-500",
        isSelected
          ? "border-clinical-600 bg-clinical-50 ring-1 ring-clinical-200"
          : "border-gray-200 bg-white"
      )}
    >
      {/* Thumbnail */}
      <div className="relative w-16 h-16 rounded-lg overflow-hidden shrink-0 border border-gray-200 group-hover:border-clinical-300">
        <img
          src={patch.thumbnailUrl}
          alt={`Evidence patch ${rank}`}
          className="w-full h-full object-cover"
        />
        <div className="absolute top-0.5 left-0.5 bg-navy-900/80 text-white text-2xs font-bold px-1 py-0.5 rounded shadow">
          #{rank}
        </div>
      </div>

      {/* Info */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between mb-1.5">
          <span className="text-sm font-medium text-gray-900">
            Patch {patch.patchId.slice(0, 8)}
          </span>
          <Badge
            variant={
              attentionPercent >= 70
                ? "danger"
                : attentionPercent >= 40
                ? "warning"
                : "info"
            }
            size="sm"
            className="font-mono"
          >
            {attentionPercent}%
          </Badge>
        </div>

        <div className="flex items-center gap-2 text-xs text-gray-500">
          <MapPin className="h-3 w-3" />
          <span className="font-mono">
            ({patch.coordinates.x.toLocaleString()}, {patch.coordinates.y.toLocaleString()})
          </span>
        </div>

        {patch.morphologyDescription && (
          <p className="text-xs text-gray-600 mt-1.5 line-clamp-2 leading-relaxed">
            {patch.morphologyDescription}
          </p>
        )}
      </div>

      {/* Navigate indicator */}
      <div className="shrink-0 opacity-0 group-hover:opacity-100 transition-opacity">
        <ZoomIn className="h-4 w-4 text-clinical-600" />
      </div>
    </button>
  );
}
