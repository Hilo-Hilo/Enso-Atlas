"use client";

import React, { useState } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { cn, formatProbability } from "@/lib/utils";
import {
  Grid3X3,
  Maximize2,
  ChevronLeft,
  ChevronRight,
  Eye,
  MapPin,
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

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Grid3X3 className="h-4 w-4" />
            Evidence Patches
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-2">
            {[...Array(6)].map((_, i) => (
              <div
                key={i}
                className="aspect-square bg-gray-100 rounded-lg animate-pulse"
              />
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!patches.length) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Grid3X3 className="h-4 w-4" />
            Evidence Patches
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-gray-500">
            <Grid3X3 className="h-8 w-8 mx-auto mb-2 text-gray-400" />
            <p className="text-sm">No evidence patches available.</p>
            <p className="text-xs mt-1">Run analysis to extract key regions.</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Grid3X3 className="h-4 w-4" />
            Evidence Patches
            <Badge variant="info" size="sm">
              {patches.length} regions
            </Badge>
          </CardTitle>
          <div className="flex items-center gap-1">
            <Button
              variant={viewMode === "grid" ? "primary" : "ghost"}
              size="sm"
              onClick={() => setViewMode("grid")}
              className="p-1.5"
            >
              <Grid3X3 className="h-3.5 w-3.5" />
            </Button>
            <Button
              variant={viewMode === "list" ? "primary" : "ghost"}
              size="sm"
              onClick={() => setViewMode("list")}
              className="p-1.5"
            >
              <Maximize2 className="h-3.5 w-3.5" />
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {/* Patch Grid/List */}
        {viewMode === "grid" ? (
          <div className="grid grid-cols-3 gap-2">
            {visiblePatches.map((patch, index) => (
              <PatchThumbnail
                key={patch.id}
                patch={patch}
                rank={currentPage * patchesPerPage + index + 1}
                isSelected={selectedPatchId === patch.id}
                onClick={() => onPatchClick?.(patch.coordinates)}
              />
            ))}
          </div>
        ) : (
          <div className="space-y-2">
            {visiblePatches.map((patch, index) => (
              <PatchListItem
                key={patch.id}
                patch={patch}
                rank={currentPage * patchesPerPage + index + 1}
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
            >
              <ChevronLeft className="h-4 w-4" />
            </Button>
            <span className="text-xs text-gray-500">
              Page {currentPage + 1} of {totalPages}
            </span>
            <Button
              variant="ghost"
              size="sm"
              onClick={() =>
                setCurrentPage((p) => Math.min(totalPages - 1, p + 1))
              }
              disabled={currentPage === totalPages - 1}
            >
              <ChevronRight className="h-4 w-4" />
            </Button>
          </div>
        )}

        {/* Legend */}
        <div className="pt-2 border-t border-gray-100">
          <p className="text-xs text-gray-500">
            Click on a patch to navigate to its location in the slide viewer.
            Higher attention weights indicate regions more influential to the
            prediction.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}

interface PatchThumbnailProps {
  patch: EvidencePatch;
  rank: number;
  isSelected: boolean;
  onClick: () => void;
}

function PatchThumbnail({ patch, rank, isSelected, onClick }: PatchThumbnailProps) {
  const attentionPercent = Math.round(patch.attentionWeight * 100);

  return (
    <button
      onClick={onClick}
      className={cn(
        "relative aspect-square rounded-lg overflow-hidden border-2 transition-all",
        "hover:border-clinical-500 hover:shadow-md focus:outline-none focus:ring-2 focus:ring-clinical-500",
        isSelected
          ? "border-clinical-600 ring-2 ring-clinical-200"
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
      <div className="absolute top-1 left-1 bg-black/70 text-white text-xs font-medium px-1.5 py-0.5 rounded">
        #{rank}
      </div>

      {/* Attention Score */}
      <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-1.5">
        <div className="flex items-center justify-between">
          <Eye className="h-3 w-3 text-white/80" />
          <span className="text-xs font-mono text-white">
            {attentionPercent}%
          </span>
        </div>
      </div>
    </button>
  );
}

interface PatchListItemProps {
  patch: EvidencePatch;
  rank: number;
  isSelected: boolean;
  onClick: () => void;
}

function PatchListItem({ patch, rank, isSelected, onClick }: PatchListItemProps) {
  const attentionPercent = Math.round(patch.attentionWeight * 100);

  return (
    <button
      onClick={onClick}
      className={cn(
        "w-full flex items-center gap-3 p-2 rounded-lg border transition-all text-left",
        "hover:border-clinical-500 hover:bg-clinical-50 focus:outline-none focus:ring-2 focus:ring-clinical-500",
        isSelected
          ? "border-clinical-600 bg-clinical-50"
          : "border-gray-200 bg-white"
      )}
    >
      {/* Thumbnail */}
      <div className="relative w-16 h-16 rounded overflow-hidden shrink-0">
        <img
          src={patch.thumbnailUrl}
          alt={`Evidence patch ${rank}`}
          className="w-full h-full object-cover"
        />
        <div className="absolute top-0.5 left-0.5 bg-black/70 text-white text-xs font-medium px-1 py-0.5 rounded text-[10px]">
          #{rank}
        </div>
      </div>

      {/* Info */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between mb-1">
          <span className="text-sm font-medium text-gray-900">
            Patch {patch.patchId.slice(0, 8)}
          </span>
          <Badge
            variant={
              attentionPercent >= 70
                ? "success"
                : attentionPercent >= 40
                ? "warning"
                : "default"
            }
            size="sm"
          >
            {attentionPercent}% attention
          </Badge>
        </div>
        <div className="flex items-center gap-1 text-xs text-gray-500">
          <MapPin className="h-3 w-3" />
          <span>
            ({patch.coordinates.x}, {patch.coordinates.y})
          </span>
        </div>
        {patch.morphologyDescription && (
          <p className="text-xs text-gray-600 mt-1 truncate">
            {patch.morphologyDescription}
          </p>
        )}
      </div>
    </button>
  );
}
