"use client";

import React, { useState, useMemo } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { SkeletonEvidenceGrid } from "@/components/ui/Skeleton";
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
  AlertCircle,
  RefreshCw,
  Circle,
  Filter,
  X,
  Search,
} from "lucide-react";
import type { EvidencePatch, PatchCoordinates, TissueType } from "@/types";

// Extended tissue type to include artifact
type ExtendedTissueType = TissueType | "artifact";

interface TissueTypeInfo {
  label: string;
  shortLabel: string;
  color: string;
  bgColor: string;
  borderColor: string;
  hoverBg: string;
}

const TISSUE_TYPES: Record<ExtendedTissueType, TissueTypeInfo> = {
  tumor: {
    label: "Tumor Region",
    shortLabel: "Tumor",
    color: "text-red-700 dark:text-red-300",
    bgColor: "bg-red-100 dark:bg-red-900/30",
    borderColor: "border-red-300 dark:border-red-800",
    hoverBg: "hover:bg-red-200 dark:hover:bg-red-900/40",
  },
  stroma: {
    label: "Stromal Tissue",
    shortLabel: "Stroma",
    color: "text-blue-700 dark:text-blue-300",
    bgColor: "bg-blue-100 dark:bg-blue-900/30",
    borderColor: "border-blue-300 dark:border-blue-800",
    hoverBg: "hover:bg-blue-200 dark:hover:bg-blue-900/40",
  },
  necrosis: {
    label: "Necrosis",
    shortLabel: "Necrosis",
    color: "text-gray-700 dark:text-gray-300",
    bgColor: "bg-gray-200 dark:bg-navy-700",
    borderColor: "border-gray-400 dark:border-navy-500",
    hoverBg: "hover:bg-gray-300 dark:hover:bg-navy-600",
  },
  inflammatory: {
    label: "Inflammatory Infiltrate",
    shortLabel: "Inflam.",
    color: "text-purple-700 dark:text-purple-300",
    bgColor: "bg-purple-100 dark:bg-purple-900/30",
    borderColor: "border-purple-300 dark:border-purple-800",
    hoverBg: "hover:bg-purple-200 dark:hover:bg-purple-900/40",
  },
  normal: {
    label: "Normal Tissue",
    shortLabel: "Normal",
    color: "text-green-700 dark:text-green-300",
    bgColor: "bg-green-100 dark:bg-green-900/30",
    borderColor: "border-green-300 dark:border-green-800",
    hoverBg: "hover:bg-green-200 dark:hover:bg-green-900/40",
  },
  artifact: {
    label: "Artifact",
    shortLabel: "Artifact",
    color: "text-amber-700 dark:text-amber-300",
    bgColor: "bg-amber-100 dark:bg-amber-900/30",
    borderColor: "border-amber-300 dark:border-amber-800",
    hoverBg: "hover:bg-amber-200 dark:hover:bg-amber-900/40",
  },
  unknown: {
    label: "Unclassified",
    shortLabel: "Unknown",
    color: "text-gray-500 dark:text-gray-400",
    bgColor: "bg-gray-100 dark:bg-navy-800",
    borderColor: "border-gray-200 dark:border-navy-600",
    hoverBg: "hover:bg-gray-200 dark:hover:bg-navy-700",
  },
};

// Get tissue type - prefer backend classification, fall back to inference from description
function getTissueType(patch: EvidencePatch): ExtendedTissueType {
  // Use backend tissue type if available
  if (patch.tissueType && patch.tissueType !== "unknown") {
    return patch.tissueType as ExtendedTissueType;
  }
  // Fall back to inference from morphology description
  if (!patch.morphologyDescription) return "unknown";
  const lower = patch.morphologyDescription.toLowerCase();
  
  if (lower.includes("necrotic") || lower.includes("necrosis")) {
    return "necrosis";
  }
  if (lower.includes("lymphocytic") || lower.includes("inflammatory") || lower.includes("infiltrate")) {
    return "inflammatory";
  }
  if (lower.includes("stromal") || lower.includes("stroma") || lower.includes("desmoplasia")) {
    return "stroma";
  }
  if (lower.includes("carcinoma") || lower.includes("tumor") || lower.includes("papillary") || 
      lower.includes("mitotic") || lower.includes("atypia") || lower.includes("cribriform") ||
      lower.includes("solid growth")) {
    return "tumor";
  }
  if (lower.includes("normal") || lower.includes("benign")) {
    return "normal";
  }
  if (lower.includes("artifact") || lower.includes("blur") || lower.includes("fold")) {
    return "artifact";
  }
  return "unknown";
}

interface EvidencePanelProps {
  patches: EvidencePatch[];
  isLoading?: boolean;
  onPatchClick?: (coords: PatchCoordinates) => void;
  onPatchZoom?: (patch: EvidencePatch) => void;
  onFindSimilar?: (patch: EvidencePatch) => void;
  selectedPatchId?: string;
  error?: string | null;
  onRetry?: () => void;
  isSearchingVisual?: boolean;
}

export function EvidencePanel({
  patches,
  isLoading,
  onPatchClick,
  onPatchZoom,
  onFindSimilar,
  selectedPatchId,
  error,
  onRetry,
  isSearchingVisual,
}: EvidencePanelProps) {
  const [viewMode, setViewMode] = useState<"grid" | "list">("grid");
  const [currentPage, setCurrentPage] = useState(0);
  const [tissueFilter, setTissueFilter] = useState<ExtendedTissueType | null>(null);
  const patchesPerPage = viewMode === "grid" ? 6 : 4;

  // Count patches by tissue type for filter buttons
  const tissueTypeCounts = useMemo(() => {
    const counts: Record<ExtendedTissueType, number> = {
      tumor: 0,
      stroma: 0,
      necrosis: 0,
      inflammatory: 0,
      normal: 0,
      artifact: 0,
      unknown: 0,
    };
    patches.forEach((patch) => {
      const type = getTissueType(patch);
      counts[type]++;
    });
    return counts;
  }, [patches]);

  // Filter patches by tissue type
  const filteredPatches = useMemo(() => {
    if (!tissueFilter) return patches;
    return patches.filter((patch) => getTissueType(patch) === tissueFilter);
  }, [patches, tissueFilter]);

  // Reset page when filter changes
  const handleFilterChange = (type: ExtendedTissueType | null) => {
    setTissueFilter(type);
    setCurrentPage(0);
  };

  const totalPages = Math.ceil(filteredPatches.length / patchesPerPage);
  const visiblePatches = filteredPatches.slice(
    currentPage * patchesPerPage,
    (currentPage + 1) * patchesPerPage
  );

  // Sort patches by attention weight for ranking (based on original list)
  const sortedPatches = [...patches].sort(
    (a, b) => b.attentionWeight - a.attentionWeight
  );
  const getRank = (patchId: string) =>
    sortedPatches.findIndex((p) => p.id === patchId) + 1;

  // Error state
  if (error && !isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Layers className="h-4 w-4 text-red-500" />
            Evidence Patches
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-6">
            <div className="w-12 h-12 mx-auto mb-3 rounded-full bg-red-100 dark:bg-red-900/30 flex items-center justify-center">
              <AlertCircle className="h-6 w-6 text-red-500" />
            </div>
            <p className="text-sm font-medium text-red-700 dark:text-red-300 mb-1">
              Failed to load patches
            </p>
            <p className="text-xs text-red-600 dark:text-red-400 mb-3">{error}</p>
            {onRetry && (
              <Button
                variant="ghost"
                size="sm"
                onClick={onRetry}
                leftIcon={<RefreshCw className="h-3 w-3" />}
              >
                Retry
              </Button>
            )}
          </div>
        </CardContent>
      </Card>
    );
  }

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Layers className="h-4 w-4 text-clinical-600 animate-pulse" />
            Evidence Patches
          </CardTitle>
        </CardHeader>
        <CardContent>
          <SkeletonEvidenceGrid />
          <p className="text-xs text-gray-500 dark:text-gray-400 text-center mt-3 animate-pulse">
            Extracting top attention regions...
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
            <Layers className="h-4 w-4 text-gray-400 dark:text-gray-500" />
            Evidence Patches
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-gray-500 dark:text-gray-400">
            <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gray-100 dark:bg-navy-700 flex items-center justify-center">
              <Grid3X3 className="h-8 w-8 text-gray-400 dark:text-gray-500" />
            </div>
            <p className="text-sm font-medium text-gray-600 dark:text-gray-300">
              No evidence patches
            </p>
            <p className="text-xs mt-1.5 text-gray-500 dark:text-gray-400 max-w-[200px] mx-auto">
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
          <div className="flex items-center gap-1 bg-surface-secondary dark:bg-navy-900 rounded-lg p-0.5">
            <button
              onClick={() => setViewMode("grid")}
              className={cn(
                "p-1.5 rounded-md transition-all",
                viewMode === "grid"
                  ? "bg-white dark:bg-navy-700 shadow-clinical text-clinical-700 dark:text-clinical-300"
                  : "text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200"
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
                  ? "bg-white dark:bg-navy-700 shadow-clinical text-clinical-700 dark:text-clinical-300"
                  : "text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200"
              )}
              title="List view"
            >
              <List className="h-4 w-4" />
            </button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-3 pt-0">
        {/* Tissue Type Filter */}
        <div className="flex flex-wrap gap-1.5 pb-2 border-b border-gray-100 dark:border-navy-700">
          <button
            onClick={() => handleFilterChange(null)}
            className={cn(
              "inline-flex items-center gap-1 px-2 py-1 rounded-md text-xs font-medium transition-all",
              tissueFilter === null
                ? "bg-clinical-100 dark:bg-clinical-900/40 text-clinical-700 dark:text-clinical-300 ring-1 ring-clinical-300 dark:ring-clinical-700"
                : "bg-gray-100 dark:bg-navy-700 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-navy-600"
            )}
          >
            <Filter className="h-3 w-3" />
            All ({patches.length})
          </button>
          {(Object.keys(TISSUE_TYPES) as ExtendedTissueType[])
            .filter((type) => tissueTypeCounts[type] > 0)
            .map((type) => {
              const info = TISSUE_TYPES[type];
              return (
                <button
                  key={type}
                  onClick={() => handleFilterChange(type)}
                  className={cn(
                    "inline-flex items-center gap-1 px-2 py-1 rounded-md text-xs font-medium transition-all border",
                    tissueFilter === type
                      ? cn(info.bgColor, info.color, info.borderColor, "ring-1")
                      : cn("bg-white dark:bg-navy-800 border-gray-200 dark:border-navy-600 text-gray-600 dark:text-gray-300", info.hoverBg)
                  )}
                >
                  <Circle className={cn("h-2 w-2", tissueFilter === type ? "fill-current" : "")} />
                  {info.shortLabel} ({tissueTypeCounts[type]})
                </button>
              );
            })}
          {tissueFilter && (
            <button
              onClick={() => handleFilterChange(null)}
              className="inline-flex items-center gap-1 px-2 py-1 rounded-md text-xs font-medium text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-navy-700"
              title="Clear filter"
            >
              <X className="h-3 w-3" />
            </button>
          )}
        </div>

        {/* Filter active indicator */}
        {tissueFilter && (
          <div className={cn(
            "flex items-center gap-2 px-2 py-1.5 rounded-md text-xs",
            TISSUE_TYPES[tissueFilter].bgColor,
            TISSUE_TYPES[tissueFilter].color
          )}>
            <Filter className="h-3 w-3" />
            <span>Showing {filteredPatches.length} {TISSUE_TYPES[tissueFilter].label.toLowerCase()} patches</span>
          </div>
        )}

        {/* Empty state when filter yields no results */}
        {filteredPatches.length === 0 && tissueFilter && (
          <div className="text-center py-6 text-gray-500 dark:text-gray-400">
            <p className="text-sm">No patches match this filter</p>
            <button
              onClick={() => handleFilterChange(null)}
              className="text-xs text-clinical-600 dark:text-clinical-400 hover:text-clinical-700 dark:hover:text-clinical-300 mt-2"
            >
              Clear filter
            </button>
          </div>
        )}

        {/* Patch Grid/List */}
        {filteredPatches.length > 0 && (
          viewMode === "grid" ? (
            <div className="grid grid-cols-3 gap-2">
              {visiblePatches.map((patch) => (
                <PatchThumbnail
                  key={patch.id}
                  patch={patch}
                  rank={getRank(patch.id)}
                  isSelected={selectedPatchId === patch.id}
                  onClick={() => onPatchClick?.(patch.coordinates)}
                  onZoom={() => onPatchZoom?.(patch)}
                  onFindSimilar={onFindSimilar ? () => onFindSimilar(patch) : undefined}
                  isSearching={isSearchingVisual}
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
                  onZoom={() => onPatchZoom?.(patch)}
                  onFindSimilar={onFindSimilar ? () => onFindSimilar(patch) : undefined}
                  isSearching={isSearchingVisual}
                />
              ))}
            </div>
          )
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
                      ? "bg-clinical-600 dark:bg-clinical-400 w-4"
                      : "bg-gray-300 dark:bg-navy-600 hover:bg-gray-400 dark:hover:bg-navy-500"
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
        <div className="pt-3 border-t border-gray-100 dark:border-navy-700">
          <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400 mb-2">
            <span className="font-medium">Attention Weight</span>
            <span>Click to navigate</span>
          </div>
          <div className="heatmap-legend h-2 rounded-full" />
          <div className="flex justify-between text-2xs text-gray-400 dark:text-gray-500 mt-1">
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
  onZoom?: () => void;
  onFindSimilar?: () => void;
  isSearching?: boolean;
}

function PatchThumbnail({
  patch,
  rank,
  isSelected,
  onClick,
  onZoom,
  onFindSimilar,
  isSearching,
}: PatchThumbnailProps) {
  const [thumbnailError, setThumbnailError] = useState(false);
  const [thumbnailLoading, setThumbnailLoading] = useState(!!patch.thumbnailUrl);
  
  const attentionPercent = Math.round(patch.attentionWeight * 100);
  const attentionColor =
    attentionPercent >= 70
      ? "bg-red-500"
      : attentionPercent >= 40
      ? "bg-amber-500"
      : "bg-blue-500";

  // Get tissue type from backend classification or infer from description
  const tissueType = getTissueType(patch);
  const tissueInfo = TISSUE_TYPES[tissueType];
  
  // Check if we have a valid thumbnail URL
  const hasThumbnail = patch.thumbnailUrl && !thumbnailError;

  return (
    <button
      onClick={onClick}
      className={cn(
        "relative aspect-square rounded-xl overflow-hidden border-2 transition-all duration-200 group",
        "hover:border-clinical-400 hover:shadow-lg hover:shadow-clinical-200/50 hover:scale-[1.03]",
        "focus:outline-none focus:ring-2 focus:ring-clinical-500 dark:focus:ring-clinical-400 focus:ring-offset-2 dark:focus:ring-offset-navy-900",
        "active:scale-[0.98]",
        isSelected
          ? "border-clinical-500 dark:border-clinical-400 ring-2 ring-clinical-200 dark:ring-clinical-700 shadow-lg shadow-clinical-200/40 dark:shadow-clinical-900/50"
          : "border-gray-200 dark:border-navy-600 hover:border-clinical-300 dark:hover:border-clinical-500"
      )}
    >
      {/* Patch Image or Tissue Type Placeholder */}
      {hasThumbnail ? (
        <>
          {thumbnailLoading && (
            <div className={cn(
              "absolute inset-0 flex items-center justify-center animate-pulse",
              tissueInfo.bgColor
            )}>
              <Layers className="h-6 w-6 text-gray-400 dark:text-gray-500" />
            </div>
          )}
          <img
            src={patch.thumbnailUrl}
            alt={`Evidence patch ${rank}`}
            className={cn(
              "w-full h-full object-cover transition-opacity",
              thumbnailLoading ? "opacity-0" : "opacity-100"
            )}
            onLoad={() => setThumbnailLoading(false)}
            onError={() => {
              setThumbnailError(true);
              setThumbnailLoading(false);
            }}
          />
        </>
      ) : (
        <div className={cn(
          "w-full h-full flex flex-col items-center justify-center",
          tissueInfo.bgColor
        )}>
          <Layers className={cn("h-6 w-6 mb-1", tissueInfo.color)} />
          <span className={cn("text-2xs font-medium", tissueInfo.color)}>
            {tissueInfo.shortLabel}
          </span>
        </div>
      )}

      {/* Rank Badge */}
      <div className="absolute top-1.5 left-1.5 bg-gradient-to-br from-navy-800 to-navy-900 backdrop-blur-sm text-white text-xs font-bold px-2 py-0.5 rounded-full shadow-lg border border-white/10">
        #{rank}
      </div>

      {/* Attention Score Indicator */}
      <div className="absolute top-1.5 right-1.5">
        <div
          className={cn(
            "w-7 h-7 rounded-full flex items-center justify-center text-2xs font-bold text-white shadow-lg border-2 border-white/20",
            "transition-transform group-hover:scale-110",
            attentionColor
          )}
        >
          {attentionPercent}
        </div>
      </div>

      {/* Tissue Type Tag */}
      <div className={cn(
        "absolute bottom-1 left-1 right-1 px-1.5 py-0.5 rounded text-2xs font-semibold text-center truncate",
        tissueInfo.bgColor,
        tissueInfo.color
      )}>
        {tissueInfo.label}
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
            <div className="flex items-center gap-1">
              {onFindSimilar && (
                <div
                  role="button"
                  tabIndex={0}
                  onClick={(e) => {
                    e.stopPropagation();
                    if (!isSearching) onFindSimilar();
                  }}
                  onKeyDown={(e) => {
                    if ((e.key === "Enter" || e.key === " ") && !isSearching) {
                      e.stopPropagation();
                      e.preventDefault();
                      onFindSimilar();
                    }
                  }}
                  className={cn(
                    "p-1 rounded transition-colors cursor-pointer",
                    isSearching
                      ? "bg-clinical-500/50 cursor-wait"
                      : "bg-white hover:bg-clinical-500/60"
                  )}
                  title="Find similar patches"
                >
                  <Search className={cn("h-4 w-4", isSearching && "animate-pulse")} />
                </div>
              )}
              <div
                role="button"
                tabIndex={0}
                onClick={(e) => {
                  e.stopPropagation();
                  onZoom?.();
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.stopPropagation();
                    e.preventDefault();
                    onZoom?.();
                  }
                }}
                className="p-1 bg-white dark:bg-navy-800 rounded hover:bg-white dark:hover:bg-navy-700 transition-colors cursor-pointer"
                title="View enlarged"
              >
                <ZoomIn className="h-4 w-4 text-navy-700 dark:text-gray-200" />
              </div>
            </div>
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
  onZoom?: () => void;
  onFindSimilar?: () => void;
  isSearching?: boolean;
}

function PatchListItem({
  patch,
  rank,
  isSelected,
  onClick,
  onZoom,
  onFindSimilar,
  isSearching,
}: PatchListItemProps) {
  const attentionPercent = Math.round(patch.attentionWeight * 100);

  // Get tissue type from backend classification or infer from description
  const tissueType = getTissueType(patch);
  const tissueInfo = TISSUE_TYPES[tissueType];

  return (
    <button
      onClick={onClick}
      className={cn(
        "w-full flex items-center gap-3 p-3 rounded-lg border transition-all text-left group",
        "hover:border-clinical-500 hover:bg-clinical-50/50 dark:hover:bg-clinical-900/30 hover:shadow-clinical",
        "focus:outline-none focus:ring-2 focus:ring-clinical-500",
        isSelected
          ? "border-clinical-600 dark:border-clinical-500 bg-clinical-50 dark:bg-clinical-900/30 ring-1 ring-clinical-200 dark:ring-clinical-700"
          : "border-gray-200 dark:border-navy-600 bg-white dark:bg-navy-800"
      )}
    >
      {/* Thumbnail */}
      <div className="relative w-16 h-16 rounded-lg overflow-hidden shrink-0 border border-gray-200 dark:border-navy-600 group-hover:border-clinical-300 dark:group-hover:border-clinical-500">
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
          <span className="text-sm font-medium text-gray-900 dark:text-gray-100">
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

        {/* Tissue Type Tag */}
        <div className="flex items-center gap-2 mb-1.5">
          <span className={cn(
            "inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium",
            tissueInfo.bgColor,
            tissueInfo.color,
            tissueInfo.borderColor,
            "border"
          )}>
            <Circle className="h-2 w-2 fill-current" />
            {tissueInfo.label}
          </span>
        </div>

        <div className="flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400">
          <MapPin className="h-3 w-3" />
          <span className="font-mono">
            ({patch.coordinates.x.toLocaleString()}, {patch.coordinates.y.toLocaleString()})
          </span>
        </div>

        {patch.morphologyDescription && (
          <p className="text-xs text-gray-600 dark:text-gray-300 mt-1.5 line-clamp-2 leading-relaxed">
            {patch.morphologyDescription}
          </p>
        )}
      </div>

      {/* Action buttons */}
      <div className="shrink-0 flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-all">
        {onFindSimilar && (
          <div
            role="button"
            tabIndex={0}
            onClick={(e) => {
              e.stopPropagation();
              if (!isSearching) onFindSimilar();
            }}
            onKeyDown={(e) => {
              if ((e.key === "Enter" || e.key === " ") && !isSearching) {
                e.stopPropagation();
                e.preventDefault();
                onFindSimilar();
              }
            }}
            className={cn(
              "p-1.5 rounded-lg transition-all cursor-pointer",
              isSearching
                ? "bg-clinical-100 dark:bg-clinical-900/40 cursor-wait"
                : "hover:bg-clinical-100 dark:hover:bg-clinical-900/40"
            )}
            title="Find similar patches"
          >
            <Search className={cn("h-4 w-4 text-clinical-600 dark:text-clinical-400", isSearching && "animate-pulse")} />
          </div>
        )}
        <div
          role="button"
          tabIndex={0}
          onClick={(e) => {
            e.stopPropagation();
            onZoom?.();
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.stopPropagation();
              e.preventDefault();
              onZoom?.();
            }
          }}
          className="p-1.5 rounded-lg hover:bg-clinical-100 dark:hover:bg-clinical-900/40 cursor-pointer"
          title="View enlarged"
        >
          <ZoomIn className="h-4 w-4 text-clinical-600 dark:text-clinical-400" />
        </div>
      </div>
    </button>
  );
}
