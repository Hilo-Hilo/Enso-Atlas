"use client";

import React, { useState, useEffect, useMemo, useCallback } from "react";
import NextImage from "next/image";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { Spinner } from "@/components/ui/Spinner";
import { InlineProgress } from "@/components/ui/ProgressStepper";
import { cn } from "@/lib/utils";
import {
  FolderOpen,
  Image as ImageIcon,
  Upload,
  RefreshCw,
  Check,
  AlertCircle,
  Search,
  SortAsc,
  SortDesc,
  ChevronDown,
  Microscope,
  Hash,
  Layers,
  ShieldCheck,
  ShieldAlert,
  ShieldX,
  User,
  Calendar,
  Activity,
  ImageOff,
  Cpu,
  Clock,
  Pencil,
} from "lucide-react";
import { getSlides, getSlideQC, getThumbnailUrl, renameSlide } from "@/lib/api";
import { cleanSlideName, deduplicateSlides } from "@/lib/slideUtils";
import { ANALYSIS_STEPS } from "@/hooks/useAnalysis";
import type { SlideInfo, SlideQCMetrics, PatientContext } from "@/types";
import { useProject } from "@/contexts/ProjectContext";

interface SlideSelectorProps {
  selectedModels: string[];
  onModelsChange: (models: string[]) => void;
  resolutionLevel: number;
  onResolutionChange: (level: number) => void;
  selectedSlideId: string | null;
  onSlideSelect: (slide: SlideInfo) => void;
  onAnalyze: () => void;
  onGenerateEmbeddings?: () => void;  // Called when user clicks "Generate Embeddings"
  isAnalyzing?: boolean;
  analysisStep?: number;
  // Level 0 embedding state
  isGeneratingEmbeddings?: boolean;
  embeddingProgress?: {
    phase: string;
    progress: number;
    message: string;
  } | null;
  // Embedding status for selected slide (passed through to ModelPicker)
  embeddingStatus?: {
    hasLevel0: boolean;
    hasLevel1: boolean;
  };
}

type SortField = "filename" | "date" | "dimensions";
type SortOrder = "asc" | "desc";

// Lazy loading thumbnail component
function SlideThumbnail({ 
  slideId, 
  thumbnailUrl, 
  filename,
  hasWsi,
  projectId,
  className 
}: { 
  slideId: string; 
  thumbnailUrl?: string; 
  filename: string;
  hasWsi?: boolean;
  projectId?: string;
  className?: string;
}) {
  const [imageLoaded, setImageLoaded] = useState(false);
  const [imageError, setImageError] = useState(false);
  const [isVisible, setIsVisible] = useState(false);

  // Use Intersection Observer for lazy loading
  const imageRef = useCallback((node: HTMLDivElement | null) => {
    if (!node) return;

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setIsVisible(true);
            observer.disconnect();
          }
        });
      },
      { rootMargin: "50px" }
    );

    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  // Always attempt thumbnail fetch when visible. Backend may materialize WSI
  // on-demand even if hasWsi is currently stale/false in cached slide metadata.
  const imgSrc = thumbnailUrl || (isVisible ? getThumbnailUrl(slideId, projectId) : undefined);

  return (
    <div 
      ref={imageRef}
      className={cn(
        "w-16 h-16 rounded-lg bg-gray-100 dark:bg-navy-700 shrink-0 overflow-hidden border border-gray-200 dark:border-navy-600 transition-colors relative",
        className
      )}
    >
      {isVisible && imgSrc && !imageError ? (
        <>
          {/* Loading skeleton while image loads */}
          {!imageLoaded && (
            <div className="absolute inset-0 bg-gray-200 dark:bg-navy-600 animate-pulse flex items-center justify-center">
              <Microscope className="h-5 w-5 text-gray-400 dark:text-gray-500" />
            </div>
          )}
          <NextImage
            src={imgSrc}
            alt={filename}
            fill
            sizes="64px"
            unoptimized
            className={cn(
              "object-cover transition-opacity duration-300",
              imageLoaded ? "opacity-100" : "opacity-0"
            )}
            onLoad={() => setImageLoaded(true)}
            onError={() => setImageError(true)}
            loading="lazy"
          />
        </>
      ) : imageError ? (
        <div className="w-full h-full flex flex-col items-center justify-center bg-gradient-to-br from-slate-100 to-slate-200 dark:from-navy-700 dark:to-navy-600">
          <Layers className="h-5 w-5 text-slate-400 dark:text-slate-500" />
          <span className="text-[7px] font-medium text-slate-400 dark:text-slate-500 mt-0.5 leading-none">Embeddings</span>
        </div>
      ) : (
        <div className="w-full h-full flex items-center justify-center pattern-dots">
          <Microscope className="h-6 w-6 text-gray-400 dark:text-gray-500" />
        </div>
      )}

      {/* Status indicator */}
      {hasWsi === false && imageError && (
        <div className="absolute bottom-0.5 right-0.5 px-1 py-0.5 rounded bg-slate-700/90 text-[8px] leading-none text-white">
          Emb
        </div>
      )}
    </div>
  );
}

export function SlideSelector({
  selectedModels,
  onModelsChange,
  resolutionLevel,
  onResolutionChange,
  selectedSlideId,
  onSlideSelect,
  onAnalyze,
  onGenerateEmbeddings,
  isAnalyzing,
  analysisStep = -1,
  isGeneratingEmbeddings = false,
  embeddingProgress = null,
  embeddingStatus,
}: SlideSelectorProps) {
  const { currentProject } = useProject();
  const [slides, setSlides] = useState<SlideInfo[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [sortField, setSortField] = useState<SortField>("filename");
  const [sortOrder, setSortOrder] = useState<SortOrder>("asc");
  const [showFilters, setShowFilters] = useState(false);
  const [showAllCases, setShowAllCases] = useState(false);
  const [qcMetrics, setQcMetrics] = useState<Record<string, SlideQCMetrics>>({});

  const loadSlides = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await getSlides({ projectId: currentProject.id });
      // Deduplicate slides (remove non-UUID duplicates) and filter test files
      setSlides(deduplicateSlides(response.slides));
      // QC metrics are fetched lazily when a slide is selected (not all 208 at once).
      // This avoids 208 parallel requests that block the sidebar from loading.
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load slides");
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadSlides();
    // Re-fetch slides when the active project changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentProject.id]);

  // Filter and sort slides
  const filteredSlides = useMemo(() => {
    let result = [...slides];

    // Search filter
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      result = result.filter(
        (slide) =>
          slide.filename.toLowerCase().includes(query) ||
          slide.id.toLowerCase().includes(query) ||
          cleanSlideName(slide.filename).toLowerCase().includes(query) ||
          (slide.displayName && slide.displayName.toLowerCase().includes(query))
      );
    }

    // Sort
    result.sort((a, b) => {
      let comparison = 0;
      switch (sortField) {
        case "filename":
          comparison = a.filename.localeCompare(b.filename);
          break;
        case "date":
          comparison =
            new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime();
          break;
        case "dimensions":
          comparison =
            a.dimensions.width * a.dimensions.height -
            b.dimensions.width * b.dimensions.height;
          break;
      }
      return sortOrder === "asc" ? comparison : -comparison;
    });

    return result;
  }, [slides, searchQuery, sortField, sortOrder]);

  const selectedSlide = slides.find((s) => s.id === selectedSlideId);
  const visibleSlides = showAllCases ? filteredSlides : filteredSlides.slice(0, 5);
  const visibleCaseRows = Math.min(visibleSlides.length, 5);
  const caseListHeaderHeightPx =
    filteredSlides.length > 0 ? 30 : 0; // "N cases found" row
  const caseRowHeightPx = 41; // row height incl. border
  const caseListHeightPx = caseListHeaderHeightPx + visibleCaseRows * caseRowHeightPx;
  const shouldScrollCaseList = filteredSlides.length > 5;

  useEffect(() => {
    setShowAllCases(false);
  }, [searchQuery, sortField, sortOrder, slides.length]);

  const toggleSort = (field: SortField) => {
    if (sortField === field) {
      setSortOrder(sortOrder === "asc" ? "desc" : "asc");
    } else {
      setSortField(field);
      setSortOrder("asc");
    }
  };

  return (
    <Card className="flex flex-col w-full max-w-full overflow-hidden">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <FolderOpen className="h-4 w-4 text-clinical-600" />
            Case Selection
          </CardTitle>
          <div className="flex items-center gap-1">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowFilters(!showFilters)}
              className={cn("p-1.5", showFilters && "bg-clinical-50 dark:bg-clinical-900/30")}
              title="Sort options"
            >
              <SortAsc className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={loadSlides}
              disabled={isLoading}
              className="p-1.5"
              title="Refresh list"
            >
              <RefreshCw
                className={cn("h-4 w-4", isLoading && "animate-spin")}
              />
            </Button>
          </div>
        </div>

        {/* Search Input */}
        <div className="relative mt-3">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
          <input
            type="text"
            placeholder="Search by filename or case ID..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="search-input"
          />
        </div>

        {/* Sort Options */}
        {showFilters && (
          <div className="mt-3 p-3 bg-surface-secondary dark:bg-navy-900 rounded-lg border border-surface-border dark:border-navy-600 animate-fade-in">
            <p className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-2">
              Sort By
            </p>
            <div className="flex flex-wrap gap-2">
              <SortButton
                label="Name"
                isActive={sortField === "filename"}
                order={sortField === "filename" ? sortOrder : undefined}
                onClick={() => toggleSort("filename")}
              />
              <SortButton
                label="Date"
                isActive={sortField === "date"}
                order={sortField === "date" ? sortOrder : undefined}
                onClick={() => toggleSort("date")}
              />
              <SortButton
                label="Size"
                isActive={sortField === "dimensions"}
                order={sortField === "dimensions" ? sortOrder : undefined}
                onClick={() => toggleSort("dimensions")}
              />
            </div>
          </div>
        )}
      </CardHeader>

      <CardContent className="flex flex-col space-y-3 pt-0 overflow-x-hidden">
        {/* Error State */}
        {error && (
          <div className="p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg animate-fade-in">
            <div className="flex items-start gap-2">
              <AlertCircle className="h-4 w-4 text-red-500 dark:text-red-400 shrink-0 mt-0.5" />
              <div className="flex-1">
                <span className="text-sm text-red-700 dark:text-red-300 block">{error}</span>
                <button
                  onClick={loadSlides}
                  className="mt-2 text-xs font-medium text-red-700 dark:text-red-300 hover:text-red-900 dark:hover:text-red-100 underline"
                >
                  Retry
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Loading State */}
        {isLoading && (
          <div
            className="w-full grid place-items-center text-center"
            style={{ minHeight: `${Math.max(caseListHeightPx, 220)}px` }}
          >
            <div className="flex flex-col items-center justify-center">
              <Spinner size="md" className="mx-auto" />
              <p className="text-sm text-gray-500 dark:text-gray-400 mt-2">Loading cases...</p>
            </div>
          </div>
        )}

        {/* Slide List */}
        {!isLoading && !error && (
          <div
            className={cn(
              "overflow-x-hidden",
              shouldScrollCaseList ? "overflow-y-auto scrollbar-hide" : "overflow-y-hidden"
            )}
            style={
              filteredSlides.length > 0
                ? { height: `${caseListHeightPx}px` }
                : undefined
            }
          >
            {filteredSlides.length === 0 ? (
              <div className="text-center py-8 text-gray-500 dark:text-gray-400">
                {slides.length === 0 ? (
                  <>
                    <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gray-100 dark:bg-navy-700 flex items-center justify-center">
                      <ImageIcon className="h-8 w-8 text-gray-400 dark:text-gray-500" />
                    </div>
                    <p className="text-sm font-medium text-gray-600 dark:text-gray-300">
                      No slides available
                    </p>
                    <p className="text-xs mt-1">Upload a WSI to get started</p>
                  </>
                ) : (
                  <>
                    <Search className="h-8 w-8 mx-auto mb-2 text-gray-400 dark:text-gray-500" />
                    <p className="text-sm">No matching slides</p>
                    <p className="text-xs mt-1">Try a different search term</p>
                  </>
                )}
              </div>
            ) : (
              <>
                <div className="flex items-center justify-between px-1 pb-2">
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    {filteredSlides.length} case{filteredSlides.length !== 1 ? "s" : ""} found
                  </p>
                  {filteredSlides.length > 5 && (
                    <button
                      type="button"
                      onClick={() => setShowAllCases((prev) => !prev)}
                      className="text-xs text-clinical-600 hover:text-clinical-700 dark:text-clinical-400 dark:hover:text-clinical-300 font-medium"
                    >
                      {showAllCases ? "Show less" : `Show all (${filteredSlides.length})`}
                    </button>
                  )}
                </div>
                <div className="bg-white dark:bg-navy-800 rounded-lg overflow-hidden">
                  {visibleSlides.map((slide) => (
                    <SlideItem
                      key={slide.id}
                      slide={slide}
                      isSelected={selectedSlideId === slide.id}
                      onClick={() => onSlideSelect(slide)}
                      qcMetrics={qcMetrics[slide.id]}
                      currentProjectId={currentProject.id}
                    />
                  ))}
                </div>
              </>
            )}
          </div>
        )}

        {/* Selected Slide Summary */}
        {selectedSlide && (
          <div className="px-3 py-2 bg-clinical-50 dark:bg-clinical-900/20 border border-clinical-200 dark:border-clinical-800 rounded-md animate-slide-up">
            <p className="text-2xs uppercase tracking-wide text-clinical-700 dark:text-clinical-300 font-semibold mb-1">
              Selected case
            </p>
            <p className="text-sm font-medium text-clinical-900 dark:text-clinical-100 truncate" title={selectedSlide.filename}>
              {selectedSlide.displayName || cleanSlideName(selectedSlide.filename)}
            </p>
            <p className="text-2xs text-clinical-600 dark:text-clinical-400 font-mono truncate" title={selectedSlide.id}>
              {cleanSlideName(selectedSlide.id)}
            </p>
          </div>
        )}

        {/* Analysis controls moved to separate AnalysisControls panel */}
      </CardContent>
    </Card>
  );
}

// Sort Button Component
function SortButton({
  label,
  isActive,
  order,
  onClick,
}: {
  label: string;
  isActive: boolean;
  order?: SortOrder;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "flex items-center gap-1 px-2.5 py-1.5 rounded-md text-xs font-medium transition-colors",
        isActive
          ? "bg-clinical-600 text-white"
          : "bg-white dark:bg-navy-700 text-gray-600 dark:text-gray-300 border border-gray-200 dark:border-navy-600 hover:bg-gray-50 dark:hover:bg-navy-600"
      )}
    >
      {label}
      {isActive && order && (
        order === "asc" ? (
          <SortAsc className="h-3 w-3" />
        ) : (
          <SortDesc className="h-3 w-3" />
        )
      )}
    </button>
  );
}

// QC Badge Component
function QCBadge({ qc }: { qc: SlideQCMetrics }) {
  const getQCConfig = () => {
    switch (qc.overallQuality) {
      case "good":
        return {
          icon: ShieldCheck,
          color: "text-green-600 dark:text-green-400",
          bg: "bg-green-50 dark:bg-green-900/30",
          border: "border-green-200 dark:border-green-800",
          label: "Good Quality",
        };
      case "acceptable":
        return {
          icon: ShieldAlert,
          color: "text-yellow-600 dark:text-yellow-400",
          bg: "bg-yellow-50 dark:bg-yellow-900/30",
          border: "border-yellow-200 dark:border-yellow-800",
          label: "Acceptable",
        };
      case "poor":
        return {
          icon: ShieldX,
          color: "text-red-600 dark:text-red-400",
          bg: "bg-red-50 dark:bg-red-900/30",
          border: "border-red-200 dark:border-red-800",
          label: "Poor Quality",
        };
    }
  };

  const config = getQCConfig();
  const Icon = config.icon;

  // Build tooltip details
  const issues = [];
  if (qc.blurScore > 0.2) issues.push("Blur detected");
  if (qc.tissueCoverage < 0.5) issues.push("Low tissue coverage");
  if (qc.stainUniformity < 0.6) issues.push("Uneven staining");
  if (qc.artifactDetected) issues.push("Artifacts present");
  if (qc.penMarks) issues.push("Pen marks");
  if (qc.foldDetected) issues.push("Tissue folds");

  const tooltipText = [
    config.label,
    `Tissue: ${Math.round(qc.tissueCoverage * 100)}%`,
    `Sharpness: ${Math.round((1 - qc.blurScore) * 100)}%`,
    `Stain: ${Math.round(qc.stainUniformity * 100)}%`,
    ...(issues.length > 0 ? [`Issues: ${issues.join(", ")}`] : []),
  ].join("\n");

  return (
    <div
      className={cn(
        "flex items-center gap-1 px-1.5 py-0.5 rounded text-2xs font-medium",
        config.bg,
        config.border,
        "border"
      )}
      title={tooltipText}
    >
      <Icon className={cn("h-3 w-3", config.color)} />
      <span className={config.color}>{qc.overallQuality.toUpperCase()}</span>
    </div>
  );
}

// Slide Item Component
interface SlideItemProps {
  slide: SlideInfo;
  isSelected: boolean;
  onClick: () => void;
  qcMetrics?: SlideQCMetrics;
  currentProjectId?: string;
}

// Inline rename component for the selected slide
function SlideRenameInline({
  slideId,
  currentName,
  onRenamed,
}: {
  slideId: string;
  currentName: string;
  onRenamed: (newName: string | null) => void;
}) {
  const [isEditing, setIsEditing] = useState(false);
  const [value, setValue] = useState(currentName);

  useEffect(() => {
    setValue(currentName);
  }, [currentName, slideId]);

  const handleSave = async () => {
    const trimmed = value.trim();
    try {
      await renameSlide(slideId, trimmed || null);
      onRenamed(trimmed || null);
    } catch (err) {
      console.error("Failed to rename slide:", err);
    }
    setIsEditing(false);
  };

  if (!isEditing) {
    return (
      <button
        onClick={() => setIsEditing(true)}
        className="mt-2 flex items-center gap-1.5 text-xs text-clinical-600 dark:text-clinical-400 hover:text-clinical-800 dark:hover:text-clinical-300 transition-colors"
      >
        <Pencil className="h-3 w-3" />
        <span>{currentName ? "Edit alias" : "Add alias"}</span>
      </button>
    );
  }

  return (
    <div className="mt-2 flex items-center gap-1.5">
      <input
        type="text"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") handleSave();
          if (e.key === "Escape") setIsEditing(false);
        }}
        placeholder="Display name..."
        autoFocus
        className="flex-1 text-xs px-2 py-1 border border-clinical-300 dark:border-clinical-700 rounded bg-white dark:bg-navy-800 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-1 focus:ring-clinical-500"
      />
      <button
        onClick={handleSave}
        className="text-xs px-2 py-1 bg-clinical-600 text-white rounded hover:bg-clinical-700"
      >
        Save
      </button>
      <button
        onClick={() => setIsEditing(false)}
        className="text-xs px-2 py-1 text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200"
      >
        Cancel
      </button>
    </div>
  );
}

function SlideItem({ slide, isSelected, onClick }: SlideItemProps) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "w-full flex items-center justify-between gap-2 px-3 py-2 text-left transition-colors border-b border-gray-200 dark:border-navy-600 last:border-b-0",
        isSelected ? "bg-clinical-50 dark:bg-clinical-900/20" : "bg-white dark:bg-navy-800 hover:bg-gray-50 dark:hover:bg-navy-700"
      )}
    >
      <div className="flex-1 min-w-0 pr-2">
        <p className="text-sm text-gray-900 dark:text-gray-100 truncate" title={slide.filename}>
          {slide.displayName || cleanSlideName(slide.filename)}
        </p>
      </div>
      {isSelected && (
        <div className="shrink-0 h-2.5 w-2.5 rounded-full bg-clinical-600" />
      )}
      {!isSelected && slide.hasEmbeddings && (
        <div className="shrink-0 h-2.5 w-2.5 rounded-full bg-sky-400/70" />
      )}
      {!isSelected && !slide.hasEmbeddings && (
        <div className="shrink-0 h-2.5 w-2.5 rounded-full bg-gray-300" />
      )}
    </button>
  );
}
