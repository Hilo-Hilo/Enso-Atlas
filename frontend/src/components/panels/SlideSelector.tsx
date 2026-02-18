"use client";

import { ModelPicker } from "./ModelPicker";
import React, { useState, useEffect, useMemo, useCallback } from "react";
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
  forceReembed: boolean;
  onForceReembedChange: (force: boolean) => void;
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
  className 
}: { 
  slideId: string; 
  thumbnailUrl?: string; 
  filename: string;
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

  // Get the thumbnail URL - either from the slide data or construct from API
  const imgSrc = thumbnailUrl || (isVisible ? getThumbnailUrl(slideId) : undefined);

  return (
    <div 
      ref={imageRef}
      className={cn(
        "w-16 h-16 rounded-lg bg-gray-100 shrink-0 overflow-hidden border border-gray-200 transition-colors relative",
        className
      )}
    >
      {isVisible && imgSrc && !imageError ? (
        <>
          {/* Loading skeleton while image loads */}
          {!imageLoaded && (
            <div className="absolute inset-0 bg-gray-200 animate-pulse flex items-center justify-center">
              <Microscope className="h-5 w-5 text-gray-400" />
            </div>
          )}
          <img
            src={imgSrc}
            alt={filename}
            className={cn(
              "w-full h-full object-cover transition-opacity duration-300",
              imageLoaded ? "opacity-100" : "opacity-0"
            )}
            onLoad={() => setImageLoaded(true)}
            onError={() => setImageError(true)}
            loading="lazy"
          />
        </>
      ) : imageError ? (
        <div className="w-full h-full flex flex-col items-center justify-center bg-gradient-to-br from-slate-100 to-slate-200">
          <Layers className="h-5 w-5 text-slate-400" />
          <span className="text-[7px] font-medium text-slate-400 mt-0.5 leading-none">Embeddings</span>
        </div>
      ) : (
        <div className="w-full h-full flex items-center justify-center pattern-dots">
          <Microscope className="h-6 w-6 text-gray-400" />
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
  forceReembed,
  onForceReembedChange,
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

  const toggleSort = (field: SortField) => {
    if (sortField === field) {
      setSortOrder(sortOrder === "asc" ? "desc" : "asc");
    } else {
      setSortField(field);
      setSortOrder("asc");
    }
  };

  return (
    <Card className="flex flex-col h-full">
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
              className={cn("p-1.5", showFilters && "bg-clinical-50")}
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
          <div className="mt-3 p-3 bg-surface-secondary rounded-lg border border-surface-border animate-fade-in">
            <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-2">
              Sort By
            </p>
            <div className="flex gap-2">
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

      <CardContent className="flex-1 flex flex-col min-h-0 space-y-4 pt-0">
        {/* Error State */}
        {error && (
          <div className="p-3 bg-red-50 border border-red-200 rounded-lg animate-fade-in">
            <div className="flex items-start gap-2">
              <AlertCircle className="h-4 w-4 text-red-500 shrink-0 mt-0.5" />
              <div className="flex-1">
                <span className="text-sm text-red-700 block">{error}</span>
                <button
                  onClick={loadSlides}
                  className="mt-2 text-xs font-medium text-red-700 hover:text-red-900 underline"
                >
                  Retry
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Loading State */}
        {isLoading && (
          <div className="flex-1 flex items-center justify-center py-8">
            <div className="text-center">
              <Spinner size="md" />
              <p className="text-sm text-gray-500 mt-2">Loading cases...</p>
            </div>
          </div>
        )}

        {/* Slide List */}
        {!isLoading && !error && (
          <div className="flex-1 overflow-y-auto space-y-2 scrollbar-hide">
            {filteredSlides.length === 0 ? (
              <div className="text-center py-8 text-gray-500">
                {slides.length === 0 ? (
                  <>
                    <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gray-100 flex items-center justify-center">
                      <ImageIcon className="h-8 w-8 text-gray-400" />
                    </div>
                    <p className="text-sm font-medium text-gray-600">
                      No slides available
                    </p>
                    <p className="text-xs mt-1">Upload a WSI to get started</p>
                  </>
                ) : (
                  <>
                    <Search className="h-8 w-8 mx-auto mb-2 text-gray-400" />
                    <p className="text-sm">No matching slides</p>
                    <p className="text-xs mt-1">Try a different search term</p>
                  </>
                )}
              </div>
            ) : (
              <>
                <p className="text-xs text-gray-500 px-1">
                  {filteredSlides.length} case{filteredSlides.length !== 1 ? "s" : ""} found
                </p>
                {filteredSlides.map((slide) => (
                  <SlideItem
                    key={slide.id}
                    slide={slide}
                    isSelected={selectedSlideId === slide.id}
                    onClick={() => onSlideSelect(slide)}
                    qcMetrics={qcMetrics[slide.id]}
                  />
                ))}
              </>
            )}
          </div>
        )}

        {/* Selected Slide Summary */}
        {selectedSlide && (
          <div className="p-4 bg-clinical-50 border border-clinical-200 rounded-lg animate-slide-up">
            <div className="flex items-center gap-2 mb-3">
              <SlideThumbnail
                slideId={selectedSlide.id}
                thumbnailUrl={selectedSlide.thumbnailUrl}
                filename={selectedSlide.filename}
                className="w-12 h-12 border-clinical-300"
              />
              <div className="flex-1 min-w-0">
                <p className="text-sm font-semibold text-clinical-900 truncate" title={selectedSlide.filename}>
                  {selectedSlide.displayName || cleanSlideName(selectedSlide.filename)}
                </p>
                {selectedSlide.displayName && (
                  <p className="text-2xs text-clinical-500 truncate" title={selectedSlide.filename}>{cleanSlideName(selectedSlide.filename)}</p>
                )}
                <p className="text-xs text-clinical-600 font-mono" title={selectedSlide.id}>
                  {cleanSlideName(selectedSlide.id)}
                </p>
              </div>
              <Check className="h-5 w-5 text-clinical-600 shrink-0" />
            </div>

            {/* Slide Metadata */}
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="flex items-center gap-1.5 text-clinical-700">
                <Layers className="h-3 w-3" />
                <span>
                  {selectedSlide.dimensions.width.toLocaleString()} x{" "}
                  {selectedSlide.dimensions.height.toLocaleString()}
                </span>
              </div>
              {selectedSlide.magnification && (
                <div className="flex items-center gap-1.5 text-clinical-700">
                  <Hash className="h-3 w-3" />
                  <span>{selectedSlide.magnification}x magnification</span>
                </div>
              )}
              {selectedSlide.numPatches && (
                <div className="flex items-center gap-1.5 text-clinical-700">
                  <ImageIcon className="h-3 w-3" />
                  <span>{selectedSlide.numPatches.toLocaleString()} patches</span>
                </div>
              )}
              {selectedSlide.label && (
                <div className="col-span-2">
                  <Badge
                    variant={
                      selectedSlide.label.toLowerCase() === (currentProject.positive_class || "").toLowerCase() || selectedSlide.label === "1" || selectedSlide.label.toLowerCase() === "sensitive" || selectedSlide.label.toLowerCase().includes("responder")
                        ? "success"
                        : selectedSlide.label.toLowerCase() === (currentProject.classes?.find(c => c !== currentProject.positive_class) || "").toLowerCase() || selectedSlide.label === "0" || selectedSlide.label.toLowerCase() === "resistant" || selectedSlide.label.toLowerCase().includes("non")
                        ? "danger"
                        : "default"
                    }
                    size="sm"
                  >
                    {selectedSlide.label}
                  </Badge>
                </div>
              )}
            </div>

            {/* Patient Demographics Card */}
            {selectedSlide.patient && (
              <div className="mt-3 p-3 bg-white border border-clinical-200 rounded-lg">
                <div className="flex items-center gap-2 mb-2">
                  <User className="h-3.5 w-3.5 text-clinical-600" />
                  <span className="text-xs font-semibold text-clinical-800">Patient Context</span>
                </div>
                <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs">
                  {selectedSlide.patient.age && (
                    <div className="flex items-center gap-1.5 text-gray-700">
                      <Calendar className="h-3 w-3 text-gray-400" />
                      <span>Age: {selectedSlide.patient.age}</span>
                    </div>
                  )}
                  {selectedSlide.patient.sex && (
                    <div className="flex items-center gap-1.5 text-gray-700">
                      <User className="h-3 w-3 text-gray-400" />
                      <span>Sex: {selectedSlide.patient.sex}</span>
                    </div>
                  )}
                  {selectedSlide.patient.stage && (
                    <div className="flex items-center gap-1.5 text-gray-700">
                      <Activity className="h-3 w-3 text-gray-400" />
                      <span>Stage: {selectedSlide.patient.stage}</span>
                    </div>
                  )}
                  {selectedSlide.patient.grade && (
                    <div className="flex items-center gap-1.5 text-gray-700">
                      <Hash className="h-3 w-3 text-gray-400" />
                      <span>Grade: {selectedSlide.patient.grade}</span>
                    </div>
                  )}
                  {selectedSlide.patient.prior_lines !== undefined && (
                    <div className="flex items-center gap-1.5 text-gray-700 col-span-2">
                      <Activity className="h-3 w-3 text-gray-400" />
                      <span>Prior Lines: {selectedSlide.patient.prior_lines}</span>
                    </div>
                  )}
                  {selectedSlide.patient.histology && (
                    <div className="col-span-2 mt-1">
                      <Badge variant="info" size="sm">
                        {selectedSlide.patient.histology}
                      </Badge>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Inline Rename */}
            <SlideRenameInline
              slideId={selectedSlide.id}
              currentName={selectedSlide.displayName || ""}
              onRenamed={(newName) => {
                setSlides((prev) =>
                  prev.map((s) =>
                    s.id === selectedSlide.id
                      ? { ...s, displayName: newName || null }
                      : s
                  )
                );
              }}
            />
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
          : "bg-white text-gray-600 border border-gray-200 hover:bg-gray-50"
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
          color: "text-green-600",
          bg: "bg-green-50",
          border: "border-green-200",
          label: "Good Quality",
        };
      case "acceptable":
        return {
          icon: ShieldAlert,
          color: "text-yellow-600",
          bg: "bg-yellow-50",
          border: "border-yellow-200",
          label: "Acceptable",
        };
      case "poor":
        return {
          icon: ShieldX,
          color: "text-red-600",
          bg: "bg-red-50",
          border: "border-red-200",
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
        className="mt-2 flex items-center gap-1.5 text-xs text-clinical-600 hover:text-clinical-800 transition-colors"
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
        className="flex-1 text-xs px-2 py-1 border border-clinical-300 rounded focus:outline-none focus:ring-1 focus:ring-clinical-500"
      />
      <button
        onClick={handleSave}
        className="text-xs px-2 py-1 bg-clinical-600 text-white rounded hover:bg-clinical-700"
      >
        Save
      </button>
      <button
        onClick={() => setIsEditing(false)}
        className="text-xs px-2 py-1 text-gray-500 hover:text-gray-700"
      >
        Cancel
      </button>
    </div>
  );
}

function SlideItem({ slide, isSelected, onClick, qcMetrics }: SlideItemProps) {
  const formattedDate = new Date(slide.createdAt).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
  });

  return (
    <button
      onClick={onClick}
      className={cn(
        "w-full flex items-center gap-3 p-3 rounded-lg border transition-all text-left group",
        "hover:border-clinical-400 hover:bg-clinical-50/50 hover:shadow-clinical",
        isSelected
          ? "border-clinical-500 bg-clinical-50 ring-1 ring-clinical-200"
          : "border-gray-200 bg-white"
      )}
    >
      {/* Thumbnail with lazy loading */}
      <SlideThumbnail
        slideId={slide.id}
        thumbnailUrl={slide.thumbnailUrl}
        filename={slide.filename}
        className="group-hover:border-clinical-300"
      />

      {/* Info */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <p className="text-sm font-medium text-gray-900 truncate flex-1" title={slide.filename}>
            {slide.displayName || cleanSlideName(slide.filename)}
          </p>
          {qcMetrics && <QCBadge qc={qcMetrics} />}
        </div>
        {slide.displayName && (
          <p className="text-2xs text-gray-400 truncate" title={slide.filename}>{cleanSlideName(slide.filename)}</p>
        )}
        <div className="flex items-center gap-2 mt-1.5">
          <span className="text-xs text-gray-500 font-mono">
            {slide.dimensions.width.toLocaleString()} x {slide.dimensions.height.toLocaleString()}
          </span>
          {slide.magnification && (
            <Badge variant="default" size="sm" className="text-2xs">
              {slide.magnification}x
            </Badge>
          )}
        </div>
        <div className="flex items-center gap-2 mt-1">
          <span className="text-2xs text-gray-400">{formattedDate}</span>
          {slide.hasEmbeddings && (
            <span className="text-2xs text-clinical-600 font-medium">
              Embeddings ready
            </span>
          )}
        </div>
      </div>

      {/* Selection Indicator */}
      {isSelected && (
        <div className="shrink-0 w-6 h-6 rounded-full bg-clinical-600 flex items-center justify-center">
          <Check className="h-4 w-4 text-white" />
        </div>
      )}
    </button>
  );
}
