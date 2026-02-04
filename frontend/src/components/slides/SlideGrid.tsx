"use client";

import React from "react";
import { Card, CardContent } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Skeleton } from "@/components/ui/Skeleton";
import { cn, truncateText } from "@/lib/utils";
import { getThumbnailUrl } from "@/lib/api";
import type { SlideInfo, ExtendedSlideInfo } from "@/types";
import {
  Star,
  MoreVertical,
  Microscope,
  Eye,
  FolderPlus,
  Trash2,
  Layers,
  CheckCircle2,
  Circle,
} from "lucide-react";

interface SlideGridProps {
  slides: (SlideInfo & Partial<ExtendedSlideInfo>)[];
  selectedIds: Set<string>;
  onSelectSlide: (id: string, selected: boolean) => void;
  onSelectAll: (selected: boolean) => void;
  onStarSlide: (id: string) => void;
  onViewSlide: (id: string) => void;
  onAnalyzeSlide: (id: string) => void;
  onAddToGroup: (id: string) => void;
  onDeleteSlide: (id: string) => void;
  isLoading?: boolean;
}

// Tag colors
const TAG_COLORS: Record<string, string> = {
  red: "bg-red-100 text-red-700",
  orange: "bg-orange-100 text-orange-700",
  amber: "bg-amber-100 text-amber-700",
  green: "bg-green-100 text-green-700",
  teal: "bg-teal-100 text-teal-700",
  blue: "bg-blue-100 text-blue-700",
  indigo: "bg-indigo-100 text-indigo-700",
  violet: "bg-violet-100 text-violet-700",
  purple: "bg-purple-100 text-purple-700",
  pink: "bg-pink-100 text-pink-700",
};

function SlideCardSkeleton() {
  return (
    <Card className="overflow-hidden">
      <div className="aspect-[4/3] bg-gray-100">
        <Skeleton className="w-full h-full" />
      </div>
      <CardContent className="p-3 space-y-2">
        <Skeleton className="h-4 w-3/4" />
        <div className="flex gap-1">
          <Skeleton className="h-5 w-16 rounded-full" />
          <Skeleton className="h-5 w-12 rounded-full" />
        </div>
        <div className="flex justify-between items-center">
          <Skeleton className="h-4 w-20" />
          <Skeleton className="h-6 w-6 rounded-full" />
        </div>
      </CardContent>
    </Card>
  );
}

function SlideCard({
  slide,
  isSelected,
  onSelect,
  onStar,
  onView,
  onAnalyze,
  onAddToGroup,
  onDelete,
}: {
  slide: SlideInfo & Partial<ExtendedSlideInfo>;
  isSelected: boolean;
  onSelect: (selected: boolean) => void;
  onStar: () => void;
  onView: () => void;
  onAnalyze: () => void;
  onAddToGroup: () => void;
  onDelete: () => void;
}) {
  const [showMenu, setShowMenu] = React.useState(false);
  const [imageError, setImageError] = React.useState(false);
  const [isImageLoaded, setIsImageLoaded] = React.useState(false);
  const menuRef = React.useRef<HTMLDivElement>(null);

  // Close menu when clicking outside
  React.useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setShowMenu(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  React.useEffect(() => {
    setImageError(false);
    setIsImageLoaded(false);
  }, [slide.id]);

  const thumbnailUrl = getThumbnailUrl(slide.id);

  return (
    <Card
      className={cn(
        "overflow-hidden transition-all duration-200 group cursor-pointer",
        isSelected && "ring-2 ring-clinical-500 shadow-lg"
      )}
    >
      {/* Thumbnail */}
      <div className="relative aspect-[4/3] bg-gray-100 overflow-hidden">
        {!imageError && (
          <img
            src={thumbnailUrl}
            alt={slide.filename}
            className={cn(
              "w-full h-full object-cover group-hover:scale-105 transition-all duration-300",
              isImageLoaded ? "opacity-100" : "opacity-0"
            )}
            onLoad={() => setIsImageLoaded(true)}
            onError={() => {
              setImageError(true);
              setIsImageLoaded(true);
            }}
            loading="lazy"
            decoding="async"
          />
        )}
        {!isImageLoaded && !imageError && (
          <div className="absolute inset-0">
            <Skeleton className="w-full h-full" />
          </div>
        )}
        {imageError && (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-50">
            <Microscope className="h-12 w-12 text-gray-300" />
          </div>
        )}

        {/* Selection checkbox overlay */}
        <button
          onClick={(e) => {
            e.stopPropagation();
            onSelect(!isSelected);
          }}
          className={cn(
            "absolute top-2 left-2 p-1 rounded-full transition-all",
            isSelected
              ? "bg-clinical-500 text-white"
              : "bg-white/80 text-gray-400 opacity-0 group-hover:opacity-100 hover:bg-white"
          )}
        >
          {isSelected ? (
            <CheckCircle2 className="h-5 w-5" />
          ) : (
            <Circle className="h-5 w-5" />
          )}
        </button>

        {/* Star button */}
        <button
          onClick={(e) => {
            e.stopPropagation();
            onStar();
          }}
          className={cn(
            "absolute top-2 right-2 p-1 rounded-full transition-all",
            slide.starred
              ? "bg-amber-100 text-amber-500"
              : "bg-white/80 text-gray-400 opacity-0 group-hover:opacity-100 hover:bg-white hover:text-amber-500"
          )}
        >
          <Star className={cn("h-5 w-5", slide.starred && "fill-amber-500")} />
        </button>

        {/* Label badge */}
        {slide.label && (
          <div className="absolute bottom-2 left-2">
            <Badge
              variant={slide.label === "platinum_sensitive" ? "success" : "danger"}
              size="sm"
            >
              {slide.label === "platinum_sensitive" ? "Sensitive" : "Resistant"}
            </Badge>
          </div>
        )}

        {/* Embeddings indicator */}
        {slide.hasEmbeddings && (
          <div className="absolute bottom-2 right-2">
            <div className="p-1 bg-clinical-100 rounded-full" title="Has embeddings">
              <Layers className="h-3 w-3 text-clinical-600" />
            </div>
          </div>
        )}
      </div>

      {/* Content */}
      <CardContent className="p-3" onClick={onView}>
        {/* Slide name */}
        <h3 className="font-medium text-sm text-gray-900 truncate mb-1.5" title={slide.filename}>
          {truncateText(slide.id, 30)}
        </h3>

        {/* Tags */}
        {slide.tags && slide.tags.length > 0 && (
          <div className="flex flex-wrap gap-1 mb-2">
            {slide.tags.slice(0, 3).map((tag, idx) => (
              <span
                key={tag}
                className={cn(
                  "px-1.5 py-0.5 text-xs rounded-full",
                  TAG_COLORS[Object.keys(TAG_COLORS)[idx % Object.keys(TAG_COLORS).length]]
                )}
              >
                {tag}
              </span>
            ))}
            {slide.tags.length > 3 && (
              <span className="px-1.5 py-0.5 text-xs bg-gray-100 text-gray-500 rounded-full">
                +{slide.tags.length - 3}
              </span>
            )}
          </div>
        )}

        {/* Footer */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-xs text-gray-500">
            {slide.numPatches !== undefined && (
              <span className="flex items-center gap-1">
                <Layers className="h-3 w-3" />
                {slide.numPatches.toLocaleString()}
              </span>
            )}
          </div>

          {/* Quick actions menu */}
          <div className="relative" ref={menuRef}>
            <button
              onClick={(e) => {
                e.stopPropagation();
                setShowMenu(!showMenu);
              }}
              className="p-1 hover:bg-gray-100 rounded transition-colors"
            >
              <MoreVertical className="h-4 w-4 text-gray-400" />
            </button>

            {showMenu && (
              <div className="absolute right-0 bottom-full mb-1 w-40 bg-white rounded-lg shadow-lg border border-gray-200 py-1 z-10">
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onView();
                    setShowMenu(false);
                  }}
                  className="w-full flex items-center gap-2 px-3 py-2 text-sm text-gray-700 hover:bg-gray-50"
                >
                  <Eye className="h-4 w-4" />
                  View Slide
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onAnalyze();
                    setShowMenu(false);
                  }}
                  className="w-full flex items-center gap-2 px-3 py-2 text-sm text-gray-700 hover:bg-gray-50"
                >
                  <Microscope className="h-4 w-4" />
                  Analyze
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onAddToGroup();
                    setShowMenu(false);
                  }}
                  className="w-full flex items-center gap-2 px-3 py-2 text-sm text-gray-700 hover:bg-gray-50"
                >
                  <FolderPlus className="h-4 w-4" />
                  Add to Group
                </button>
                <hr className="my-1 border-gray-100" />
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onDelete();
                    setShowMenu(false);
                  }}
                  className="w-full flex items-center gap-2 px-3 py-2 text-sm text-red-600 hover:bg-red-50"
                >
                  <Trash2 className="h-4 w-4" />
                  Delete
                </button>
              </div>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export function SlideGrid({
  slides,
  selectedIds,
  onSelectSlide,
  onSelectAll,
  onStarSlide,
  onViewSlide,
  onAnalyzeSlide,
  onAddToGroup,
  onDeleteSlide,
  isLoading,
}: SlideGridProps) {
  if (isLoading && slides.length === 0) {
    return (
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5 gap-4">
        {Array.from({ length: 12 }).map((_, i) => (
          <SlideCardSkeleton key={i} />
        ))}
      </div>
    );
  }

  if (slides.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-center">
        <Microscope className="h-16 w-16 text-gray-300 mb-4" />
        <h3 className="text-lg font-medium text-gray-900 mb-2">No slides found</h3>
        <p className="text-sm text-gray-500 max-w-sm">
          Try adjusting your filters or search criteria to find slides.
        </p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5 gap-4">
      {slides.map((slide) => (
        <SlideCard
          key={slide.id}
          slide={slide}
          isSelected={selectedIds.has(slide.id)}
          onSelect={(selected) => onSelectSlide(slide.id, selected)}
          onStar={() => onStarSlide(slide.id)}
          onView={() => onViewSlide(slide.id)}
          onAnalyze={() => onAnalyzeSlide(slide.id)}
          onAddToGroup={() => onAddToGroup(slide.id)}
          onDelete={() => onDeleteSlide(slide.id)}
        />
      ))}
    </div>
  );
}
