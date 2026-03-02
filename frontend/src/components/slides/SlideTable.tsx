"use client";

import React from "react";
import { Badge } from "@/components/ui/Badge";
import { Skeleton } from "@/components/ui/Skeleton";
import { cn, truncateText, formatDate } from "@/lib/utils";
import { useProject } from "@/contexts/ProjectContext";
import type { SlideInfo, ExtendedSlideInfo } from "@/types";
import {
  Star,
  MoreVertical,
  Microscope,
  Eye,
  FolderPlus,
  Trash2,
  Layers,
  ChevronUp,
  ChevronDown,
  CheckCircle2,
  Circle,
  Minus,
} from "lucide-react";
import { getVirtualWindow } from "./virtualization";

interface SlideTableProps {
  slides: (SlideInfo & Partial<ExtendedSlideInfo>)[];
  selectedIds: Set<string>;
  onSelectSlide: (id: string, selected: boolean) => void;
  onSelectAll: (selected: boolean) => void;
  onStarSlide: (id: string) => void;
  onViewSlide: (id: string) => void;
  onAnalyzeSlide: (id: string) => void;
  onAddToGroup: (id: string) => void;
  onDeleteSlide: (id: string) => void;
  sortBy?: string;
  sortOrder?: "asc" | "desc";
  onSort?: (field: string) => void;
  isLoading?: boolean;
}

const ROW_HEIGHT = 72;
const VIRTUALIZE_THRESHOLD = 60;

// Tag colors
const TAG_COLORS: Record<string, string> = {
  red: "bg-red-100 text-red-700",
  orange: "bg-orange-100 text-orange-700",
  amber: "bg-amber-100 text-amber-700",
  green: "bg-green-100 text-green-700",
  blue: "bg-blue-100 text-blue-700",
  purple: "bg-purple-100 text-purple-700",
  pink: "bg-pink-100 text-pink-700",
};

type SortableColumn = "name" | "label" | "patches" | "date" | "starred";

function SortableHeader({
  label,
  field,
  currentSort,
  currentOrder,
  onSort,
}: {
  label: string;
  field: SortableColumn;
  currentSort?: string;
  currentOrder?: "asc" | "desc";
  onSort?: (field: string) => void;
}) {
  const isActive = currentSort === field;

  return (
    <button
      onClick={() => onSort?.(field)}
      className={cn(
        "flex items-center gap-1 text-left font-medium hover:text-gray-900 dark:hover:text-gray-100 transition-colors",
        isActive ? "text-clinical-600 dark:text-clinical-400" : "text-gray-500 dark:text-gray-400"
      )}
    >
      {label}
      {isActive ? (
        currentOrder === "asc" ? (
          <ChevronUp className="h-4 w-4" />
        ) : (
          <ChevronDown className="h-4 w-4" />
        )
      ) : (
        <div className="w-4" />
      )}
    </button>
  );
}

function TableRowSkeleton() {
  return (
    <tr className="border-b border-gray-100 h-[72px]">
      <td className="py-3 px-4">
        <Skeleton className="h-5 w-5 rounded" />
      </td>
      <td className="py-3 px-4">
        <Skeleton className="h-4 w-40" />
      </td>
      <td className="py-3 px-4">
        <Skeleton className="h-5 w-20 rounded-full" />
      </td>
      <td className="py-3 px-4">
        <div className="flex gap-1">
          <Skeleton className="h-5 w-14 rounded-full" />
          <Skeleton className="h-5 w-14 rounded-full" />
        </div>
      </td>
      <td className="py-3 px-4">
        <Skeleton className="h-4 w-12" />
      </td>
      <td className="py-3 px-4">
        <Skeleton className="h-4 w-24" />
      </td>
      <td className="py-3 px-4">
        <Skeleton className="h-5 w-5 rounded" />
      </td>
      <td className="py-3 px-4">
        <Skeleton className="h-5 w-5 rounded" />
      </td>
    </tr>
  );
}

function SlideRow({
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
  const { currentProject } = useProject();
  const [showMenu, setShowMenu] = React.useState(false);
  const menuRef = React.useRef<HTMLDivElement>(null);

  // Get display labels from project context
  const positiveLabel = currentProject.positive_class
    ? currentProject.positive_class.charAt(0).toUpperCase() + currentProject.positive_class.slice(1)
    : "Positive";
  const negativeLabel = currentProject.classes?.find((c) => c !== currentProject.positive_class)
    ? (currentProject.classes.find((c) => c !== currentProject.positive_class) as string).charAt(0).toUpperCase() +
      (currentProject.classes.find((c) => c !== currentProject.positive_class) as string).slice(1)
    : "Negative";

  React.useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setShowMenu(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  return (
    <tr
      className={cn(
        "h-[72px] border-b border-gray-100 dark:border-navy-700 hover:bg-gray-50 dark:hover:bg-navy-800/50 transition-colors",
        isSelected && "bg-clinical-50 dark:bg-clinical-900/20"
      )}
    >
      {/* Selection checkbox */}
      <td className="py-3 px-4 align-middle">
        <button
          onClick={() => onSelect(!isSelected)}
          className="flex items-center justify-center"
        >
          {isSelected ? (
            <CheckCircle2 className="h-5 w-5 text-clinical-500" />
          ) : (
            <Circle className="h-5 w-5 text-gray-300 hover:text-gray-400" />
          )}
        </button>
      </td>

      {/* Name */}
      <td className="py-3 px-4 align-middle">
        <button
          onClick={onView}
          className="flex items-center gap-2 hover:text-clinical-600 transition-colors"
        >
          <span className="font-medium text-gray-900 dark:text-gray-100">{truncateText(slide.id, 40)}</span>
          {slide.hasEmbeddings && (
            <span className="p-0.5 bg-clinical-100 rounded" title="Has embeddings">
              <Layers className="h-3 w-3 text-clinical-600" />
            </span>
          )}
        </button>
      </td>

      {/* Label */}
      <td className="py-3 px-4 align-middle">
        {slide.label ? (
          <Badge
            variant={slide.label === "1" ? "success" : "danger"}
            size="sm"
          >
            {slide.label === "1" ? positiveLabel : negativeLabel}
          </Badge>
        ) : (
          <span className="text-gray-400 text-sm">—</span>
        )}
      </td>

      {/* Tags */}
      <td className="py-3 px-4 align-middle">
        {slide.tags && slide.tags.length > 0 ? (
          <div className="flex items-center gap-1 overflow-hidden">
            {slide.tags.slice(0, 2).map((tag, idx) => (
              <span
                key={tag}
                className={cn(
                  "px-1.5 py-0.5 text-xs rounded-full whitespace-nowrap",
                  TAG_COLORS[Object.keys(TAG_COLORS)[idx % Object.keys(TAG_COLORS).length]]
                )}
              >
                {tag}
              </span>
            ))}
            {slide.tags.length > 2 && (
              <span className="px-1.5 py-0.5 text-xs bg-gray-100 text-gray-500 rounded-full whitespace-nowrap">
                +{slide.tags.length - 2}
              </span>
            )}
          </div>
        ) : (
          <span className="text-gray-400 text-sm">—</span>
        )}
      </td>

      {/* Patches */}
      <td className="py-3 px-4 align-middle">
        <span className="text-sm text-gray-600 dark:text-gray-300">
          {slide.numPatches !== undefined ? slide.numPatches.toLocaleString() : "—"}
        </span>
      </td>

      {/* Date */}
      <td className="py-3 px-4 align-middle">
        <span className="text-sm text-gray-500 dark:text-gray-400">
          {formatDate(slide.createdAt)}
        </span>
      </td>

      {/* Starred */}
      <td className="py-3 px-4 align-middle">
        <button
          onClick={onStar}
          className={cn(
            "p-1 rounded transition-colors",
            slide.starred
              ? "text-amber-500 hover:bg-amber-50"
              : "text-gray-300 hover:text-amber-500 hover:bg-gray-100"
          )}
        >
          <Star className={cn("h-5 w-5", slide.starred && "fill-amber-500")} />
        </button>
      </td>

      {/* Actions */}
      <td className="py-3 px-4 align-middle">
        <div className="relative" ref={menuRef}>
          <button
            onClick={() => setShowMenu(!showMenu)}
            className="p-1 hover:bg-gray-100 dark:hover:bg-navy-700 rounded transition-colors"
          >
            <MoreVertical className="h-4 w-4 text-gray-400" />
          </button>

          {showMenu && (
            <div className="absolute right-0 top-full mt-1 w-40 bg-white dark:bg-navy-800 rounded-lg shadow-lg border border-gray-200 dark:border-navy-600 py-1 z-10">
              <button
                onClick={() => {
                  onView();
                  setShowMenu(false);
                }}
                className="w-full flex items-center gap-2 px-3 py-2 text-sm text-gray-700 dark:text-gray-200 hover:bg-gray-50 dark:hover:bg-navy-700"
              >
                <Eye className="h-4 w-4" />
                View Slide
              </button>
              <button
                onClick={() => {
                  onAnalyze();
                  setShowMenu(false);
                }}
                className="w-full flex items-center gap-2 px-3 py-2 text-sm text-gray-700 dark:text-gray-200 hover:bg-gray-50 dark:hover:bg-navy-700"
              >
                <Microscope className="h-4 w-4" />
                Analyze
              </button>
              <button
                onClick={() => {
                  onAddToGroup();
                  setShowMenu(false);
                }}
                className="w-full flex items-center gap-2 px-3 py-2 text-sm text-gray-700 dark:text-gray-200 hover:bg-gray-50 dark:hover:bg-navy-700"
              >
                <FolderPlus className="h-4 w-4" />
                Add to Group
              </button>
              <hr className="my-1 border-gray-100 dark:border-navy-600" />
              <button
                onClick={() => {
                  onDelete();
                  setShowMenu(false);
                }}
                className="w-full flex items-center gap-2 px-3 py-2 text-sm text-red-600 hover:bg-red-50 dark:hover:bg-red-900/20"
              >
                <Trash2 className="h-4 w-4" />
                Delete
              </button>
            </div>
          )}
        </div>
      </td>
    </tr>
  );
}

export function SlideTable({
  slides,
  selectedIds,
  onSelectSlide,
  onSelectAll,
  onStarSlide,
  onViewSlide,
  onAnalyzeSlide,
  onAddToGroup,
  onDeleteSlide,
  sortBy,
  sortOrder,
  onSort,
  isLoading,
}: SlideTableProps) {
  const allSelected = slides.length > 0 && slides.every((s) => selectedIds.has(s.id));
  const someSelected = slides.some((s) => selectedIds.has(s.id));

  const containerRef = React.useRef<HTMLDivElement>(null);
  const [scrollTop, setScrollTop] = React.useState(0);
  const [viewportHeight, setViewportHeight] = React.useState(640);
  const shouldVirtualize = slides.length >= VIRTUALIZE_THRESHOLD;

  React.useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const updateViewport = () => {
      setScrollTop(el.scrollTop);
      setViewportHeight(Math.max(320, el.clientHeight || 0));
    };

    updateViewport();
    el.addEventListener("scroll", updateViewport, { passive: true });

    const resizeObserver =
      typeof ResizeObserver !== "undefined" ? new ResizeObserver(updateViewport) : null;
    resizeObserver?.observe(el);

    return () => {
      el.removeEventListener("scroll", updateViewport);
      resizeObserver?.disconnect();
    };
  }, []);

  const windowRange = shouldVirtualize
    ? getVirtualWindow({
        itemCount: slides.length,
        itemHeight: ROW_HEIGHT,
        viewportHeight,
        scrollTop,
        overscan: 8,
      })
    : {
        startIndex: 0,
        endIndex: slides.length,
        topSpacerHeight: 0,
        bottomSpacerHeight: 0,
      };

  const visibleSlides = slides.slice(windowRange.startIndex, windowRange.endIndex);

  if (isLoading && slides.length === 0) {
    return (
      <div className="h-full overflow-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-gray-200 bg-gray-50">
              <th className="py-3 px-4 text-left w-12">
                <Skeleton className="h-5 w-5 rounded" />
              </th>
              <th className="py-3 px-4 text-left">
                <Skeleton className="h-4 w-16" />
              </th>
              <th className="py-3 px-4 text-left">
                <Skeleton className="h-4 w-12" />
              </th>
              <th className="py-3 px-4 text-left">
                <Skeleton className="h-4 w-12" />
              </th>
              <th className="py-3 px-4 text-left">
                <Skeleton className="h-4 w-16" />
              </th>
              <th className="py-3 px-4 text-left">
                <Skeleton className="h-4 w-12" />
              </th>
              <th className="py-3 px-4 text-left w-12">
                <Skeleton className="h-4 w-4" />
              </th>
              <th className="py-3 px-4 w-12" />
            </tr>
          </thead>
          <tbody>
            {Array.from({ length: 10 }).map((_, i) => (
              <TableRowSkeleton key={i} />
            ))}
          </tbody>
        </table>
      </div>
    );
  }

  if (slides.length === 0) {
    return (
      <div className="h-full flex flex-col items-center justify-center py-16 text-center">
        <Microscope className="h-16 w-16 text-gray-300 dark:text-gray-600 mb-4" />
        <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-2">No slides found</h3>
        <p className="text-sm text-gray-500 dark:text-gray-400 max-w-sm">
          Try adjusting your filters or search criteria to find slides.
        </p>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="h-full overflow-auto" data-testid="slide-table-scroll-container">
      <table className="w-full">
        <thead className="sticky top-0 z-10">
          <tr className="border-b border-gray-200 dark:border-navy-700 bg-gray-50 dark:bg-navy-900">
            {/* Select all */}
            <th className="py-3 px-4 text-left w-12">
              <button
                onClick={() => onSelectAll(!allSelected)}
                className="flex items-center justify-center"
              >
                {allSelected ? (
                  <CheckCircle2 className="h-5 w-5 text-clinical-500" />
                ) : someSelected ? (
                  <div className="h-5 w-5 rounded border-2 border-clinical-500 flex items-center justify-center">
                    <Minus className="h-3 w-3 text-clinical-500" />
                  </div>
                ) : (
                  <Circle className="h-5 w-5 text-gray-300 hover:text-gray-400" />
                )}
              </button>
            </th>
            <th className="py-3 px-4 text-left text-xs uppercase tracking-wide">
              <SortableHeader
                label="Name"
                field="name"
                currentSort={sortBy}
                currentOrder={sortOrder}
                onSort={onSort}
              />
            </th>
            <th className="py-3 px-4 text-left text-xs uppercase tracking-wide">
              <SortableHeader
                label="Label"
                field="label"
                currentSort={sortBy}
                currentOrder={sortOrder}
                onSort={onSort}
              />
            </th>
            <th className="py-3 px-4 text-left text-xs uppercase tracking-wide text-gray-500 font-medium">
              Tags
            </th>
            <th className="py-3 px-4 text-left text-xs uppercase tracking-wide">
              <SortableHeader
                label="Patches"
                field="patches"
                currentSort={sortBy}
                currentOrder={sortOrder}
                onSort={onSort}
              />
            </th>
            <th className="py-3 px-4 text-left text-xs uppercase tracking-wide">
              <SortableHeader
                label="Date"
                field="date"
                currentSort={sortBy}
                currentOrder={sortOrder}
                onSort={onSort}
              />
            </th>
            <th className="py-3 px-4 text-left text-xs uppercase tracking-wide w-12">
              <SortableHeader
                label=""
                field="starred"
                currentSort={sortBy}
                currentOrder={sortOrder}
                onSort={onSort}
              />
            </th>
            <th className="py-3 px-4 w-12" />
          </tr>
        </thead>
        <tbody>
          {windowRange.topSpacerHeight > 0 && (
            <tr aria-hidden="true">
              <td colSpan={8} style={{ height: windowRange.topSpacerHeight, padding: 0 }} />
            </tr>
          )}

          {visibleSlides.map((slide) => (
            <SlideRow
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

          {windowRange.bottomSpacerHeight > 0 && (
            <tr aria-hidden="true">
              <td colSpan={8} style={{ height: windowRange.bottomSpacerHeight, padding: 0 }} />
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
