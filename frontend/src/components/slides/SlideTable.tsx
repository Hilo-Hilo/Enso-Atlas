"use client";

import React from "react";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Skeleton } from "@/components/ui/Skeleton";
import { cn, truncateText, formatDate } from "@/lib/utils";
import type { SlideInfo, ExtendedSlideInfo, SlideFilters } from "@/types";
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
        "flex items-center gap-1 text-left font-medium hover:text-gray-900 transition-colors",
        isActive ? "text-clinical-600" : "text-gray-500"
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
    <tr className="border-b border-gray-100">
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
  const [showMenu, setShowMenu] = React.useState(false);
  const menuRef = React.useRef<HTMLDivElement>(null);

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
        "border-b border-gray-100 hover:bg-gray-50 transition-colors",
        isSelected && "bg-clinical-50"
      )}
    >
      {/* Selection checkbox */}
      <td className="py-3 px-4">
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
      <td className="py-3 px-4">
        <button
          onClick={onView}
          className="flex items-center gap-2 hover:text-clinical-600 transition-colors"
        >
          <span className="font-medium text-gray-900">{truncateText(slide.id, 40)}</span>
          {slide.hasEmbeddings && (
            <span className="p-0.5 bg-clinical-100 rounded" title="Has embeddings">
              <Layers className="h-3 w-3 text-clinical-600" />
            </span>
          )}
        </button>
      </td>

      {/* Label */}
      <td className="py-3 px-4">
        {slide.label ? (
          <Badge
            variant={slide.label === "platinum_sensitive" ? "success" : "danger"}
            size="sm"
          >
            {slide.label === "platinum_sensitive" ? "Sensitive" : "Resistant"}
          </Badge>
        ) : (
          <span className="text-gray-400 text-sm">—</span>
        )}
      </td>

      {/* Tags */}
      <td className="py-3 px-4">
        {slide.tags && slide.tags.length > 0 ? (
          <div className="flex flex-wrap gap-1">
            {slide.tags.slice(0, 2).map((tag, idx) => (
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
            {slide.tags.length > 2 && (
              <span className="px-1.5 py-0.5 text-xs bg-gray-100 text-gray-500 rounded-full">
                +{slide.tags.length - 2}
              </span>
            )}
          </div>
        ) : (
          <span className="text-gray-400 text-sm">—</span>
        )}
      </td>

      {/* Patches */}
      <td className="py-3 px-4">
        <span className="text-sm text-gray-600">
          {slide.numPatches !== undefined ? slide.numPatches.toLocaleString() : "—"}
        </span>
      </td>

      {/* Date */}
      <td className="py-3 px-4">
        <span className="text-sm text-gray-500">
          {formatDate(slide.createdAt)}
        </span>
      </td>

      {/* Starred */}
      <td className="py-3 px-4">
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
      <td className="py-3 px-4">
        <div className="relative" ref={menuRef}>
          <button
            onClick={() => setShowMenu(!showMenu)}
            className="p-1 hover:bg-gray-100 rounded transition-colors"
          >
            <MoreVertical className="h-4 w-4 text-gray-400" />
          </button>

          {showMenu && (
            <div className="absolute right-0 top-full mt-1 w-40 bg-white rounded-lg shadow-lg border border-gray-200 py-1 z-10">
              <button
                onClick={() => {
                  onView();
                  setShowMenu(false);
                }}
                className="w-full flex items-center gap-2 px-3 py-2 text-sm text-gray-700 hover:bg-gray-50"
              >
                <Eye className="h-4 w-4" />
                View Slide
              </button>
              <button
                onClick={() => {
                  onAnalyze();
                  setShowMenu(false);
                }}
                className="w-full flex items-center gap-2 px-3 py-2 text-sm text-gray-700 hover:bg-gray-50"
              >
                <Microscope className="h-4 w-4" />
                Analyze
              </button>
              <button
                onClick={() => {
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
                onClick={() => {
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

  if (isLoading) {
    return (
      <div className="overflow-x-auto">
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
    <div className="overflow-x-auto">
      <table className="w-full">
        <thead>
          <tr className="border-b border-gray-200 bg-gray-50">
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
          {slides.map((slide) => (
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
        </tbody>
      </table>
    </div>
  );
}
