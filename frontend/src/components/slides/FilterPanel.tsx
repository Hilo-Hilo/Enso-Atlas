"use client";

import React, { useState } from "react";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { cn } from "@/lib/utils";
import type { Tag, Group, SlideFilters } from "@/types";
import {
  Search,
  Tags,
  FolderOpen,
  Activity,
  Hash,
  Calendar,
  Star,
  X,
  ChevronDown,
  ChevronRight,
  Filter,
  Sparkles,
} from "lucide-react";

interface FilterPanelProps {
  filters: SlideFilters;
  onFiltersChange: (filters: SlideFilters) => void;
  tags: Tag[];
  groups: Group[];
  isLoading?: boolean;
  onCreateTag?: () => void;
  onCreateGroup?: () => void;
}

// Color palette for tags
const TAG_COLORS: Record<string, string> = {
  red: "bg-red-100 text-red-800 border-red-200",
  orange: "bg-orange-100 text-orange-800 border-orange-200",
  amber: "bg-amber-100 text-amber-800 border-amber-200",
  yellow: "bg-yellow-100 text-yellow-800 border-yellow-200",
  lime: "bg-lime-100 text-lime-800 border-lime-200",
  green: "bg-green-100 text-green-800 border-green-200",
  emerald: "bg-emerald-100 text-emerald-800 border-emerald-200",
  teal: "bg-teal-100 text-teal-800 border-teal-200",
  cyan: "bg-cyan-100 text-cyan-800 border-cyan-200",
  sky: "bg-sky-100 text-sky-800 border-sky-200",
  blue: "bg-blue-100 text-blue-800 border-blue-200",
  indigo: "bg-indigo-100 text-indigo-800 border-indigo-200",
  violet: "bg-violet-100 text-violet-800 border-violet-200",
  purple: "bg-purple-100 text-purple-800 border-purple-200",
  fuchsia: "bg-fuchsia-100 text-fuchsia-800 border-fuchsia-200",
  pink: "bg-pink-100 text-pink-800 border-pink-200",
  rose: "bg-rose-100 text-rose-800 border-rose-200",
};

// Collapsible section component
function FilterSection({
  title,
  icon: Icon,
  children,
  defaultOpen = true,
  badge,
}: {
  title: string;
  icon: React.ElementType;
  children: React.ReactNode;
  defaultOpen?: boolean;
  badge?: number;
}) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div className="border-b border-gray-100 last:border-b-0">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between px-4 py-3 hover:bg-gray-50 transition-colors"
      >
        <div className="flex items-center gap-2">
          <Icon className="h-4 w-4 text-gray-500" />
          <span className="text-sm font-medium text-gray-700">{title}</span>
          {badge !== undefined && badge > 0 && (
            <span className="px-1.5 py-0.5 text-xs font-medium bg-clinical-100 text-clinical-700 rounded-full">
              {badge}
            </span>
          )}
        </div>
        {isOpen ? (
          <ChevronDown className="h-4 w-4 text-gray-400" />
        ) : (
          <ChevronRight className="h-4 w-4 text-gray-400" />
        )}
      </button>
      {isOpen && <div className="px-4 pb-3">{children}</div>}
    </div>
  );
}

export function FilterPanel({
  filters,
  onFiltersChange,
  tags,
  groups,
  isLoading,
  onCreateTag,
  onCreateGroup,
}: FilterPanelProps) {
  const activeFilterCount =
    (filters.search ? 1 : 0) +
    (filters.tags?.length || 0) +
    (filters.groupId ? 1 : 0) +
    (filters.label ? 1 : 0) +
    (filters.hasEmbeddings !== undefined ? 1 : 0) +
    (filters.minPatches !== undefined ? 1 : 0) +
    (filters.maxPatches !== undefined ? 1 : 0) +
    (filters.starred ? 1 : 0) +
    (filters.dateFrom ? 1 : 0) +
    (filters.dateTo ? 1 : 0);

  const handleClearAll = () => {
    onFiltersChange({
      page: 1,
      perPage: filters.perPage || 20,
      sortBy: filters.sortBy,
      sortOrder: filters.sortOrder,
    });
  };

  const toggleTag = (tagName: string) => {
    const currentTags = filters.tags || [];
    const newTags = currentTags.includes(tagName)
      ? currentTags.filter((t) => t !== tagName)
      : [...currentTags, tagName];
    onFiltersChange({ ...filters, tags: newTags.length ? newTags : undefined, page: 1 });
  };

  return (
    <div className="h-full flex flex-col bg-white border-r border-gray-200">
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-200 bg-gradient-to-r from-gray-50 to-white">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Filter className="h-5 w-5 text-clinical-600" />
            <h2 className="text-sm font-semibold text-gray-900">Filters</h2>
            {activeFilterCount > 0 && (
              <Badge variant="clinical" size="sm">
                {activeFilterCount} active
              </Badge>
            )}
          </div>
          {activeFilterCount > 0 && (
            <Button
              variant="ghost"
              size="sm"
              onClick={handleClearAll}
              className="text-xs text-gray-500 hover:text-gray-700"
            >
              Clear all
            </Button>
          )}
        </div>
      </div>

      {/* Scrollable filter sections */}
      <div className="flex-1 overflow-y-auto">
        {/* Search */}
        <FilterSection title="Search" icon={Search} defaultOpen={true}>
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
            <input
              type="text"
              placeholder="Slide name or ID..."
              value={filters.search || ""}
              onChange={(e) =>
                onFiltersChange({ ...filters, search: e.target.value || undefined, page: 1 })
              }
              className="w-full pl-9 pr-3 py-2 text-sm border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-clinical-500/20 focus:border-clinical-500 transition-all"
            />
            {filters.search && (
              <button
                onClick={() => onFiltersChange({ ...filters, search: undefined, page: 1 })}
                className="absolute right-2 top-1/2 -translate-y-1/2 p-1 hover:bg-gray-100 rounded"
              >
                <X className="h-3 w-3 text-gray-400" />
              </button>
            )}
          </div>
        </FilterSection>

        {/* Tags */}
        <FilterSection
          title="Tags"
          icon={Tags}
          badge={filters.tags?.length}
        >
          <div className="space-y-2">
            {tags.length === 0 ? (
              <p className="text-xs text-gray-500 py-2">No tags created yet</p>
            ) : (
              <div className="flex flex-wrap gap-1.5">
                {tags.map((tag) => {
                  const isSelected = filters.tags?.includes(tag.name);
                  const colorClass = tag.color
                    ? TAG_COLORS[tag.color] || TAG_COLORS.blue
                    : TAG_COLORS.blue;
                  return (
                    <button
                      key={tag.name}
                      onClick={() => toggleTag(tag.name)}
                      className={cn(
                        "px-2 py-1 text-xs font-medium rounded-full border transition-all",
                        isSelected
                          ? "ring-2 ring-clinical-500 ring-offset-1"
                          : "hover:ring-1 hover:ring-gray-300",
                        colorClass
                      )}
                    >
                      {tag.name}
                      <span className="ml-1 opacity-70">({tag.count})</span>
                    </button>
                  );
                })}
              </div>
            )}
            {onCreateTag && (
              <Button
                variant="ghost"
                size="sm"
                onClick={onCreateTag}
                className="w-full text-xs mt-2"
              >
                <Sparkles className="h-3 w-3 mr-1" />
                Create Tag
              </Button>
            )}
          </div>
        </FilterSection>

        {/* Groups */}
        <FilterSection title="Groups" icon={FolderOpen}>
          <div className="space-y-1">
            <button
              onClick={() => onFiltersChange({ ...filters, groupId: undefined, page: 1 })}
              className={cn(
                "w-full flex items-center justify-between px-2 py-1.5 text-sm rounded-lg transition-colors",
                !filters.groupId
                  ? "bg-clinical-50 text-clinical-700"
                  : "text-gray-600 hover:bg-gray-50"
              )}
            >
              <span>All Slides</span>
            </button>
            {groups.map((group) => (
              <button
                key={group.id}
                onClick={() =>
                  onFiltersChange({
                    ...filters,
                    groupId: filters.groupId === group.id ? undefined : group.id,
                    page: 1,
                  })
                }
                className={cn(
                  "w-full flex items-center justify-between px-2 py-1.5 text-sm rounded-lg transition-colors",
                  filters.groupId === group.id
                    ? "bg-clinical-50 text-clinical-700"
                    : "text-gray-600 hover:bg-gray-50"
                )}
              >
                <span className="truncate">{group.name}</span>
                <span className="text-xs text-gray-400">{group.slideIds.length}</span>
              </button>
            ))}
            {onCreateGroup && (
              <Button
                variant="ghost"
                size="sm"
                onClick={onCreateGroup}
                className="w-full text-xs mt-2"
              >
                <FolderOpen className="h-3 w-3 mr-1" />
                Create Group
              </Button>
            )}
          </div>
        </FilterSection>

        {/* Label */}
        <FilterSection title="Label" icon={Activity}>
          <div className="flex flex-wrap gap-2">
            {["platinum_sensitive", "platinum_resistant"].map((label) => (
              <button
                key={label}
                onClick={() =>
                  onFiltersChange({
                    ...filters,
                    label: filters.label === label ? undefined : label,
                    page: 1,
                  })
                }
                className={cn(
                  "px-3 py-1.5 text-xs font-medium rounded-lg border transition-all",
                  filters.label === label
                    ? label === "platinum_sensitive"
                      ? "bg-green-50 text-green-700 border-green-200 ring-2 ring-green-500/30"
                      : "bg-red-50 text-red-700 border-red-200 ring-2 ring-red-500/30"
                    : "bg-gray-50 text-gray-600 border-gray-200 hover:bg-gray-100"
                )}
              >
                {label === "platinum_sensitive" ? "Sensitive" : "Resistant"}
              </button>
            ))}
          </div>
        </FilterSection>

        {/* Embeddings */}
        <FilterSection title="Embeddings" icon={Sparkles}>
          <div className="flex flex-wrap gap-2">
            <button
              onClick={() =>
                onFiltersChange({
                  ...filters,
                  hasEmbeddings: filters.hasEmbeddings === true ? undefined : true,
                  page: 1,
                })
              }
              className={cn(
                "px-3 py-1.5 text-xs font-medium rounded-lg border transition-all",
                filters.hasEmbeddings === true
                  ? "bg-clinical-50 text-clinical-700 border-clinical-200 ring-2 ring-clinical-500/30"
                  : "bg-gray-50 text-gray-600 border-gray-200 hover:bg-gray-100"
              )}
            >
              Has Embeddings
            </button>
            <button
              onClick={() =>
                onFiltersChange({
                  ...filters,
                  hasEmbeddings: filters.hasEmbeddings === false ? undefined : false,
                  page: 1,
                })
              }
              className={cn(
                "px-3 py-1.5 text-xs font-medium rounded-lg border transition-all",
                filters.hasEmbeddings === false
                  ? "bg-amber-50 text-amber-700 border-amber-200 ring-2 ring-amber-500/30"
                  : "bg-gray-50 text-gray-600 border-gray-200 hover:bg-gray-100"
              )}
            >
              No Embeddings
            </button>
          </div>
        </FilterSection>

        {/* Patch Count */}
        <FilterSection title="Patch Count" icon={Hash}>
          <div className="flex items-center gap-2">
            <input
              type="number"
              placeholder="Min"
              value={filters.minPatches ?? ""}
              onChange={(e) =>
                onFiltersChange({
                  ...filters,
                  minPatches: e.target.value ? Number(e.target.value) : undefined,
                  page: 1,
                })
              }
              className="w-full px-2 py-1.5 text-sm border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-clinical-500/20"
            />
            <span className="text-gray-400">-</span>
            <input
              type="number"
              placeholder="Max"
              value={filters.maxPatches ?? ""}
              onChange={(e) =>
                onFiltersChange({
                  ...filters,
                  maxPatches: e.target.value ? Number(e.target.value) : undefined,
                  page: 1,
                })
              }
              className="w-full px-2 py-1.5 text-sm border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-clinical-500/20"
            />
          </div>
        </FilterSection>

        {/* Date Range */}
        <FilterSection title="Date Added" icon={Calendar} defaultOpen={false}>
          <div className="space-y-2">
            <div>
              <label className="text-xs text-gray-500 mb-1 block">From</label>
              <input
                type="date"
                value={filters.dateFrom || ""}
                onChange={(e) =>
                  onFiltersChange({
                    ...filters,
                    dateFrom: e.target.value || undefined,
                    page: 1,
                  })
                }
                className="w-full px-2 py-1.5 text-sm border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-clinical-500/20"
              />
            </div>
            <div>
              <label className="text-xs text-gray-500 mb-1 block">To</label>
              <input
                type="date"
                value={filters.dateTo || ""}
                onChange={(e) =>
                  onFiltersChange({
                    ...filters,
                    dateTo: e.target.value || undefined,
                    page: 1,
                  })
                }
                className="w-full px-2 py-1.5 text-sm border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-clinical-500/20"
              />
            </div>
          </div>
        </FilterSection>

        {/* Starred */}
        <FilterSection title="Favorites" icon={Star} defaultOpen={false}>
          <button
            onClick={() =>
              onFiltersChange({
                ...filters,
                starred: filters.starred ? undefined : true,
                page: 1,
              })
            }
            className={cn(
              "w-full flex items-center justify-center gap-2 px-3 py-2 text-sm font-medium rounded-lg border transition-all",
              filters.starred
                ? "bg-amber-50 text-amber-700 border-amber-200 ring-2 ring-amber-500/30"
                : "bg-gray-50 text-gray-600 border-gray-200 hover:bg-gray-100"
            )}
          >
            <Star className={cn("h-4 w-4", filters.starred && "fill-amber-500")} />
            Show Starred Only
          </button>
        </FilterSection>
      </div>
    </div>
  );
}
