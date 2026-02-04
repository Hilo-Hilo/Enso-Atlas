"use client";

import React, { useState, useCallback, useEffect, useMemo } from "react";
import { useRouter } from "next/navigation";
import { Header } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { Skeleton } from "@/components/ui/Skeleton";
import { useToast } from "@/components/ui";
import {
  FilterPanel,
  SlideGrid,
  SlideTable,
  BulkActions,
  CreateTagModal,
  CreateGroupModal,
  ConfirmDeleteModal,
  AddToGroupModal,
} from "@/components/slides";
import { cn, debounce } from "@/lib/utils";
import {
  getSlides,
  searchSlides,
  getAllTags,
  getGroups,
  createGroup,
  toggleSlideStar,
  bulkAddTags,
  bulkAddToGroup,
  addSlidesToGroup,
  healthCheck,
} from "@/lib/api";
import type { SlideInfo, ExtendedSlideInfo, SlideFilters, Tag, Group } from "@/types";
import {
  LayoutGrid,
  Table2,
  RefreshCw,
  ChevronLeft,
  ChevronRight,
  ArrowLeft,
  PanelLeft,
  Microscope,
  Home,
  Layers,
} from "lucide-react";

// Pagination component
function Pagination({
  currentPage,
  totalPages,
  onPageChange,
  totalItems,
  perPage,
}: {
  currentPage: number;
  totalPages: number;
  onPageChange: (page: number) => void;
  totalItems: number;
  perPage: number;
}) {
  const startItem = (currentPage - 1) * perPage + 1;
  const endItem = Math.min(currentPage * perPage, totalItems);

  return (
    <div className="flex items-center justify-between px-4 py-3 border-t border-gray-200 bg-white">
      <div className="text-sm text-gray-500">
        Showing <span className="font-medium">{startItem}</span> to{" "}
        <span className="font-medium">{endItem}</span> of{" "}
        <span className="font-medium">{totalItems}</span> slides
      </div>
      <div className="flex items-center gap-2">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => onPageChange(currentPage - 1)}
          disabled={currentPage === 1}
        >
          <ChevronLeft className="h-4 w-4" />
          Previous
        </Button>
        <div className="flex items-center gap-1">
          {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
            let page: number;
            if (totalPages <= 5) {
              page = i + 1;
            } else if (currentPage <= 3) {
              page = i + 1;
            } else if (currentPage >= totalPages - 2) {
              page = totalPages - 4 + i;
            } else {
              page = currentPage - 2 + i;
            }
            return (
              <button
                key={page}
                onClick={() => onPageChange(page)}
                className={cn(
                  "w-8 h-8 text-sm font-medium rounded-lg transition-colors",
                  currentPage === page
                    ? "bg-clinical-500 text-white"
                    : "text-gray-600 hover:bg-gray-100"
                )}
              >
                {page}
              </button>
            );
          })}
        </div>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => onPageChange(currentPage + 1)}
          disabled={currentPage === totalPages}
        >
          Next
          <ChevronRight className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
}

export default function SlidesPage() {
  const router = useRouter();
  const { showToast } = useToast();

  // Connection state
  const [isConnected, setIsConnected] = useState(false);

  // View mode
  const [viewMode, setViewMode] = useState<"grid" | "table">("grid");
  const [showFilterPanel, setShowFilterPanel] = useState(true);

  // Data state
  const [slides, setSlides] = useState<(SlideInfo & Partial<ExtendedSlideInfo>)[]>([]);
  const [tags, setTags] = useState<Tag[]>([]);
  const [groups, setGroups] = useState<Group[]>([]);
  const [totalSlides, setTotalSlides] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);

  // Filter state
  const [filters, setFilters] = useState<SlideFilters>({
    page: 1,
    perPage: 20,
    sortBy: "date",
    sortOrder: "desc",
  });

  // Selection state
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());

  // Modal state
  const [createTagModalOpen, setCreateTagModalOpen] = useState(false);
  const [createGroupModalOpen, setCreateGroupModalOpen] = useState(false);
  const [deleteModalOpen, setDeleteModalOpen] = useState(false);
  const [addToGroupModalOpen, setAddToGroupModalOpen] = useState(false);
  const [slideForGroupModal, setSlideForGroupModal] = useState<string | null>(null);

  // Processing state
  const [isProcessing, setIsProcessing] = useState(false);

  // Check backend connection
  useEffect(() => {
    const checkConnection = async () => {
      try {
        await healthCheck();
        setIsConnected(true);
      } catch {
        setIsConnected(false);
      }
    };
    checkConnection();
    const interval = setInterval(checkConnection, 30000);
    return () => clearInterval(interval);
  }, []);

  // Fetch tags and groups on mount
  useEffect(() => {
    const fetchMetadata = async () => {
      try {
        const [tagsData, groupsData] = await Promise.all([
          getAllTags().catch(() => []),
          getGroups().catch(() => []),
        ]);
        setTags(tagsData);
        setGroups(groupsData);
      } catch (error) {
        console.error("Failed to fetch metadata:", error);
      }
    };
    fetchMetadata();
  }, []);

  // Fetch slides when filters change
  const fetchSlides = useCallback(async () => {
    setIsLoading(true);
    try {
      // Use searchSlides if we have any filters, otherwise use getSlides
      const hasFilters =
        filters.search ||
        filters.tags?.length ||
        filters.groupId ||
        filters.label ||
        filters.hasEmbeddings !== undefined ||
        filters.minPatches !== undefined ||
        filters.maxPatches !== undefined ||
        filters.starred ||
        filters.dateFrom ||
        filters.dateTo;

      if (hasFilters) {
        const result = await searchSlides(filters);
        setSlides(result.slides);
        setTotalSlides(result.total);
      } else {
        const result = await getSlides({
          page: filters.page,
          perPage: filters.perPage,
        });
        setSlides(result.slides);
        setTotalSlides(result.total);
      }
    } catch (error) {
      console.error("Failed to fetch slides:", error);
      showToast({
        type: "error",
        message: "Failed to load slides. Please try again.",
      });
    } finally {
      setIsLoading(false);
    }
  }, [filters, showToast]);

  useEffect(() => {
    fetchSlides();
  }, [fetchSlides]);

  // Handlers
  const handleRefresh = async () => {
    setIsRefreshing(true);
    await fetchSlides();
    setIsRefreshing(false);
    showToast({ type: "success", message: "Slides refreshed" });
  };

  const handleFiltersChange = useCallback((newFilters: SlideFilters) => {
    setFilters(newFilters);
    setSelectedIds(new Set());
  }, []);

  const handleSelectSlide = useCallback((id: string, selected: boolean) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (selected) {
        next.add(id);
      } else {
        next.delete(id);
      }
      return next;
    });
  }, []);

  const handleSelectAll = useCallback(
    (selected: boolean) => {
      if (selected) {
        setSelectedIds(new Set(slides.map((s) => s.id)));
      } else {
        setSelectedIds(new Set());
      }
    },
    [slides]
  );

  const handleStarSlide = useCallback(
    async (id: string) => {
      try {
        const newStarred = await toggleSlideStar(id);
        setSlides((prev) =>
          prev.map((s) => (s.id === id ? { ...s, starred: newStarred } : s))
        );
        showToast({
          type: "success",
          message: newStarred ? "Slide starred" : "Slide unstarred",
        });
      } catch (error) {
        showToast({ type: "error", message: "Failed to update star status" });
      }
    },
    [showToast]
  );

  const handleViewSlide = useCallback(
    (id: string) => {
      router.push(`/?slide=${id}`);
    },
    [router]
  );

  const handleAnalyzeSlide = useCallback(
    (id: string) => {
      router.push(`/?slide=${id}&analyze=true`);
    },
    [router]
  );

  const handleAddToGroup = useCallback((id: string) => {
    setSlideForGroupModal(id);
    setAddToGroupModalOpen(true);
  }, []);

  const handleDeleteSlide = useCallback((id: string) => {
    setSelectedIds(new Set([id]));
    setDeleteModalOpen(true);
  }, []);

  const handleSort = useCallback(
    (field: string) => {
      setFilters((prev) => ({
        ...prev,
        sortBy: field,
        sortOrder: prev.sortBy === field && prev.sortOrder === "asc" ? "desc" : "asc",
        page: 1,
      }));
    },
    []
  );

  const handlePageChange = useCallback((page: number) => {
    setFilters((prev) => ({ ...prev, page }));
    setSelectedIds(new Set());
  }, []);

  // Bulk action handlers
  const handleBulkAddTags = useCallback(
    async (tagNames: string[]) => {
      if (selectedIds.size === 0) return;
      setIsProcessing(true);
      try {
        await bulkAddTags(Array.from(selectedIds), tagNames);
        showToast({
          type: "success",
          message: `Added ${tagNames.length} tag(s) to ${selectedIds.size} slide(s)`,
        });
        fetchSlides();
        // Refresh tags
        const newTags = await getAllTags();
        setTags(newTags);
      } catch (error) {
        showToast({ type: "error", message: "Failed to add tags" });
      } finally {
        setIsProcessing(false);
      }
    },
    [selectedIds, showToast, fetchSlides]
  );

  const handleBulkAddToGroup = useCallback(
    async (groupId: string) => {
      if (selectedIds.size === 0) return;
      setIsProcessing(true);
      try {
        await bulkAddToGroup(Array.from(selectedIds), groupId);
        showToast({
          type: "success",
          message: `Added ${selectedIds.size} slide(s) to group`,
        });
        // Refresh groups
        const newGroups = await getGroups();
        setGroups(newGroups);
      } catch (error) {
        showToast({ type: "error", message: "Failed to add to group" });
      } finally {
        setIsProcessing(false);
      }
    },
    [selectedIds, showToast]
  );

  const handleBulkRemoveFromGroup = useCallback(
    async (groupId: string) => {
      showToast({
        type: "info",
        message: "Remove from group feature coming soon",
      });
    },
    [showToast]
  );

  const handleExportCsv = useCallback(() => {
    const selectedSlides = slides.filter((s) => selectedIds.has(s.id));
    const headers = ["ID", "Filename", "Label", "Patches", "Has Embeddings", "Starred"];
    const rows = selectedSlides.map((s) => [
      s.id,
      s.filename,
      s.label || "",
      s.numPatches?.toString() || "",
      s.hasEmbeddings ? "Yes" : "No",
      s.starred ? "Yes" : "No",
    ]);

    const csv = [headers.join(","), ...rows.map((r) => r.join(","))].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `slides-export-${new Date().toISOString().split("T")[0]}.csv`;
    a.click();
    URL.revokeObjectURL(url);

    showToast({
      type: "success",
      message: `Exported ${selectedSlides.length} slides to CSV`,
    });
  }, [slides, selectedIds, showToast]);

  const handleBatchAnalyze = useCallback(() => {
    router.push(`/?batch=${Array.from(selectedIds).join(",")}`);
  }, [selectedIds, router]);

  const handleBulkDelete = useCallback(async () => {
    showToast({
      type: "info",
      message: "Delete feature requires backend implementation",
    });
    setDeleteModalOpen(false);
    setSelectedIds(new Set());
  }, [showToast]);

  const handleCreateTag = useCallback(
    async (name: string, color: string) => {
      // Tags are created on-demand when added to slides
      // For now, just show a success message
      showToast({ type: "success", message: `Tag "${name}" created` });
      setCreateTagModalOpen(false);
    },
    [showToast]
  );

  const handleCreateGroup = useCallback(
    async (name: string, description?: string) => {
      try {
        const newGroup = await createGroup(name, description);
        setGroups((prev) => [...prev, newGroup]);
        showToast({ type: "success", message: `Group "${name}" created` });
        setCreateGroupModalOpen(false);
      } catch (error) {
        showToast({ type: "error", message: "Failed to create group" });
      }
    },
    [showToast]
  );

  const handleAddSlideToGroup = useCallback(
    async (groupId: string) => {
      if (!slideForGroupModal) return;
      try {
        await addSlidesToGroup(groupId, [slideForGroupModal]);
        showToast({ type: "success", message: "Slide added to group" });
        setAddToGroupModalOpen(false);
        setSlideForGroupModal(null);
        // Refresh groups
        const newGroups = await getGroups();
        setGroups(newGroups);
      } catch (error) {
        showToast({ type: "error", message: "Failed to add slide to group" });
      }
    },
    [slideForGroupModal, showToast]
  );

  // Computed values
  const totalPages = Math.ceil(totalSlides / (filters.perPage || 20));

  return (
    <div className="h-screen flex flex-col bg-gray-50">
      {/* Header */}
      <Header
        isConnected={isConnected}
        version="0.1.0"
        institutionName="Research Laboratory"
      />

      {/* Page Header */}
      <div className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => router.push("/")}
              className="gap-1.5"
            >
              <ArrowLeft className="h-4 w-4" />
              Back
            </Button>
            <div className="h-6 w-px bg-gray-200" />
            <div className="flex items-center gap-3">
              <div className="p-2 bg-clinical-50 rounded-lg">
                <Layers className="h-5 w-5 text-clinical-600" />
              </div>
              <div>
                <h1 className="text-xl font-semibold text-gray-900">Slide Manager</h1>
                <p className="text-sm text-gray-500">
                  {totalSlides} slide{totalSlides !== 1 ? "s" : ""} total
                </p>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {/* Refresh button */}
            <Button
              variant="ghost"
              size="sm"
              onClick={handleRefresh}
              isLoading={isRefreshing}
            >
              <RefreshCw className="h-4 w-4" />
            </Button>

            {/* Toggle filter panel */}
            <Button
              variant={showFilterPanel ? "secondary" : "ghost"}
              size="sm"
              onClick={() => setShowFilterPanel(!showFilterPanel)}
              className="gap-1.5"
            >
              <PanelLeft className="h-4 w-4" />
              Filters
            </Button>

            {/* View mode toggle */}
            <div className="flex items-center bg-gray-100 rounded-lg p-1">
              <button
                onClick={() => setViewMode("grid")}
                className={cn(
                  "flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors",
                  viewMode === "grid"
                    ? "bg-white text-gray-900 shadow-sm"
                    : "text-gray-500 hover:text-gray-700"
                )}
              >
                <LayoutGrid className="h-4 w-4" />
                Grid
              </button>
              <button
                onClick={() => setViewMode("table")}
                className={cn(
                  "flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors",
                  viewMode === "table"
                    ? "bg-white text-gray-900 shadow-sm"
                    : "text-gray-500 hover:text-gray-700"
                )}
              >
                <Table2 className="h-4 w-4" />
                Table
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Filter panel */}
        {showFilterPanel && (
          <div className="w-72 shrink-0 border-r border-gray-200 overflow-y-auto">
            <FilterPanel
              filters={filters}
              onFiltersChange={handleFiltersChange}
              tags={tags}
              groups={groups}
              isLoading={isLoading}
              onCreateTag={() => setCreateTagModalOpen(true)}
              onCreateGroup={() => setCreateGroupModalOpen(true)}
            />
          </div>
        )}

        {/* Slides content */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Slides view */}
          <div className="flex-1 overflow-y-auto p-4">
            {viewMode === "grid" ? (
              <SlideGrid
                slides={slides}
                selectedIds={selectedIds}
                onSelectSlide={handleSelectSlide}
                onSelectAll={handleSelectAll}
                onStarSlide={handleStarSlide}
                onViewSlide={handleViewSlide}
                onAnalyzeSlide={handleAnalyzeSlide}
                onAddToGroup={handleAddToGroup}
                onDeleteSlide={handleDeleteSlide}
                isLoading={isLoading && slides.length === 0}
              />
            ) : (
              <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
                <SlideTable
                  slides={slides}
                  selectedIds={selectedIds}
                  onSelectSlide={handleSelectSlide}
                  onSelectAll={handleSelectAll}
                  onStarSlide={handleStarSlide}
                  onViewSlide={handleViewSlide}
                  onAnalyzeSlide={handleAnalyzeSlide}
                  onAddToGroup={handleAddToGroup}
                  onDeleteSlide={handleDeleteSlide}
                  sortBy={filters.sortBy}
                  sortOrder={filters.sortOrder}
                  onSort={handleSort}
                  isLoading={isLoading && slides.length === 0}
                />
              </div>
            )}
          </div>

          {/* Pagination */}
          {!isLoading && totalSlides > 0 && (
            <Pagination
              currentPage={filters.page || 1}
              totalPages={totalPages}
              onPageChange={handlePageChange}
              totalItems={totalSlides}
              perPage={filters.perPage || 20}
            />
          )}
        </div>
      </div>

      {/* Bulk actions bar */}
      <BulkActions
        selectedCount={selectedIds.size}
        onClearSelection={() => setSelectedIds(new Set())}
        onAddTags={handleBulkAddTags}
        onAddToGroup={handleBulkAddToGroup}
        onRemoveFromGroup={handleBulkRemoveFromGroup}
        onExportCsv={handleExportCsv}
        onBatchAnalyze={handleBatchAnalyze}
        onDelete={() => setDeleteModalOpen(true)}
        groups={groups}
        availableTags={tags.map((t) => t.name)}
        isProcessing={isProcessing}
      />

      {/* Modals */}
      <CreateTagModal
        isOpen={createTagModalOpen}
        onClose={() => setCreateTagModalOpen(false)}
        onCreateTag={handleCreateTag}
        existingTags={tags.map((t) => t.name)}
      />
      <CreateGroupModal
        isOpen={createGroupModalOpen}
        onClose={() => setCreateGroupModalOpen(false)}
        onCreateGroup={handleCreateGroup}
        existingGroups={groups.map((g) => g.name)}
      />
      <ConfirmDeleteModal
        isOpen={deleteModalOpen}
        onClose={() => setDeleteModalOpen(false)}
        onConfirm={handleBulkDelete}
        count={selectedIds.size}
      />
      {slideForGroupModal && (
        <AddToGroupModal
          isOpen={addToGroupModalOpen}
          onClose={() => {
            setAddToGroupModalOpen(false);
            setSlideForGroupModal(null);
          }}
          onAddToGroup={handleAddSlideToGroup}
          groups={groups}
          slideId={slideForGroupModal}
        />
      )}

      {/* Footer */}
      <Footer />
    </div>
  );
}
