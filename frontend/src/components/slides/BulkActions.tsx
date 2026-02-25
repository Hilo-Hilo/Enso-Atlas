"use client";

import React, { useState } from "react";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { cn } from "@/lib/utils";
import type { Group } from "@/types";
import {
  Tags,
  FolderPlus,
  FolderMinus,
  Download,
  Microscope,
  Trash2,
  X,
  ChevronDown,
} from "lucide-react";

interface BulkActionsProps {
  selectedCount: number;
  onClearSelection: () => void;
  onAddTags: (tags: string[]) => void;
  onAddToGroup: (groupId: string) => void;
  onRemoveFromGroup?: (groupId: string) => void;
  onExportCsv: () => void;
  onBatchAnalyze: () => void;
  onDelete?: () => void;
  groups: Group[];
  availableTags: string[];
  isProcessing?: boolean;
  enableRemoveFromGroup?: boolean;
  enableDelete?: boolean;
}

function DropdownMenu({
  trigger,
  children,
  align = "left",
}: {
  trigger: React.ReactNode;
  children: React.ReactNode;
  align?: "left" | "right";
}) {
  const [isOpen, setIsOpen] = useState(false);
  const menuRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  return (
    <div className="relative" ref={menuRef}>
      <div onClick={() => setIsOpen(!isOpen)}>{trigger}</div>
      {isOpen && (
        <div
          className={cn(
            "absolute bottom-full mb-2 w-56 bg-white rounded-lg shadow-lg border border-gray-200 py-2 z-20",
            align === "right" ? "right-0" : "left-0"
          )}
        >
          {React.Children.map(children, (child) =>
            React.isValidElement(child)
              ? React.cloneElement(child as React.ReactElement<{ onClick?: () => void }>, {
                  onClick: () => {
                    (child.props as { onClick?: () => void }).onClick?.();
                    setIsOpen(false);
                  },
                })
              : child
          )}
        </div>
      )}
    </div>
  );
}

function TagSelector({
  availableTags,
  onSelect,
}: {
  availableTags: string[];
  onSelect: (tags: string[]) => void;
}) {
  const [selectedTags, setSelectedTags] = useState<Set<string>>(new Set());

  const toggleTag = (tag: string) => {
    const newSelected = new Set(selectedTags);
    if (newSelected.has(tag)) {
      newSelected.delete(tag);
    } else {
      newSelected.add(tag);
    }
    setSelectedTags(newSelected);
  };

  return (
    <div className="p-2">
      <p className="text-xs text-gray-500 mb-2 px-2">Select tags to add:</p>
      <div className="max-h-48 overflow-y-auto space-y-1">
        {availableTags.map((tag) => (
          <button
            key={tag}
            onClick={() => toggleTag(tag)}
            className={cn(
              "w-full flex items-center justify-between px-2 py-1.5 rounded text-sm transition-colors",
              selectedTags.has(tag)
                ? "bg-clinical-50 text-clinical-700"
                : "hover:bg-gray-50 text-gray-700"
            )}
          >
            <span>{tag}</span>
            {selectedTags.has(tag) && (
              <span className="text-clinical-500">✓</span>
            )}
          </button>
        ))}
      </div>
      {selectedTags.size > 0 && (
        <div className="mt-2 pt-2 border-t border-gray-100">
          <Button
            variant="primary"
            size="sm"
            onClick={() => onSelect(Array.from(selectedTags))}
            className="w-full"
          >
            Add {selectedTags.size} tag{selectedTags.size > 1 ? "s" : ""}
          </Button>
        </div>
      )}
    </div>
  );
}

export function BulkActions({
  selectedCount,
  onClearSelection,
  onAddTags,
  onAddToGroup,
  onRemoveFromGroup,
  onExportCsv,
  onBatchAnalyze,
  onDelete,
  groups,
  availableTags,
  isProcessing,
  enableRemoveFromGroup = false,
  enableDelete = false,
}: BulkActionsProps) {
  if (selectedCount === 0) return null;

  return (
    <div className="fixed bottom-0 left-0 right-0 z-30 animate-slide-up">
      <div className="mx-auto max-w-6xl px-4 pb-4">
        <div className="bg-navy-900 rounded-xl shadow-2xl border border-navy-700/50 px-4 py-3">
          <div className="flex items-center justify-between gap-4 flex-wrap">
            {/* Selection info */}
            <div className="flex items-center gap-3">
              <Badge variant="clinical" size="md" className="font-semibold">
                {selectedCount} selected
              </Badge>
              <button
                onClick={onClearSelection}
                className="flex items-center gap-1 text-sm text-gray-400 hover:text-white transition-colors"
              >
                <X className="h-4 w-4" />
                Clear
              </button>
            </div>

            {/* Actions */}
            <div className="flex items-center gap-2 flex-wrap">
              {/* Add Tags */}
              <DropdownMenu
                trigger={
                  <Button variant="secondary" size="sm" className="gap-1.5">
                    <Tags className="h-4 w-4" />
                    Add Tags
                    <ChevronDown className="h-3 w-3" />
                  </Button>
                }
              >
                {availableTags.length > 0 ? (
                  <TagSelector
                    availableTags={availableTags}
                    onSelect={onAddTags}
                  />
                ) : (
                  <p className="px-4 py-2 text-sm text-gray-500">No tags available</p>
                )}
              </DropdownMenu>

              {/* Add to Group */}
              <DropdownMenu
                trigger={
                  <Button variant="secondary" size="sm" className="gap-1.5">
                    <FolderPlus className="h-4 w-4" />
                    Add to Group
                    <ChevronDown className="h-3 w-3" />
                  </Button>
                }
              >
                {groups.length > 0 ? (
                  <div className="max-h-48 overflow-y-auto">
                    {groups.map((group) => (
                      <button
                        key={group.id}
                        onClick={() => onAddToGroup(group.id)}
                        className="w-full flex items-center justify-between px-4 py-2 text-sm text-gray-700 hover:bg-gray-50"
                      >
                        <span>{group.name}</span>
                        <span className="text-xs text-gray-400">{group.slideIds.length}</span>
                      </button>
                    ))}
                  </div>
                ) : (
                  <p className="px-4 py-2 text-sm text-gray-500">No groups available</p>
                )}
              </DropdownMenu>

              {enableRemoveFromGroup && onRemoveFromGroup && (
                <DropdownMenu
                  trigger={
                    <Button variant="secondary" size="sm" className="gap-1.5">
                      <FolderMinus className="h-4 w-4" />
                      Remove from Group
                      <ChevronDown className="h-3 w-3" />
                    </Button>
                  }
                >
                  {groups.length > 0 ? (
                    <div className="max-h-48 overflow-y-auto">
                      {groups.map((group) => (
                        <button
                          key={group.id}
                          onClick={() => onRemoveFromGroup(group.id)}
                          className="w-full flex items-center justify-between px-4 py-2 text-sm text-gray-700 hover:bg-gray-50"
                        >
                          <span>{group.name}</span>
                          <span className="text-xs text-gray-400">{group.slideIds.length}</span>
                        </button>
                      ))}
                    </div>
                  ) : (
                    <p className="px-4 py-2 text-sm text-gray-500">No groups available</p>
                  )}
                </DropdownMenu>
              )}

              {/* Divider */}
              <div className="h-6 w-px bg-navy-700 mx-1" />

              {/* Export CSV */}
              <Button
                variant="secondary"
                size="sm"
                onClick={onExportCsv}
                className="gap-1.5"
              >
                <Download className="h-4 w-4" />
                Export CSV
              </Button>

              {/* Batch Analyze */}
              <Button
                variant="primary"
                size="sm"
                onClick={onBatchAnalyze}
                isLoading={isProcessing}
                className="gap-1.5"
              >
                <Microscope className="h-4 w-4" />
                Batch Analyze
              </Button>

              {enableDelete && onDelete && (
                <>
                  {/* Divider */}
                  <div className="h-6 w-px bg-navy-700 mx-1" />

                  {/* Delete */}
                  <Button
                    variant="danger"
                    size="sm"
                    onClick={onDelete}
                    className="gap-1.5"
                  >
                    <Trash2 className="h-4 w-4" />
                    Delete
                  </Button>
                </>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
