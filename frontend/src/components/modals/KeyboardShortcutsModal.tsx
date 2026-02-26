"use client";

import React, { useEffect, useCallback } from "react";
import { Button } from "@/components/ui/Button";
import {
  X,
  Keyboard,
  Navigation,
  ZoomIn,
  LayoutGrid,
  MousePointer,
} from "lucide-react";
import { cn } from "@/lib/utils";
import {
  type KeyboardShortcut,
  formatShortcutKey,
  groupShortcutsByCategory,
} from "@/hooks/useKeyboardShortcuts";

interface KeyboardShortcutsModalProps {
  isOpen: boolean;
  onClose: () => void;
  shortcuts: KeyboardShortcut[];
}

const CATEGORY_ICONS: Record<string, React.ReactNode> = {
  Navigation: <Navigation className="h-4 w-4" />,
  Viewer: <ZoomIn className="h-4 w-4" />,
  Panels: <LayoutGrid className="h-4 w-4" />,
  Actions: <MousePointer className="h-4 w-4" />,
};

const CATEGORY_ORDER = ["Navigation", "Viewer", "Panels", "Actions"];

export function KeyboardShortcutsModal({
  isOpen,
  onClose,
  shortcuts,
}: KeyboardShortcutsModalProps) {
  // Handle escape key
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === "Escape" && isOpen) {
        onClose();
      }
    },
    [isOpen, onClose]
  );

  useEffect(() => {
    if (typeof window === "undefined") return;

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [handleKeyDown]);

  // Prevent scroll when modal is open
  useEffect(() => {
    if (typeof window === "undefined") return;

    if (isOpen) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "";
    }
    return () => {
      document.body.style.overflow = "";
    };
  }, [isOpen]);

  if (!isOpen) return null;

  // Filter out hidden shortcuts (aliases) before grouping
  const visibleShortcuts = shortcuts.filter((s) => !s.hidden);
  const groupedShortcuts = groupShortcutsByCategory(visibleShortcuts);

  // Sort categories by predefined order
  const sortedCategories = Array.from(groupedShortcuts.keys()).sort((a, b) => {
    const aIndex = CATEGORY_ORDER.indexOf(a);
    const bIndex = CATEGORY_ORDER.indexOf(b);
    if (aIndex === -1 && bIndex === -1) return a.localeCompare(b);
    if (aIndex === -1) return 1;
    if (bIndex === -1) return -1;
    return aIndex - bIndex;
  });

  return (
    <div className="fixed inset-0 z-[300] flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm animate-fade-in"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative z-10 w-full max-w-2xl mx-4 bg-white dark:bg-navy-800 rounded-xl shadow-2xl overflow-hidden animate-scale-in">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 dark:border-navy-700 bg-gray-50 dark:bg-navy-900">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-clinical-600 flex items-center justify-center">
              <Keyboard className="h-5 w-5 text-white" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                Keyboard Shortcuts
              </h2>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Quick access for power users
              </p>
            </div>
          </div>
          <Button variant="ghost" size="sm" onClick={onClose} className="p-2">
            <X className="h-5 w-5" />
          </Button>
        </div>

        {/* Content */}
        <div className="p-6 max-h-[70vh] overflow-y-auto">
          <div className="grid grid-cols-2 gap-6">
            {sortedCategories.map((category) => (
              <div key={category} className="space-y-3">
                <div className="flex items-center gap-2 text-sm font-semibold text-gray-700 dark:text-gray-300">
                  {CATEGORY_ICONS[category] || <Keyboard className="h-4 w-4" />}
                  <span>{category}</span>
                </div>
                <div className="space-y-2">
                  {groupedShortcuts.get(category)?.map((shortcut, index) => (
                    <ShortcutItem key={index} shortcut={shortcut} />
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-gray-200 dark:border-navy-700 bg-gray-50 dark:bg-navy-900">
          <p className="text-xs text-gray-500 dark:text-gray-400 text-center">
            Press <kbd className="kbd">?</kbd> to toggle this help dialog
          </p>
        </div>
      </div>
    </div>
  );
}

interface ShortcutItemProps {
  shortcut: KeyboardShortcut;
}

function ShortcutItem({ shortcut }: ShortcutItemProps) {
  const displayKey = formatShortcutKey(shortcut.key, shortcut.modifiers);

  return (
    <div className="flex items-center justify-between py-1.5">
      <span className="text-sm text-gray-600 dark:text-gray-300">{shortcut.description}</span>
      <kbd className="kbd">{displayKey}</kbd>
    </div>
  );
}
