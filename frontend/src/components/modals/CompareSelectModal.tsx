"use client";

import React, { useCallback, useEffect } from "react";
import { Button } from "@/components/ui/Button";
import { X, GitCompare, Search } from "lucide-react";
import type { SlideInfo } from "@/types";

interface CompareSelectModalProps {
  isOpen: boolean;
  onClose: () => void;
  slideA: SlideInfo;
  slides: SlideInfo[];
  onSelectSlideB: (slide: SlideInfo) => void;
}

export function CompareSelectModal({
  isOpen,
  onClose,
  slideA,
  slides,
  onSelectSlideB,
}: CompareSelectModalProps) {
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

  const candidates = slides.filter((s) => s.id !== slideA.id);

  return (
    <div className="fixed inset-0 z-[300] flex items-center justify-center">
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />

      <div className="relative z-10 w-full max-w-2xl mx-4 bg-white dark:bg-navy-800 rounded-xl shadow-2xl overflow-hidden">
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 dark:border-navy-700 bg-gray-50 dark:bg-navy-900">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-clinical-600 flex items-center justify-center">
              <GitCompare className="h-5 w-5 text-white" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">Compare Cases</h2>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Select a second slide to compare against <span className="font-medium text-gray-700 dark:text-gray-300">{slideA.filename}</span>
              </p>
            </div>
          </div>
          <Button variant="ghost" size="sm" onClick={onClose} className="p-2">
            <X className="h-5 w-5" />
          </Button>
        </div>

        <div className="p-6 max-h-[70vh] overflow-y-auto">
          {candidates.length === 0 ? (
            <div className="text-center py-10 text-gray-500 dark:text-gray-400">
              <Search className="h-8 w-8 mx-auto mb-2 text-gray-400 dark:text-gray-500" />
              <p className="text-sm">No other slides available to compare.</p>
            </div>
          ) : (
            <div className="space-y-2">
              {candidates.map((slide) => (
                <button
                  key={slide.id}
                  onClick={() => onSelectSlideB(slide)}
                  className="w-full text-left p-4 rounded-lg border border-gray-200 dark:border-navy-700 hover:border-clinical-400 dark:hover:border-clinical-500 hover:bg-clinical-50/50 dark:hover:bg-clinical-900/30 transition-colors"
                >
                  <div className="flex items-center justify-between gap-3">
                    <div className="min-w-0">
                      <p className="text-sm font-medium text-gray-900 dark:text-gray-100 truncate">{slide.filename}</p>
                      <p className="text-xs text-gray-500 dark:text-gray-400 font-mono truncate">{slide.id}</p>
                    </div>
                    {slide.label && (
                      <span className="text-xs font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-navy-700 border border-gray-200 dark:border-navy-600 rounded px-2 py-1 shrink-0">
                        {slide.label}
                      </span>
                    )}
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="px-6 py-4 border-t border-gray-200 dark:border-navy-700 bg-gray-50 dark:bg-navy-900">
          <p className="text-xs text-gray-500 dark:text-gray-400">
            Tip: run analysis on both slides to compare predictions, evidence patches, and heatmaps.
          </p>
        </div>
      </div>
    </div>
  );
}
