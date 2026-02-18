"use client";

import React, { useEffect, useCallback } from "react";
import { cn, humanizeIdentifier } from "@/lib/utils";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { useProject } from "@/contexts/ProjectContext";
import {
  X,
  ZoomIn,
  ZoomOut,
  ChevronLeft,
  ChevronRight,
  MapPin,
  Target,
  Crosshair,
  Maximize2,
  Download,
} from "lucide-react";
import type { EvidencePatch } from "@/types";

interface PatchZoomModalProps {
  isOpen: boolean;
  onClose: () => void;
  patch: EvidencePatch | null;
  allPatches: EvidencePatch[];
  onNavigate?: (direction: "prev" | "next") => void;
  slideId?: string;
}

export function PatchZoomModal({
  isOpen,
  onClose,
  patch,
  allPatches,
  onNavigate,
  slideId,
}: PatchZoomModalProps) {
  const [zoomLevel, setZoomLevel] = React.useState(1);
  const { currentProject } = useProject();
  const predictionTargetLabel = humanizeIdentifier(currentProject.prediction_target).toLowerCase();

  // Sort patches by attention weight for rank
  const sortedPatches = [...allPatches].sort(
    (a, b) => b.attentionWeight - a.attentionWeight
  );
  const currentRank = patch
    ? sortedPatches.findIndex((p) => p.id === patch.id) + 1
    : 0;
  const currentIndex = patch
    ? allPatches.findIndex((p) => p.id === patch.id)
    : -1;

  // Keyboard navigation
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (!isOpen) return;

      switch (e.key) {
        case "Escape":
          onClose();
          break;
        case "ArrowLeft":
          if (currentIndex > 0) onNavigate?.("prev");
          break;
        case "ArrowRight":
          if (currentIndex < allPatches.length - 1) onNavigate?.("next");
          break;
        case "+":
        case "=":
          setZoomLevel((z) => Math.min(z + 0.25, 3));
          break;
        case "-":
          setZoomLevel((z) => Math.max(z - 0.25, 0.5));
          break;
      }
    },
    [isOpen, onClose, onNavigate, currentIndex, allPatches.length]
  );

  useEffect(() => {
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [handleKeyDown]);

  // Reset zoom when patch changes
  useEffect(() => {
    setZoomLevel(1);
  }, [patch?.id]);

  // Prevent scroll on body when modal is open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "";
    }
    return () => {
      document.body.style.overflow = "";
    };
  }, [isOpen]);

  if (!isOpen || !patch) return null;

  const attentionPercent = Math.round(patch.attentionWeight * 100);
  const attentionColor =
    attentionPercent >= 70
      ? "bg-red-500"
      : attentionPercent >= 40
      ? "bg-amber-500"
      : "bg-blue-500";

  // Generate morphology description based on attention weight
  const getMorphologyDescription = (patch: EvidencePatch) => {
    if (patch.morphologyDescription) return patch.morphologyDescription;

    const weight = patch.attentionWeight;
    if (weight >= 0.7) {
      return `High attention region showing distinctive cellular patterns that strongly influence the model's prediction. This area contains morphological features characteristic of the predicted ${predictionTargetLabel} class.`;
    } else if (weight >= 0.4) {
      return "Moderate attention region with notable tissue architecture. The cellular composition and spatial arrangement in this patch contribute to the overall prediction confidence.";
    } else {
      return "Lower attention region that provides supporting context. While less influential than top-ranked patches, the tissue patterns here still contribute to the holistic slide assessment.";
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/80 backdrop-blur-sm animate-fade-in"
        onClick={onClose}
      />

      {/* Modal Content */}
      <div className="relative z-10 w-full max-w-4xl mx-4 bg-white rounded-xl shadow-2xl overflow-hidden animate-scale-in">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 bg-gray-50">
          <div className="flex items-center gap-3">
            <div
              className={cn(
                "w-10 h-10 rounded-lg flex items-center justify-center text-white font-bold text-lg",
                attentionColor
              )}
            >
              #{currentRank}
            </div>
            <div>
              <h2 className="text-lg font-semibold text-gray-900">
                Evidence Patch Analysis
              </h2>
              <p className="text-sm text-gray-500">
                Patch {patch.patchId.slice(0, 12)}... | Attention:{" "}
                {attentionPercent}%
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {/* Zoom Controls */}
            <div className="flex items-center gap-1 bg-gray-100 rounded-lg p-1">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setZoomLevel((z) => Math.max(z - 0.25, 0.5))}
                disabled={zoomLevel <= 0.5}
                className="p-1.5"
              >
                <ZoomOut className="h-4 w-4" />
              </Button>
              <span className="text-xs font-mono px-2 min-w-[3rem] text-center">
                {Math.round(zoomLevel * 100)}%
              </span>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setZoomLevel((z) => Math.min(z + 0.25, 3))}
                disabled={zoomLevel >= 3}
                className="p-1.5"
              >
                <ZoomIn className="h-4 w-4" />
              </Button>
            </div>

            {/* Close Button */}
            <Button
              variant="ghost"
              size="sm"
              onClick={onClose}
              className="p-2"
            >
              <X className="h-5 w-5" />
            </Button>
          </div>
        </div>

        {/* Main Content */}
        <div className="flex">
          {/* Image Panel */}
          <div className="flex-1 relative bg-gray-900 aspect-square max-h-[60vh] overflow-hidden">
            {/* Navigation Arrows */}
            {currentIndex > 0 && (
              <button
                onClick={() => onNavigate?.("prev")}
                className="absolute left-4 top-1/2 -translate-y-1/2 z-10 w-10 h-10 rounded-full bg-white/90 hover:bg-white shadow-lg flex items-center justify-center transition-all"
              >
                <ChevronLeft className="h-6 w-6 text-gray-700" />
              </button>
            )}
            {currentIndex < allPatches.length - 1 && (
              <button
                onClick={() => onNavigate?.("next")}
                className="absolute right-4 top-1/2 -translate-y-1/2 z-10 w-10 h-10 rounded-full bg-white/90 hover:bg-white shadow-lg flex items-center justify-center transition-all"
              >
                <ChevronRight className="h-6 w-6 text-gray-700" />
              </button>
            )}

            {/* Patch Image */}
            <div
              className="w-full h-full flex items-center justify-center p-8 overflow-auto"
              style={{ cursor: zoomLevel > 1 ? "grab" : "default" }}
            >
              <img
                src={patch.thumbnailUrl}
                alt={`Evidence patch ${currentRank}`}
                className="max-w-full max-h-full object-contain rounded-lg shadow-xl transition-transform duration-200"
                style={{ transform: `scale(${zoomLevel})` }}
              />
            </div>

            {/* Image Counter */}
            <div className="absolute bottom-4 left-1/2 -translate-x-1/2 bg-black/70 text-white text-sm font-medium px-4 py-1.5 rounded-full">
              {currentIndex + 1} / {allPatches.length}
            </div>
          </div>

          {/* Info Panel */}
          <div className="w-80 bg-white border-l border-gray-200 p-6 overflow-y-auto max-h-[60vh]">
            {/* Attention Score */}
            <div className="mb-6">
              <h3 className="text-sm font-semibold text-gray-900 mb-3 flex items-center gap-2">
                <Target className="h-4 w-4 text-clinical-600" />
                Attention Score
              </h3>
              <div className="relative pt-2">
                <div className="h-3 bg-gray-100 rounded-full overflow-hidden">
                  <div
                    className={cn("h-full rounded-full transition-all", attentionColor)}
                    style={{ width: `${attentionPercent}%` }}
                  />
                </div>
                <div className="flex justify-between mt-2 text-xs text-gray-500">
                  <span>Low</span>
                  <span className="font-semibold text-gray-900">
                    {attentionPercent}%
                  </span>
                  <span>High</span>
                </div>
              </div>
            </div>

            {/* Coordinates */}
            <div className="mb-6">
              <h3 className="text-sm font-semibold text-gray-900 mb-3 flex items-center gap-2">
                <MapPin className="h-4 w-4 text-clinical-600" />
                Location
              </h3>
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-gray-50 rounded-lg p-3">
                  <p className="text-xs text-gray-500 mb-1">X Position</p>
                  <p className="text-sm font-mono font-medium text-gray-900">
                    {patch.coordinates.x.toLocaleString()}
                  </p>
                </div>
                <div className="bg-gray-50 rounded-lg p-3">
                  <p className="text-xs text-gray-500 mb-1">Y Position</p>
                  <p className="text-sm font-mono font-medium text-gray-900">
                    {patch.coordinates.y.toLocaleString()}
                  </p>
                </div>
                <div className="bg-gray-50 rounded-lg p-3">
                  <p className="text-xs text-gray-500 mb-1">Patch Size</p>
                  <p className="text-sm font-mono font-medium text-gray-900">
                    {patch.coordinates.width}x{patch.coordinates.height}
                  </p>
                </div>
                <div className="bg-gray-50 rounded-lg p-3">
                  <p className="text-xs text-gray-500 mb-1">Level</p>
                  <p className="text-sm font-mono font-medium text-gray-900">
                    {patch.coordinates.level}
                  </p>
                </div>
              </div>
            </div>

            {/* Morphology Description */}
            <div className="mb-6">
              <h3 className="text-sm font-semibold text-gray-900 mb-3 flex items-center gap-2">
                <Crosshair className="h-4 w-4 text-clinical-600" />
                Morphology Insights
              </h3>
              <div className="bg-clinical-50 border border-clinical-200 rounded-lg p-4">
                <p className="text-sm text-gray-700 leading-relaxed">
                  {getMorphologyDescription(patch)}
                </p>
              </div>
            </div>

            {/* Rank Badge */}
            <div className="mb-6">
              <h3 className="text-sm font-semibold text-gray-900 mb-3">
                Evidence Ranking
              </h3>
              <div className="flex items-center gap-3">
                <Badge
                  variant={
                    currentRank <= 3
                      ? "success"
                      : currentRank <= 6
                      ? "warning"
                      : "default"
                  }
                  size="md"
                >
                  Top {currentRank} of {allPatches.length}
                </Badge>
                {currentRank <= 3 && (
                  <span className="text-xs text-clinical-600 font-medium">
                    Primary Evidence
                  </span>
                )}
              </div>
            </div>

            {/* Keyboard Shortcuts */}
            <div className="border-t border-gray-200 pt-4">
              <p className="text-xs text-gray-500 mb-2 font-medium">
                Keyboard Shortcuts
              </p>
              <div className="space-y-1.5 text-xs text-gray-500">
                <div className="flex justify-between">
                  <span>Navigate patches</span>
                  <span className="font-mono bg-gray-100 px-1.5 py-0.5 rounded">
                    Arrow Left/Right
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Zoom in/out</span>
                  <span className="font-mono bg-gray-100 px-1.5 py-0.5 rounded">
                    +/-
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Close modal</span>
                  <span className="font-mono bg-gray-100 px-1.5 py-0.5 rounded">
                    Esc
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
