"use client";

import React, { useState, useEffect } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { Spinner } from "@/components/ui/Spinner";
import { cn } from "@/lib/utils";
import {
  FolderOpen,
  Image,
  Upload,
  RefreshCw,
  Check,
  AlertCircle,
} from "lucide-react";
import { getSlides } from "@/lib/api";
import type { SlideInfo } from "@/types";

interface SlideSelectorProps {
  selectedSlideId: string | null;
  onSlideSelect: (slide: SlideInfo) => void;
  onAnalyze: () => void;
  isAnalyzing?: boolean;
}

export function SlideSelector({
  selectedSlideId,
  onSlideSelect,
  onAnalyze,
  isAnalyzing,
}: SlideSelectorProps) {
  const [slides, setSlides] = useState<SlideInfo[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadSlides = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await getSlides();
      setSlides(response.slides);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load slides");
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadSlides();
  }, []);

  const selectedSlide = slides.find((s) => s.id === selectedSlideId);

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <FolderOpen className="h-4 w-4" />
            Slide Selection
          </CardTitle>
          <Button
            variant="ghost"
            size="sm"
            onClick={loadSlides}
            disabled={isLoading}
            className="p-1.5"
          >
            <RefreshCw
              className={cn("h-4 w-4", isLoading && "animate-spin")}
            />
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Error State */}
        {error && (
          <div className="p-3 bg-red-50 border border-red-200 rounded-lg flex items-center gap-2">
            <AlertCircle className="h-4 w-4 text-red-500" />
            <span className="text-sm text-red-700">{error}</span>
          </div>
        )}

        {/* Loading State */}
        {isLoading && (
          <div className="flex items-center justify-center py-8">
            <Spinner size="md" />
          </div>
        )}

        {/* Slide List */}
        {!isLoading && !error && (
          <div className="space-y-2 max-h-[240px] overflow-y-auto">
            {slides.length === 0 ? (
              <div className="text-center py-6 text-gray-500">
                <Image className="h-8 w-8 mx-auto mb-2 text-gray-400" />
                <p className="text-sm">No slides available</p>
                <p className="text-xs mt-1">Upload a WSI to get started</p>
              </div>
            ) : (
              slides.map((slide) => (
                <SlideItem
                  key={slide.id}
                  slide={slide}
                  isSelected={selectedSlideId === slide.id}
                  onClick={() => onSlideSelect(slide)}
                />
              ))
            )}
          </div>
        )}

        {/* Selected Slide Info */}
        {selectedSlide && (
          <div className="p-3 bg-clinical-50 border border-clinical-200 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <Check className="h-4 w-4 text-clinical-600" />
              <span className="text-sm font-medium text-clinical-800">
                Selected Slide
              </span>
            </div>
            <p className="text-sm text-clinical-700 truncate">
              {selectedSlide.filename}
            </p>
            <div className="flex items-center gap-3 mt-2 text-xs text-clinical-600">
              <span>
                {selectedSlide.dimensions.width} x{" "}
                {selectedSlide.dimensions.height}
              </span>
              {selectedSlide.magnification && (
                <span>{selectedSlide.magnification}x</span>
              )}
            </div>
          </div>
        )}

        {/* Analyze Button */}
        <Button
          variant="primary"
          size="lg"
          onClick={onAnalyze}
          disabled={!selectedSlideId || isAnalyzing}
          isLoading={isAnalyzing}
          className="w-full"
        >
          {isAnalyzing ? "Analyzing..." : "Analyze Slide"}
        </Button>

        {/* Upload Hint */}
        <div className="flex items-center justify-center gap-2 text-xs text-gray-500">
          <Upload className="h-3.5 w-3.5" />
          <span>Drag and drop WSI files or use the backend upload</span>
        </div>
      </CardContent>
    </Card>
  );
}

interface SlideItemProps {
  slide: SlideInfo;
  isSelected: boolean;
  onClick: () => void;
}

function SlideItem({ slide, isSelected, onClick }: SlideItemProps) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "w-full flex items-center gap-3 p-3 rounded-lg border transition-all text-left",
        "hover:border-clinical-400 hover:bg-clinical-50",
        isSelected
          ? "border-clinical-500 bg-clinical-50 ring-1 ring-clinical-200"
          : "border-gray-200 bg-white"
      )}
    >
      {/* Thumbnail */}
      <div className="w-12 h-12 rounded bg-gray-100 shrink-0 overflow-hidden">
        {slide.thumbnailUrl ? (
          <img
            src={slide.thumbnailUrl}
            alt={slide.filename}
            className="w-full h-full object-cover"
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center">
            <Image className="h-5 w-5 text-gray-400" />
          </div>
        )}
      </div>

      {/* Info */}
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-gray-900 truncate">
          {slide.filename}
        </p>
        <div className="flex items-center gap-2 mt-1">
          <span className="text-xs text-gray-500">
            {slide.dimensions.width} x {slide.dimensions.height}
          </span>
          {slide.magnification && (
            <Badge variant="default" size="sm">
              {slide.magnification}x
            </Badge>
          )}
        </div>
      </div>

      {/* Selection Indicator */}
      {isSelected && (
        <div className="shrink-0">
          <Check className="h-5 w-5 text-clinical-600" />
        </div>
      )}
    </button>
  );
}
