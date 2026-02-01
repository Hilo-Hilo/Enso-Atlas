"use client";

import React, { useEffect, useRef, useState } from "react";
import OpenSeadragon from "openseadragon";
import { cn } from "@/lib/utils";
import { Toggle } from "@/components/ui/Toggle";
import { Slider } from "@/components/ui/Slider";
import { Button } from "@/components/ui/Button";
import {
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Layers,
  Maximize2,
} from "lucide-react";
import type { PatchCoordinates, HeatmapData } from "@/types";

interface WSIViewerProps {
  slideId: string;
  dziUrl: string;
  heatmap?: HeatmapData;
  onRegionClick?: (coords: PatchCoordinates) => void;
  className?: string;
}

export function WSIViewer({
  slideId,
  dziUrl,
  heatmap,
  onRegionClick,
  className,
}: WSIViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<OpenSeadragon.Viewer | null>(null);
  const heatmapOverlayRef = useRef<HTMLImageElement | null>(null);

  const [isReady, setIsReady] = useState(false);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [heatmapOpacity, setHeatmapOpacity] = useState(0.5);
  const [zoom, setZoom] = useState(1);
  const [showControls, setShowControls] = useState(true);

  // Initialize OpenSeadragon viewer
  useEffect(() => {
    if (!containerRef.current || !dziUrl) return;

    const containerId = `wsi-viewer-${slideId}`;
    containerRef.current.id = containerId;

    const viewer = OpenSeadragon({
      id: containerId,
      prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/images/",
      tileSources: dziUrl,
      showNavigator: true,
      navigatorPosition: "BOTTOM_RIGHT",
      navigatorHeight: "100px",
      navigatorWidth: "150px",
      navigatorAutoFade: false,
      showRotationControl: false,
      showFullPageControl: false,
      showZoomControl: false,
      gestureSettingsMouse: {
        clickToZoom: false,
        dblClickToZoom: true,
        scrollToZoom: true,
      },
      minZoomLevel: 0.1,
      maxZoomLevel: 40,
      visibilityRatio: 0.5,
      constrainDuringPan: true,
      animationTime: 0.3,
      zoomPerClick: 2,
      zoomPerScroll: 1.2,
      crossOriginPolicy: "Anonymous",
    });

    viewer.addHandler("open", () => {
      setIsReady(true);
    });

    viewer.addHandler("zoom", (event: OpenSeadragon.ZoomEvent) => {
      if (event.zoom) {
        setZoom(event.zoom);
      }
    });

    viewer.addHandler("canvas-click", (event: OpenSeadragon.CanvasClickEvent) => {
      if (onRegionClick && event.quick) {
        const tiledImage = viewer.world.getItemAt(0);
        if (tiledImage) {
          const viewportPoint = viewer.viewport.pointFromPixel(event.position);
          const imagePoint = tiledImage.viewportToImageCoordinates(viewportPoint);
          onRegionClick({
            x: Math.floor(imagePoint.x),
            y: Math.floor(imagePoint.y),
            level: 0,
            width: 224,
            height: 224,
          });
        }
      }
    });

    viewerRef.current = viewer;

    return () => {
      viewer.destroy();
      viewerRef.current = null;
      setIsReady(false);
    };
  }, [dziUrl, slideId, onRegionClick]);

  // Handle heatmap overlay
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || !isReady || !heatmap) return;

    // Add heatmap as an overlay
    if (heatmapOverlayRef.current) {
      viewer.removeOverlay(heatmapOverlayRef.current);
    }

    const img = document.createElement("img");
    img.src = heatmap.imageUrl;
    img.style.opacity = showHeatmap ? String(heatmapOpacity) : "0";
    img.style.pointerEvents = "none";
    img.className = "transition-opacity duration-300";
    heatmapOverlayRef.current = img;

    viewer.addOverlay({
      element: img,
      location: new OpenSeadragon.Rect(0, 0, 1, 1),
    });
  }, [heatmap, isReady, showHeatmap, heatmapOpacity]);

  // Update heatmap opacity
  useEffect(() => {
    if (heatmapOverlayRef.current) {
      heatmapOverlayRef.current.style.opacity = showHeatmap
        ? String(heatmapOpacity)
        : "0";
    }
  }, [showHeatmap, heatmapOpacity]);

  const handleZoomIn = () => {
    viewerRef.current?.viewport.zoomBy(1.5);
  };

  const handleZoomOut = () => {
    viewerRef.current?.viewport.zoomBy(0.67);
  };

  const handleReset = () => {
    viewerRef.current?.viewport.goHome();
  };

  const handleFullscreen = () => {
    if (containerRef.current) {
      if (document.fullscreenElement) {
        document.exitFullscreen();
      } else {
        containerRef.current.requestFullscreen();
      }
    }
  };

  // Navigate to specific coordinates
  const navigateTo = (coords: PatchCoordinates) => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    const tiledImage = viewer.world.getItemAt(0);
    if (!tiledImage) return;

    const imageRect = new OpenSeadragon.Rect(
      coords.x,
      coords.y,
      coords.width,
      coords.height
    );
    const viewportRect = tiledImage.imageToViewportRectangle(imageRect);
    // Scale the rect by creating a new one with expanded dimensions
    const scaledRect = new OpenSeadragon.Rect(
      viewportRect.x - viewportRect.width,
      viewportRect.y - viewportRect.height,
      viewportRect.width * 3,
      viewportRect.height * 3
    );
    viewer.viewport.fitBounds(scaledRect, false);
  };

  // Expose navigateTo function via ref
  React.useImperativeHandle(
    React.createRef<{ navigateTo: (coords: PatchCoordinates) => void }>(),
    () => ({ navigateTo }),
    []
  );

  return (
    <div className={cn("relative bg-gray-900 rounded-lg overflow-hidden", className)}>
      {/* Viewer Container */}
      <div
        ref={containerRef}
        className="w-full h-full min-h-[400px]"
        style={{ background: "#1a1a2e" }}
      />

      {/* Loading Indicator */}
      {!isReady && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-clinical-400 mx-auto" />
            <p className="mt-4 text-sm text-gray-400">Loading slide...</p>
          </div>
        </div>
      )}

      {/* Viewer Controls */}
      {isReady && showControls && (
        <div className="absolute top-4 left-4 flex flex-col gap-2">
          {/* Zoom Controls */}
          <div className="flex flex-col gap-1 bg-white/90 backdrop-blur rounded-lg shadow-lg p-1">
            <Button
              variant="ghost"
              size="sm"
              onClick={handleZoomIn}
              title="Zoom In"
              className="p-2"
            >
              <ZoomIn className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={handleZoomOut}
              title="Zoom Out"
              className="p-2"
            >
              <ZoomOut className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={handleReset}
              title="Reset View"
              className="p-2"
            >
              <RotateCcw className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={handleFullscreen}
              title="Fullscreen"
              className="p-2"
            >
              <Maximize2 className="h-4 w-4" />
            </Button>
          </div>
        </div>
      )}

      {/* Heatmap Controls */}
      {isReady && heatmap && showControls && (
        <div className="absolute top-4 right-4 bg-white/90 backdrop-blur rounded-lg shadow-lg p-3 min-w-[180px]">
          <div className="flex items-center gap-2 mb-3">
            <Layers className="h-4 w-4 text-gray-600" />
            <span className="text-sm font-medium text-gray-700">Heatmap</span>
          </div>
          <Toggle
            checked={showHeatmap}
            onChange={setShowHeatmap}
            label="Show overlay"
            size="sm"
          />
          {showHeatmap && (
            <div className="mt-3">
              <Slider
                label="Opacity"
                min={0}
                max={1}
                step={0.05}
                value={heatmapOpacity}
                onChange={(e) => setHeatmapOpacity(Number(e.target.value))}
                formatValue={(v) => `${Math.round(v * 100)}%`}
              />
            </div>
          )}
        </div>
      )}

      {/* Zoom Level Indicator */}
      {isReady && (
        <div className="absolute bottom-4 left-4 bg-white/90 backdrop-blur rounded px-2 py-1">
          <span className="text-xs font-mono text-gray-600">
            {zoom.toFixed(1)}x
          </span>
        </div>
      )}
    </div>
  );
}

// Export navigateTo helper for external use
export type WSIViewerRef = {
  navigateTo: (coords: PatchCoordinates) => void;
};
