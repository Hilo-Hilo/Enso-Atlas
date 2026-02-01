"use client";

import React, { useEffect, useRef, useState, useCallback } from "react";
import OpenSeadragon from "openseadragon";
import { cn } from "@/lib/utils";
import { Toggle } from "@/components/ui/Toggle";
import { Slider } from "@/components/ui/Slider";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import {
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Layers,
  Maximize2,
  Move,
  Crosshair,
  Ruler,
  Grid3X3,
  Eye,
  EyeOff,
  Home,
  Minus,
  Plus,
  Settings2,
} from "lucide-react";
import type { PatchCoordinates, HeatmapData } from "@/types";

interface WSIViewerProps {
  slideId: string;
  dziUrl: string;
  heatmap?: HeatmapData;
  mpp?: number; // microns per pixel
  onRegionClick?: (coords: PatchCoordinates) => void;
  className?: string;
}

export function WSIViewer({
  slideId,
  dziUrl,
  heatmap,
  mpp = 0.5,
  onRegionClick,
  className,
}: WSIViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<OpenSeadragon.Viewer | null>(null);
  const heatmapOverlayRef = useRef<HTMLImageElement | null>(null);

  const [isReady, setIsReady] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [heatmapOpacity, setHeatmapOpacity] = useState(0.5);
  const [zoom, setZoom] = useState(1);
  const [showToolbar, setShowToolbar] = useState(true);
  const [showHeatmapPanel, setShowHeatmapPanel] = useState(false);
  const [activeTool, setActiveTool] = useState<"pan" | "crosshair">("pan");

  // Calculate scale bar based on current zoom
  const getScaleBarInfo = useCallback(() => {
    // Calculate effective magnification
    const baseMag = 40; // Assuming 40x base magnification
    const effectiveMag = zoom * baseMag;

    // Calculate scale bar width in microns for a 100px bar
    const scaleBarPx = 100;
    const scaleBarMicrons = scaleBarPx * mpp / zoom;

    // Round to nice values
    let displayValue: number;
    let displayUnit: string;

    if (scaleBarMicrons >= 1000) {
      displayValue = Math.round(scaleBarMicrons / 1000);
      displayUnit = "mm";
    } else if (scaleBarMicrons >= 100) {
      displayValue = Math.round(scaleBarMicrons / 100) * 100;
      displayUnit = "um";
    } else if (scaleBarMicrons >= 10) {
      displayValue = Math.round(scaleBarMicrons / 10) * 10;
      displayUnit = "um";
    } else {
      displayValue = Math.round(scaleBarMicrons);
      displayUnit = "um";
    }

    return { displayValue, displayUnit, effectiveMag };
  }, [zoom, mpp]);

  // Initialize OpenSeadragon viewer
  useEffect(() => {
    if (!containerRef.current || !dziUrl) return;

    setLoadError(null);
    setIsReady(false);

    const containerId = `wsi-viewer-${slideId}`;
    containerRef.current.id = containerId;

    const viewer = OpenSeadragon({
      id: containerId,
      prefixUrl:
        "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/images/",
      tileSources: dziUrl,
      showNavigator: true,
      navigatorPosition: "BOTTOM_RIGHT",
      navigatorHeight: "120px",
      navigatorWidth: "160px",
      navigatorAutoFade: false,
      navigatorBackground: "rgba(0, 0, 0, 0.6)",
      navigatorBorderColor: "rgba(255, 255, 255, 0.2)",
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
      setLoadError(null);
    });

    viewer.addHandler("open-failed", (event: OpenSeadragon.OpenFailedEvent) => {
      const message = event.message || "Failed to load slide";
      if (message.includes("404") || message.includes("Unable to open")) {
        setLoadError("WSI preview unavailable - embeddings only");
      } else {
        setLoadError(message);
      }
      setIsReady(false);
    });

    viewer.addHandler("zoom", (event: OpenSeadragon.ZoomEvent) => {
      if (event.zoom) {
        setZoom(event.zoom);
      }
    });

    viewer.addHandler("canvas-click", (event: OpenSeadragon.CanvasClickEvent) => {
      if (onRegionClick && event.quick && activeTool === "crosshair") {
        const tiledImage = viewer.world.getItemAt(0);
        if (tiledImage) {
          const viewportPoint = viewer.viewport.pointFromPixel(event.position);
          const imagePoint =
            tiledImage.viewportToImageCoordinates(viewportPoint);
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
  }, [dziUrl, slideId, onRegionClick, activeTool]);

  // Handle heatmap overlay
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || !isReady || !heatmap) return;

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

  const handleZoomIn = () => viewerRef.current?.viewport.zoomBy(1.5);
  const handleZoomOut = () => viewerRef.current?.viewport.zoomBy(0.67);
  const handleReset = () => viewerRef.current?.viewport.goHome();
  const handleFullscreen = () => {
    if (containerRef.current) {
      if (document.fullscreenElement) {
        document.exitFullscreen();
      } else {
        containerRef.current.requestFullscreen();
      }
    }
  };

  const scaleInfo = getScaleBarInfo();

  return (
    <div
      className={cn(
        "relative bg-navy-900 rounded-xl overflow-hidden shadow-clinical-lg border border-navy-700",
        className
      )}
    >
      {/* Viewer Container */}
      <div
        ref={containerRef}
        className="w-full h-full min-h-[400px]"
        style={{ background: "#0f172a" }}
      />

      {/* Loading Indicator */}
      {!isReady && !loadError && (
        <div className="absolute inset-0 flex items-center justify-center bg-navy-900">
          <div className="text-center">
            <div className="relative w-16 h-16 mb-4 mx-auto">
              <div className="absolute inset-0 rounded-full border-4 border-navy-700" />
              <div className="absolute inset-0 rounded-full border-4 border-clinical-500 border-t-transparent animate-spin" />
            </div>
            <p className="text-sm text-gray-300 font-medium">
              Loading slide...
            </p>
            <p className="text-xs text-gray-500 mt-1">
              Preparing tile pyramid
            </p>
          </div>
        </div>
      )}

      {/* Error State */}
      {loadError && (
        <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-navy-800 to-navy-900">
          <div className="text-center max-w-md px-6">
            <div className="w-20 h-20 mx-auto mb-4 rounded-2xl bg-navy-700 flex items-center justify-center">
              <Grid3X3 className="w-10 h-10 text-gray-400" />
            </div>
            <h3 className="text-lg font-semibold text-gray-200 mb-2">
              {loadError}
            </h3>
            <p className="text-sm text-gray-400 leading-relaxed">
              The whole slide image is not available for visualization.
              Predictions are computed from pre-extracted patch embeddings.
            </p>
          </div>
        </div>
      )}

      {/* Professional Toolbar */}
      {isReady && showToolbar && (
        <div className="absolute top-4 left-4 viewer-toolbar">
          {/* Navigation Tools */}
          <button
            onClick={() => setActiveTool("pan")}
            className={cn(
              "toolbar-button",
              activeTool === "pan" && "toolbar-button-active"
            )}
            title="Pan Tool (drag to move)"
          >
            <Move className="h-4 w-4" />
          </button>
          <button
            onClick={() => setActiveTool("crosshair")}
            className={cn(
              "toolbar-button",
              activeTool === "crosshair" && "toolbar-button-active"
            )}
            title="Select Region (click to navigate)"
          >
            <Crosshair className="h-4 w-4" />
          </button>

          <div className="toolbar-divider" />

          {/* Zoom Controls */}
          <button
            onClick={handleZoomOut}
            className="toolbar-button"
            title="Zoom Out"
          >
            <Minus className="h-4 w-4" />
          </button>
          <div className="px-2 min-w-[52px] text-center">
            <span className="text-xs font-mono text-gray-700">
              {zoom < 1 ? zoom.toFixed(2) : zoom.toFixed(1)}x
            </span>
          </div>
          <button
            onClick={handleZoomIn}
            className="toolbar-button"
            title="Zoom In"
          >
            <Plus className="h-4 w-4" />
          </button>

          <div className="toolbar-divider" />

          {/* View Controls */}
          <button
            onClick={handleReset}
            className="toolbar-button"
            title="Reset View"
          >
            <Home className="h-4 w-4" />
          </button>
          <button
            onClick={handleFullscreen}
            className="toolbar-button"
            title="Fullscreen"
          >
            <Maximize2 className="h-4 w-4" />
          </button>
        </div>
      )}

      {/* Heatmap Controls */}
      {isReady && heatmap && (
        <div className="absolute top-4 right-4">
          <div className="viewer-toolbar flex-col items-stretch gap-2 p-3 min-w-[200px]">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Layers className="h-4 w-4 text-gray-600" />
                <span className="text-sm font-medium text-gray-700">
                  Attention Heatmap
                </span>
              </div>
              <button
                onClick={() => setShowHeatmapPanel(!showHeatmapPanel)}
                className="p-1 rounded hover:bg-gray-100"
              >
                <Settings2 className="h-3.5 w-3.5 text-gray-400" />
              </button>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-xs text-gray-500">Show overlay</span>
              <Toggle
                checked={showHeatmap}
                onChange={setShowHeatmap}
                size="sm"
              />
            </div>

            {showHeatmapPanel && showHeatmap && (
              <div className="pt-2 border-t border-gray-100 animate-fade-in">
                <Slider
                  label="Opacity"
                  min={0}
                  max={1}
                  step={0.05}
                  value={heatmapOpacity}
                  onChange={(e) => setHeatmapOpacity(Number(e.target.value))}
                  formatValue={(v) => `${Math.round(v * 100)}%`}
                />
                <div className="mt-3">
                  <p className="text-xs text-gray-500 mb-1.5">Color Scale</p>
                  <div className="heatmap-legend" />
                  <div className="flex justify-between text-2xs text-gray-400 mt-1">
                    <span>Low attention</span>
                    <span>High attention</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Scale Bar */}
      {isReady && (
        <div className="absolute bottom-4 left-4 scale-bar">
          <div
            className="scale-bar-line"
            style={{ width: "100px" }}
          />
          <span>
            {scaleInfo.displayValue} {scaleInfo.displayUnit}
          </span>
        </div>
      )}

      {/* Magnification Indicator */}
      {isReady && (
        <div className="absolute bottom-4 left-36 bg-black/70 text-white px-2 py-1 rounded text-xs font-mono">
          {scaleInfo.effectiveMag.toFixed(1)}x effective
        </div>
      )}

      {/* Coordinates Display */}
      {isReady && activeTool === "crosshair" && (
        <div className="absolute bottom-4 right-4 bg-black/70 text-white px-3 py-1.5 rounded text-xs">
          <span className="text-gray-400 mr-1">Mode:</span>
          <span className="font-medium">Click to select region</span>
        </div>
      )}
    </div>
  );
}

export type WSIViewerRef = {
  navigateTo: (coords: PatchCoordinates) => void;
};
