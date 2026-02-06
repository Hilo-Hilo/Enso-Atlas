"use client";

import React, { useEffect, useRef, useState, useCallback } from "react";
import OpenSeadragon from "openseadragon";
import { cn } from "@/lib/utils";
import { Toggle } from "@/components/ui/Toggle";
import { Slider } from "@/components/ui/Slider";
import {
  Layers,
  Maximize2,
  Move,
  Crosshair,
  Grid3X3,
  Home,
  Minus,
  Plus,
  Settings2,
  ImageIcon,
} from "lucide-react";
import type { PatchCoordinates, HeatmapData } from "@/types";

// Heatmap model options
const HEATMAP_MODELS = [
  { id: null, name: "Legacy Model" },
  { id: "platinum_sensitivity", name: "Platinum Sensitivity" },
  { id: "tumor_grade", name: "Tumor Grade" },
  { id: "survival_5y", name: "5-Year Survival" },
  { id: "survival_3y", name: "3-Year Survival" },
  { id: "survival_1y", name: "1-Year Survival" },
] as const;

// Viewer control interface for keyboard shortcuts
export interface WSIViewerControls {
  zoomIn: () => void;
  zoomOut: () => void;
  resetZoom: () => void;
  toggleHeatmap: () => void;
  toggleHeatmapOnly: () => void;
  toggleFullscreen: () => void;
}

interface WSIViewerProps {
  slideId: string;
  dziUrl: string;
  heatmap?: HeatmapData;
  mpp?: number; // microns per pixel
  onRegionClick?: (coords: PatchCoordinates) => void;
  targetCoordinates?: PatchCoordinates | null; // Navigate to this location when set
  className?: string;
  // Heatmap model selection
  heatmapModel?: string | null;
  onHeatmapModelChange?: (model: string | null) => void;
  availableModels?: Array<{id: string; name: string}>;
  // Heatmap resolution level
  heatmapLevel?: number; // 0-4, default 2
  onHeatmapLevelChange?: (level: number) => void;
  onControlsReady?: (controls: WSIViewerControls) => void;
}

export function WSIViewer({
  slideId,
  dziUrl,
  heatmap,
  mpp = 0.5,
  onRegionClick,
  targetCoordinates,
  className,
  heatmapModel,
  onHeatmapModelChange,
  availableModels = [],
  heatmapLevel = 2,
  onHeatmapLevelChange,
  onControlsReady,
}: WSIViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<OpenSeadragon.Viewer | null>(null);
  const heatmapOverlayRef = useRef<HTMLElement | null>(null);
  const heatmapTiledImageRef = useRef<any>(null);
  
  // Store callbacks in refs so they don't trigger viewer recreation
  const onRegionClickRef = useRef(onRegionClick);
  useEffect(() => {
    onRegionClickRef.current = onRegionClick;
  }, [onRegionClick]);

  const [isReady, setIsReady] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [showHeatmap, setShowHeatmap] = useState(true); // Default to showing heatmap
  const [heatmapOnly, setHeatmapOnly] = useState(false); // Hide pathology, show only attention
  const [heatmapOpacity, setHeatmapOpacity] = useState(0.6);
  const [zoom, setZoom] = useState(1);
  const [showToolbar, setShowToolbar] = useState(true);
  const [showHeatmapPanel, setShowHeatmapPanel] = useState(false);
  const [activeTool, setActiveTool] = useState<"pan" | "crosshair">("pan");
  const [heatmapLoaded, setHeatmapLoaded] = useState(false);
  const [heatmapError, setHeatmapError] = useState(false);
  const heatmapImageUrl = heatmap?.imageUrl;
  
  // Store activeTool in ref for click handler
  const activeToolRef = useRef(activeTool);
  useEffect(() => {
    activeToolRef.current = activeTool;
  }, [activeTool]);

  // Calculate scale bar based on current zoom
  const getScaleBarInfo = useCallback(() => {
    const baseMag = 40; // Assuming 40x base magnification
    const effectiveMag = zoom * baseMag;

    const scaleBarPx = 100;
    const scaleBarMicrons = scaleBarPx * mpp / zoom;

    let displayValue: number;
    let displayUnit: string;

    if (scaleBarMicrons >= 1000) {
      displayValue = Math.round(scaleBarMicrons / 1000);
      displayUnit = "mm";
    } else if (scaleBarMicrons >= 100) {
      displayValue = Math.round(scaleBarMicrons / 100) * 100;
      displayUnit = "µm";
    } else if (scaleBarMicrons >= 10) {
      displayValue = Math.round(scaleBarMicrons / 10) * 10;
      displayUnit = "µm";
    } else {
      displayValue = Math.round(scaleBarMicrons);
      displayUnit = "µm";
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

    // Use ref for click handler to avoid recreating viewer when tool/callback changes
    viewer.addHandler("canvas-click", (event: OpenSeadragon.CanvasClickEvent) => {
      if (onRegionClickRef.current && event.quick && activeToolRef.current === "crosshair") {
        const tiledImage = viewer.world.getItemAt(0);
        if (tiledImage) {
          const viewportPoint = viewer.viewport.pointFromPixel(event.position);
          const imagePoint =
            tiledImage.viewportToImageCoordinates(viewportPoint);
          onRegionClickRef.current({
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
  }, [dziUrl, slideId]); // Removed onRegionClick and activeTool - they use refs now

  // Navigate to target coordinates when they change
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || !isReady || !targetCoordinates) return;

    const tiledImage = viewer.world.getItemAt(0);
    if (!tiledImage) return;

    // Convert image coordinates to viewport coordinates
    const imagePoint = new OpenSeadragon.Point(
      targetCoordinates.x + (targetCoordinates.width || 224) / 2,  // Center of patch
      targetCoordinates.y + (targetCoordinates.height || 224) / 2
    );
    const viewportPoint = tiledImage.imageToViewportCoordinates(imagePoint);

    // Calculate appropriate zoom level to show the patch in context
    // Aim for the patch to be about 1/4 of the viewport width
    const patchWidth = targetCoordinates.width || 224;
    const imageWidth = tiledImage.getContentSize().x;
    const patchViewportWidth = patchWidth / imageWidth;
    const targetZoom = 0.25 / patchViewportWidth;  // Patch takes up 25% of viewport
    const clampedZoom = Math.min(Math.max(targetZoom, 1), 20);  // Clamp between 1x and 20x

    // Pan and zoom to the target location
    viewer.viewport.panTo(viewportPoint, false);
    viewer.viewport.zoomTo(clampedZoom, viewportPoint, false);
  }, [targetCoordinates, isReady]);

  // Handle heatmap overlay using OSD's SimpleImage layer (not HTML overlay).
  // HTML overlays drift at high zoom due to sub-pixel CSS rounding.
  // addSimpleImage renders the heatmap in the same canvas/WebGL pipeline
  // as the slide tiles, so it's pixel-perfect at every zoom level.
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || !isReady || !heatmapImageUrl) return;

    // Remove existing heatmap tiled image if present
    if (heatmapTiledImageRef.current) {
      try {
        viewer.world.removeItem(heatmapTiledImageRef.current);
      } catch {
        // Ignore
      }
      heatmapTiledImageRef.current = null;
    }

    // Get the slide's bounds so the heatmap aligns exactly
    const slideImage = viewer.world.getItemAt(0);
    if (!slideImage) {
      console.warn("No slide image found for heatmap overlay");
      return;
    }
    const bounds = slideImage.getBounds(false);

    // Add heatmap as a simple image layer (rendered in OSD's tile pipeline)
    viewer.addSimpleImage({
      url: heatmapImageUrl,
      x: bounds.x,
      y: bounds.y,
      width: bounds.width,
      opacity: 0, // Start hidden, update via showHeatmap effect
      success: (event: any) => {
        heatmapTiledImageRef.current = event.item;
        // Apply pixelated rendering for the heatmap
        if (event.item?._drawer?.canvas) {
          event.item._drawer.canvas.style.imageRendering = "pixelated";
        }
        setHeatmapLoaded(true);
        setHeatmapError(false);
        console.log("Heatmap added as tiled image layer");
      },
      error: () => {
        setHeatmapError(true);
        setHeatmapLoaded(false);
        console.error("Failed to load heatmap image:", heatmapImageUrl);
      },
    });

    return () => {
      if (heatmapTiledImageRef.current && viewerRef.current) {
        try {
          viewerRef.current.world.removeItem(heatmapTiledImageRef.current);
        } catch {
          // Ignore
        }
        heatmapTiledImageRef.current = null;
      }
    };
  }, [heatmapImageUrl, isReady]);

  // Update heatmap opacity via the tiled image layer
  useEffect(() => {
    if (heatmapTiledImageRef.current && heatmapLoaded) {
      const targetOpacity = showHeatmap ? heatmapOpacity : 0;
      heatmapTiledImageRef.current.setOpacity(targetOpacity);
    }
  }, [showHeatmap, heatmapOpacity, heatmapLoaded]);

  // Hide/show main pathology tiles when heatmapOnly changes
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || !isReady) return;
    
    const tiledImage = viewer.world.getItemAt(0);
    if (tiledImage) {
      tiledImage.setOpacity(heatmapOnly ? 0 : 1);
    }
  }, [heatmapOnly, isReady]);

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

  // Expose controls to parent via callback
  useEffect(() => {
    if (onControlsReady && isReady) {
      onControlsReady({
        zoomIn: handleZoomIn,
        zoomOut: handleZoomOut,
        resetZoom: handleReset,
        toggleHeatmap: () => setShowHeatmap((prev) => !prev),
        toggleHeatmapOnly: () => {
          setHeatmapOnly((prev) => {
            if (!prev) setShowHeatmap(true);
            return !prev;
          });
        },
        toggleFullscreen: handleFullscreen,
      });
    }
  }, [isReady, onControlsReady]);

  const scaleInfo = getScaleBarInfo();

  // Model selector component
  const ModelSelector = () => (
    <div className="mb-3">
      <label className="text-xs text-gray-500 mb-1.5 block">Model</label>
      <select
        value={heatmapModel ?? ""}
        onChange={(e) => {
          const value = e.target.value;
          onHeatmapModelChange?.(value === "" ? null : value);
        }}
        className="w-full px-2 py-1.5 text-xs bg-white border border-gray-200 rounded-md text-gray-700 focus:outline-none focus:ring-2 focus:ring-clinical-500 focus:border-transparent"
      >
        {HEATMAP_MODELS.map((model) => (
          <option key={model.id ?? "legacy"} value={model.id ?? ""}>
            {model.name}
          </option>
        ))}
      </select>
    </div>
  );

  // When WSI isn't available but heatmap is, show standalone heatmap view
  if (loadError && heatmap) {
    return (
      <div
        className={cn(
          "relative bg-navy-900 rounded-xl overflow-hidden shadow-clinical-lg border border-navy-700",
          className
        )}
      >
        {/* Standalone Heatmap Display */}
        <div className="w-full h-full min-h-[400px] flex items-center justify-center bg-gradient-to-br from-navy-800 to-navy-900 p-8">
          <div className="text-center w-full max-w-2xl">
            {/* Info Header */}
            <div className="flex items-center justify-center gap-2 mb-4">
              <ImageIcon className="w-5 h-5 text-gray-400" />
              <span className="text-sm text-gray-400">
                Attention Heatmap (WSI not available)
              </span>
            </div>
            
            {/* Heatmap Image */}
            <div className="relative bg-navy-700 rounded-lg overflow-hidden shadow-xl">
              <img
                src={heatmap.imageUrl}
                alt="Attention heatmap"
                className="w-full h-auto max-h-[60vh] object-contain"
                style={{ 
                  opacity: showHeatmap ? heatmapOpacity : 0,
                  transition: "opacity 0.3s ease"
                }}
                onLoad={() => {
                  setHeatmapLoaded(true);
                  setHeatmapError(false);
                }}
                onError={() => {
                  setHeatmapError(true);
                  setHeatmapLoaded(false);
                }}
              />
              
              {/* Loading/Error states */}
              {!heatmapLoaded && !heatmapError && (
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-clinical-500" />
                </div>
              )}
              
              {heatmapError && (
                <div className="absolute inset-0 flex items-center justify-center text-gray-400">
                  <Grid3X3 className="w-12 h-12" />
                </div>
              )}
            </div>

            {/* Color Scale Legend */}
            <div className="mt-4 px-4">
              <div className="heatmap-legend h-3 rounded" />
              <div className="flex justify-between text-xs text-gray-400 mt-1">
                <span>Low attention</span>
                <span>High attention</span>
              </div>
            </div>

            {/* Info text */}
            <p className="text-xs text-gray-500 mt-4 px-4">
              The whole slide image is not available. This heatmap shows
              attention weights from pre-extracted patch embeddings.
            </p>
          </div>
        </div>

        {/* Heatmap Controls */}
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
                <ModelSelector />
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
        </div>
      </div>
    );
  }

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
        style={{ background: heatmapOnly ? "#000000" : "#0f172a" }}
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

      {/* Error State (no heatmap available) */}
      {loadError && !heatmap && (
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
            title="Select Region (click to analyze)"
          >
            <Crosshair className="h-4 w-4" />
          </button>

          <div className="toolbar-divider" />

          {/* Zoom Controls */}
          <button
            onClick={handleZoomOut}
            className="toolbar-button"
            title="Zoom Out (-)"
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
            title="Zoom In (+)"
          >
            <Plus className="h-4 w-4" />
          </button>

          <div className="toolbar-divider" />

          {/* View Controls */}
          <button
            onClick={handleReset}
            className="toolbar-button"
            title="Reset View (0)"
          >
            <Home className="h-4 w-4" />
          </button>
          <button
            onClick={handleFullscreen}
            className="toolbar-button"
            title="Fullscreen (F)"
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
                {heatmapLoaded && (
                  <span className="w-2 h-2 rounded-full bg-green-500" title="Loaded" />
                )}
                {heatmapError && (
                  <span className="w-2 h-2 rounded-full bg-red-500" title="Failed to load" />
                )}
              </div>
              <button
                onClick={() => setShowHeatmapPanel(!showHeatmapPanel)}
                className="p-1 rounded hover:bg-gray-100"
              >
                <Settings2 className="h-3.5 w-3.5 text-gray-400" />
              </button>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-xs text-gray-500">Show overlay (H)</span>
              <Toggle
                checked={showHeatmap}
                onChange={setShowHeatmap}
                size="sm"
              />
            </div>

            <div className="flex items-center justify-between">
              <span className="text-xs text-gray-500">Heatmap only (J)</span>
              <Toggle
                checked={heatmapOnly}
                onChange={(checked) => {
                  setHeatmapOnly(checked);
                  if (checked) setShowHeatmap(true);
                }}
                size="sm"
              />
            </div>

            {showHeatmapPanel && showHeatmap && (
              <div className="pt-2 border-t border-gray-100 animate-fade-in">
                <ModelSelector />
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
                  <div className="heatmap-legend h-3 rounded" />
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

      {/* Coordinates Display - when crosshair tool is active */}
      {isReady && activeTool === "crosshair" && (
        <div className="absolute bottom-4 right-4 bg-black/70 text-white px-3 py-1.5 rounded text-xs">
          <span className="text-gray-400 mr-1">Mode:</span>
          <span className="font-medium">Click to select region</span>
        </div>
      )}
    </div>
  );
}

// Export navigateTo method type for ref usage
export type WSIViewerRef = {
  navigateTo: (coords: PatchCoordinates) => void;
  zoomIn: () => void;
  zoomOut: () => void;
  resetZoom: () => void;
  toggleHeatmap: () => void;
};
