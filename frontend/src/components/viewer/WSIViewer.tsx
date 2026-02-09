"use client";

import React, { useEffect, useRef, useState, useCallback } from "react";
import OpenSeadragon from "openseadragon";

// Patch Canvas2D context creation to suppress "willReadFrequently" warnings.
// OpenSeadragon reads tile pixel data frequently via getImageData; Chrome warns
// unless the context was created with { willReadFrequently: true }.
if (typeof window !== "undefined") {
  const origGetContext = HTMLCanvasElement.prototype.getContext;
  // @ts-expect-error — overloaded signature mismatch is fine for this patch
  HTMLCanvasElement.prototype.getContext = function (
    type: string,
    attrs?: Record<string, unknown>
  ) {
    if (type === "2d") {
      attrs = { ...attrs, willReadFrequently: true };
    }
    return origGetContext.call(this, type, attrs);
  };
}
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
import type { PatchCoordinates, HeatmapData, Annotation } from "@/types";

// Heatmap model options
const HEATMAP_MODELS = [
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
  zoomTo: (level: number) => void;
  getZoom: () => number;
  toggleHeatmap: () => void;
  toggleHeatmapOnly: () => void;
  toggleFullscreen: () => void;
  toggleGrid: () => void;
}

type AnnotationTool = "pointer" | "circle" | "rectangle" | "freehand" | "point";

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
  onZoomChange?: (zoom: number) => void;
  // Annotation support
  annotations?: Annotation[];
  activeAnnotationTool?: AnnotationTool;
  onAnnotationCreate?: (annotation: { type: string; coordinates: Annotation["coordinates"] }) => void;
  onAnnotationSelect?: (annotationId: string) => void;
  onAnnotationDelete?: (annotationId: string) => void;
  selectedAnnotationId?: string | null;
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
  onZoomChange,
  annotations = [],
  activeAnnotationTool = "pointer",
  onAnnotationCreate,
  onAnnotationSelect,
  onAnnotationDelete,
  selectedAnnotationId,
}: WSIViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerShellRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<OpenSeadragon.Viewer | null>(null);
  const heatmapOverlayRef = useRef<HTMLElement | null>(null);
  const heatmapTiledImageRef = useRef<any>(null);
  
  // Store callbacks in refs so they don't trigger viewer recreation
  const onRegionClickRef = useRef(onRegionClick);
  useEffect(() => {
    onRegionClickRef.current = onRegionClick;
  }, [onRegionClick]);

  const onZoomChangeRef = useRef(onZoomChange);
  useEffect(() => {
    onZoomChangeRef.current = onZoomChange;
  }, [onZoomChange]);

  const [isReady, setIsReady] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [showHeatmap, setShowHeatmap] = useState(true); // Default to showing heatmap
  const [heatmapOnly, setHeatmapOnly] = useState(false); // Hide pathology, show only attention
  const [showGrid, setShowGrid] = useState(false);
  const gridCanvasRef = useRef<HTMLCanvasElement | null>(null);
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

  // Annotation drawing state
  const svgOverlayRef = useRef<SVGSVGElement | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const drawStartRef = useRef<{ x: number; y: number } | null>(null);
  const freehandPointsRef = useRef<Array<{ x: number; y: number }>>([]);
  const [drawingPreview, setDrawingPreview] = useState<{
    type: string;
    x: number; y: number; width: number; height: number;
    points?: Array<{ x: number; y: number }>;
  } | null>(null);

  // Store annotation callbacks in refs
  const onAnnotationCreateRef = useRef(onAnnotationCreate);
  useEffect(() => { onAnnotationCreateRef.current = onAnnotationCreate; }, [onAnnotationCreate]);
  const activeAnnotationToolRef = useRef(activeAnnotationTool);
  useEffect(() => { activeAnnotationToolRef.current = activeAnnotationTool; }, [activeAnnotationTool]);

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
        onZoomChangeRef.current?.(event.zoom);
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
      } catch (err) {
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
        } catch (err) {
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

  // Draw 224px patch grid overlay on a canvas
  useEffect(() => {
    const viewer = viewerRef.current;
    const canvas = gridCanvasRef.current;
    if (!viewer || !canvas || !isReady) return;

    const PATCH = 224;

    const draw = () => {
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const container = canvas.parentElement;
      if (!container) return;

      const w = container.clientWidth;
      const h = container.clientHeight;
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }

      ctx.clearRect(0, 0, w, h);
      if (!showGrid) return;

      const tiledImage = viewer.world.getItemAt(0);
      if (!tiledImage) return;

      const contentSize = tiledImage.getContentSize();
      const imgW = contentSize.x;
      const imgH = contentSize.y;

      // Determine visible image-coordinate bounds
      const topLeftVP = viewer.viewport.pointFromPixel(new OpenSeadragon.Point(0, 0));
      const bottomRightVP = viewer.viewport.pointFromPixel(new OpenSeadragon.Point(w, h));
      const topLeftImg = tiledImage.viewportToImageCoordinates(topLeftVP);
      const bottomRightImg = tiledImage.viewportToImageCoordinates(bottomRightVP);

      const minX = Math.max(0, Math.floor(topLeftImg.x / PATCH) * PATCH);
      const maxX = Math.min(imgW, Math.ceil(bottomRightImg.x / PATCH) * PATCH);
      const minY = Math.max(0, Math.floor(topLeftImg.y / PATCH) * PATCH);
      const maxY = Math.min(imgH, Math.ceil(bottomRightImg.y / PATCH) * PATCH);

      ctx.strokeStyle = "rgba(0, 255, 255, 0.3)";
      ctx.lineWidth = 0.5;
      ctx.beginPath();

      // Vertical lines
      for (let ix = minX; ix <= maxX; ix += PATCH) {
        const vpTop = tiledImage.imageToViewportCoordinates(new OpenSeadragon.Point(ix, minY));
        const vpBot = tiledImage.imageToViewportCoordinates(new OpenSeadragon.Point(ix, maxY));
        const sTop = viewer.viewport.pixelFromPoint(vpTop);
        const sBot = viewer.viewport.pixelFromPoint(vpBot);
        ctx.moveTo(Math.round(sTop.x) + 0.5, sTop.y);
        ctx.lineTo(Math.round(sBot.x) + 0.5, sBot.y);
      }

      // Horizontal lines
      for (let iy = minY; iy <= maxY; iy += PATCH) {
        const vpLeft = tiledImage.imageToViewportCoordinates(new OpenSeadragon.Point(minX, iy));
        const vpRight = tiledImage.imageToViewportCoordinates(new OpenSeadragon.Point(maxX, iy));
        const sLeft = viewer.viewport.pixelFromPoint(vpLeft);
        const sRight = viewer.viewport.pixelFromPoint(vpRight);
        ctx.moveTo(sLeft.x, Math.round(sLeft.y) + 0.5);
        ctx.lineTo(sRight.x, Math.round(sRight.y) + 0.5);
      }

      ctx.stroke();
    };

    draw();
    viewer.addHandler("animation", draw);
    viewer.addHandler("resize", draw);

    return () => {
      viewer.removeHandler("animation", draw);
      viewer.removeHandler("resize", draw);
      // Clear canvas on cleanup
      const ctx = canvas.getContext("2d");
      if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
    };
  }, [isReady, showGrid]);

  // Helper: convert screen pixel position to image coordinates
  const screenToImageCoords = useCallback((screenX: number, screenY: number): { x: number; y: number } | null => {
    const viewer = viewerRef.current;
    if (!viewer) return null;
    const tiledImage = viewer.world.getItemAt(0);
    if (!tiledImage) return null;
    const containerEl = containerRef.current;
    if (!containerEl) return null;
    const rect = containerEl.getBoundingClientRect();
    const pixelPoint = new OpenSeadragon.Point(screenX - rect.left, screenY - rect.top);
    const viewportPoint = viewer.viewport.pointFromPixel(pixelPoint);
    const imagePoint = tiledImage.viewportToImageCoordinates(viewportPoint);
    return { x: Math.round(imagePoint.x), y: Math.round(imagePoint.y) };
  }, []);

  // Keep mouse navigation enabled so users can always pan/zoom.
  // Drawing temporarily disables navigation only while dragging.
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || !isReady) return;
    viewer.setMouseNavEnabled(true);
    const gestureSettingsMouse = (viewer as any).gestureSettingsMouse;
    if (gestureSettingsMouse) {
      gestureSettingsMouse.scrollToZoom = true;
    }
  }, [isReady]);

  // Annotation mouse handlers attached to the container
  useEffect(() => {
    const container = viewerShellRef.current;
    if (!container || !isReady) return;

    const handleMouseDown = (e: MouseEvent) => {
      const tool = activeAnnotationToolRef.current;
      if (tool === "pointer") return;
      if (e.button !== 0) return; // left click only

      const imgCoords = screenToImageCoords(e.clientX, e.clientY);
      if (!imgCoords) return;

      if (tool === "point") {
        // Point is single-click, create immediately
        onAnnotationCreateRef.current?.({
          type: "point",
          coordinates: { x: imgCoords.x, y: imgCoords.y, width: 0, height: 0 },
        });
        return;
      }

      e.preventDefault();
      e.stopPropagation();
      const viewer = viewerRef.current;
      if (viewer) viewer.setMouseNavEnabled(false);
      setIsDrawing(true);
      drawStartRef.current = imgCoords;
      freehandPointsRef.current = [imgCoords];

      if (tool === "freehand") {
        setDrawingPreview({ type: "freehand", x: 0, y: 0, width: 0, height: 0, points: [imgCoords] });
      } else {
        setDrawingPreview({ type: tool, x: imgCoords.x, y: imgCoords.y, width: 0, height: 0 });
      }
    };

    const handleMouseMove = (e: MouseEvent) => {
      if (!isDrawing || !drawStartRef.current) return;
      const tool = activeAnnotationToolRef.current;
      const imgCoords = screenToImageCoords(e.clientX, e.clientY);
      if (!imgCoords) return;

      if (tool === "freehand") {
        freehandPointsRef.current.push(imgCoords);
        setDrawingPreview({
          type: "freehand", x: 0, y: 0, width: 0, height: 0,
          points: [...freehandPointsRef.current],
        });
      } else {
        const start = drawStartRef.current;
        const x = Math.min(start.x, imgCoords.x);
        const y = Math.min(start.y, imgCoords.y);
        const w = Math.abs(imgCoords.x - start.x);
        const h = Math.abs(imgCoords.y - start.y);
        setDrawingPreview({ type: tool, x, y, width: w, height: h });
      }
    };

    const handleMouseUp = (e: MouseEvent) => {
      if (!isDrawing || !drawStartRef.current) return;
      const tool = activeAnnotationToolRef.current;
      setIsDrawing(false);
      const viewer = viewerRef.current;
      if (viewer) viewer.setMouseNavEnabled(true);
      setDrawingPreview(null);

      const imgCoords = screenToImageCoords(e.clientX, e.clientY);
      if (!imgCoords) {
        drawStartRef.current = null;
        freehandPointsRef.current = [];
        return;
      }

      if (tool === "freehand") {
        const pts = freehandPointsRef.current;
        if (pts.length < 2) return;
        // Compute bounding box
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        for (const p of pts) {
          if (p.x < minX) minX = p.x;
          if (p.y < minY) minY = p.y;
          if (p.x > maxX) maxX = p.x;
          if (p.y > maxY) maxY = p.y;
        }
        onAnnotationCreateRef.current?.({
          type: "freehand",
          coordinates: { x: minX, y: minY, width: maxX - minX, height: maxY - minY, points: pts },
        });
      } else {
        const start = drawStartRef.current;
        const x = Math.min(start.x, imgCoords.x);
        const y = Math.min(start.y, imgCoords.y);
        const w = Math.abs(imgCoords.x - start.x);
        const h = Math.abs(imgCoords.y - start.y);
        if (w < 5 && h < 5) return; // too small, ignore
        onAnnotationCreateRef.current?.({
          type: tool,
          coordinates: { x, y, width: w, height: h },
        });
      }

      drawStartRef.current = null;
      freehandPointsRef.current = [];
    };

    container.addEventListener("mousedown", handleMouseDown);
    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);

    return () => {
      container.removeEventListener("mousedown", handleMouseDown);
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isReady, isDrawing, screenToImageCoords]);

  // Convert image coordinates to viewport pixel coordinates for SVG rendering
  const imageToScreenCoords = useCallback((imgX: number, imgY: number): { x: number; y: number } | null => {
    const viewer = viewerRef.current;
    if (!viewer) return null;
    const tiledImage = viewer.world.getItemAt(0);
    if (!tiledImage) return null;
    const viewportPoint = tiledImage.imageToViewportCoordinates(new OpenSeadragon.Point(imgX, imgY));
    const pixelPoint = viewer.viewport.pixelFromPoint(viewportPoint);
    return { x: pixelPoint.x, y: pixelPoint.y };
  }, []);

  // Force SVG re-render on zoom/pan
  const [renderTick, setRenderTick] = useState(0);
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || !isReady) return;
    const handler = () => setRenderTick((t) => t + 1);
    viewer.addHandler("zoom", handler);
    viewer.addHandler("pan", handler);
    viewer.addHandler("animation", handler);
    return () => {
      viewer.removeHandler("zoom", handler);
      viewer.removeHandler("pan", handler);
      viewer.removeHandler("animation", handler);
    };
  }, [isReady]);

  // Render an annotation as SVG element
  const renderAnnotationSVG = useCallback((ann: Annotation, isSelected: boolean) => {
    const coords = ann.coordinates;
    const color = ann.color || "#3b82f6";
    const strokeWidth = isSelected ? 3 : 2;
    const opacity = isSelected ? 1 : 0.7;

    if (ann.type === "freehand" && coords.points && coords.points.length > 1) {
      const screenPoints = coords.points.map((p) => imageToScreenCoords(p.x, p.y)).filter(Boolean) as Array<{ x: number; y: number }>;
      if (screenPoints.length < 2) return null;
      const pathData = screenPoints.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x} ${p.y}`).join(" ");
      return (
        <path
          key={ann.id}
          d={pathData}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth}
          opacity={opacity}
          strokeLinejoin="round"
          strokeLinecap="round"
          style={{ cursor: "pointer", pointerEvents: "stroke" }}
          onClick={(e) => { e.stopPropagation(); onAnnotationSelect?.(ann.id); }}
        />
      );
    }

    if (ann.type === "point") {
      const sc = imageToScreenCoords(coords.x, coords.y);
      if (!sc) return null;
      return (
        <g key={ann.id} onClick={(e) => { e.stopPropagation(); onAnnotationSelect?.(ann.id); }} style={{ cursor: "pointer" }}>
          <circle cx={sc.x} cy={sc.y} r={isSelected ? 8 : 6} fill={color} opacity={opacity} stroke={isSelected ? "#fff" : "none"} strokeWidth={2} />
          <line x1={sc.x - 10} y1={sc.y} x2={sc.x + 10} y2={sc.y} stroke={color} strokeWidth={1.5} opacity={0.6} />
          <line x1={sc.x} y1={sc.y - 10} x2={sc.x} y2={sc.y + 10} stroke={color} strokeWidth={1.5} opacity={0.6} />
        </g>
      );
    }

    if (ann.type === "circle") {
      const cx = coords.x + coords.width / 2;
      const cy = coords.y + coords.height / 2;
      const scCenter = imageToScreenCoords(cx, cy);
      const scEdge = imageToScreenCoords(cx + coords.width / 2, cy);
      if (!scCenter || !scEdge) return null;
      const rx = Math.abs(scEdge.x - scCenter.x);
      const scEdgeY = imageToScreenCoords(cx, cy + coords.height / 2);
      const ry = scEdgeY ? Math.abs(scEdgeY.y - scCenter.y) : rx;
      return (
        <ellipse
          key={ann.id}
          cx={scCenter.x} cy={scCenter.y} rx={rx} ry={ry}
          fill="none" stroke={color} strokeWidth={strokeWidth} opacity={opacity}
          strokeDasharray={isSelected ? "none" : "6 3"}
          style={{ cursor: "pointer", pointerEvents: "stroke" }}
          onClick={(e) => { e.stopPropagation(); onAnnotationSelect?.(ann.id); }}
        />
      );
    }

    // Default: rectangle (also handles marker, note, measurement)
    const topLeft = imageToScreenCoords(coords.x, coords.y);
    const bottomRight = imageToScreenCoords(coords.x + coords.width, coords.y + coords.height);
    if (!topLeft || !bottomRight) return null;
    const w = bottomRight.x - topLeft.x;
    const h = bottomRight.y - topLeft.y;
    if (Math.abs(w) < 1 && Math.abs(h) < 1) return null;
    return (
      <rect
        key={ann.id}
        x={Math.min(topLeft.x, bottomRight.x)} y={Math.min(topLeft.y, bottomRight.y)}
        width={Math.abs(w)} height={Math.abs(h)}
        fill="none" stroke={color} strokeWidth={strokeWidth} opacity={opacity}
        strokeDasharray={isSelected ? "none" : "6 3"}
        style={{ cursor: "pointer", pointerEvents: "stroke" }}
        onClick={(e) => { e.stopPropagation(); onAnnotationSelect?.(ann.id); }}
      />
    );
  }, [imageToScreenCoords, onAnnotationSelect]);

  // Render the drawing preview
  const renderDrawingPreview = useCallback(() => {
    if (!drawingPreview) return null;
    const color = "#f59e0b";

    if (drawingPreview.type === "freehand" && drawingPreview.points) {
      const screenPoints = drawingPreview.points.map((p) => imageToScreenCoords(p.x, p.y)).filter(Boolean) as Array<{ x: number; y: number }>;
      if (screenPoints.length < 2) return null;
      const pathData = screenPoints.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x} ${p.y}`).join(" ");
      return <path d={pathData} fill="none" stroke={color} strokeWidth={2} strokeDasharray="4 2" />;
    }

    const topLeft = imageToScreenCoords(drawingPreview.x, drawingPreview.y);
    const bottomRight = imageToScreenCoords(
      drawingPreview.x + drawingPreview.width,
      drawingPreview.y + drawingPreview.height
    );
    if (!topLeft || !bottomRight) return null;
    const w = bottomRight.x - topLeft.x;
    const h = bottomRight.y - topLeft.y;

    if (drawingPreview.type === "circle") {
      return (
        <ellipse
          cx={topLeft.x + w / 2} cy={topLeft.y + h / 2}
          rx={Math.abs(w / 2)} ry={Math.abs(h / 2)}
          fill="none" stroke={color} strokeWidth={2} strokeDasharray="4 2"
        />
      );
    }

    return (
      <rect
        x={Math.min(topLeft.x, bottomRight.x)} y={Math.min(topLeft.y, bottomRight.y)}
        width={Math.abs(w)} height={Math.abs(h)}
        fill="none" stroke={color} strokeWidth={2} strokeDasharray="4 2"
      />
    );
  }, [drawingPreview, imageToScreenCoords]);

  const handleZoomIn = () => viewerRef.current?.viewport.zoomBy(1.5);
  const handleZoomOut = () => viewerRef.current?.viewport.zoomBy(0.67);
  const handleReset = () => viewerRef.current?.viewport.goHome();
  const handleZoomTo = (level: number) => viewerRef.current?.viewport.zoomTo(level);
  const handleGetZoom = () => viewerRef.current?.viewport.getZoom() ?? 1;
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
        zoomTo: handleZoomTo,
        getZoom: handleGetZoom,
        toggleHeatmap: () => setShowHeatmap((prev) => !prev),
        toggleHeatmapOnly: () => {
          setHeatmapOnly((prev) => {
            if (!prev) setShowHeatmap(true);
            return !prev;
          });
        },
        toggleFullscreen: handleFullscreen,
        toggleGrid: () => setShowGrid((prev) => !prev),
      });
    }
  }, [isReady, onControlsReady]);

  const scaleInfo = getScaleBarInfo();

  // Model selector component
  const ModelSelector = () => (
    <div className="mb-3">
      <label className="text-xs text-gray-500 mb-1.5 block">Model</label>
      <select
        value={heatmapModel ?? "platinum_sensitivity"}
        onChange={(e) => {
          onHeatmapModelChange?.(e.target.value);
        }}
        className="w-full px-2 py-1.5 text-xs bg-white border border-gray-200 rounded-md text-gray-700 focus:outline-none focus:ring-2 focus:ring-clinical-500 focus:border-transparent"
      >
        {HEATMAP_MODELS.map((model) => (
          <option key={model.id} value={model.id}>
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
      ref={viewerShellRef}
      className={cn(
        "relative bg-navy-900 rounded-xl overflow-hidden shadow-clinical-lg border border-navy-700",
        className
      )}
      style={{ cursor: activeAnnotationTool !== "pointer" ? "crosshair" : "default" }}
    >
      {/* Viewer Container */}
      <div
        ref={containerRef}
        className="w-full h-full min-h-[400px]"
        style={{ background: heatmapOnly ? "#000000" : "#0f172a" }}
      />

      {/* Patch grid overlay canvas */}
      <canvas
        ref={gridCanvasRef}
        className="absolute top-0 left-0 w-full h-full z-10"
        style={{ pointerEvents: "none" }}
      />

      {/* Annotation overlay (SVG rendered in screen coordinates) */}
      {isReady && (
        <svg
          key={renderTick}
          ref={svgOverlayRef}
          className="absolute inset-0 w-full h-full"
          style={{ pointerEvents: activeAnnotationTool !== "pointer" ? "auto" : "none" }}
          viewBox={`0 0 ${containerRef.current?.clientWidth || 1} ${containerRef.current?.clientHeight || 1}`}
          preserveAspectRatio="none"
        >
          {annotations.map((ann) => renderAnnotationSVG(ann, selectedAnnotationId === ann.id))}
          {renderDrawingPreview()}
        </svg>
      )}

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

      {/* Overlay Controls (always visible when viewer is ready) */}
      {isReady && (
        <div className="absolute top-4 right-4">
          <div className="viewer-toolbar flex-col items-stretch gap-2 p-3 min-w-[200px]">
            {heatmap && (
              <>
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
              </>
            )}

            <div className="flex items-center justify-between">
              <span className="text-xs text-gray-500">Patch grid (G)</span>
              <Toggle
                checked={showGrid}
                onChange={setShowGrid}
                size="sm"
              />
            </div>

            {heatmap && showHeatmapPanel && showHeatmap && (
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

      {/* Scale Bar + Magnification (single container to prevent overlap) */}
      {isReady && (
        <div className="absolute bottom-4 left-4 flex items-end gap-3">
          <div className="scale-bar">
            <div
              className="scale-bar-line"
              style={{ width: "100px" }}
            />
            <span>
              {scaleInfo.displayValue} {scaleInfo.displayUnit}
            </span>
          </div>
          <div className="bg-black/70 text-white px-2 py-1 rounded text-xs font-mono whitespace-nowrap">
            {scaleInfo.effectiveMag.toFixed(1)}x
          </div>
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
