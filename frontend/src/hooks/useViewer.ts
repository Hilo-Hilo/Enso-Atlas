// Enso Atlas - Viewer Hook
// OpenSeadragon viewer state management

import { useState, useCallback, useRef, useEffect } from "react";
import OpenSeadragon from "openseadragon";
import type { ViewerState, PatchCoordinates } from "@/types";

interface UseViewerOptions {
  containerId: string;
  dziUrl?: string;
  heatmapUrl?: string;
}

interface UseViewerReturn {
  viewer: OpenSeadragon.Viewer | null;
  viewerState: ViewerState;
  isReady: boolean;
  setHeatmapVisible: (visible: boolean) => void;
  setHeatmapOpacity: (opacity: number) => void;
  goToRegion: (coords: PatchCoordinates) => void;
  resetView: () => void;
  zoomIn: () => void;
  zoomOut: () => void;
}

const defaultViewerState: ViewerState = {
  zoom: 1,
  center: { x: 0.5, y: 0.5 },
  rotation: 0,
  showHeatmap: false,
  heatmapOpacity: 0.5,
};

export function useViewer(options: UseViewerOptions): UseViewerReturn {
  const { containerId, dziUrl, heatmapUrl } = options;
  const viewerRef = useRef<OpenSeadragon.Viewer | null>(null);
  const heatmapOverlayRef = useRef<OpenSeadragon.TiledImage | null>(null);

  const [viewerState, setViewerState] = useState<ViewerState>(defaultViewerState);
  const [isReady, setIsReady] = useState(false);

  // Initialize viewer
  useEffect(() => {
    if (!dziUrl) return;

    const viewer = OpenSeadragon({
      id: containerId,
      prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/images/",
      tileSources: dziUrl,
      showNavigator: true,
      navigatorPosition: "BOTTOM_RIGHT",
      navigatorHeight: "120px",
      navigatorWidth: "180px",
      showRotationControl: false,
      showFullPageControl: false,
      gestureSettingsMouse: {
        clickToZoom: false,
        dblClickToZoom: true,
      },
      minZoomLevel: 0.1,
      maxZoomLevel: 40,
      visibilityRatio: 0.5,
      constrainDuringPan: true,
      animationTime: 0.5,
      zoomPerClick: 2,
      zoomPerScroll: 1.2,
    });

    viewer.addHandler("open", () => {
      setIsReady(true);
    });

    viewer.addHandler("zoom", (event: OpenSeadragon.ZoomEvent) => {
      setViewerState((prev) => ({
        ...prev,
        zoom: event.zoom || prev.zoom,
      }));
    });

    viewer.addHandler("pan", (event: OpenSeadragon.PanEvent) => {
      if (event.center) {
        setViewerState((prev) => ({
          ...prev,
          center: { x: event.center!.x, y: event.center!.y },
        }));
      }
    });

    viewerRef.current = viewer;

    return () => {
      viewer.destroy();
      viewerRef.current = null;
      setIsReady(false);
    };
  }, [containerId, dziUrl]);

  // Add heatmap overlay when available
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || !heatmapUrl || !isReady) return;

    viewer.addTiledImage({
      tileSource: heatmapUrl,
      opacity: viewerState.showHeatmap ? viewerState.heatmapOpacity : 0,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      success: (event: any) => {
        heatmapOverlayRef.current = event.item as OpenSeadragon.TiledImage;
      },
    });
  }, [heatmapUrl, isReady, viewerState.heatmapOpacity, viewerState.showHeatmap]);

  const setHeatmapVisible = useCallback((visible: boolean) => {
    setViewerState((prev) => ({
      ...prev,
      showHeatmap: visible,
    }));

    if (heatmapOverlayRef.current) {
      heatmapOverlayRef.current.setOpacity(
        visible ? viewerState.heatmapOpacity : 0
      );
    }
  }, [viewerState.heatmapOpacity]);

  const setHeatmapOpacity = useCallback((opacity: number) => {
    setViewerState((prev) => ({
      ...prev,
      heatmapOpacity: opacity,
    }));

    if (heatmapOverlayRef.current && viewerState.showHeatmap) {
      heatmapOverlayRef.current.setOpacity(opacity);
    }
  }, [viewerState.showHeatmap]);

  const goToRegion = useCallback((coords: PatchCoordinates) => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    const viewport = viewer.viewport;
    const tiledImage = viewer.world.getItemAt(0);

    if (!tiledImage) return;

    // Convert image coordinates to viewport coordinates
    const imageRect = new OpenSeadragon.Rect(
      coords.x,
      coords.y,
      coords.width,
      coords.height
    );
    const viewportRect = viewport.imageToViewportRectangle(imageRect);

    // Pan and zoom to the region
    viewport.fitBounds(viewportRect, false);
  }, []);

  const resetView = useCallback(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    viewer.viewport.goHome();
  }, []);

  const zoomIn = useCallback(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    viewer.viewport.zoomBy(1.5);
  }, []);

  const zoomOut = useCallback(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    viewer.viewport.zoomBy(0.67);
  }, []);

  return {
    viewer: viewerRef.current,
    viewerState,
    isReady,
    setHeatmapVisible,
    setHeatmapOpacity,
    goToRegion,
    resetView,
    zoomIn,
    zoomOut,
  };
}
