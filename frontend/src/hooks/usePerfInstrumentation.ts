"use client";

import { useEffect, useRef } from "react";
import { measureOnNextPaint } from "@/lib/perfLogger";

export function useRoutePerfLogger(pathname: string): void {
  const previousRouteRef = useRef<string | null>(null);

  useEffect(() => {
    if (!pathname || typeof window === "undefined") return;

    const startedAt = performance.now();
    measureOnNextPaint(
      "route_render_ms",
      {
        route: pathname,
        from_route: previousRouteRef.current ?? undefined,
      },
      startedAt
    );

    previousRouteRef.current = pathname;
  }, [pathname]);
}

export function usePanelSwitchPerf(panelGroup: string, panel: string): void {
  const previousPanelRef = useRef<string | null>(null);

  useEffect(() => {
    if (!panel || typeof window === "undefined") return;

    const startedAt = performance.now();
    measureOnNextPaint(
      "panel_render_ms",
      {
        panel_group: panelGroup,
        panel,
        from_panel: previousPanelRef.current ?? undefined,
        route: window.location.pathname,
      },
      startedAt
    );

    previousPanelRef.current = panel;
  }, [panelGroup, panel]);
}
