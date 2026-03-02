"use client";

import { useEffect } from "react";
import { usePathname } from "next/navigation";
import { logPerfMetric, setupWebVitalsLogging } from "@/lib/perfLogger";
import { useRoutePerfLogger } from "@/hooks/usePerfInstrumentation";

export function PerfObserver() {
  const pathname = usePathname() || "/";

  useRoutePerfLogger(pathname);

  useEffect(() => {
    return setupWebVitalsLogging();
  }, []);

  useEffect(() => {
    if (typeof window === "undefined" || typeof performance === "undefined") return;

    const navEntries = performance.getEntriesByType("navigation");
    const nav = navEntries[0] as PerformanceNavigationTiming | undefined;
    if (!nav) return;

    logPerfMetric("navigation_ttfb_ms", {
      value: nav.responseStart,
      unit: "ms",
      route: pathname,
    });
  }, [pathname]);

  return null;
}
