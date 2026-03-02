export interface PerfLogPayload {
  value: number;
  unit?: string;
  route?: string;
  panel?: string;
  panel_group?: string;
  from_panel?: string;
  from_route?: string;
  [key: string]: unknown;
}

const PERF_FLAG = process.env.NEXT_PUBLIC_ENABLE_PERF_LOGS;

export function isPerfLoggingEnabled(): boolean {
  if (typeof window === "undefined") return false;

  if (PERF_FLAG != null) {
    return ["1", "true", "yes", "on"].includes(PERF_FLAG.toLowerCase());
  }

  // Default: enabled in development, off in production.
  return process.env.NODE_ENV !== "production";
}

export function logPerfMetric(name: string, payload: PerfLogPayload): void {
  if (!isPerfLoggingEnabled()) return;

  const ts = new Date().toISOString();
  // Structured console output (copy/paste friendly for analysis scripts).
  console.info("[enso-perf]", {
    metric: name,
    ts,
    ...payload,
  });
}

export function measureOnNextPaint(
  metric: string,
  payload: Omit<PerfLogPayload, "value" | "unit">,
  startedAtMs: number = performance.now()
): void {
  if (typeof window === "undefined" || typeof window.requestAnimationFrame !== "function") {
    return;
  }

  window.requestAnimationFrame(() => {
    window.requestAnimationFrame(() => {
      const durationMs = performance.now() - startedAtMs;
      logPerfMetric(metric, {
        ...payload,
        value: durationMs,
        unit: "ms",
      });
    });
  });
}

export function setupWebVitalsLogging(): () => void {
  if (
    typeof window === "undefined" ||
    typeof PerformanceObserver === "undefined" ||
    !isPerfLoggingEnabled()
  ) {
    return () => {};
  }

  const observers: PerformanceObserver[] = [];

  const setups: Array<{ type: string; metric: string; value: (entry: PerformanceEntry) => number }> = [
    {
      type: "largest-contentful-paint",
      metric: "web_vitals_lcp_ms",
      value: (entry) => entry.startTime,
    },
    {
      type: "first-input",
      metric: "web_vitals_fid_ms",
      value: (entry) => {
        const fidEntry = entry as PerformanceEntry & { processingStart?: number };
        if (typeof fidEntry.processingStart !== "number") return 0;
        return fidEntry.processingStart - fidEntry.startTime;
      },
    },
    {
      type: "layout-shift",
      metric: "web_vitals_cls",
      value: (entry) => {
        const clsEntry = entry as PerformanceEntry & { hadRecentInput?: boolean; value?: number };
        if (clsEntry.hadRecentInput) return 0;
        return typeof clsEntry.value === "number" ? clsEntry.value : 0;
      },
    },
    {
      type: "paint",
      metric: "web_vitals_paint_ms",
      value: (entry) => entry.startTime,
    },
  ];

  for (const setup of setups) {
    try {
      const observer = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          const value = setup.value(entry);
          if (!Number.isFinite(value) || value <= 0) continue;

          logPerfMetric(setup.metric, {
            value,
            unit: setup.metric.includes("_ms") ? "ms" : undefined,
            entry_type: setup.type,
            name: entry.name,
          });
        }
      });

      observer.observe({ type: setup.type, buffered: true });
      observers.push(observer);
    } catch {
      // Some entry types are browser/version specific.
    }
  }

  return () => {
    observers.forEach((observer) => observer.disconnect());
  };
}
