# Performance Baseline (Dev Observability)

This document defines a **lightweight, toggleable** perf baseline for Enso Atlas.
It is designed for day-to-day development with minimal runtime risk.

## What was added

### Backend (FastAPI)

- Request timing middleware (enabled by default in dev-style runs):
  - `method`
  - normalized `path`
  - `status`
  - `duration_ms`
  - `request_id`
- Structured log line prefix: `request_timing {...}`
- In-memory rolling latency tracker with percentile summaries.
- New endpoint:
  - `GET /api/perf/latency-summary`

### Frontend (Next.js)

- Client-side perf logger (`frontend/src/lib/perfLogger.ts`) for:
  - route render timing (`route_render_ms`)
  - panel switch timing (`panel_render_ms`)
  - web-vitals-style entries (LCP/FID/CLS/paint when supported)
- Global observer mounted in layout (`PerfObserver`).
- Panel timing hooks in main app page for right sidebar and mobile panel switches.

### Tooling

- `scripts/benchmark_dev.sh` for quick local API latency sampling.

---

## Toggles

### Backend env vars

- `ENSO_PERF_ENABLED` (default: `true`)
  - `0/false` disables middleware tracking.
- `ENSO_PERF_SUMMARY_ENABLED` (default: same as `ENSO_PERF_ENABLED`)
  - controls `/api/perf/latency-summary`.
- `ENSO_PERF_MAX_SAMPLES` (default: `2000`)
  - in-memory rolling window size.
- `ENSO_PERF_ROUTE_LIMIT` (default: `25`)
  - max route rows returned in summary.

### Frontend env vars

- `NEXT_PUBLIC_ENABLE_PERF_LOGS`
  - `true/1`: force-enable perf console logs
  - `false/0`: disable perf console logs
  - unset: enabled in dev, disabled in production

---

## How to run

### 1) Start backend/frontend normally

No schema/database changes are required.

### 2) Generate API baseline samples

```bash
./scripts/benchmark_dev.sh
```

Optional:

```bash
SAMPLES=20 ENSO_API_BASE_URL=http://localhost:8000 ./scripts/benchmark_dev.sh
```

The script prints per-endpoint `p50/p95/max` and writes raw CSV under `benchmarks/`.

### 3) Read live in-memory summary

```bash
curl -s http://localhost:8000/api/perf/latency-summary | python3 -m json.tool
```

### 4) Inspect frontend metrics (dev)

Open browser devtools console and filter for:

- `[enso-perf] route_render_ms`
- `[enso-perf] panel_render_ms`
- `[enso-perf] web_vitals_*`

---

## Interpreting the baseline

Use these as initial guardrails (adjust per environment):

- API `p50` stable and low variance across runs.
- API `p95` highlights tail latency regressions.
- Panel/route timings should remain consistent after UI changes.
- Track status mix (`2xx/4xx/5xx`) during benchmarks to avoid false perf conclusions.

For regression checks, compare against a prior CSV and `/api/perf/latency-summary` snapshot from a known-good commit.
