import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  AtlasApiError,
  __apiClientTestUtils,
  addTagsToSlide,
  getEmbeddingTaskStatus,
  getSlides,
} from "../api";

function jsonResponse(payload: unknown, status: number = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

describe("api client network optimizations", () => {
  beforeEach(() => {
    __apiClientTestUtils.clearRequestCaches();
    vi.restoreAllMocks();
  });

  it("dedupes concurrent read requests", async () => {
    const fetchMock = vi.fn(async () => jsonResponse([]));
    vi.stubGlobal("fetch", fetchMock);

    const [first, second] = await Promise.all([getSlides(), getSlides()]);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(first.total).toBe(0);
    expect(second.total).toBe(0);
  });

  it("invalidates short read cache after writes", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        jsonResponse({
          task_id: "task-1",
          slide_id: "slide-1",
          level: 0,
          status: "running",
          progress: 10,
          message: "running",
          num_patches: 100,
          processing_time_seconds: 2,
          error: null,
          elapsed_seconds: 2,
        })
      )
      .mockResolvedValueOnce(jsonResponse({ success: true }))
      .mockResolvedValueOnce(
        jsonResponse({
          task_id: "task-1",
          slide_id: "slide-1",
          level: 0,
          status: "running",
          progress: 25,
          message: "running",
          num_patches: 100,
          processing_time_seconds: 4,
          error: null,
          elapsed_seconds: 4,
        })
      );

    vi.stubGlobal("fetch", fetchMock);

    const firstStatus = await getEmbeddingTaskStatus("task-1");
    const cachedStatus = await getEmbeddingTaskStatus("task-1");

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(cachedStatus.progress).toBe(firstStatus.progress);

    await addTagsToSlide("slide-1", ["reviewed"]);

    const refreshedStatus = await getEmbeddingTaskStatus("task-1");
    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(refreshedStatus.progress).toBe(25);
  });

  it("does not retry non-idempotent writes by default", async () => {
    const fetchMock = vi.fn(async () =>
      jsonResponse({ detail: "temporary upstream issue" }, 503)
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(addTagsToSlide("slide-1", ["urgent"]))
      .rejects
      .toBeInstanceOf(AtlasApiError);

    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("uses adaptive polling cadence helpers", () => {
    const stalledPendingDelay = __apiClientTestUtils.getAdaptivePollDelay({
      baseIntervalMs: 2000,
      minIntervalMs: 700,
      maxIntervalMs: 10000,
      status: "pending",
      progress: 10,
      previousProgress: 10,
      stablePollCount: 3,
      includeJitter: false,
    });

    const nearCompleteDelay = __apiClientTestUtils.getAdaptivePollDelay({
      baseIntervalMs: 2000,
      minIntervalMs: 700,
      maxIntervalMs: 10000,
      status: "running",
      progress: 95,
      previousProgress: 90,
      stablePollCount: 0,
      includeJitter: false,
    });

    const errorBackoffDelay = __apiClientTestUtils.getAdaptivePollDelay({
      baseIntervalMs: 2000,
      minIntervalMs: 700,
      maxIntervalMs: 10000,
      progress: 20,
      previousProgress: 20,
      stablePollCount: 4,
      consecutiveErrors: 3,
      includeJitter: false,
    });

    expect(nearCompleteDelay).toBeLessThan(stalledPendingDelay);
    expect(errorBackoffDelay).toBe(10000);
  });
});
