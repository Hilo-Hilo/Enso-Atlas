import React from "react";
import { describe, it, expect, vi, beforeAll } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { SlideTable } from "../SlideTable";
import type { SlideInfo } from "@/types";

beforeAll(() => {
  class ResizeObserverMock {
    observe() {}
    unobserve() {}
    disconnect() {}
  }

  (globalThis as { ResizeObserver?: typeof ResizeObserver }).ResizeObserver =
    ResizeObserverMock as unknown as typeof ResizeObserver;
});

function makeSlide(i: number): SlideInfo {
  return {
    id: `slide-${i}`,
    filename: `slide-${i}.svs`,
    dimensions: { width: 1000, height: 800 },
    magnification: 40,
    mpp: 0.25,
    createdAt: "2026-01-01T00:00:00.000Z",
    numPatches: i,
    hasEmbeddings: i % 2 === 0,
  };
}

describe("SlideTable virtualization", () => {
  it("renders a windowed subset while preserving row selection behavior", () => {
    const slides = Array.from({ length: 200 }, (_, i) => makeSlide(i));
    const onSelectSlide = vi.fn();
    const onSelectAll = vi.fn();

    const { container } = render(
      <div style={{ height: 640 }}>
        <SlideTable
          slides={slides}
          selectedIds={new Set()}
          onSelectSlide={onSelectSlide}
          onSelectAll={onSelectAll}
          onStarSlide={vi.fn()}
          onViewSlide={vi.fn()}
          onAnalyzeSlide={vi.fn()}
          onAddToGroup={vi.fn()}
          onDeleteSlide={vi.fn()}
          sortBy="date"
          sortOrder="desc"
          onSort={vi.fn()}
        />
      </div>
    );

    // Regression guard: far rows should not be in DOM before scrolling.
    expect(screen.queryByText("slide-199")).toBeNull();
    expect(screen.getByText("slide-0")).toBeTruthy();

    // Header select-all remains functional.
    const headerButton = container.querySelector("thead th button") as HTMLButtonElement;
    fireEvent.click(headerButton);
    expect(onSelectAll).toHaveBeenCalledWith(true);

    // Per-row select remains functional.
    const firstRow = screen.getByText("slide-0").closest("tr");
    expect(firstRow).toBeTruthy();
    const rowButtons = within(firstRow as HTMLElement).getAllByRole("button");
    fireEvent.click(rowButtons[0]);
    expect(onSelectSlide).toHaveBeenCalledWith("slide-0", true);

    // Should only render a small subset (+ spacer rows), not all 200.
    const renderedRows = container.querySelectorAll("tbody tr").length;
    expect(renderedRows).toBeLessThan(40);
  });
});
