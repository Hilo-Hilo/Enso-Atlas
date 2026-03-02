export interface VirtualWindowInput {
  itemCount: number;
  itemHeight: number;
  viewportHeight: number;
  scrollTop: number;
  overscan?: number;
}

export interface VirtualWindow {
  startIndex: number;
  endIndex: number;
  topSpacerHeight: number;
  bottomSpacerHeight: number;
}

/**
 * Compute a stable virtual window for fixed-height lists.
 */
export function getVirtualWindow({
  itemCount,
  itemHeight,
  viewportHeight,
  scrollTop,
  overscan = 4,
}: VirtualWindowInput): VirtualWindow {
  if (itemCount <= 0 || itemHeight <= 0 || viewportHeight <= 0) {
    return {
      startIndex: 0,
      endIndex: Math.max(0, itemCount),
      topSpacerHeight: 0,
      bottomSpacerHeight: 0,
    };
  }

  const visibleCount = Math.max(1, Math.ceil(viewportHeight / itemHeight));
  const rawStart = Math.floor(scrollTop / itemHeight);
  const startIndex = Math.max(0, rawStart - overscan);
  const endIndex = Math.min(itemCount, rawStart + visibleCount + overscan);

  const topSpacerHeight = startIndex * itemHeight;
  const bottomSpacerHeight = Math.max(0, (itemCount - endIndex) * itemHeight);

  return {
    startIndex,
    endIndex,
    topSpacerHeight,
    bottomSpacerHeight,
  };
}

/**
 * Keep column count aligned with Tailwind breakpoints used by SlideGrid.
 */
export function getGridColumnCount(width: number): number {
  if (!Number.isFinite(width) || width <= 0) return 1;
  if (width >= 1536) return 5; // 2xl
  if (width >= 1280) return 4; // xl
  if (width >= 1024) return 3; // lg
  if (width >= 640) return 2; // sm
  return 1;
}
