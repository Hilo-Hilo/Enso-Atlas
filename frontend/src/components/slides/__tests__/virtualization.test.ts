import { describe, expect, it } from "vitest";
import { getGridColumnCount, getVirtualWindow } from "../virtualization";

describe("slides virtualization helpers", () => {
  it("returns bounded virtual ranges for large lists", () => {
    const window = getVirtualWindow({
      itemCount: 1000,
      itemHeight: 72,
      viewportHeight: 360,
      scrollTop: 720,
      overscan: 4,
    });

    expect(window.startIndex).toBe(6);
    expect(window.endIndex).toBe(19);
    expect(window.topSpacerHeight).toBe(432);
    expect(window.bottomSpacerHeight).toBe((1000 - 19) * 72);
  });

  it("falls back to a safe range when viewport is unavailable", () => {
    const window = getVirtualWindow({
      itemCount: 30,
      itemHeight: 72,
      viewportHeight: 0,
      scrollTop: 0,
    });

    expect(window.startIndex).toBe(0);
    expect(window.endIndex).toBe(30);
    expect(window.topSpacerHeight).toBe(0);
    expect(window.bottomSpacerHeight).toBe(0);
  });

  it("maps widths to Tailwind grid breakpoints", () => {
    expect(getGridColumnCount(320)).toBe(1);
    expect(getGridColumnCount(700)).toBe(2);
    expect(getGridColumnCount(1100)).toBe(3);
    expect(getGridColumnCount(1300)).toBe(4);
    expect(getGridColumnCount(1600)).toBe(5);
  });
});
