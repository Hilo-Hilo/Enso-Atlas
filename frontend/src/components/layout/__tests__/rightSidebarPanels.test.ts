import { describe, expect, it } from "vitest";

import {
  ensureActiveRightPanel,
  getRightPanelAfterViewModeChange,
  getRightSidebarPanelOptions,
} from "../rightSidebarPanels";

describe("right sidebar panel routing", () => {
  it("switches to pathologist workspace when entering pathologist mode", () => {
    expect(getRightPanelAfterViewModeChange("pathologist", "medgemma")).toBe(
      "pathologist-workspace"
    );
  });

  it("returns to medgemma when leaving pathologist workspace", () => {
    expect(getRightPanelAfterViewModeChange("oncologist", "pathologist-workspace")).toBe(
      "medgemma"
    );
  });

  it("preserves current panel when staying in oncologist-compatible panels", () => {
    expect(getRightPanelAfterViewModeChange("oncologist", "semantic-search")).toBe(
      "semantic-search"
    );
  });
});

describe("right sidebar panel options", () => {
  it("returns pathologist toolset with pathology-specific ordering", () => {
    const options = getRightSidebarPanelOptions({
      userViewMode: "pathologist",
      showMultiModelPanel: true,
      primaryPredictionPanelLabel: "Resistance to Therapy",
    });

    expect(options.map((o) => o.value)).toEqual([
      "pathologist-workspace",
      "semantic-search",
      "evidence",
      "outlier-detector",
    ]);
  });

  it("includes multi-model panel in oncologist mode when available", () => {
    const options = getRightSidebarPanelOptions({
      userViewMode: "oncologist",
      showMultiModelPanel: true,
      primaryPredictionPanelLabel: "Tumor Stage",
    });

    expect(options.map((o) => o.value)).toContain("multi-model");
    expect(options.find((o) => o.value === "prediction")?.label).toBe("Tumor Stage");
  });

  it("omits multi-model panel in oncologist mode when unavailable", () => {
    const options = getRightSidebarPanelOptions({
      userViewMode: "oncologist",
      showMultiModelPanel: false,
      primaryPredictionPanelLabel: "Resistance to Therapy",
    });

    expect(options.map((o) => o.value)).not.toContain("multi-model");
  });
});

describe("ensureActiveRightPanel", () => {
  const oncologistOptions = getRightSidebarPanelOptions({
    userViewMode: "oncologist",
    showMultiModelPanel: false,
    primaryPredictionPanelLabel: "Resistance to Therapy",
  });

  it("keeps active panel when it exists in current options", () => {
    expect(ensureActiveRightPanel("medgemma", oncologistOptions)).toBe("medgemma");
  });

  it("falls back to first available panel when active panel disappears", () => {
    expect(ensureActiveRightPanel("pathologist-workspace", oncologistOptions)).toBe(
      "medgemma"
    );
  });
});
