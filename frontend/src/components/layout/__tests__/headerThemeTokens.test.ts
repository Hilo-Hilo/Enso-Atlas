import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

// Regression coverage for issues #117 and #118: the top-nav controls listed
// below must carry both light- and dark-mode tokens. If a future change drops
// the dark surface or the light text/background, header labels become
// unreadable in one of the two themes.

const HEADER_SRC = readFileSync(
  path.resolve(__dirname, "..", "Header.tsx"),
  "utf8",
);

const DEMO_SRC = readFileSync(
  path.resolve(__dirname, "..", "..", "demo", "DemoMode.tsx"),
  "utf8",
);

// Find a className string that contains all of the given unique substrings.
// Searches double-quoted attribute values across the whole source (multi-line
// supported). Returns the matched string or throws.
function findClassNameContaining(
  source: string,
  needles: string[],
  label: string,
): string {
  const re = /"([^"]+)"/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(source)) !== null) {
    const value = m[1];
    if (needles.every((needle) => value.includes(needle))) {
      return value;
    }
  }
  throw new Error(`could not locate className for ${label} (needles: ${needles.join(", ")})`);
}

function expectThemeAwareSurface(classes: string, controlName: string) {
  expect(
    /\bbg-(?:white|sky-\d{2,3})(?:\/\d{1,3})?\b/.test(classes),
    `${controlName} should declare a light-mode background`,
  ).toBe(true);

  expect(
    /\bdark:bg-(?:navy|slate|gray|sky)-\d{2,3}(?:\/\d{1,3})?\b/.test(classes),
    `${controlName} should declare a dark-mode background`,
  ).toBe(true);

  expect(
    /\btext-(?:slate|sky|navy|gray)-\d{2,3}\b/.test(classes),
    `${controlName} should declare a dark text color for light mode`,
  ).toBe(true);

  expect(
    /\bdark:text-(?:gray|slate|white|sky)(?:-\d{2,3})?\b/.test(classes),
    `${controlName} should declare a light text color for dark mode`,
  ).toBe(true);
}

describe("top-nav controls carry theme-aware surfaces (#117, #118)", () => {
  it("project switcher trigger has light and dark variants", () => {
    // The trigger button uses a `truncate max-w-[200px]` span inside, and is
    // the only button with `rounded-lg border border-sky-200 bg-white/90`
    // that does not have a layout prefix like `hidden lg:flex`. Anchor on
    // `flex items-center gap-2 px-3 py-1.5`, the button's leading classes.
    const classes = findClassNameContaining(
      HEADER_SRC,
      ["flex items-center gap-2 px-3 py-1.5 rounded-lg", "border-sky-200"],
      "project switcher trigger",
    );
    expectThemeAwareSurface(classes, "project switcher trigger");
  });

  it("slides nav link has light and dark variants", () => {
    const classes = findClassNameContaining(
      HEADER_SRC,
      ["hidden lg:flex items-center gap-2 rounded-lg", "xl:px-3"],
      "slides nav link",
    );
    // Two nav links share this shape (Slides and Projects); make sure we
    // got one of them and that it is theme-aware.
    expectThemeAwareSurface(classes, "slides/projects nav link");
  });

  it("utility overflow button has light and dark variants", () => {
    const classes = findClassNameContaining(
      HEADER_SRC,
      ["rounded-lg border border-sky-200 bg-white/90 p-2"],
      "utility overflow button",
    );
    expectThemeAwareSurface(classes, "utility overflow button");
  });

  it("inactive demo-mode toggle has light and dark variants", () => {
    // The inactive branch lives inside the cn() ternary; locate it by the
    // ternary structure rather than a className= attribute.
    const ternary = DEMO_SRC.match(
      /disabled\s*\?\s*"[^"]+"\s*:\s*isActive\s*\?\s*"[^"]+"\s*:\s*"([^"]+)"/,
    );
    expect(
      ternary,
      "could not locate DemoToggle inactive className branch",
    ).toBeTruthy();
    expectThemeAwareSurface(ternary![1], "inactive demo-mode toggle");
  });
});
