#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";

const root = process.cwd();
const read = (rel) => fs.readFileSync(path.join(root, rel), "utf8");

const viewerPath = "src/components/viewer/WSIViewer.tsx";
const modelProxyPath = "src/app/api/heatmap/[slideId]/[modelId]/route.ts";
const baseProxyPath = "src/app/api/heatmap/[slideId]/route.ts";

const viewer = read(viewerPath);
const modelProxy = read(modelProxyPath);
const baseProxy = read(baseProxyPath);

const worldGetItemMatches = viewer.match(/world\.getItemAt\(0\)/g) ?? [];

const checks = [
  {
    name: "WSIViewer keeps explicit slide item ref (no repeated world.getItemAt(0) lookups)",
    pass: worldGetItemMatches.length <= 1 && /const slideTiledImageRef = useRef/.test(viewer),
  },
  {
    name: "Heatmap metadata headers are parsed in WSIViewer",
    pass:
      /X-Coverage-Width/.test(viewer) &&
      /X-Coverage-Height/.test(viewer) &&
      /heatmapMetaRef\.current\s*=\s*\{/.test(viewer),
  },
  {
    name: "Heatmap overlay world geometry uses BOTH width and height scaling",
    pass:
      /const heatmapWorldWidth = coverageW \* widthScale/.test(viewer) &&
      /const heatmapWorldHeight = coverageH \* heightScale/.test(viewer) &&
      /height:\s*heatmapWorldHeight/.test(viewer),
  },
  {
    name: "Grid patch spacing derives from heatmap metadata (patchX/patchY)",
    pass:
      /const patchX = heatmapMeta\?\.patchWidthPx/.test(viewer) &&
      /const patchY = heatmapMeta\?\.patchHeightPx/.test(viewer) &&
      /ix \+= patchX/.test(viewer) &&
      /iy \+= patchY/.test(viewer),
  },
  {
    name: "Model heatmap proxy forwards project_id to backend",
    pass: /const projectId = searchParams\.get\("project_id"\)/.test(modelProxy) && /backendParams\.set\("project_id", projectId\)/.test(modelProxy),
  },
  {
    name: "Base heatmap proxy forwards project_id/alpha_power and coverage headers",
    pass:
      /searchParams\.get\("project_id"\)/.test(baseProxy) &&
      /searchParams\.get\("alpha_power"\)/.test(baseProxy) &&
      /X-Coverage-Width/.test(baseProxy),
  },
];

let failed = false;
for (const check of checks) {
  if (check.pass) {
    console.log(`✅ ${check.name}`);
  } else {
    failed = true;
    console.error(`❌ ${check.name}`);
  }
}

if (failed) {
  console.error("\nHeatmap/grid alignment checks failed.");
  process.exit(1);
}

console.log("\nAll heatmap/grid alignment checks passed.");
