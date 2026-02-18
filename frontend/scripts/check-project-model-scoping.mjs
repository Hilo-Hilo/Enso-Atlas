#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";

const root = process.cwd();

const files = {
  page: "src/app/page.tsx",
  viewer: "src/components/viewer/WSIViewer.tsx",
  modelPicker: "src/components/panels/ModelPicker.tsx",
  multiModelPanel: "src/components/panels/MultiModelPredictionPanel.tsx",
  batchPanel: "src/components/panels/BatchAnalysisPanel.tsx",
  api: "src/lib/api.ts",
};

const read = (relPath) => fs.readFileSync(path.join(root, relPath), "utf8");

const contents = Object.fromEntries(
  Object.entries(files).map(([key, relPath]) => [key, read(relPath)])
);

const checks = [
  {
    name: "page.tsx should not import or reference AVAILABLE_MODELS",
    pass: !/AVAILABLE_MODELS/.test(contents.page),
  },
  {
    name: "WSIViewer should not use hardcoded HEATMAP_MODELS/AVAILABLE_MODELS fallbacks",
    pass:
      !/HEATMAP_MODELS/.test(contents.viewer) &&
      !/AVAILABLE_MODELS/.test(contents.viewer) &&
      !/\|\|\s*"tumor_stage"/.test(contents.viewer),
  },
  {
    name: "ModelPicker should not hardcode ovarian model IDs",
    pass:
      !/platinum_sensitivity/.test(contents.modelPicker) &&
      !/survival_5y/.test(contents.modelPicker) &&
      !/survival_3y/.test(contents.modelPicker) &&
      !/survival_1y/.test(contents.modelPicker),
  },
  {
    name: "BatchAnalysisPanel should not reference AVAILABLE_MODELS",
    pass: !/AVAILABLE_MODELS/.test(contents.batchPanel),
  },
  {
    name: "MultiModelPredictionPanel should not reference AVAILABLE_MODELS",
    pass: !/AVAILABLE_MODELS/.test(contents.multiModelPanel),
  },
  {
    name: "api.getProjectAvailableModels should guard empty/default project IDs",
    pass: /if\s*\(\s*!projectId\s*\|\|\s*projectId\s*===\s*"default"\s*\)/.test(contents.api),
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
  console.error("\nProject model scoping checks failed.");
  process.exit(1);
}

console.log("\nAll project model scoping checks passed.");
