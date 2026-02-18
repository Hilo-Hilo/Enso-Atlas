// Mock data for demo mode - allows the guided tour to work without a backend connection

import type {
  SlideInfo,
  AnalysisResponse,
  EvidencePatch,
  SimilarCase,
  StructuredReport,
  MultiModelResponse,
  ModelPrediction,
  SlideQCMetrics,
} from "@/types";

export const DEMO_SLIDE: SlideInfo = {
  id: "TCGA-DEMO-001",
  filename: "TCGA-DEMO-001.svs",
  displayName: "Demo: Cancer Biopsy Sample",
  dimensions: { width: 98304, height: 65536 },
  magnification: 20,
  mpp: 0.5,
  createdAt: new Date().toISOString(),
  label: "sensitive",
  hasEmbeddings: true,
  hasLevel0Embeddings: true,
  numPatches: 4200,
  patient: {
    age: 58,
    sex: "Female",
    stage: "III",
    grade: "High",
    prior_lines: 0,
    histology: "Adenocarcinoma",
  },
};

function makePatch(idx: number, weight: number): EvidencePatch {
  return {
    id: `demo-patch-${idx}`,
    patchId: `patch_${idx}`,
    coordinates: {
      x: 10000 + idx * 2000,
      y: 8000 + (idx % 3) * 1500,
      level: 0,
      width: 224,
      height: 224,
    },
    attentionWeight: weight,
    thumbnailUrl: `data:image/svg+xml,${encodeURIComponent(
      `<svg xmlns="http://www.w3.org/2000/svg" width="224" height="224"><rect width="224" height="224" fill="#f0e6f6"/><rect x="20" y="20" width="184" height="184" rx="8" fill="#e8d5f5" stroke="#c084fc" stroke-width="2"/><text x="112" y="105" text-anchor="middle" font-family="system-ui" font-size="14" fill="#7c3aed">Patch ${idx + 1}</text><text x="112" y="130" text-anchor="middle" font-family="system-ui" font-size="11" fill="#9f7aea">${(weight * 100).toFixed(0)}% attention</text></svg>`
    )}`,
    morphologyDescription: [
      "Dense tumor cell clusters with high nuclear-to-cytoplasmic ratio",
      "Solid sheets of carcinoma cells with prominent nucleoli",
      "Stromal invasion with desmoplastic reaction",
      "Glandular architecture with irregular borders",
      "Tumor-infiltrating lymphocytes at invasive front",
      "High mitotic activity with atypical mitoses",
    ][idx % 6],
    tissueType: (["tumor", "tumor", "stroma", "tumor", "inflammatory", "tumor"] as const)[idx % 6],
    tissueConfidence: 0.85 + Math.random() * 0.1,
  };
}

export const DEMO_EVIDENCE_PATCHES: EvidencePatch[] = [
  makePatch(0, 0.95),
  makePatch(1, 0.88),
  makePatch(2, 0.82),
  makePatch(3, 0.76),
  makePatch(4, 0.71),
  makePatch(5, 0.65),
];

export const DEMO_SIMILAR_CASES: SimilarCase[] = [
  { slideId: "TCGA-04-1332", similarity: 0.94, distance: 0.06, label: "sensitive" },
  { slideId: "TCGA-13-0890", similarity: 0.91, distance: 0.09, label: "sensitive" },
  { slideId: "TCGA-24-1562", similarity: 0.87, distance: 0.13, label: "resistant" },
  { slideId: "TCGA-61-2008", similarity: 0.85, distance: 0.15, label: "sensitive" },
  { slideId: "TCGA-25-1318", similarity: 0.82, distance: 0.18, label: "sensitive" },
];

export const DEMO_PREDICTION = {
  label: "Favorable Response",
  score: 0.87,
  confidence: 0.91,
};

export const DEMO_QC_METRICS: SlideQCMetrics = {
  slideId: "TCGA-DEMO-001",
  tissueCoverage: 0.72,
  blurScore: 0.12,
  stainUniformity: 0.88,
  artifactDetected: false,
  penMarks: false,
  foldDetected: false,
  overallQuality: "good",
};

export const DEMO_ANALYSIS_RESULT: AnalysisResponse = {
  slideInfo: DEMO_SLIDE,
  prediction: DEMO_PREDICTION,
  evidencePatches: DEMO_EVIDENCE_PATCHES,
  similarCases: DEMO_SIMILAR_CASES,
  heatmap: {
    imageUrl: "",
    minValue: 0,
    maxValue: 1,
    colorScale: "viridis",
  },
  processingTimeMs: 12400,
};

export const DEMO_MULTI_MODEL: MultiModelResponse = {
  slideId: "TCGA-DEMO-001",
  predictions: {
    treatment_response: {
      modelId: "treatment_response",
      modelName: "Treatment Response",
      category: "cancer_specific",
      score: 0.87,
      label: "Favorable",
      positiveLabel: "Favorable",
      negativeLabel: "Unfavorable",
      confidence: 0.91,
      auc: 0.78,
      nTrainingSlides: 342,
      description: "Predicts treatment response from histopathology patterns",
    },
    tumor_grade: {
      modelId: "tumor_grade",
      modelName: "Tumor Grade",
      category: "general_pathology",
      score: 0.92,
      label: "High Grade",
      positiveLabel: "High Grade",
      negativeLabel: "Low Grade",
      confidence: 0.94,
      auc: 0.85,
      nTrainingSlides: 520,
      description: "Classifies tumor grade from histopathology",
    },
  },
  byCategory: {
    cancerSpecific: [
      {
        modelId: "treatment_response",
        modelName: "Treatment Response",
        category: "cancer_specific",
        score: 0.87,
        label: "Favorable",
        positiveLabel: "Favorable",
        negativeLabel: "Unfavorable",
        confidence: 0.91,
        auc: 0.78,
        nTrainingSlides: 342,
        description: "Predicts treatment response from histopathology patterns",
      },
    ],
    generalPathology: [
      {
        modelId: "tumor_grade",
        modelName: "Tumor Grade",
        category: "general_pathology",
        score: 0.92,
        label: "High Grade",
        positiveLabel: "High Grade",
        negativeLabel: "Low Grade",
        confidence: 0.94,
        auc: 0.85,
        nTrainingSlides: 520,
        description: "Classifies tumor grade from histopathology",
      },
    ],
  },
  nPatches: 4200,
  processingTimeMs: 12400,
};

export const DEMO_REPORT: StructuredReport = {
  caseId: "TCGA-DEMO-001",
  task: "Treatment response prediction from histopathology",
  generatedAt: new Date().toISOString(),
  patientContext: DEMO_SLIDE.patient,
  modelOutput: DEMO_PREDICTION,
  evidence: DEMO_EVIDENCE_PATCHES.slice(0, 4).map((p, i) => ({
    patchId: p.patchId,
    coordsLevel0: [p.coordinates.x, p.coordinates.y] as [number, number],
    morphologyDescription: p.morphologyDescription || "",
    whyThisPatchMatters: [
      "Highest attention region — tumor architecture suggests favorable response phenotype",
      "Prominent nucleoli and high mitotic rate correlate with treatment response",
      "Stromal reaction pattern associated with immune-mediated response",
      "Glandular growth pattern linked to favorable treatment response",
    ][i],
  })),
  similarExamples: DEMO_SIMILAR_CASES.slice(0, 3).map((c) => ({
    exampleId: c.slideId,
    label: c.label || "unknown",
    distance: c.distance || 0,
  })),
  limitations: [
    "Research model — not validated for clinical decision-making",
    "Prediction based on morphology alone; genomic and clinical factors not included",
    "Training cohort limited to TCGA cancer data",
  ],
  suggestedNextSteps: [
    "Correlate with relevant biomarker testing results",
    "Consider germline and somatic testing results",
    "Discuss with tumor board before treatment decision",
  ],
  safetyStatement:
    "This AI analysis is for research and decision support only. All treatment decisions must be made by qualified clinicians in consultation with the patient.",
  summary:
    "AI analysis of this cancer biopsy predicts favorable treatment response with 91% confidence. The model identified dense tumor cell clusters and stromal invasion patterns consistent with favorable response phenotypes observed in the training cohort. Four of five most similar historical cases were also classified as favorable responders, supporting the prediction. These findings should be integrated with genomic profiling and clinical context before therapeutic decision-making.",
};
