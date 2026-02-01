// Mock data for development and testing
// This file provides sample data when the backend is not available

import type {
  SlideInfo,
  AnalysisResponse,
  EvidencePatch,
  SimilarCase,
  StructuredReport,
  PredictionResult,
  HeatmapData,
} from "@/types";

// Mock slides for the slide selector
export const mockSlides: SlideInfo[] = [
  {
    id: "slide-001",
    filename: "ovarian_sample_001.svs",
    dimensions: { width: 45056, height: 32768 },
    magnification: 40,
    mpp: 0.25,
    thumbnailUrl: "/api/placeholder/thumbnail",
    createdAt: new Date().toISOString(),
  },
  {
    id: "slide-002",
    filename: "ovarian_sample_002.svs",
    dimensions: { width: 40960, height: 28672 },
    magnification: 40,
    mpp: 0.25,
    thumbnailUrl: "/api/placeholder/thumbnail",
    createdAt: new Date().toISOString(),
  },
  {
    id: "slide-003",
    filename: "ovarian_sample_003.ndpi",
    dimensions: { width: 51200, height: 35840 },
    magnification: 20,
    mpp: 0.5,
    thumbnailUrl: "/api/placeholder/thumbnail",
    createdAt: new Date().toISOString(),
  },
];

// Mock prediction result
export const mockPrediction: PredictionResult = {
  label: "Bevacizumab Responder",
  probability: 0.78,
  confidence: "high",
  calibrationNote:
    "Model calibration based on validation cohort (n=156). Probability reflects estimated likelihood of treatment response.",
};

// Mock evidence patches
export const mockEvidencePatches: EvidencePatch[] = [
  {
    id: "patch-001",
    patchId: "p-a1b2c3d4",
    coordinates: { x: 12500, y: 8750, level: 0, width: 224, height: 224 },
    attentionWeight: 0.92,
    thumbnailUrl: "/api/placeholder/patch",
    morphologyDescription: "High-grade serous carcinoma with papillary architecture",
  },
  {
    id: "patch-002",
    patchId: "p-e5f6g7h8",
    coordinates: { x: 15200, y: 11300, level: 0, width: 224, height: 224 },
    attentionWeight: 0.87,
    thumbnailUrl: "/api/placeholder/patch",
    morphologyDescription: "Dense lymphocytic infiltrate at tumor margin",
  },
  {
    id: "patch-003",
    patchId: "p-i9j0k1l2",
    coordinates: { x: 18900, y: 14500, level: 0, width: 224, height: 224 },
    attentionWeight: 0.81,
    thumbnailUrl: "/api/placeholder/patch",
    morphologyDescription: "Necrotic debris with viable tumor cells",
  },
  {
    id: "patch-004",
    patchId: "p-m3n4o5p6",
    coordinates: { x: 21600, y: 16800, level: 0, width: 224, height: 224 },
    attentionWeight: 0.75,
    thumbnailUrl: "/api/placeholder/patch",
    morphologyDescription: "Stromal desmoplasia with scattered tumor nests",
  },
  {
    id: "patch-005",
    patchId: "p-q7r8s9t0",
    coordinates: { x: 9800, y: 12100, level: 0, width: 224, height: 224 },
    attentionWeight: 0.69,
    thumbnailUrl: "/api/placeholder/patch",
    morphologyDescription: "Solid growth pattern with high mitotic activity",
  },
  {
    id: "patch-006",
    patchId: "p-u1v2w3x4",
    coordinates: { x: 24300, y: 19200, level: 0, width: 224, height: 224 },
    attentionWeight: 0.63,
    thumbnailUrl: "/api/placeholder/patch",
    morphologyDescription: "Cribriform pattern with prominent nucleoli",
  },
];

// Mock similar cases
export const mockSimilarCases: SimilarCase[] = [
  {
    caseId: "case-ref-001",
    slideId: "slide-ref-001",
    patchId: "ref-patch-001",
    distance: 0.12,
    label: "Responder",
    thumbnailUrl: "/api/placeholder/patch",
    coordinates: { x: 14200, y: 9100, level: 0, width: 224, height: 224 },
  },
  {
    caseId: "case-ref-002",
    slideId: "slide-ref-002",
    patchId: "ref-patch-002",
    distance: 0.18,
    label: "Responder",
    thumbnailUrl: "/api/placeholder/patch",
    coordinates: { x: 11800, y: 7400, level: 0, width: 224, height: 224 },
  },
  {
    caseId: "case-ref-003",
    slideId: "slide-ref-003",
    patchId: "ref-patch-003",
    distance: 0.24,
    label: "Non-Responder",
    thumbnailUrl: "/api/placeholder/patch",
    coordinates: { x: 19500, y: 13200, level: 0, width: 224, height: 224 },
  },
  {
    caseId: "case-ref-004",
    slideId: "slide-ref-004",
    patchId: "ref-patch-004",
    distance: 0.31,
    label: "Responder",
    thumbnailUrl: "/api/placeholder/patch",
    coordinates: { x: 16700, y: 10800, level: 0, width: 224, height: 224 },
  },
  {
    caseId: "case-ref-005",
    slideId: "slide-ref-005",
    patchId: "ref-patch-005",
    distance: 0.38,
    label: "Non-Responder",
    thumbnailUrl: "/api/placeholder/patch",
    coordinates: { x: 22100, y: 15600, level: 0, width: 224, height: 224 },
  },
];

// Mock heatmap data
export const mockHeatmap: HeatmapData = {
  imageUrl: "/api/placeholder/heatmap",
  opacity: 0.5,
  bounds: { x: 0, y: 0, width: 45056, height: 32768 },
};

// Mock structured report
export const mockReport: StructuredReport = {
  caseId: "slide-001",
  task: "Bevacizumab Response Prediction",
  generatedAt: new Date().toISOString(),
  modelOutput: mockPrediction,
  evidence: [
    {
      patchId: "patch-001",
      coordsLevel0: [12500, 8750],
      morphologyDescription:
        "This region shows high-grade serous carcinoma with prominent papillary architecture and nuclear atypia.",
      whyThisPatchMatters:
        "Papillary architecture with this degree of atypia is associated with treatment response in the training cohort.",
    },
    {
      patchId: "patch-002",
      coordsLevel0: [15200, 11300],
      morphologyDescription:
        "Dense lymphocytic infiltrate observed at the tumor-stroma interface, suggesting active immune response.",
      whyThisPatchMatters:
        "Tumor-infiltrating lymphocytes are a positive prognostic factor associated with anti-angiogenic therapy response.",
    },
    {
      patchId: "patch-003",
      coordsLevel0: [18900, 14500],
      morphologyDescription:
        "Region shows necrotic debris with surrounding viable tumor cells and inflammatory infiltrate.",
      whyThisPatchMatters:
        "The pattern of necrosis distribution may indicate tumor vasculature characteristics relevant to anti-VEGF therapy.",
    },
  ],
  similarExamples: [
    { exampleId: "case-ref-001", label: "Responder", distance: 0.12 },
    { exampleId: "case-ref-002", label: "Responder", distance: 0.18 },
    { exampleId: "case-ref-003", label: "Non-Responder", distance: 0.24 },
  ],
  limitations: [
    "This prediction is based on a model trained on a limited cohort (n=78 patients, 288 WSIs) and may not generalize to all populations.",
    "Scanner and staining variations between the training cohort and the current slide may affect prediction accuracy.",
    "The model has not been validated in a prospective clinical setting.",
    "Attention-based evidence highlights correlation, not causation - morphological patterns may not directly drive treatment response.",
  ],
  suggestedNextSteps: [
    "Consider additional IHC markers (CD31, VEGF) to confirm vascular phenotype.",
    "Review clinical history for prior anti-angiogenic therapy exposure.",
    "Discuss case in tumor board with full clinical context before treatment decisions.",
    "Consider NGS panel to identify additional targetable alterations.",
  ],
  safetyStatement:
    "This analysis is for research and decision support purposes only. It is NOT a diagnostic device and should NOT be used as the sole basis for treatment decisions. All findings must be interpreted by qualified healthcare professionals in the context of the complete clinical picture, patient history, and standard of care guidelines.",
  summary: `ENSO ATLAS ANALYSIS SUMMARY

Case ID: slide-001
Analysis Date: ${new Date().toLocaleDateString()}
Task: Bevacizumab Response Prediction

PREDICTION:
The model predicts this case as "Bevacizumab Responder" with high confidence (probability: 78%).

KEY FINDINGS:
The analysis identified several morphological features associated with treatment response:

1. Prominent papillary architecture with high-grade nuclear features in the primary tumor regions
2. Significant tumor-infiltrating lymphocyte presence at the tumor-stromal interface
3. Characteristic necrosis pattern that may indicate tumor vasculature susceptibility to anti-angiogenic therapy

SUPPORTING EVIDENCE:
The model identified 6 high-attention regions that contributed most to this prediction. The top 3 evidence patches showed morphological patterns including papillary carcinoma architecture, immune infiltration, and necrosis distribution that align with responder cases in the reference cohort.

SIMILAR CASES:
Comparison with the reference cohort identified 3 similar cases with matching morphological patterns. Two of three similar cases were confirmed responders in the training data.

This report was generated by Enso Atlas using Path Foundation embeddings and MedGemma-powered structured reporting. Please review the limitations section carefully before using this information in clinical decision-making.`,
};

// Mock analysis response
export const mockAnalysisResponse: AnalysisResponse = {
  slideInfo: mockSlides[0],
  prediction: mockPrediction,
  evidencePatches: mockEvidencePatches,
  similarCases: mockSimilarCases,
  heatmap: mockHeatmap,
  report: mockReport,
  processingTimeMs: 12450,
};
