"use client";

import React, { useState, useCallback, useMemo, useEffect } from "react";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import {
  Crosshair,
  Ruler,
  PenTool,
  Circle,
  Square,
  MessageSquare,
  Trash2,
  Save,
  Download,
  FileText,
  Microscope,
  Target,
  AlertTriangle,
  CheckCircle,
  Info,
  ChevronDown,
  ChevronUp,
  Eye,
  EyeOff,
  Layers,
  RefreshCw,
} from "lucide-react";
import { cn } from "@/lib/utils";
import type { AnalysisResponse, EvidencePatch, Annotation, StructuredReport } from "@/types";

interface PathologistViewProps {
  analysisResult?: AnalysisResponse | null;
  annotations: Annotation[];
  onAddAnnotation: (annotation: Omit<Annotation, "id" | "createdAt">) => void;
  onDeleteAnnotation: (id: string) => void;
  onPatchClick: (patchId: string) => void;
  onSwitchToOncologistView: () => void;
  selectedPatchId?: string;
  slideId: string;
  viewerZoom?: number;
  onZoomTo?: (level: number) => void;
  onAnnotationToolChange?: (tool: "pointer" | "circle" | "rectangle" | "freehand" | "point") => void;
  selectedAnnotationId?: string | null;
  onSelectAnnotation?: (annotationId: string) => void;
  onExportPdf?: () => void;
  report?: StructuredReport | null;
  mpp?: number;
}

type AnnotationTool = "pointer" | "circle" | "rectangle" | "freehand" | "measure" | "note";

const MAGNIFICATION_OPTIONS = [
  { value: 0.5, label: "5x", viewerZoom: 0.125 },
  { value: 1, label: "10x", viewerZoom: 0.25 },
  { value: 2, label: "20x", viewerZoom: 0.5 },
  { value: 4, label: "40x", viewerZoom: 1 },
  { value: 10, label: "100x", viewerZoom: 2.5 },
];

const TUMOR_GRADES = [
  { grade: 1, label: "Grade 1 - Well Differentiated", description: "Cells closely resemble normal tissue" },
  { grade: 2, label: "Grade 2 - Moderately Differentiated", description: "Cells somewhat abnormal" },
  { grade: 3, label: "Grade 3 - Poorly Differentiated", description: "Cells highly abnormal" },
  { grade: 4, label: "Grade 4 - Undifferentiated", description: "Cells bear little resemblance to normal" },
];

// localStorage key prefix for annotations
const ANNOTATION_STORAGE_KEY = "pathologist-annotations-";

export function PathologistView({
  analysisResult,
  annotations,
  onAddAnnotation,
  onDeleteAnnotation,
  onPatchClick,
  onSwitchToOncologistView,
  selectedPatchId,
  slideId,
  viewerZoom = 1,
  onZoomTo,
  onAnnotationToolChange,
  selectedAnnotationId,
  onSelectAnnotation,
  onExportPdf,
  report,
  mpp,
}: PathologistViewProps) {
  const [activeTool, setActiveTool] = useState<AnnotationTool>("pointer");
  const [showAnnotations, setShowAnnotations] = useState(true);
  const [noteText, setNoteText] = useState("");
  const [selectedGrade, setSelectedGrade] = useState<number | null>(null);
  const [saveStatus, setSaveStatus] = useState<"idle" | "saving" | "saved">("idle");
  
  // Collapsible sections
  const [expandedSections, setExpandedSections] = useState({
    navigation: true,
    annotations: true,
    grading: true,
    mitotic: true,
    morphology: true,
  });

  // Mock mitotic count state
  const [mitoticCount, setMitoticCount] = useState(0);
  const [mitoticFields, setMitoticFields] = useState(0);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setActiveTool("pointer");
        onAnnotationToolChange?.("pointer");
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onAnnotationToolChange]);

  // Derive current magnification label from viewer zoom
  const baseMagnification = useMemo(() => {
    if (mpp && mpp > 0) {
      return 10 / mpp;
    }
    return 40;
  }, [mpp]);

  const effectiveMagnification = useMemo(() => viewerZoom * baseMagnification, [viewerZoom, baseMagnification]);

  const currentMagnification = useMemo(() => {
    // Find closest magnification option to current effective magnification
    let closest = MAGNIFICATION_OPTIONS[0];
    let minDist = Infinity;
    for (const opt of MAGNIFICATION_OPTIONS) {
      const dist = Math.abs(effectiveMagnification - opt.value);
      if (dist < minDist) {
        minDist = dist;
        closest = opt;
      }
    }
    return closest;
  }, [effectiveMagnification]);

  const toggleSection = (section: keyof typeof expandedSections) => {
    setExpandedSections(prev => ({ ...prev, [section]: !prev[section] }));
  };

  const handleMagnificationClick = useCallback((opt: typeof MAGNIFICATION_OPTIONS[number]) => {
    if (onZoomTo) {
      const targetViewerZoom = opt.value / baseMagnification;
      onZoomTo(targetViewerZoom);
    }
  }, [onZoomTo, baseMagnification]);

  const handleToolChange = useCallback((tool: AnnotationTool) => {
    const nextTool: AnnotationTool = activeTool === tool ? "pointer" : tool;
    setActiveTool(nextTool);
    if (nextTool === "note" || nextTool === "pointer") {
      onAnnotationToolChange?.("pointer");
      return;
    }
    if (nextTool === "measure") {
      onAnnotationToolChange?.("point");
      return;
    }
    onAnnotationToolChange?.(nextTool);
  }, [activeTool, onAnnotationToolChange]);

  const handleAddNote = useCallback(() => {
    if (!noteText.trim()) return;
    
    onAddAnnotation({
      slideId,
      type: "note",
      coordinates: { x: 0, y: 0, width: 0, height: 0 },
      text: noteText,
      color: "#3b82f6",
    });
    setNoteText("");
  }, [noteText, slideId, onAddAnnotation]);

  const handleMarkMitotic = useCallback(() => {
    setMitoticCount(prev => prev + 1);
    onAddAnnotation({
      slideId,
      type: "marker",
      coordinates: { x: 0, y: 0, width: 10, height: 10 },
      text: `Mitotic figure #${mitoticCount + 1}`,
      color: "#ef4444",
      category: "mitotic",
    });
  }, [slideId, mitoticCount, onAddAnnotation]);

  const handleNewField = useCallback(() => {
    if (mitoticFields > 0) {
      onAddAnnotation({
        slideId,
        type: "note",
        coordinates: { x: 0, y: 0, width: 0, height: 0 },
        text: `Field ${mitoticFields}: ${mitoticCount} mitotic figures`,
        color: "#f59e0b",
        category: "mitotic-summary",
      });
    }
    setMitoticFields(prev => prev + 1);
    setMitoticCount(0);
  }, [slideId, mitoticCount, mitoticFields, onAddAnnotation]);

  // Save annotations to localStorage
  const handleSaveAnnotations = useCallback(() => {
    setSaveStatus("saving");
    try {
      const key = ANNOTATION_STORAGE_KEY + slideId;
      const data = {
        savedAt: new Date().toISOString(),
        slideId,
        selectedGrade,
        mitoticCount,
        mitoticFields,
        annotations: annotations.map(ann => ({
          id: ann.id,
          type: ann.type,
          text: ann.text,
          category: ann.category,
          color: ann.color,
          coordinates: ann.coordinates,
          createdAt: ann.createdAt,
        })),
      };
      localStorage.setItem(key, JSON.stringify(data));
      setSaveStatus("saved");
      setTimeout(() => setSaveStatus("idle"), 2000);
    } catch {
      setSaveStatus("idle");
    }
  }, [slideId, annotations, selectedGrade, mitoticCount, mitoticFields]);

  // Load saved grade/mitotic state from localStorage on mount
  useEffect(() => {
    try {
      const key = ANNOTATION_STORAGE_KEY + slideId;
      const raw = localStorage.getItem(key);
      if (raw) {
        const data = JSON.parse(raw);
        if (data.selectedGrade != null) setSelectedGrade(data.selectedGrade);
        if (data.mitoticCount != null) setMitoticCount(data.mitoticCount);
        if (data.mitoticFields != null) setMitoticFields(data.mitoticFields);
      }
    } catch {
      // ignore parse errors
    }
  }, [slideId]);

  // Handle export - formatted text report or PDF
  const handleExportTextReport = useCallback(() => {
    const selectedGradeInfo = TUMOR_GRADES.find(g => g.grade === selectedGrade);
    const now = new Date();

    const lines: string[] = [
      "=".repeat(60),
      "PATHOLOGIST REVIEW REPORT",
      "=".repeat(60),
      "",
      `Slide ID: ${slideId}`,
      `Date: ${now.toLocaleDateString()} ${now.toLocaleTimeString()}`,
      "",
      "--- TUMOR GRADING ---",
      selectedGradeInfo
        ? `Grade: ${selectedGradeInfo.label}`
        : "Grade: Not assessed",
      "",
      "--- MITOTIC ASSESSMENT ---",
      `Fields counted: ${mitoticFields}`,
      `Current field count: ${mitoticCount}`,
      "",
      "--- ANNOTATIONS ---",
      `Total: ${annotations.length}`,
    ];

    for (const ann of annotations) {
      lines.push(`  [${ann.type}] ${ann.text || "(no text)"} (${ann.createdAt})`);
    }

    if (analysisResult) {
      lines.push("");
      lines.push("--- AI ANALYSIS SUMMARY ---");
      lines.push(`Prediction: ${analysisResult.prediction?.label ?? "N/A"}`);
      lines.push(`Confidence: ${analysisResult.prediction?.confidence != null ? (analysisResult.prediction.confidence * 100).toFixed(1) + "%" : "N/A"}`);
      lines.push(`Evidence patches: ${analysisResult.evidencePatches?.length ?? 0}`);
    }

    lines.push("");
    lines.push("--- DISCLAIMER ---");
    lines.push("This report is for research and decision support only.");
    lines.push("Not a substitute for professional medical judgment.");
    lines.push("=".repeat(60));

    const blob = new Blob([lines.join("\n")], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `pathologist-review-${slideId}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  }, [slideId, selectedGrade, mitoticFields, mitoticCount, annotations, analysisResult]);

  // Get morphology descriptions for patches
  const patchesWithMorphology = useMemo(() => {
    if (!analysisResult) return [];
    return analysisResult.evidencePatches.map((patch, idx) => ({
      ...patch,
      morphologyDescription: patch.morphologyDescription || generateMockMorphology(patch, idx),
      isGenerated: !patch.morphologyDescription,
    }));
  }, [analysisResult]);

  const secondaryButtonDarkClass = "dark:bg-navy-700 dark:text-gray-100 dark:border-navy-500 dark:hover:bg-navy-600 dark:focus:ring-offset-navy-900";

  return (
    <div className="flex flex-col gap-4 pb-2">
      {/* Header with mode indicator */}
      <div className="flex items-center justify-between rounded-lg border border-violet-200 bg-violet-50 px-4 py-3 dark:border-violet-800 dark:bg-violet-900/20">
        <div className="flex items-center gap-3">
          <div className="rounded-lg bg-violet-600 p-2 dark:bg-violet-500">
            <Microscope className="h-5 w-5 text-white" />
          </div>
          <div>
            <h2 className="font-semibold text-violet-900 dark:text-violet-100">Pathologist Review Mode</h2>
            <p className="text-sm text-violet-600 dark:text-violet-300">Full WSI analysis with annotation tools</p>
          </div>
        </div>
        <Button
          variant="secondary"
          size="sm"
          onClick={onSwitchToOncologistView}
          className="border-violet-300 text-violet-700 hover:bg-violet-100 dark:border-violet-700 dark:bg-violet-900/40 dark:text-violet-200 dark:hover:bg-violet-800/70"
        >
          Switch to Oncologist View
        </Button>
      </div>

      {/* Magnification Control */}
      <Card className="p-4">
        <button
          onClick={() => toggleSection("navigation")}
          className="flex items-center justify-between w-full mb-3"
        >
          <h3 className="flex items-center gap-2 font-medium text-gray-900 dark:text-gray-100">
            <Target className="h-4 w-4 text-violet-600 dark:text-violet-400" />
            Navigation &amp; Magnification
          </h3>
          {expandedSections.navigation ? (
            <ChevronUp className="h-4 w-4 text-gray-400 dark:text-gray-500" />
          ) : (
            <ChevronDown className="h-4 w-4 text-gray-400 dark:text-gray-500" />
          )}
        </button>

        {expandedSections.navigation && (
          <div className="space-y-4">
            <div className="flex flex-wrap gap-2">
              {MAGNIFICATION_OPTIONS.map((opt) => (
                <button
                  key={opt.label}
                  onClick={() => handleMagnificationClick(opt)}
                  className={cn(
                    "rounded-md px-3 py-1.5 text-sm font-medium transition-all",
                    currentMagnification.label === opt.label
                      ? "bg-violet-600 text-white dark:bg-violet-500"
                      : "bg-gray-100 text-gray-700 hover:bg-gray-200 dark:bg-navy-700 dark:text-gray-200 dark:hover:bg-navy-600"
                  )}
                >
                  {opt.label}
                </button>
              ))}
            </div>
            <div className="flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400">
              <Info className="h-3 w-3 text-violet-500 dark:text-violet-400" />
              <span>
                Viewer zoom: {viewerZoom < 1 ? viewerZoom.toFixed(2) : viewerZoom.toFixed(1)}x
                {" | "}Effective: {effectiveMagnification.toFixed(1)}x
                {" | "}Nearest preset: {currentMagnification.label}
              </span>
            </div>
          </div>
        )}
      </Card>

      {/* Annotation Tools */}
      <Card className="p-4">
        <button
          onClick={() => toggleSection("annotations")}
          className="flex items-center justify-between w-full mb-3"
        >
          <h3 className="flex items-center gap-2 font-medium text-gray-900 dark:text-gray-100">
            <PenTool className="h-4 w-4 text-violet-600 dark:text-violet-400" />
            Annotation Tools
          </h3>
          {expandedSections.annotations ? (
            <ChevronUp className="h-4 w-4 text-gray-400 dark:text-gray-500" />
          ) : (
            <ChevronDown className="h-4 w-4 text-gray-400 dark:text-gray-500" />
          )}
        </button>

        {expandedSections.annotations && (
          <div className="space-y-4">
            {/* Tool buttons */}
            <div className="flex flex-wrap gap-2">
              <ToolButton
                icon={<Crosshair className="h-4 w-4" />}
                label="Pointer"
                active={activeTool === "pointer"}
                onClick={() => handleToolChange("pointer")}
              />
              <ToolButton
                icon={<Circle className="h-4 w-4" />}
                label="Circle"
                active={activeTool === "circle"}
                onClick={() => handleToolChange("circle")}
              />
              <ToolButton
                icon={<Square className="h-4 w-4" />}
                label="Rectangle"
                active={activeTool === "rectangle"}
                onClick={() => handleToolChange("rectangle")}
              />
              <ToolButton
                icon={<PenTool className="h-4 w-4" />}
                label="Freehand"
                active={activeTool === "freehand"}
                onClick={() => handleToolChange("freehand")}
              />
              <ToolButton
                icon={<Ruler className="h-4 w-4" />}
                label="Measure"
                active={activeTool === "measure"}
                onClick={() => handleToolChange("measure")}
              />
              <ToolButton
                icon={<MessageSquare className="h-4 w-4" />}
                label="Note"
                active={activeTool === "note"}
                onClick={() => handleToolChange("note")}
              />
            </div>

            {/* Active tool indicator */}
            {activeTool !== "pointer" && activeTool !== "note" && (
              <div className="rounded-md border border-violet-200 bg-violet-50 px-3 py-2 text-xs text-violet-700 dark:border-violet-800 dark:bg-violet-900/20 dark:text-violet-300">
                Tool active: {activeTool}. Draw on the slide. Click the active tool again or press Esc to return to pan mode.
              </div>
            )}

            {/* Note input */}
            {activeTool === "note" && (
              <div className="flex gap-2">
                <input
                  type="text"
                  value={noteText}
                  onChange={(e) => setNoteText(e.target.value)}
                  placeholder="Add a note..."
                  className="flex-1 rounded-lg border border-gray-200 bg-white px-3 py-2 text-sm text-gray-900 focus:outline-none focus:ring-2 focus:ring-violet-500 dark:border-navy-600 dark:bg-navy-900 dark:text-gray-100 dark:placeholder:text-gray-400"
                  onKeyDown={(e) => {
                    if (e.key === "Enter") handleAddNote();
                  }}
                />
                <Button size="sm" onClick={handleAddNote} disabled={!noteText.trim()}>
                  Add
                </Button>
              </div>
            )}

            {/* Annotations list */}
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600 dark:text-gray-300">
                {annotations.length} annotation{annotations.length !== 1 ? "s" : ""}
              </span>
              <button
                onClick={() => setShowAnnotations(!showAnnotations)}
                className="flex items-center gap-1 text-sm text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
              >
                {showAnnotations ? (
                  <>
                    <Eye className="h-3 w-3" /> Visible
                  </>
                ) : (
                  <>
                    <EyeOff className="h-3 w-3" /> Hidden
                  </>
                )}
              </button>
            </div>

            {showAnnotations && annotations.length > 0 && (
              <div className="max-h-32 overflow-y-auto space-y-1 dark:rounded-lg dark:border dark:border-navy-600 dark:bg-navy-900/30 dark:p-1">
                {annotations.map((ann) => (
                  <div
                    key={ann.id}
                    className={cn(
                      "flex cursor-pointer items-center justify-between rounded px-2 py-1 text-sm",
                      selectedAnnotationId === ann.id
                        ? "border border-violet-300 bg-violet-100 dark:border-violet-700 dark:bg-violet-900/30"
                        : "bg-gray-50 dark:bg-navy-700/70"
                    )}
                    onClick={() => onSelectAnnotation?.(ann.id)}
                  >
                    <span className="flex-1 truncate text-gray-800 dark:text-gray-100">{ann.text || ann.notes || ann.label || ann.type}</span>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onDeleteAnnotation(ann.id);
                      }}
                      className="p-1 text-red-500 hover:text-red-700 dark:text-red-400 dark:hover:text-red-300"
                    >
                      <Trash2 className="h-3 w-3" />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </Card>

      {/* Tumor Grading Assistant */}
      <Card className="p-4">
        <button
          onClick={() => toggleSection("grading")}
          className="flex items-center justify-between w-full mb-3"
        >
          <h3 className="flex items-center gap-2 font-medium text-gray-900 dark:text-gray-100">
            <Layers className="h-4 w-4 text-violet-600 dark:text-violet-400" />
            Tumor Grading Assistant
          </h3>
          {expandedSections.grading ? (
            <ChevronUp className="h-4 w-4 text-gray-400 dark:text-gray-500" />
          ) : (
            <ChevronDown className="h-4 w-4 text-gray-400 dark:text-gray-500" />
          )}
        </button>

        {expandedSections.grading && (
          <div className="max-h-64 space-y-3 overflow-y-auto pr-1">
            {TUMOR_GRADES.map((grade) => (
              <button
                key={grade.grade}
                onClick={() => setSelectedGrade(grade.grade)}
                className={cn(
                  "w-full rounded-lg border p-3 text-left transition-all",
                  selectedGrade === grade.grade
                    ? "border-violet-500 bg-violet-50 dark:border-violet-700 dark:bg-violet-900/20"
                    : "border-gray-200 hover:border-gray-300 dark:border-navy-600 dark:bg-navy-900/20 dark:hover:border-navy-500"
                )}
              >
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-gray-900 dark:text-gray-100">{grade.label}</span>
                  {selectedGrade === grade.grade && (
                    <CheckCircle className="h-4 w-4 text-violet-600 dark:text-violet-400" />
                  )}
                </div>
                <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">{grade.description}</p>
              </button>
            ))}
          </div>
        )}
      </Card>

      {/* Mitotic Figure Counter */}
      <Card className="p-4">
        <button
          onClick={() => toggleSection("mitotic")}
          className="flex items-center justify-between w-full mb-3"
        >
          <h3 className="flex items-center gap-2 font-medium text-gray-900 dark:text-gray-100">
            <Target className="h-4 w-4 text-violet-600 dark:text-violet-400" />
            Mitotic Figure Counter
          </h3>
          {expandedSections.mitotic ? (
            <ChevronUp className="h-4 w-4 text-gray-400 dark:text-gray-500" />
          ) : (
            <ChevronDown className="h-4 w-4 text-gray-400 dark:text-gray-500" />
          )}
        </button>

        {expandedSections.mitotic && (
          <div className="space-y-4">
            <div className="flex items-center justify-center gap-4 rounded-lg bg-gray-50 p-4 dark:border dark:border-navy-600 dark:bg-navy-900/40">
              <div className="text-center">
                <div className="text-3xl font-bold text-violet-600 dark:text-violet-400">{mitoticCount}</div>
                <div className="text-xs text-gray-500 dark:text-gray-400">Current Field</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-gray-400 dark:text-gray-300">{mitoticFields}</div>
                <div className="text-xs text-gray-500 dark:text-gray-400">Fields Counted</div>
              </div>
            </div>

            <div className="flex gap-2">
              <Button
                variant="primary"
                size="sm"
                className="flex-1"
                onClick={handleMarkMitotic}
              >
                <Target className="mr-1 h-4 w-4" />
                Mark Mitotic
              </Button>
              <Button
                variant="secondary"
                size="sm"
                className={secondaryButtonDarkClass}
                onClick={handleNewField}
              >
                New Field
              </Button>
            </div>

            <div className="flex items-start gap-2 text-xs text-gray-500 dark:text-gray-400">
              <AlertTriangle className="mt-0.5 h-3 w-3 text-amber-500 dark:text-amber-400" />
              <span>
                Count mitotic figures in 10 consecutive HPFs (40x) for accurate assessment.
                Mock counter for demo purposes.
              </span>
            </div>
          </div>
        )}
      </Card>

      {/* Morphology Descriptions */}
      <Card className="p-4">
        <button
          onClick={() => toggleSection("morphology")}
          className="flex items-center justify-between w-full mb-3"
        >
          <h3 className="flex items-center gap-2 font-medium text-gray-900 dark:text-gray-100">
            <Microscope className="h-4 w-4 text-violet-600 dark:text-violet-400" />
            Patch Morphology Analysis
          </h3>
          {expandedSections.morphology ? (
            <ChevronUp className="h-4 w-4 text-gray-400 dark:text-gray-500" />
          ) : (
            <ChevronDown className="h-4 w-4 text-gray-400 dark:text-gray-500" />
          )}
        </button>

        {expandedSections.morphology && (
          <div className="max-h-96 space-y-3 overflow-y-auto">
            {patchesWithMorphology.length === 0 && (
              <p className="py-4 text-center text-sm text-gray-500 dark:text-gray-400">
                Run analysis to see patch morphology descriptions.
              </p>
            )}
            {patchesWithMorphology.slice(0, 8).map((patch, idx) => (
              <div
                key={patch.id}
                onClick={() => onPatchClick(patch.id)}
                className={cn(
                  "cursor-pointer rounded-lg border p-3 transition-all",
                  selectedPatchId === patch.id
                    ? "border-violet-500 bg-violet-50 dark:border-violet-700 dark:bg-violet-900/20"
                    : "border-gray-200 hover:border-gray-300 dark:border-navy-600 dark:bg-navy-900/20 dark:hover:border-navy-500"
                )}
              >
                <div className="mb-2 flex items-center justify-between">
                  <Badge variant="default" size="sm">
                    Patch {idx + 1}
                  </Badge>
                  <span className="text-xs text-gray-500 dark:text-gray-400">
                    Attention: {(patch.attentionWeight * 100).toFixed(1)}%
                  </span>
                </div>
                {patch.tissueType && (
                  <Badge
                    variant={
                      patch.tissueType === "tumor"
                        ? "danger"
                        : patch.tissueType === "stroma"
                          ? "warning"
                          : "info"
                    }
                    size="sm"
                    className="mb-2"
                  >
                    {patch.tissueType}
                    {patch.tissueConfidence && ` (${(patch.tissueConfidence * 100).toFixed(0)}%)`}
                  </Badge>
                )}
                <p className="text-sm leading-relaxed text-gray-600 dark:text-gray-300">
                  {patch.morphologyDescription}
                </p>
                {patch.isGenerated && (
                  <p className="mt-1 text-2xs italic text-gray-400 dark:text-gray-500">
                    AI-generated description (fallback)
                  </p>
                )}
              </div>
            ))}
          </div>
        )}
      </Card>

      {/* Export Actions */}
      <div className="mt-2 flex flex-col gap-2 dark:rounded-lg dark:border dark:border-navy-700 dark:bg-navy-900/40 dark:p-3">
        <div className="flex gap-2">
          <Button
            variant="secondary"
            size="sm"
            className={cn("flex-1", secondaryButtonDarkClass)}
            onClick={handleSaveAnnotations}
            disabled={saveStatus === "saving"}
          >
            {saveStatus === "saving" ? (
              <>
                <RefreshCw className="mr-1 h-4 w-4 animate-spin" />
                Saving...
              </>
            ) : saveStatus === "saved" ? (
              <>
                <CheckCircle className="mr-1 h-4 w-4 text-green-600 dark:text-green-400" />
                Saved to browser
              </>
            ) : (
              <>
                <Save className="mr-1 h-4 w-4" />
                Save Annotations
              </>
            )}
          </Button>
        </div>
        <div className="flex gap-2">
          {onExportPdf && report && (
            <Button
              variant="secondary"
              size="sm"
              className={cn("flex-1", secondaryButtonDarkClass)}
              onClick={onExportPdf}
            >
              <FileText className="mr-1 h-4 w-4" />
              Export PDF
            </Button>
          )}
          <Button
            variant="secondary"
            size="sm"
            className={cn("flex-1", secondaryButtonDarkClass)}
            onClick={handleExportTextReport}
          >
            <Download className="mr-1 h-4 w-4" />
            Export Text Report
          </Button>
        </div>
        <p className="text-center text-2xs text-gray-400 dark:text-gray-500">
          Annotations saved to browser localStorage
        </p>
      </div>
    </div>
  );
}

// Tool button component
function ToolButton({
  icon,
  label,
  active,
  onClick,
}: {
  icon: React.ReactNode;
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "flex items-center gap-1.5 rounded-lg px-3 py-2 text-sm font-medium transition-all",
        active
          ? "bg-violet-600 text-white dark:bg-violet-500"
          : "bg-gray-100 text-gray-700 hover:bg-gray-200 dark:bg-navy-700 dark:text-gray-200 dark:hover:bg-navy-600"
      )}
      title={label}
    >
      {icon}
      <span className="hidden sm:inline">{label}</span>
    </button>
  );
}

// Mock morphology description generator (fallback when real data unavailable)
function generateMockMorphology(patch: EvidencePatch, idx: number): string {
  const descriptions = [
    "Displays atypical cellular proliferation with enlarged nuclei and irregular chromatin distribution. Increased nuclear-to-cytoplasmic ratio suggests high-grade features.",
    "Stromal component with reactive fibroblasts and scattered inflammatory cells. Desmoplastic reaction evident around tumor nests.",
    "High cellular density with solid growth pattern. Mitotic activity observed in multiple cells. Nuclear pleomorphism present.",
    "Well-formed glandular structures with moderate differentiation. Some areas show cribriform architecture with back-to-back glands.",
    "Necrotic debris with ghost outlines of cells. Adjacent viable tumor shows high-grade features with geographic necrosis pattern.",
    "Dense lymphocytic infiltrate surrounding tumor nests. Pattern suggests tumor-infiltrating lymphocytes (TILs) with potential prognostic significance.",
    "Clear cell morphology with distinct cell borders. Abundant cytoplasm with eccentric nuclei. Consistent with specific histological subtype.",
    "Sheet-like growth with loss of polarity. High mitotic rate with atypical mitotic figures. Anaplastic features noted.",
  ];
  
  return descriptions[idx % descriptions.length];
}
