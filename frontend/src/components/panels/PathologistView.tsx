"use client";

import React, { useState, useCallback, useMemo } from "react";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { Slider } from "@/components/ui/Slider";
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
} from "lucide-react";
import { cn } from "@/lib/utils";
import type { AnalysisResponse, EvidencePatch, Annotation } from "@/types";

interface PathologistViewProps {
  analysisResult: AnalysisResponse;
  annotations: Annotation[];
  onAddAnnotation: (annotation: Omit<Annotation, "id" | "createdAt">) => void;
  onDeleteAnnotation: (id: string) => void;
  onPatchClick: (patchId: string) => void;
  onSwitchToOncologistView: () => void;
  selectedPatchId?: string;
  slideId: string;
}

type AnnotationTool = "pointer" | "circle" | "rectangle" | "freehand" | "measure" | "note";

const MAGNIFICATION_OPTIONS = [
  { value: 5, label: "5x" },
  { value: 10, label: "10x" },
  { value: 20, label: "20x" },
  { value: 40, label: "40x" },
  { value: 100, label: "100x" },
];

const TUMOR_GRADES = [
  { grade: 1, label: "Grade 1 - Well Differentiated", description: "Cells closely resemble normal tissue" },
  { grade: 2, label: "Grade 2 - Moderately Differentiated", description: "Cells somewhat abnormal" },
  { grade: 3, label: "Grade 3 - Poorly Differentiated", description: "Cells highly abnormal" },
  { grade: 4, label: "Grade 4 - Undifferentiated", description: "Cells bear little resemblance to normal" },
];

export function PathologistView({
  analysisResult,
  annotations,
  onAddAnnotation,
  onDeleteAnnotation,
  onPatchClick,
  onSwitchToOncologistView,
  selectedPatchId,
  slideId,
}: PathologistViewProps) {
  const [activeTool, setActiveTool] = useState<AnnotationTool>("pointer");
  const [selectedMagnification, setSelectedMagnification] = useState(20);
  const [showAnnotations, setShowAnnotations] = useState(true);
  const [noteText, setNoteText] = useState("");
  const [selectedGrade, setSelectedGrade] = useState<number | null>(null);
  
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

  const toggleSection = (section: keyof typeof expandedSections) => {
    setExpandedSections(prev => ({ ...prev, [section]: !prev[section] }));
  };

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
      // Save previous field count
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

  // Get morphology descriptions for patches
  const patchesWithMorphology = useMemo(() => {
    return analysisResult.evidencePatches.map((patch, idx) => ({
      ...patch,
      morphologyDescription: patch.morphologyDescription || generateMockMorphology(patch, idx),
    }));
  }, [analysisResult.evidencePatches]);

  return (
    <div className="h-full flex flex-col gap-4 overflow-y-auto">
      {/* Header with mode indicator */}
      <div className="flex items-center justify-between bg-violet-50 border border-violet-200 rounded-lg px-4 py-3">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-violet-600 rounded-lg">
            <Microscope className="h-5 w-5 text-white" />
          </div>
          <div>
            <h2 className="font-semibold text-violet-900">Pathologist Review Mode</h2>
            <p className="text-sm text-violet-600">Full WSI analysis with annotation tools</p>
          </div>
        </div>
        <Button
          variant="secondary"
          size="sm"
          onClick={onSwitchToOncologistView}
          className="border-violet-300 text-violet-700 hover:bg-violet-100"
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
          <h3 className="font-medium text-gray-900 flex items-center gap-2">
            <Target className="h-4 w-4 text-violet-600" />
            Navigation & Magnification
          </h3>
          {expandedSections.navigation ? (
            <ChevronUp className="h-4 w-4 text-gray-400" />
          ) : (
            <ChevronDown className="h-4 w-4 text-gray-400" />
          )}
        </button>
        
        {expandedSections.navigation && (
          <div className="space-y-4">
            <div className="flex flex-wrap gap-2">
              {MAGNIFICATION_OPTIONS.map((opt) => (
                <button
                  key={opt.value}
                  onClick={() => setSelectedMagnification(opt.value)}
                  className={cn(
                    "px-3 py-1.5 rounded-md text-sm font-medium transition-all",
                    selectedMagnification === opt.value
                      ? "bg-violet-600 text-white"
                      : "bg-gray-100 text-gray-700 hover:bg-gray-200"
                  )}
                >
                  {opt.label}
                </button>
              ))}
            </div>
            <div className="text-xs text-gray-500 flex items-center gap-2">
              <Info className="h-3 w-3" />
              Higher magnification for detailed cellular analysis
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
          <h3 className="font-medium text-gray-900 flex items-center gap-2">
            <PenTool className="h-4 w-4 text-violet-600" />
            Annotation Tools
          </h3>
          {expandedSections.annotations ? (
            <ChevronUp className="h-4 w-4 text-gray-400" />
          ) : (
            <ChevronDown className="h-4 w-4 text-gray-400" />
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
                onClick={() => setActiveTool("pointer")}
              />
              <ToolButton
                icon={<Circle className="h-4 w-4" />}
                label="Circle"
                active={activeTool === "circle"}
                onClick={() => setActiveTool("circle")}
              />
              <ToolButton
                icon={<Square className="h-4 w-4" />}
                label="Rectangle"
                active={activeTool === "rectangle"}
                onClick={() => setActiveTool("rectangle")}
              />
              <ToolButton
                icon={<PenTool className="h-4 w-4" />}
                label="Freehand"
                active={activeTool === "freehand"}
                onClick={() => setActiveTool("freehand")}
              />
              <ToolButton
                icon={<Ruler className="h-4 w-4" />}
                label="Measure"
                active={activeTool === "measure"}
                onClick={() => setActiveTool("measure")}
              />
              <ToolButton
                icon={<MessageSquare className="h-4 w-4" />}
                label="Note"
                active={activeTool === "note"}
                onClick={() => setActiveTool("note")}
              />
            </div>

            {/* Note input */}
            {activeTool === "note" && (
              <div className="flex gap-2">
                <input
                  type="text"
                  value={noteText}
                  onChange={(e) => setNoteText(e.target.value)}
                  placeholder="Add a note..."
                  className="flex-1 px-3 py-2 text-sm border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-violet-500"
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
              <span className="text-sm text-gray-600">
                {annotations.length} annotation{annotations.length !== 1 ? "s" : ""}
              </span>
              <button
                onClick={() => setShowAnnotations(!showAnnotations)}
                className="flex items-center gap-1 text-sm text-gray-500 hover:text-gray-700"
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
              <div className="max-h-32 overflow-y-auto space-y-1">
                {annotations.map((ann) => (
                  <div
                    key={ann.id}
                    className="flex items-center justify-between py-1 px-2 bg-gray-50 rounded text-sm"
                  >
                    <span className="truncate flex-1">{ann.text || ann.type}</span>
                    <button
                      onClick={() => onDeleteAnnotation(ann.id)}
                      className="p-1 text-red-500 hover:text-red-700"
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
          <h3 className="font-medium text-gray-900 flex items-center gap-2">
            <Layers className="h-4 w-4 text-violet-600" />
            Tumor Grading Assistant
          </h3>
          {expandedSections.grading ? (
            <ChevronUp className="h-4 w-4 text-gray-400" />
          ) : (
            <ChevronDown className="h-4 w-4 text-gray-400" />
          )}
        </button>

        {expandedSections.grading && (
          <div className="space-y-3">
            {TUMOR_GRADES.map((grade) => (
              <button
                key={grade.grade}
                onClick={() => setSelectedGrade(grade.grade)}
                className={cn(
                  "w-full text-left p-3 rounded-lg border transition-all",
                  selectedGrade === grade.grade
                    ? "border-violet-500 bg-violet-50"
                    : "border-gray-200 hover:border-gray-300"
                )}
              >
                <div className="flex items-center justify-between">
                  <span className="font-medium text-sm">{grade.label}</span>
                  {selectedGrade === grade.grade && (
                    <CheckCircle className="h-4 w-4 text-violet-600" />
                  )}
                </div>
                <p className="text-xs text-gray-500 mt-1">{grade.description}</p>
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
          <h3 className="font-medium text-gray-900 flex items-center gap-2">
            <Target className="h-4 w-4 text-violet-600" />
            Mitotic Figure Counter
          </h3>
          {expandedSections.mitotic ? (
            <ChevronUp className="h-4 w-4 text-gray-400" />
          ) : (
            <ChevronDown className="h-4 w-4 text-gray-400" />
          )}
        </button>

        {expandedSections.mitotic && (
          <div className="space-y-4">
            <div className="flex items-center justify-center gap-4 p-4 bg-gray-50 rounded-lg">
              <div className="text-center">
                <div className="text-3xl font-bold text-violet-600">{mitoticCount}</div>
                <div className="text-xs text-gray-500">Current Field</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-gray-400">{mitoticFields}</div>
                <div className="text-xs text-gray-500">Fields Counted</div>
              </div>
            </div>

            <div className="flex gap-2">
              <Button
                variant="primary"
                size="sm"
                className="flex-1"
                onClick={handleMarkMitotic}
              >
                <Target className="h-4 w-4 mr-1" />
                Mark Mitotic
              </Button>
              <Button
                variant="secondary"
                size="sm"
                onClick={handleNewField}
              >
                New Field
              </Button>
            </div>

            <div className="text-xs text-gray-500 flex items-start gap-2">
              <AlertTriangle className="h-3 w-3 mt-0.5 text-amber-500" />
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
          <h3 className="font-medium text-gray-900 flex items-center gap-2">
            <Microscope className="h-4 w-4 text-violet-600" />
            Patch Morphology Analysis
          </h3>
          {expandedSections.morphology ? (
            <ChevronUp className="h-4 w-4 text-gray-400" />
          ) : (
            <ChevronDown className="h-4 w-4 text-gray-400" />
          )}
        </button>

        {expandedSections.morphology && (
          <div className="space-y-3 max-h-96 overflow-y-auto">
            {patchesWithMorphology.slice(0, 8).map((patch, idx) => (
              <div
                key={patch.id}
                onClick={() => onPatchClick(patch.id)}
                className={cn(
                  "p-3 rounded-lg border cursor-pointer transition-all",
                  selectedPatchId === patch.id
                    ? "border-violet-500 bg-violet-50"
                    : "border-gray-200 hover:border-gray-300"
                )}
              >
                <div className="flex items-center justify-between mb-2">
                  <Badge variant="default" size="sm">
                    Patch {idx + 1}
                  </Badge>
                  <span className="text-xs text-gray-500">
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
                <p className="text-sm text-gray-600 leading-relaxed">
                  {patch.morphologyDescription}
                </p>
              </div>
            ))}
          </div>
        )}
      </Card>

      {/* Export Actions */}
      <div className="flex gap-2 mt-2">
        <Button variant="secondary" size="sm" className="flex-1">
          <Save className="h-4 w-4 mr-1" />
          Save Annotations
        </Button>
        <Button variant="secondary" size="sm" className="flex-1">
          <Download className="h-4 w-4 mr-1" />
          Export Report
        </Button>
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
        "flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm font-medium transition-all",
        active
          ? "bg-violet-600 text-white"
          : "bg-gray-100 text-gray-700 hover:bg-gray-200"
      )}
      title={label}
    >
      {icon}
      <span className="hidden sm:inline">{label}</span>
    </button>
  );
}

// Mock morphology description generator
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
