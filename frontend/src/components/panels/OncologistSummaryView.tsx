"use client";

import React from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { cn } from "@/lib/utils";
import {
  CheckCircle,
  XCircle,
  AlertTriangle,
  FlaskConical,
  Stethoscope,
  FileText,
  ZoomIn,
  ArrowRight,
  Eye,
  Layers,
  Shield,
  Circle,
} from "lucide-react";
import type { AnalysisResponse, StructuredReport, EvidencePatch } from "@/types";
import { useProject } from "@/contexts/ProjectContext";

interface OncologistSummaryViewProps {
  analysisResult: AnalysisResponse | null;
  report: StructuredReport | null;
  onPatchZoom?: (patch: EvidencePatch) => void;
  onSwitchToFullView?: () => void;
}

// Tissue type classification
type TissueType = "tumor" | "stroma" | "necrosis" | "inflammatory" | "normal" | "unknown";

function inferTissueType(description?: string): TissueType {
  if (!description) return "unknown";
  const lower = description.toLowerCase();
  if (lower.includes("necrotic") || lower.includes("necrosis")) return "necrosis";
  if (lower.includes("lymphocytic") || lower.includes("inflammatory") || lower.includes("infiltrate")) return "inflammatory";
  if (lower.includes("stromal") || lower.includes("stroma") || lower.includes("desmoplasia")) return "stroma";
  if (lower.includes("carcinoma") || lower.includes("tumor") || lower.includes("papillary") || 
      lower.includes("mitotic") || lower.includes("atypia")) return "tumor";
  if (lower.includes("normal") || lower.includes("benign")) return "normal";
  return "unknown";
}

const TISSUE_TYPES: Record<TissueType, { label: string; color: string; bgColor: string }> = {
  tumor: { label: "Tumor", color: "text-red-700", bgColor: "bg-red-100" },
  stroma: { label: "Stroma", color: "text-blue-700", bgColor: "bg-blue-100" },
  necrosis: { label: "Necrosis", color: "text-gray-700", bgColor: "bg-gray-200" },
  inflammatory: { label: "Inflammatory", color: "text-purple-700", bgColor: "bg-purple-100" },
  normal: { label: "Normal", color: "text-green-700", bgColor: "bg-green-100" },
  unknown: { label: "Unclassified", color: "text-gray-500", bgColor: "bg-gray-100" },
};

export function OncologistSummaryView({
  analysisResult,
  report,
  onPatchZoom,
  onSwitchToFullView,
}: OncologistSummaryViewProps) {
  // Project-aware labels (must be before any returns per Rules of Hooks)
  const { currentProject } = useProject();
  const positiveLabel = currentProject.positive_class || currentProject.classes?.[1] || "Responder";
  const negativeLabel = currentProject.classes?.find(c => c !== currentProject.positive_class) || currentProject.classes?.[0] || "Non-Responder";

  if (!analysisResult) {
    return (
      <div className="h-full flex items-center justify-center bg-gray-50 rounded-lg">
        <div className="text-center p-8">
          <Eye className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">
            No Analysis Results
          </h3>
          <p className="text-sm text-gray-500">
            Run an analysis to see the oncologist summary view.
          </p>
        </div>
      </div>
    );
  }

  const { prediction, evidencePatches } = analysisResult;
  const isResponder = prediction.score >= 0.5;
  const topPatches = evidencePatches.slice(0, 3);

  // Group similar cases by outcome
  const similarCases = analysisResult.similarCases || [];
  const responderCount = similarCases.filter(c => 
    c.label?.toLowerCase().includes("responder") && !c.label?.toLowerCase().includes("non")
  ).length;
  const nonResponderCount = similarCases.filter(c => 
    c.label?.toLowerCase().includes("non") || c.label?.toLowerCase().includes("negative")
  ).length;

  return (
    <div className="h-full overflow-y-auto bg-gray-50 p-6 space-y-6">
      {/* Header with view toggle */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Stethoscope className="h-6 w-6 text-clinical-600" />
          <h2 className="text-xl font-bold text-gray-900">Oncologist Summary</h2>
          <Badge variant="info" size="sm">Simplified View</Badge>
        </div>
        {onSwitchToFullView && (
          <Button
            variant="secondary"
            size="sm"
            onClick={onSwitchToFullView}
            leftIcon={<Layers className="h-4 w-4" />}
          >
            Full WSI View
          </Button>
        )}
      </div>

      {/* Uncalibrated Warning */}
      <div className="flex items-center gap-3 px-4 py-3 bg-amber-100 border-2 border-amber-400 rounded-lg">
        <FlaskConical className="h-5 w-5 text-amber-700 shrink-0" />
        <div>
          <span className="text-sm font-bold text-amber-800 uppercase tracking-wide">
            Uncalibrated - Research Use Only
          </span>
          <p className="text-xs text-amber-700 mt-0.5">
            Model output has not been validated for clinical decision-making.
          </p>
        </div>
      </div>

      {/* Main Prediction Card */}
      <Card className={cn(
        "border-2",
        isResponder ? "border-green-300 bg-green-50" : "border-red-300 bg-red-50"
      )}>
        <CardContent className="p-6">
          <div className="flex items-center gap-4 mb-4">
            {isResponder ? (
              <CheckCircle className="h-12 w-12 text-green-600" />
            ) : (
              <XCircle className="h-12 w-12 text-red-600" />
            )}
            <div>
              <h3 className={cn(
                "text-2xl font-bold",
                isResponder ? "text-green-800" : "text-red-800"
              )}>
                {prediction.label}
              </h3>
              <p className="text-lg font-mono font-semibold text-gray-700">
                {Math.round(prediction.score * 100)}% <span className="text-sm font-normal text-gray-500">raw score</span>
              </p>
            </div>
          </div>

          {/* Warning about raw probability */}
          <div className="flex items-center gap-2 text-sm text-gray-600 bg-white/50 rounded p-2">
            <AlertTriangle className="h-4 w-4" />
            <span>
              Raw model output: {Math.round(prediction.score * 100)}% (uncalibrated, interpret with caution)
            </span>
          </div>
        </CardContent>
      </Card>

      {/* Similar Cases Summary */}
      {similarCases.length > 0 && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Similar Cases in Reference Cohort</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-6">
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded-full bg-green-500" />
                <span className="text-sm font-medium">
                  {responderCount} {positiveLabel}{responderCount !== 1 ? "s" : ""}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded-full bg-red-500" />
                <span className="text-sm font-medium">
                  {nonResponderCount} {negativeLabel}{nonResponderCount !== 1 ? "s" : ""}
                </span>
              </div>
            </div>
            {responderCount > 0 && nonResponderCount > 0 && (
              <p className="text-xs text-gray-600 mt-3">
                This case shares morphological features with both {positiveLabel.toLowerCase()} and {negativeLabel.toLowerCase()} cases.
                Clinical context is essential for interpretation.
              </p>
            )}
          </CardContent>
        </Card>
      )}

      {/* Top 3 Key Regions */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base flex items-center gap-2">
            <Eye className="h-4 w-4 text-clinical-600" />
            Key Regions (Top 3 by Attention)
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-4">
            {topPatches.map((patch, index) => {
              const tissueType = inferTissueType(patch.morphologyDescription);
              const tissueInfo = TISSUE_TYPES[tissueType];
              return (
                <div
                  key={patch.id}
                  className="relative group cursor-pointer"
                  onClick={() => onPatchZoom?.(patch)}
                >
                  {/* Large thumbnail */}
                  <div className="aspect-square rounded-lg overflow-hidden border-2 border-gray-200 group-hover:border-clinical-500 transition-colors">
                    <img
                      src={patch.thumbnailUrl || "/placeholder-patch.png"}
                      alt={`Key region ${index + 1}`}
                      className="w-full h-full object-cover"
                    />
                    {/* Hover overlay */}
                    <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                      <ZoomIn className="h-8 w-8 text-white" />
                    </div>
                  </div>
                  
                  {/* Rank badge */}
                  <div className="absolute top-2 left-2 bg-navy-900/90 text-white text-sm font-bold px-2 py-1 rounded">
                    #{index + 1}
                  </div>
                  
                  {/* Attention score */}
                  <div className={cn(
                    "absolute top-2 right-2 w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold text-white",
                    patch.attentionWeight >= 0.7 ? "bg-red-500" :
                    patch.attentionWeight >= 0.4 ? "bg-amber-500" : "bg-blue-500"
                  )}>
                    {Math.round(patch.attentionWeight * 100)}
                  </div>
                  
                  {/* Tissue type and description */}
                  <div className="mt-2 space-y-1">
                    <span className={cn(
                      "inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium",
                      tissueInfo.bgColor,
                      tissueInfo.color
                    )}>
                      <Circle className="h-2 w-2 fill-current" />
                      {tissueInfo.label}
                    </span>
                    {patch.morphologyDescription && (
                      <p className="text-xs text-gray-600 line-clamp-2">
                        {patch.morphologyDescription}
                      </p>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
          <p className="text-xs text-gray-500 mt-4 text-center">
            Click any region to enlarge. These are the top attention regions driving the prediction.
          </p>
        </CardContent>
      </Card>

      {/* Suggested Next Steps */}
      {report?.suggestedNextSteps && report.suggestedNextSteps.length > 0 && (
        <Card className="border-2 border-blue-200 bg-blue-50">
          <CardHeader className="pb-3">
            <CardTitle className="text-base flex items-center gap-2 text-blue-800">
              <Stethoscope className="h-4 w-4" />
              Suggested Next Steps
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ol className="space-y-3">
              {report.suggestedNextSteps.map((step, index) => (
                <li key={index} className="flex items-start gap-3">
                  <span className="flex items-center justify-center w-6 h-6 rounded-full bg-blue-200 text-blue-700 text-sm font-bold shrink-0">
                    {index + 1}
                  </span>
                  <span className="text-sm text-blue-800">{step}</span>
                </li>
              ))}
            </ol>
          </CardContent>
        </Card>
      )}

      {/* Quick Summary Text */}
      {report?.summary && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base flex items-center gap-2">
              <FileText className="h-4 w-4" />
              Summary
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-gray-700 leading-relaxed whitespace-pre-wrap">
              {report.summary.split('\n').slice(0, 10).join('\n')}
            </p>
            {report.summary.split('\n').length > 10 && (
              <p className="text-xs text-gray-500 mt-2">
                ... (view full report for complete summary)
              </p>
            )}
          </CardContent>
        </Card>
      )}

      {/* Safety Notice */}
      <div className="p-4 bg-red-50 border-2 border-red-200 rounded-lg">
        <div className="flex items-start gap-3">
          <Shield className="h-5 w-5 text-red-600 mt-0.5 shrink-0" />
          <div>
            <h4 className="text-sm font-bold text-red-800 mb-1">
              Important Safety Notice
            </h4>
            <p className="text-sm text-red-700 leading-relaxed">
              This analysis is for research purposes only and is NOT a diagnostic tool. 
              All findings must be interpreted by qualified healthcare professionals in 
              the context of the complete clinical picture.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
