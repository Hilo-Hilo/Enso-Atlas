"use client";

import React, { useState, useMemo } from "react";
import { useProject } from "@/contexts/ProjectContext";
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
  CardFooter,
} from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { cn } from "@/lib/utils";
import {
  FileText,
  Download,
  Copy,
  Check,
  AlertTriangle,
  Lightbulb,
  ChevronDown,
  ChevronUp,
  Printer,
  Search,
  ClipboardList,
  AlertCircle,
  ArrowRight,
  FileCheck,
  RefreshCw,
  Stethoscope,
  Circle,
  Activity,
  BookOpen,
  Target,
  TrendingUp,
  ExternalLink,
  Info,
} from "lucide-react";
import type { StructuredReport, DecisionSupport, RiskLevel, PatchCoordinates } from "@/types";

// Tissue type inference for evidence patches in report
type TissueType = "tumor" | "stroma" | "necrosis" | "inflammatory" | "normal" | "unknown";

function inferTissueType(description: string): TissueType {
  const lower = description.toLowerCase();
  if (lower.includes("necrotic") || lower.includes("necrosis")) return "necrosis";
  if (lower.includes("lymphocytic") || lower.includes("inflammatory") || lower.includes("infiltrate")) return "inflammatory";
  if (lower.includes("stromal") || lower.includes("stroma") || lower.includes("desmoplasia")) return "stroma";
  if (lower.includes("carcinoma") || lower.includes("tumor") || lower.includes("papillary") || 
      lower.includes("mitotic") || lower.includes("atypia")) return "tumor";
  if (lower.includes("normal") || lower.includes("benign")) return "normal";
  return "unknown";
}

const TISSUE_TYPE_LABELS: Record<TissueType, { label: string; color: string }> = {
  tumor: { label: "Tumor", color: "text-red-600 dark:text-red-300" },
  stroma: { label: "Stroma", color: "text-blue-600 dark:text-blue-300" },
  necrosis: { label: "Necrosis", color: "text-gray-600 dark:text-gray-300" },
  inflammatory: { label: "Inflammatory", color: "text-purple-600 dark:text-purple-300" },
  normal: { label: "Normal", color: "text-green-600 dark:text-green-300" },
  unknown: { label: "Unclassified", color: "text-gray-400 dark:text-gray-500" },
};

// Risk level styling configuration
const RISK_LEVEL_CONFIG: Record<RiskLevel, { label: string; color: string; bgColor: string; borderColor: string; icon: string }> = {
  high_confidence: { 
    label: "High Confidence", 
    color: "text-emerald-700 dark:text-emerald-300", 
    bgColor: "bg-emerald-50 dark:bg-emerald-900/30", 
    borderColor: "border-emerald-300 dark:border-emerald-800",
    icon: "check-circle"
  },
  moderate_confidence: { 
    label: "Moderate Confidence", 
    color: "text-amber-700 dark:text-amber-300", 
    bgColor: "bg-amber-50 dark:bg-amber-900/30", 
    borderColor: "border-amber-300 dark:border-amber-800",
    icon: "alert-circle"
  },
  low_confidence: { 
    label: "Low Confidence", 
    color: "text-orange-700 dark:text-orange-300", 
    bgColor: "bg-orange-50 dark:bg-orange-900/30", 
    borderColor: "border-orange-300 dark:border-orange-800",
    icon: "alert-triangle"
  },
  inconclusive: { 
    label: "Inconclusive", 
    color: "text-red-700 dark:text-red-300", 
    bgColor: "bg-red-50 dark:bg-red-900/30", 
    borderColor: "border-red-300 dark:border-red-800",
    icon: "x-circle"
  },
};

interface ReportPanelProps {
  progress?: number;  // 0-100
  progressMessage?: string;
  report: StructuredReport | null;
  isLoading?: boolean;
  onGenerateReport?: () => void;
  onExportPdf?: () => void;
  onExportJson?: () => void;
  error?: string | null;
  onRetry?: () => void;
  onEvidenceClick?: (coords: PatchCoordinates) => void;
}

export function ReportPanel({
  report,
  isLoading,
  onGenerateReport,
  onExportPdf,
  onExportJson,
  error,
  progress = 0,
  progressMessage = "Generating clinical report...",
  onRetry,
  onEvidenceClick,
}: ReportPanelProps) {
  const panelTitle = "MedGemma Clinical Decision Brief";
  const panelSubtitle = "AI-authored pathology decision-support narrative";
  const { currentProject } = useProject();
  const positiveClassLower = useMemo(() => (currentProject.positive_class || "").toLowerCase(), [currentProject.positive_class]);
  const isPositiveLabel = (label: string) => {
    const lower = label.toLowerCase();
    if (positiveClassLower && lower === positiveClassLower) return true;
    return lower.includes("responder") && !lower.includes("non") || lower === "sensitive" || lower === "positive";
  };

  const [expandedSections, setExpandedSections] = useState<Set<string>>(
    new Set(["evidence"])
  );
  const [copied, setCopied] = useState(false);

  const toggleSection = (section: string) => {
    const newExpanded = new Set(expandedSections);
    if (newExpanded.has(section)) {
      newExpanded.delete(section);
    } else {
      newExpanded.add(section);
    }
    setExpandedSections(newExpanded);
  };

  const handleCopy = async () => {
    if (!report) return;
    try {
      await navigator.clipboard.writeText(report.summary);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy:", err);
    }
  };

  const handlePrint = () => {
    window.print();
  };

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <ClipboardList className="h-4 w-4 text-clinical-600 animate-pulse" />
            {panelTitle}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col items-center py-8">
            <div className="relative w-16 h-16 mb-4">
              <div className="absolute inset-0 rounded-full border-4 border-gray-100 dark:border-navy-700" />
              <div className="absolute inset-0 rounded-full border-4 border-clinical-500 border-t-transparent animate-spin" />
              <div className="absolute inset-2 rounded-full bg-gray-50 dark:bg-navy-800 flex items-center justify-center">
                <FileText className="h-6 w-6 text-clinical-600" />
              </div>
            </div>
            <p className="text-sm font-medium text-gray-700 dark:text-gray-300">
              {progressMessage}
            </p>
            {/* Progress bar */}
            <div className="w-full max-w-xs mt-4">
              <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400 mb-1">
                <span>Progress</span>
                <span>{Math.round(progress)}%</span>
              </div>
              <div className="w-full bg-gray-200 dark:bg-navy-700 rounded-full h-2">
                <div 
                  className="bg-clinical-500 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
            <p className="text-xs text-gray-400 dark:text-gray-500 mt-3">
              {progress < 30 ? "Loading slide data..." : 
               progress < 50 ? "Running analysis..." :
               progress < 80 ? "Generating MedGemma clinical narrative..." :
               "Finalizing MedGemma brief..."}
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Error state for report generation
  if (error && !isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <ClipboardList className="h-4 w-4 text-red-500" />
            {panelTitle}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-6">
            <div className="w-14 h-14 mx-auto mb-3 rounded-full bg-red-100 dark:bg-red-900/30 flex items-center justify-center">
              <AlertCircle className="h-7 w-7 text-red-500" />
            </div>
            <p className="text-sm font-medium text-red-700 dark:text-red-300 mb-1">
              Report Generation Failed
            </p>
            <p className="text-xs text-red-600 dark:text-red-400 mb-4 max-w-[200px] mx-auto">
              {error}
            </p>
            {onRetry && (
              <Button
                variant="secondary"
                size="sm"
                onClick={onRetry}
                leftIcon={<RefreshCw className="h-3.5 w-3.5" />}
              >
                Retry
              </Button>
            )}
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!report) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <ClipboardList className="h-4 w-4 text-gray-400 dark:text-gray-500" />
            {panelTitle}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8">
            <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gray-100 dark:bg-navy-700 flex items-center justify-center">
              <FileText className="h-8 w-8 text-gray-400" />
            </div>
            <p className="text-sm font-medium text-gray-600 dark:text-gray-300">
              No MedGemma brief generated yet
            </p>
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1.5 max-w-[220px] mx-auto">
              Generate a model-authored decision-support brief grounded in this
              slide analysis.
            </p>
            {onGenerateReport && (
              <Button
                onClick={onGenerateReport}
                variant="primary"
                size="sm"
                className="mt-4"
              >
                <FileCheck className="h-4 w-4 mr-1.5" />
                Generate MedGemma Brief
              </Button>
            )}
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="print:shadow-none print:border-gray-300">
      <CardHeader className="print:bg-white">
        <div>
          <CardTitle className="flex items-center gap-2">
            <ClipboardList className="h-4 w-4 text-clinical-600" />
            {panelTitle}
          </CardTitle>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">{panelSubtitle}</p>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Report Header - Print Only */}
        <div className="hidden print:block mb-6 pb-4 border-b-2 border-gray-200">
          <h1 className="text-xl font-bold text-gray-900">
            MedGemma Clinical Decision Brief
          </h1>
          <p className="text-sm text-gray-600 mt-1">
            Case: {report.caseId} | Task: {report.task}
          </p>
          <p className="text-xs text-gray-500 mt-0.5">
            Generated: {new Date(report.generatedAt).toLocaleString()}
          </p>
        </div>

        <MedGemmaSummarySection
          summary={report.summary}
          onCopy={handleCopy}
          copied={copied}
        />

        {/* CLINICAL DECISION SUPPORT - Most prominent section */}
        {report.decisionSupport && (
          <DecisionSupportSection 
            decisionSupport={report.decisionSupport}
          />
        )}

        {report.suggestedNextSteps.length > 0 && (
          <div className="border-2 rounded-lg overflow-hidden border-green-200 bg-green-50 ring-1 ring-green-200">
            <div className="w-full flex items-center gap-2 p-3 bg-blue-50 border-b border-blue-200">
              <Stethoscope className="h-4 w-4 text-blue-700" />
              <span className="text-sm font-semibold text-blue-800">
                MedGemma Suggested Follow-up
              </span>
            </div>
            <div className="p-4 bg-green-50">
              <ol className="space-y-2">
              {report.suggestedNextSteps.map((step, index) => (
                <li
                  key={index}
                  className="flex items-start gap-3 text-sm text-green-800"
                >
                  <span className="flex items-center justify-center w-6 h-6 rounded-full bg-green-200 text-green-700 text-xs font-bold shrink-0">
                    {index + 1}
                  </span>
                  <span className="leading-relaxed">{step}</span>
                </li>
              ))}
            </ol>
            </div>
          </div>
        )}

        {/* Evidence Section with Tissue Types */}
        <ReportSection
          title="MedGemma Evidence Interpretation"
          icon={<Lightbulb className="h-4 w-4" />}
          isExpanded={expandedSections.has("evidence")}
          onToggle={() => toggleSection("evidence")}
          badge={`${report.evidence.length} regions`}
        >
          <div className="space-y-3">
            {report.evidence.map((item, index) => {
              const tissueType = inferTissueType(item.morphologyDescription);
              const tissueInfo = TISSUE_TYPE_LABELS[tissueType];
              const evidenceCoords: PatchCoordinates = {
                x: item.coordsLevel0[0],
                y: item.coordsLevel0[1],
                level: 0,
                width: 224,
                height: 224,
              };
              const isClickable = Boolean(onEvidenceClick);
              return (
                <div
                  key={item.patchId}
                  className={cn(
                    "p-3 bg-gray-50 dark:bg-navy-900 rounded-lg border border-gray-100 dark:border-navy-700 transition-colors",
                    isClickable && "cursor-pointer hover:border-clinical-300 dark:hover:border-clinical-500 hover:bg-clinical-50/60 dark:hover:bg-clinical-900/30 focus-within:ring-2 focus-within:ring-clinical-300 dark:focus-within:ring-clinical-700"
                  )}
                  role={isClickable ? "button" : undefined}
                  tabIndex={isClickable ? 0 : undefined}
                  onClick={isClickable ? () => onEvidenceClick?.(evidenceCoords) : undefined}
                  onKeyDown={
                    isClickable
                      ? (event) => {
                          if (event.key === "Enter" || event.key === " ") {
                            event.preventDefault();
                            onEvidenceClick?.(evidenceCoords);
                          }
                        }
                      : undefined
                  }
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-semibold text-gray-600 dark:text-gray-300">
                        Evidence #{index + 1}
                      </span>
                      {/* Tissue Type Badge */}
                      <span className={cn(
                        "inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-white dark:bg-navy-800 border border-gray-200 dark:border-navy-600",
                        tissueInfo.color
                      )}>
                        <Circle className="h-2 w-2 fill-current" />
                        {tissueInfo.label}
                      </span>
                    </div>
                    <span className="text-xs font-mono text-gray-400 dark:text-gray-500">
                      ({item.coordsLevel0[0].toLocaleString()},{" "}
                      {item.coordsLevel0[1].toLocaleString()})
                    </span>
                  </div>
                  <p className="text-sm text-gray-700 dark:text-gray-300 mb-2 leading-relaxed">
                    {item.morphologyDescription}
                  </p>
                  <div className="flex items-start gap-2 p-2 bg-clinical-50 dark:bg-clinical-900/30 rounded border border-clinical-100 dark:border-clinical-800">
                    <ArrowRight className="h-3.5 w-3.5 text-clinical-600 mt-0.5 shrink-0" />
                    <p className="text-xs text-clinical-700 dark:text-clinical-300 italic leading-relaxed">
                      {item.whyThisPatchMatters}
                    </p>
                  </div>
                  {isClickable && (
                    <p className="text-2xs text-clinical-700 dark:text-clinical-300 mt-2 font-medium">
                      Click to focus this region in the WSI viewer
                    </p>
                  )}
                </div>
              );
            })}
          </div>
        </ReportSection>

        {/* Similar Cases Section */}
        {report.similarExamples.length > 0 && (
          <ReportSection
            title="Reference Cohort Comparison"
            icon={<Search className="h-4 w-4" />}
            isExpanded={expandedSections.has("similar")}
            onToggle={() => toggleSection("similar")}
            badge={`${report.similarExamples.length} cases`}
          >
            <div className="overflow-x-auto">
              <table className="clinical-table">
                <thead>
                  <tr>
                    <th>Case ID</th>
                    <th>Outcome</th>
                    <th>Distance</th>
                  </tr>
                </thead>
                <tbody>
                  {report.similarExamples.map((example) => (
                    <tr key={example.exampleId}>
                      <td className="font-mono text-xs">
                        {example.exampleId.slice(0, 16)}
                      </td>
                      <td>
                        {example.label && (
                          <Badge
                            variant={
                              isPositiveLabel(example.label)
                                ? "success"
                                : "danger"
                            }
                            size="sm"
                          >
                            {example.label}
                          </Badge>
                        )}
                      </td>
                      <td className="font-mono text-xs text-gray-600 dark:text-gray-300">
                        {example.distance.toFixed(4)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </ReportSection>
        )}

        {/* Limitations Section */}
        {report.limitations.length > 0 && (
          <ReportSection
            title="Limitations"
            icon={<Info className="h-4 w-4 text-gray-500 dark:text-gray-400" />}
            isExpanded={expandedSections.has("limitations")}
            onToggle={() => toggleSection("limitations")}
            variant="default"
          >
            <ul className="space-y-2">
              {report.limitations.map((limitation, index) => (
                <li
                  key={index}
                  className="flex items-start gap-2 text-sm text-gray-700 dark:text-gray-300"
                >
                  <Circle className="h-2 w-2 mt-1.5 shrink-0 fill-current text-gray-400 dark:text-gray-500" />
                  <span className="leading-relaxed">{limitation}</span>
                </li>
              ))}
            </ul>
          </ReportSection>
        )}
      </CardContent>

      {/* Export Actions */}
      <CardFooter className="flex gap-2 no-print">
        <Button
          variant="secondary"
          size="sm"
          onClick={handlePrint}
          leftIcon={<Printer className="h-3.5 w-3.5" />}
        >
          Print
        </Button>
        {onExportPdf && (
          <Button
            variant="primary"
            size="sm"
            onClick={onExportPdf}
            leftIcon={<Download className="h-3.5 w-3.5" />}
          >
            Export PDF
          </Button>
        )}
        {onExportJson && (
          <Button
            variant="ghost"
            size="sm"
            onClick={onExportJson}
            leftIcon={<Download className="h-3.5 w-3.5" />}
          >
            Export JSON
          </Button>
        )}
      </CardFooter>
    </Card>
  );
}

// Reusable Section Component
interface ReportSectionProps {
  title: string;
  icon: React.ReactNode;
  isExpanded: boolean;
  onToggle: () => void;
  children: React.ReactNode;
  badge?: string;
  variant?: "default" | "warning" | "info";
  priority?: "high" | "normal";
}

function ReportSection({
  title,
  icon,
  isExpanded,
  onToggle,
  children,
  badge,
  variant = "default",
  priority = "normal",
}: ReportSectionProps) {
  const variantStyles = {
    default: "bg-white dark:bg-navy-800 border-gray-200 dark:border-navy-600",
    warning: "bg-amber-50 dark:bg-amber-900/20 border-amber-200 dark:border-amber-800",
    info: "bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800",
  };

  const headerStyles = {
    default: "hover:bg-gray-50 dark:hover:bg-navy-700/60",
    warning: "hover:bg-amber-100/50 dark:hover:bg-amber-900/30",
    info: "hover:bg-blue-100/50 dark:hover:bg-blue-900/30",
  };

  return (
    <div
      className={cn(
        "report-section border rounded-lg overflow-hidden",
        variantStyles[variant],
        priority === "high" && "ring-1 ring-clinical-200 dark:ring-clinical-700"
      )}
    >
      <button
        onClick={onToggle}
        className={cn(
          "report-section-header w-full flex items-center justify-between p-3 transition-colors",
          headerStyles[variant]
        )}
      >
        <div className="flex items-center gap-2">
          {icon}
          <span className="text-sm font-semibold text-gray-800 dark:text-gray-200">{title}</span>
          {badge && (
            <Badge variant="default" size="sm" className="font-mono">
              {badge}
            </Badge>
          )}
        </div>
        {isExpanded ? (
          <ChevronUp className="h-4 w-4 text-gray-400 dark:text-gray-500" />
        ) : (
          <ChevronDown className="h-4 w-4 text-gray-400 dark:text-gray-500" />
        )}
      </button>
      {isExpanded && (
        <div
          className={cn(
            "report-section-content p-4 border-t animate-fade-in",
            variant === "warning"
              ? "border-amber-200"
              : variant === "info"
              ? "border-blue-200"
              : "border-gray-100 dark:border-navy-700"
          )}
        >
          {children}
        </div>
      )}
    </div>
  );
}

// Clinical Decision Support Section Component
interface DecisionSupportSectionProps {
  decisionSupport: DecisionSupport;
}

interface MedGemmaSummarySectionProps {
  summary: string;
  onCopy: () => void;
  copied: boolean;
}

function MedGemmaSummarySection({
  summary,
  onCopy,
  copied,
}: MedGemmaSummarySectionProps) {
  return (
    <div className="border-2 rounded-lg overflow-hidden border-blue-200 dark:border-blue-800 ring-2 ring-offset-1 dark:ring-offset-navy-800 ring-blue-100 dark:ring-blue-900/40">
      <div className="w-full flex items-center justify-between p-4 bg-blue-50 dark:bg-blue-900/30">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full flex items-center justify-center bg-blue-200 dark:bg-blue-800/60">
            <FileText className="h-5 w-5 text-blue-700 dark:text-blue-300" />
          </div>
          <div className="text-left">
            <span className="text-sm font-bold uppercase tracking-wide text-blue-800 dark:text-blue-200">
              Summary by MedGemma
            </span>
          </div>
        </div>
      </div>

      <div className="px-4 py-3 border-t border-green-200 dark:border-green-800 bg-green-50 dark:bg-green-900/20 space-y-3">
        <div className="flex items-start gap-3">
          <Activity className="h-5 w-5 mt-0.5 shrink-0 text-green-700 dark:text-green-300" />
          <p className="text-sm font-medium leading-relaxed text-green-800 dark:text-green-200 whitespace-pre-wrap">
            {summary}
          </p>
        </div>
        <div className="flex justify-end no-print">
          <Button
            variant="ghost"
            size="sm"
            onClick={onCopy}
            className="text-xs"
          >
            {copied ? (
              <>
                <Check className="h-3 w-3 mr-1" />
                Copied
              </>
            ) : (
              <>
                <Copy className="h-3 w-3 mr-1" />
                Copy Summary
              </>
            )}
          </Button>
        </div>
      </div>
    </div>
  );
}

function DecisionSupportSection({
  decisionSupport,
}: DecisionSupportSectionProps) {
  const riskConfig = RISK_LEVEL_CONFIG[decisionSupport.risk_level];
  const isLowConfidence = decisionSupport.confidence_level === "low" || decisionSupport.risk_level === "inconclusive";
  
  return (
    <div className="border-2 rounded-lg overflow-hidden border-blue-200 dark:border-blue-800 bg-blue-50 dark:bg-blue-900/20 ring-1 ring-blue-200 dark:ring-blue-900/50">
      <div className="w-full flex items-center gap-3 p-4 bg-blue-50 dark:bg-blue-900/30 border-b border-blue-200 dark:border-blue-800">
        <div className="w-10 h-10 rounded-full flex items-center justify-center bg-blue-200 dark:bg-blue-800/60">
          <Target className="h-5 w-5 text-blue-700 dark:text-blue-300" />
        </div>
        <div className="text-left">
          <span className="text-sm font-bold uppercase tracking-wide text-blue-800 dark:text-blue-200">
            MedGemma Decision Support
          </span>
          <p className="text-xs text-blue-700 dark:text-blue-300 mt-0.5">
            Risk level: {riskConfig.label}
          </p>
        </div>
      </div>

      <div className="px-4 py-3 border-b border-green-200 dark:border-green-800 bg-green-50 dark:bg-green-900/20">
        <div className="flex items-start gap-3">
          <Activity className="h-5 w-5 mt-0.5 shrink-0 text-green-700 dark:text-green-300" />
          <div>
            <p className="text-sm font-medium leading-relaxed text-green-800 dark:text-green-200">
              {decisionSupport.primary_recommendation}
            </p>
          </div>
        </div>
      </div>

      <div className="p-4 bg-blue-50 dark:bg-blue-900/10 space-y-4">
        {isLowConfidence && (
          <div className="flex items-start gap-3 p-3 bg-blue-100 dark:bg-blue-900/30 border border-blue-200 dark:border-blue-800 rounded-lg">
            <AlertTriangle className="h-5 w-5 text-blue-700 dark:text-blue-300 mt-0.5 shrink-0" />
            <div>
              <p className="text-sm font-semibold text-blue-900 dark:text-blue-100">
                Interpretation Caution Required
              </p>
              <p className="text-xs text-blue-800 dark:text-blue-200 mt-1 leading-relaxed">
                {decisionSupport.uncertainty_statement}
              </p>
            </div>
          </div>
        )}

        {decisionSupport.supporting_rationale && decisionSupport.supporting_rationale.length > 0 && (
          <div className="p-3 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg">
            <h4 className="text-xs font-semibold text-green-700 dark:text-green-300 uppercase tracking-wide mb-2 flex items-center gap-1.5">
              <TrendingUp className="h-3.5 w-3.5" />
              Supporting Evidence
            </h4>
            <ul className="space-y-1.5">
              {decisionSupport.supporting_rationale.map((reason, idx) => (
                <li key={idx} className="flex items-start gap-2 text-sm text-green-800 dark:text-green-200">
                  <Check className="h-4 w-4 text-green-600 dark:text-green-300 mt-0.5 shrink-0" />
                  <span>{reason}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {decisionSupport.alternative_considerations && decisionSupport.alternative_considerations.length > 0 && (
          <div className="p-3 bg-blue-100 dark:bg-blue-900/30 border border-blue-200 dark:border-blue-800 rounded-lg">
            <h4 className="text-xs font-semibold text-blue-700 dark:text-blue-300 uppercase tracking-wide mb-2 flex items-center gap-1.5">
              <Lightbulb className="h-3.5 w-3.5" />
              Alternative Considerations
            </h4>
            <ul className="space-y-1.5">
              {decisionSupport.alternative_considerations.map((alt, idx) => (
                <li key={idx} className="flex items-start gap-2 text-sm text-blue-800 dark:text-blue-200">
                  <ArrowRight className="h-4 w-4 text-blue-500 dark:text-blue-300 mt-0.5 shrink-0" />
                  <span>{alt}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {decisionSupport.guideline_references && decisionSupport.guideline_references.length > 0 && (
          <div className="border-t border-blue-200 dark:border-blue-800 pt-4">
            <h4 className="text-xs font-semibold text-blue-700 dark:text-blue-300 uppercase tracking-wide mb-2 flex items-center gap-1.5">
              <BookOpen className="h-3.5 w-3.5" />
              Clinical Guideline References
            </h4>
            <div className="space-y-2">
              {decisionSupport.guideline_references.map((ref, idx) => (
                <div key={idx} className="p-2.5 bg-blue-100 dark:bg-blue-900/30 rounded border border-blue-200 dark:border-blue-800">
                  <div className="flex items-start justify-between gap-2">
                    <div>
                      <p className="text-xs font-medium text-blue-800 dark:text-blue-200">
                        {ref.source} - {ref.section}
                      </p>
                      <p className="text-xs text-blue-700 dark:text-blue-300 mt-1 leading-relaxed">
                        {ref.recommendation}
                      </p>
                    </div>
                    {ref.url && (
                      <a
                        href={ref.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="shrink-0 p-1 text-blue-600 dark:text-blue-300 hover:text-blue-800 dark:hover:text-blue-200 hover:bg-blue-200 dark:hover:bg-blue-800/60 rounded"
                        title="Open guideline"
                      >
                        <ExternalLink className="h-3.5 w-3.5" />
                      </a>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

      </div>
    </div>
  );
}
