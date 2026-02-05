"use client";

import React, { useState } from "react";
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
  CardFooter,
} from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { cn, formatDate } from "@/lib/utils";
import {
  FileText,
  Download,
  Copy,
  Check,
  AlertTriangle,
  Lightbulb,
  Shield,
  ChevronDown,
  ChevronUp,
  Printer,
  Search,
  ClipboardList,
  AlertCircle,
  ArrowRight,
  FileCheck,
  RefreshCw,
  FlaskConical,
  Stethoscope,
  Circle,
  Activity,
  BookOpen,
  Target,
  TrendingUp,
  ExternalLink,
  Info,
} from "lucide-react";
import type { StructuredReport, DecisionSupport, RiskLevel, ConfidenceLevel } from "@/types";

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
  tumor: { label: "Tumor", color: "text-red-600" },
  stroma: { label: "Stroma", color: "text-blue-600" },
  necrosis: { label: "Necrosis", color: "text-gray-600" },
  inflammatory: { label: "Inflammatory", color: "text-purple-600" },
  normal: { label: "Normal", color: "text-green-600" },
  unknown: { label: "Unclassified", color: "text-gray-400" },
};

// Risk level styling configuration
const RISK_LEVEL_CONFIG: Record<RiskLevel, { label: string; color: string; bgColor: string; borderColor: string; icon: string }> = {
  high_confidence: { 
    label: "High Confidence", 
    color: "text-emerald-700", 
    bgColor: "bg-emerald-50", 
    borderColor: "border-emerald-300",
    icon: "check-circle"
  },
  moderate_confidence: { 
    label: "Moderate Confidence", 
    color: "text-amber-700", 
    bgColor: "bg-amber-50", 
    borderColor: "border-amber-300",
    icon: "alert-circle"
  },
  low_confidence: { 
    label: "Low Confidence", 
    color: "text-orange-700", 
    bgColor: "bg-orange-50", 
    borderColor: "border-orange-300",
    icon: "alert-triangle"
  },
  inconclusive: { 
    label: "Inconclusive", 
    color: "text-red-700", 
    bgColor: "bg-red-50", 
    borderColor: "border-red-300",
    icon: "x-circle"
  },
};

// Confidence level badge styling
const CONFIDENCE_BADGE_CONFIG: Record<ConfidenceLevel, { variant: "success" | "warning" | "danger"; label: string }> = {
  high: { variant: "success", label: "High" },
  moderate: { variant: "warning", label: "Moderate" },
  low: { variant: "danger", label: "Low" },
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
}: ReportPanelProps) {
  const [expandedSections, setExpandedSections] = useState<Set<string>>(
    new Set(["decisionSupport", "summary", "evidence"])
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
            Clinical Report
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col items-center py-8">
            <div className="relative w-16 h-16 mb-4">
              <div className="absolute inset-0 rounded-full border-4 border-gray-100" />
              <div className="absolute inset-0 rounded-full border-4 border-clinical-500 border-t-transparent animate-spin" />
              <div className="absolute inset-2 rounded-full bg-gray-50 flex items-center justify-center">
                <FileText className="h-6 w-6 text-clinical-600" />
              </div>
            </div>
            <p className="text-sm font-medium text-gray-700">
              {progressMessage}
            </p>
            {/* Progress bar */}
            <div className="w-full max-w-xs mt-4">
              <div className="flex items-center justify-between text-xs text-gray-500 mb-1">
                <span>Progress</span>
                <span>{Math.round(progress)}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-clinical-500 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
            <p className="text-xs text-gray-400 mt-3">
              {progress < 30 ? "Loading slide data..." : 
               progress < 50 ? "Running analysis..." :
               progress < 80 ? "Generating report with MedGemma..." :
               "Finalizing report..."}
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
            Clinical Report
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-6">
            <div className="w-14 h-14 mx-auto mb-3 rounded-full bg-red-100 flex items-center justify-center">
              <AlertCircle className="h-7 w-7 text-red-500" />
            </div>
            <p className="text-sm font-medium text-red-700 mb-1">
              Report Generation Failed
            </p>
            <p className="text-xs text-red-600 mb-4 max-w-[200px] mx-auto">
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
            <ClipboardList className="h-4 w-4 text-gray-400" />
            Clinical Report
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8">
            <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gray-100 flex items-center justify-center">
              <FileText className="h-8 w-8 text-gray-400" />
            </div>
            <p className="text-sm font-medium text-gray-600">
              No report generated yet
            </p>
            <p className="text-xs text-gray-500 mt-1.5 max-w-[220px] mx-auto">
              Generate a structured clinical report summarizing the analysis
              findings.
            </p>
            {onGenerateReport && (
              <Button
                onClick={onGenerateReport}
                variant="primary"
                size="sm"
                className="mt-4"
              >
                <FileCheck className="h-4 w-4 mr-1.5" />
                Generate Report
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
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <ClipboardList className="h-4 w-4 text-clinical-600" />
            Clinical Report
          </CardTitle>
          <Badge variant="info" size="sm" className="font-mono">
            {formatDate(report.generatedAt)}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* UNCALIBRATED WARNING BANNER */}
        <div className="flex items-center gap-3 px-4 py-3 bg-amber-100 border-2 border-amber-400 rounded-lg">
          <FlaskConical className="h-5 w-5 text-amber-700 shrink-0" />
          <div>
            <span className="text-sm font-bold text-amber-800 uppercase tracking-wide">
              Uncalibrated Model - Research Use Only
            </span>
            <p className="text-xs text-amber-700 mt-0.5">
              Probabilities are raw model outputs. Not validated for clinical decision-making.
            </p>
          </div>
        </div>

        {/* Report Header - Print Only */}
        <div className="hidden print:block mb-6 pb-4 border-b-2 border-gray-200">
          <h1 className="text-xl font-bold text-gray-900">
            Pathology Analysis Report
          </h1>
          <p className="text-sm text-gray-600 mt-1">
            Case: {report.caseId} | Task: {report.task}
          </p>
          <p className="text-xs text-gray-500 mt-0.5">
            Generated: {new Date(report.generatedAt).toLocaleString()}
          </p>
        </div>

        {/* CLINICAL DECISION SUPPORT - Most prominent section */}
        {report.decisionSupport && (
          <DecisionSupportSection 
            decisionSupport={report.decisionSupport}
            isExpanded={expandedSections.has("decisionSupport")}
            onToggle={() => toggleSection("decisionSupport")}
          />
        )}

        {/* ACTIONABLE NEXT STEPS - Moved to top for visibility */}
        <ReportSection
          title="Suggested Next Steps"
          icon={<Stethoscope className="h-4 w-4 text-clinical-600" />}
          isExpanded={expandedSections.has("nextSteps")}
          onToggle={() => toggleSection("nextSteps")}
          variant="info"
          priority="high"
        >
          <div className="space-y-3">
            <p className="text-xs text-blue-700 mb-3">
              Based on the analysis findings, consider the following clinical actions:
            </p>
            <ol className="space-y-2">
              {report.suggestedNextSteps.map((step, index) => (
                <li
                  key={index}
                  className="flex items-start gap-3 text-sm text-blue-800"
                >
                  <span className="flex items-center justify-center w-6 h-6 rounded-full bg-blue-200 text-blue-700 text-xs font-bold shrink-0">
                    {index + 1}
                  </span>
                  <span className="leading-relaxed">{step}</span>
                </li>
              ))}
            </ol>
            {/* Additional context based on prediction */}
            <div className="mt-4 pt-3 border-t border-blue-200">
              <p className="text-xs text-blue-700 leading-relaxed">
                <strong>Interpretation guidance:</strong> These recommendations are generated 
                based on the morphological patterns identified by the model. They should be 
                evaluated in the context of the patient&apos;s complete clinical picture, imaging 
                findings, and other laboratory results.
              </p>
            </div>
          </div>
        </ReportSection>

        {/* Summary Section */}
        <ReportSection
          title="Clinical Summary"
          icon={<FileText className="h-4 w-4" />}
          isExpanded={expandedSections.has("summary")}
          onToggle={() => toggleSection("summary")}
        >
          <div className="prose prose-sm max-w-none">
            <p className="text-gray-700 leading-relaxed whitespace-pre-wrap">
              {report.summary}
            </p>
          </div>
          <div className="mt-3 flex justify-end no-print">
            <Button
              variant="ghost"
              size="sm"
              onClick={handleCopy}
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
        </ReportSection>

        {/* Evidence Section with Tissue Types */}
        <ReportSection
          title="Supporting Evidence"
          icon={<Lightbulb className="h-4 w-4" />}
          isExpanded={expandedSections.has("evidence")}
          onToggle={() => toggleSection("evidence")}
          badge={`${report.evidence.length} patches`}
        >
          <div className="space-y-3">
            {report.evidence.map((item, index) => {
              const tissueType = inferTissueType(item.morphologyDescription);
              const tissueInfo = TISSUE_TYPE_LABELS[tissueType];
              return (
                <div
                  key={item.patchId}
                  className="p-3 bg-gray-50 rounded-lg border border-gray-100"
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-semibold text-gray-600">
                        Evidence #{index + 1}
                      </span>
                      {/* Tissue Type Badge */}
                      <span className={cn(
                        "inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-white border",
                        tissueInfo.color
                      )}>
                        <Circle className="h-2 w-2 fill-current" />
                        {tissueInfo.label}
                      </span>
                    </div>
                    <span className="text-xs font-mono text-gray-400">
                      ({item.coordsLevel0[0].toLocaleString()},{" "}
                      {item.coordsLevel0[1].toLocaleString()})
                    </span>
                  </div>
                  <p className="text-sm text-gray-700 mb-2 leading-relaxed">
                    {item.morphologyDescription}
                  </p>
                  <div className="flex items-start gap-2 p-2 bg-clinical-50 rounded border border-clinical-100">
                    <ArrowRight className="h-3.5 w-3.5 text-clinical-600 mt-0.5 shrink-0" />
                    <p className="text-xs text-clinical-700 italic leading-relaxed">
                      {item.whyThisPatchMatters}
                    </p>
                  </div>
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
                              example.label.toLowerCase().includes("responder")
                                ? "success"
                                : example.label.toLowerCase().includes("non")
                                ? "danger"
                                : "default"
                            }
                            size="sm"
                          >
                            {example.label}
                          </Badge>
                        )}
                      </td>
                      <td className="font-mono text-xs text-gray-600">
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
        <ReportSection
          title="Limitations"
          icon={<AlertTriangle className="h-4 w-4 text-amber-500" />}
          isExpanded={expandedSections.has("limitations")}
          onToggle={() => toggleSection("limitations")}
          variant="warning"
        >
          <ul className="space-y-2">
            {report.limitations.map((limitation, index) => (
              <li
                key={index}
                className="flex items-start gap-2 text-sm text-amber-800"
              >
                <AlertCircle className="h-4 w-4 text-amber-500 mt-0.5 shrink-0" />
                <span className="leading-relaxed">{limitation}</span>
              </li>
            ))}
          </ul>
        </ReportSection>

        {/* Safety Statement */}
        <div className="p-4 bg-red-50 border-2 border-red-200 rounded-lg">
          <div className="flex items-start gap-3">
            <div className="w-8 h-8 rounded-full bg-red-100 flex items-center justify-center shrink-0">
              <Shield className="h-4 w-4 text-red-600" />
            </div>
            <div>
              <h4 className="text-sm font-bold text-red-800 mb-1">
                Important Safety Notice
              </h4>
              <p className="text-sm text-red-700 leading-relaxed">
                {report.safetyStatement}
              </p>
            </div>
          </div>
        </div>
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
    default: "bg-white border-gray-200",
    warning: "bg-amber-50 border-amber-200",
    info: "bg-blue-50 border-blue-200",
  };

  const headerStyles = {
    default: "hover:bg-gray-50",
    warning: "hover:bg-amber-100/50",
    info: "hover:bg-blue-100/50",
  };

  return (
    <div
      className={cn(
        "report-section border rounded-lg overflow-hidden",
        variantStyles[variant],
        priority === "high" && "ring-1 ring-clinical-200"
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
          <span className="text-sm font-semibold text-gray-800">{title}</span>
          {badge && (
            <Badge variant="default" size="sm" className="font-mono">
              {badge}
            </Badge>
          )}
        </div>
        {isExpanded ? (
          <ChevronUp className="h-4 w-4 text-gray-400" />
        ) : (
          <ChevronDown className="h-4 w-4 text-gray-400" />
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
              : "border-gray-100"
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
  isExpanded: boolean;
  onToggle: () => void;
}

function DecisionSupportSection({
  decisionSupport,
  isExpanded,
  onToggle,
}: DecisionSupportSectionProps) {
  const riskConfig = RISK_LEVEL_CONFIG[decisionSupport.risk_level];
  const confidenceBadge = CONFIDENCE_BADGE_CONFIG[decisionSupport.confidence_level];
  const isLowConfidence = decisionSupport.confidence_level === "low" || decisionSupport.risk_level === "inconclusive";
  
  return (
    <div className={cn(
      "border-2 rounded-lg overflow-hidden",
      riskConfig.borderColor,
      "ring-2 ring-offset-1",
      decisionSupport.risk_level === "high_confidence" ? "ring-emerald-200" :
      decisionSupport.risk_level === "moderate_confidence" ? "ring-amber-200" :
      decisionSupport.risk_level === "low_confidence" ? "ring-orange-200" : "ring-red-200"
    )}>
      {/* Header with Risk Level Badge */}
      <button
        onClick={onToggle}
        className={cn(
          "w-full flex items-center justify-between p-4 transition-colors",
          riskConfig.bgColor,
          "hover:opacity-90"
        )}
      >
        <div className="flex items-center gap-3">
          <div className={cn(
            "w-10 h-10 rounded-full flex items-center justify-center",
            decisionSupport.risk_level === "high_confidence" ? "bg-emerald-200" :
            decisionSupport.risk_level === "moderate_confidence" ? "bg-amber-200" :
            decisionSupport.risk_level === "low_confidence" ? "bg-orange-200" : "bg-red-200"
          )}>
            <Target className={cn("h-5 w-5", riskConfig.color)} />
          </div>
          <div className="text-left">
            <div className="flex items-center gap-2">
              <span className={cn("text-sm font-bold uppercase tracking-wide", riskConfig.color)}>
                Clinical Decision Support
              </span>
              <Badge variant={confidenceBadge.variant} size="sm">
                {riskConfig.label}
              </Badge>
            </div>
            <p className="text-xs text-gray-600 mt-0.5">
              Confidence: {(decisionSupport.confidence_score * 100).toFixed(0)}%
            </p>
          </div>
        </div>
        {isExpanded ? (
          <ChevronUp className="h-5 w-5 text-gray-500" />
        ) : (
          <ChevronDown className="h-5 w-5 text-gray-500" />
        )}
      </button>

      {/* Always show primary recommendation summary */}
      <div className={cn("px-4 py-3 border-t", riskConfig.borderColor, riskConfig.bgColor)}>
        <div className="flex items-start gap-3">
          <Activity className={cn("h-5 w-5 mt-0.5 shrink-0", riskConfig.color)} />
          <div>
            <p className={cn("text-sm font-medium leading-relaxed", riskConfig.color)}>
              {decisionSupport.primary_recommendation}
            </p>
          </div>
        </div>
      </div>

      {/* Expanded Content */}
      {isExpanded && (
        <div className="p-4 bg-white border-t border-gray-100 space-y-4 animate-fade-in">
          
          {/* Uncertainty Warning - Prominent when confidence is low */}
          {isLowConfidence && (
            <div className="flex items-start gap-3 p-3 bg-orange-50 border border-orange-200 rounded-lg">
              <AlertTriangle className="h-5 w-5 text-orange-600 mt-0.5 shrink-0" />
              <div>
                <p className="text-sm font-semibold text-orange-800">
                  Interpretation Caution Required
                </p>
                <p className="text-xs text-orange-700 mt-1 leading-relaxed">
                  {decisionSupport.uncertainty_statement}
                </p>
              </div>
            </div>
          )}

          {/* Quality Warnings */}
          {decisionSupport.quality_warnings && decisionSupport.quality_warnings.length > 0 && (
            <div className="p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <Info className="h-4 w-4 text-yellow-700" />
                <span className="text-xs font-semibold text-yellow-800 uppercase">
                  Quality Considerations
                </span>
              </div>
              <ul className="space-y-1">
                {decisionSupport.quality_warnings.map((warning, idx) => (
                  <li key={idx} className="flex items-center gap-2 text-xs text-yellow-700">
                    <Circle className="h-1.5 w-1.5 fill-current" />
                    {warning}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Supporting Rationale */}
          {decisionSupport.supporting_rationale && decisionSupport.supporting_rationale.length > 0 && (
            <div>
              <h4 className="text-xs font-semibold text-gray-600 uppercase tracking-wide mb-2 flex items-center gap-1.5">
                <TrendingUp className="h-3.5 w-3.5" />
                Supporting Evidence
              </h4>
              <ul className="space-y-1.5">
                {decisionSupport.supporting_rationale.map((reason, idx) => (
                  <li key={idx} className="flex items-start gap-2 text-sm text-gray-700">
                    <Check className="h-4 w-4 text-emerald-500 mt-0.5 shrink-0" />
                    <span>{reason}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Interpretation Note */}
          <div className="p-3 bg-gray-50 rounded-lg border border-gray-200">
            <p className="text-xs text-gray-600 leading-relaxed">
              <strong>Interpretation:</strong> {decisionSupport.interpretation_note}
            </p>
          </div>

          {/* Alternative Considerations */}
          {decisionSupport.alternative_considerations && decisionSupport.alternative_considerations.length > 0 && (
            <div>
              <h4 className="text-xs font-semibold text-gray-600 uppercase tracking-wide mb-2 flex items-center gap-1.5">
                <Lightbulb className="h-3.5 w-3.5" />
                Alternative Considerations
              </h4>
              <ul className="space-y-1.5">
                {decisionSupport.alternative_considerations.map((alt, idx) => (
                  <li key={idx} className="flex items-start gap-2 text-sm text-gray-700">
                    <ArrowRight className="h-4 w-4 text-gray-400 mt-0.5 shrink-0" />
                    <span>{alt}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Suggested Workup */}
          {decisionSupport.suggested_workup && decisionSupport.suggested_workup.length > 0 && (
            <div>
              <h4 className="text-xs font-semibold text-gray-600 uppercase tracking-wide mb-2 flex items-center gap-1.5">
                <Stethoscope className="h-3.5 w-3.5" />
                Suggested Additional Workup
              </h4>
              <ol className="space-y-1.5">
                {decisionSupport.suggested_workup.map((step, idx) => (
                  <li key={idx} className="flex items-start gap-2 text-sm text-gray-700">
                    <span className="flex items-center justify-center w-5 h-5 rounded-full bg-clinical-100 text-clinical-700 text-xs font-bold shrink-0">
                      {idx + 1}
                    </span>
                    <span>{step}</span>
                  </li>
                ))}
              </ol>
            </div>
          )}

          {/* NCCN Guideline References */}
          {decisionSupport.guideline_references && decisionSupport.guideline_references.length > 0 && (
            <div className="border-t border-gray-200 pt-4">
              <h4 className="text-xs font-semibold text-gray-600 uppercase tracking-wide mb-2 flex items-center gap-1.5">
                <BookOpen className="h-3.5 w-3.5" />
                Clinical Guideline References
              </h4>
              <div className="space-y-2">
                {decisionSupport.guideline_references.map((ref, idx) => (
                  <div key={idx} className="p-2.5 bg-blue-50 rounded border border-blue-100">
                    <div className="flex items-start justify-between gap-2">
                      <div>
                        <p className="text-xs font-medium text-blue-800">
                          {ref.source} - {ref.section}
                        </p>
                        <p className="text-xs text-blue-700 mt-1 leading-relaxed">
                          {ref.recommendation}
                        </p>
                      </div>
                      {ref.url && (
                        <a
                          href={ref.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="shrink-0 p-1 text-blue-600 hover:text-blue-800 hover:bg-blue-100 rounded"
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

          {/* Caveat Footer */}
          <div className="pt-3 border-t border-gray-200">
            <p className="text-xs text-gray-500 italic leading-relaxed">
              {decisionSupport.caveat}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
