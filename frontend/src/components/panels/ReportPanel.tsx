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
} from "lucide-react";
import type { StructuredReport } from "@/types";

interface ReportPanelProps {
  report: StructuredReport | null;
  isLoading?: boolean;
  onGenerateReport?: () => void;
  onExportPdf?: () => void;
  onExportJson?: () => void;
}

export function ReportPanel({
  report,
  isLoading,
  onGenerateReport,
  onExportPdf,
  onExportJson,
}: ReportPanelProps) {
  const [expandedSections, setExpandedSections] = useState<Set<string>>(
    new Set(["summary", "evidence"])
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
            <ClipboardList className="h-4 w-4 text-clinical-600" />
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
              Generating report...
            </p>
            <p className="text-xs text-gray-500 mt-1">
              Synthesizing findings with MedGemma
            </p>
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

        {/* Summary Section */}
        <ReportSection
          title="Clinical Summary"
          icon={<FileText className="h-4 w-4" />}
          isExpanded={expandedSections.has("summary")}
          onToggle={() => toggleSection("summary")}
          priority="high"
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

        {/* Evidence Section */}
        <ReportSection
          title="Supporting Evidence"
          icon={<Lightbulb className="h-4 w-4" />}
          isExpanded={expandedSections.has("evidence")}
          onToggle={() => toggleSection("evidence")}
          badge={`${report.evidence.length} patches`}
        >
          <div className="space-y-3">
            {report.evidence.map((item, index) => (
              <div
                key={item.patchId}
                className="p-3 bg-gray-50 rounded-lg border border-gray-100"
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-semibold text-gray-600">
                    Evidence #{index + 1}
                  </span>
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
            ))}
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

        {/* Recommendations Section */}
        <ReportSection
          title="Suggested Next Steps"
          icon={<Lightbulb className="h-4 w-4 text-blue-500" />}
          isExpanded={expandedSections.has("nextSteps")}
          onToggle={() => toggleSection("nextSteps")}
          variant="info"
        >
          <ol className="space-y-2">
            {report.suggestedNextSteps.map((step, index) => (
              <li
                key={index}
                className="flex items-start gap-3 text-sm text-blue-800"
              >
                <span className="flex items-center justify-center w-5 h-5 rounded-full bg-blue-100 text-blue-600 text-xs font-bold shrink-0">
                  {index + 1}
                </span>
                <span className="leading-relaxed">{step}</span>
              </li>
            ))}
          </ol>
        </ReportSection>

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
            JSON
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
