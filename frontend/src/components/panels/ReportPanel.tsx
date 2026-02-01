"use client";

import React, { useState } from "react";
import { Card, CardHeader, CardTitle, CardContent, CardFooter } from "@/components/ui/Card";
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

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <FileText className="h-4 w-4" />
            Clinical Report
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4 animate-pulse">
            <div className="h-4 bg-gray-200 rounded w-3/4" />
            <div className="h-4 bg-gray-200 rounded w-full" />
            <div className="h-4 bg-gray-200 rounded w-5/6" />
            <div className="h-4 bg-gray-200 rounded w-2/3" />
          </div>
          <p className="text-sm text-gray-500 mt-4 text-center">
            Generating report with MedGemma...
          </p>
        </CardContent>
      </Card>
    );
  }

  if (!report) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <FileText className="h-4 w-4" />
            Clinical Report
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8">
            <FileText className="h-12 w-12 mx-auto mb-3 text-gray-300" />
            <p className="text-sm text-gray-600 mb-1">No report generated yet</p>
            <p className="text-xs text-gray-500 mb-4">
              Generate a structured clinical report based on the analysis results.
            </p>
            {onGenerateReport && (
              <Button onClick={onGenerateReport} variant="primary" size="sm">
                Generate Report
              </Button>
            )}
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <FileText className="h-4 w-4" />
            Clinical Report
          </CardTitle>
          <Badge variant="info" size="sm">
            {formatDate(report.generatedAt)}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Summary Section */}
        <ReportSection
          title="Summary"
          icon={<FileText className="h-4 w-4" />}
          isExpanded={expandedSections.has("summary")}
          onToggle={() => toggleSection("summary")}
        >
          <div className="prose prose-sm max-w-none">
            <p className="text-gray-700 leading-relaxed whitespace-pre-wrap">
              {report.summary}
            </p>
          </div>
          <div className="mt-3 flex justify-end">
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
                  <span className="text-xs font-medium text-gray-500">
                    Evidence #{index + 1}
                  </span>
                  <span className="text-xs font-mono text-gray-400">
                    ({item.coordsLevel0[0]}, {item.coordsLevel0[1]})
                  </span>
                </div>
                <p className="text-sm text-gray-700 mb-1">
                  {item.morphologyDescription}
                </p>
                <p className="text-xs text-clinical-600 italic">
                  {item.whyThisPatchMatters}
                </p>
              </div>
            ))}
          </div>
        </ReportSection>

        {/* Similar Cases Section */}
        {report.similarExamples.length > 0 && (
          <ReportSection
            title="Similar Cases in Cohort"
            icon={<Search className="h-4 w-4" />}
            isExpanded={expandedSections.has("similar")}
            onToggle={() => toggleSection("similar")}
            badge={`${report.similarExamples.length} cases`}
          >
            <div className="space-y-2">
              {report.similarExamples.map((example) => (
                <div
                  key={example.exampleId}
                  className="flex items-center justify-between py-2 border-b border-gray-100 last:border-0"
                >
                  <div>
                    <span className="text-sm font-medium text-gray-700">
                      {example.exampleId.slice(0, 12)}
                    </span>
                    {example.label && (
                      <Badge variant="default" size="sm" className="ml-2">
                        {example.label}
                      </Badge>
                    )}
                  </div>
                  <span className="text-xs font-mono text-gray-500">
                    d={example.distance.toFixed(3)}
                  </span>
                </div>
              ))}
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
                <span className="text-amber-500 mt-0.5">-</span>
                <span>{limitation}</span>
              </li>
            ))}
          </ul>
        </ReportSection>

        {/* Suggested Next Steps Section */}
        <ReportSection
          title="Suggested Next Steps"
          icon={<Lightbulb className="h-4 w-4 text-blue-500" />}
          isExpanded={expandedSections.has("nextSteps")}
          onToggle={() => toggleSection("nextSteps")}
          variant="info"
        >
          <ul className="space-y-2">
            {report.suggestedNextSteps.map((step, index) => (
              <li
                key={index}
                className="flex items-start gap-2 text-sm text-blue-800"
              >
                <span className="text-blue-500 font-medium">{index + 1}.</span>
                <span>{step}</span>
              </li>
            ))}
          </ul>
        </ReportSection>

        {/* Safety Statement */}
        <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
          <div className="flex items-start gap-3">
            <Shield className="h-5 w-5 text-red-500 shrink-0 mt-0.5" />
            <div>
              <h4 className="text-sm font-semibold text-red-800 mb-1">
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
      <CardFooter className="flex gap-2">
        {onExportPdf && (
          <Button
            variant="primary"
            size="sm"
            onClick={onExportPdf}
            leftIcon={<Printer className="h-3.5 w-3.5" />}
          >
            Export PDF
          </Button>
        )}
        {onExportJson && (
          <Button
            variant="secondary"
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
}

function ReportSection({
  title,
  icon,
  isExpanded,
  onToggle,
  children,
  badge,
  variant = "default",
}: ReportSectionProps) {
  const variantStyles = {
    default: "bg-white border-gray-200",
    warning: "bg-amber-50 border-amber-200",
    info: "bg-blue-50 border-blue-200",
  };

  return (
    <div className={cn("border rounded-lg overflow-hidden", variantStyles[variant])}>
      <button
        onClick={onToggle}
        className="w-full flex items-center justify-between p-3 hover:bg-gray-50/50 transition-colors"
      >
        <div className="flex items-center gap-2">
          {icon}
          <span className="text-sm font-medium text-gray-800">{title}</span>
          {badge && (
            <Badge variant="default" size="sm">
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
        <div className="px-3 pb-3 border-t border-gray-100/50">{children}</div>
      )}
    </div>
  );
}

// Import Search icon for the Similar Cases section
import { Search } from "lucide-react";
