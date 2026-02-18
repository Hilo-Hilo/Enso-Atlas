"use client";

import React, { useState, useMemo } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { SkeletonSimilarCases } from "@/components/ui/Skeleton";
import { cn, formatDistance } from "@/lib/utils";
import {
  Search,
  ChevronDown,
  ChevronUp,
  Database,
  ExternalLink,
  GitCompare,
  MapPin,
  ArrowRight,
  BarChart3,
  AlertCircle,
  RefreshCw,
  CheckCircle,
  XCircle,
  HelpCircle,
} from "lucide-react";
import type { SimilarCase } from "@/types";
import { useProject } from "@/contexts/ProjectContext";

// Helper to classify case outcome
function classifyOutcome(label?: string): "responder" | "non-responder" | "unknown" {
  if (!label) return "unknown";
  const lower = label.toLowerCase();
  if (lower.includes("positive") || lower.includes("responder") && !lower.includes("non")) {
    return "responder";
  }
  if (lower.includes("negative") || lower.includes("non")) {
    return "non-responder";
  }
  return "unknown";
}

// Group cases by outcome
interface GroupedCases {
  responders: SimilarCase[];
  nonResponders: SimilarCase[];
  unknown: SimilarCase[];
}

interface SimilarCasesPanelProps {
  cases: SimilarCase[];
  isLoading?: boolean;
  onCaseClick?: (caseId: string) => void;
  error?: string | null;
  onRetry?: () => void;
}

export function SimilarCasesPanel({
  cases,
  isLoading,
  onCaseClick,
  error,
  onRetry,
}: SimilarCasesPanelProps) {
  const [expandedCase, setExpandedCase] = useState<string | null>(null);
  const [showAll, setShowAll] = useState(false);
  const [viewMode, setViewMode] = useState<"grouped" | "list">("grouped");

  // Project-aware labels
  const { currentProject } = useProject();
  const positiveLabel = currentProject.positive_class || currentProject.classes?.[1] || "Positive";
  const negativeLabel = currentProject.classes?.find(c => c !== currentProject.positive_class) || currentProject.classes?.[0] || "Negative";

  // Group cases by outcome
  const groupedCases = useMemo<GroupedCases>(() => {
    return cases.reduce<GroupedCases>(
      (acc, c) => {
        const outcome = classifyOutcome(c.label);
        if (outcome === "responder") acc.responders.push(c);
        else if (outcome === "non-responder") acc.nonResponders.push(c);
        else acc.unknown.push(c);
        return acc;
      },
      { responders: [], nonResponders: [], unknown: [] }
    );
  }, [cases]);

  const visibleCases = showAll ? cases : cases.slice(0, 5);

  // Calculate outcome summary
  const outcomeSummary = {
    responders: groupedCases.responders.length,
    nonResponders: groupedCases.nonResponders.length,
    unknown: groupedCases.unknown.length,
  };

  // Error state
  if (error && !isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <GitCompare className="h-4 w-4 text-red-500" />
            Similar Cases
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-6">
            <div className="w-12 h-12 mx-auto mb-3 rounded-full bg-red-100 flex items-center justify-center">
              <AlertCircle className="h-6 w-6 text-red-500" />
            </div>
            <p className="text-sm font-medium text-red-700 mb-1">
              Search failed
            </p>
            <p className="text-xs text-red-600 mb-3">{error}</p>
            {onRetry && (
              <Button
                variant="ghost"
                size="sm"
                onClick={onRetry}
                leftIcon={<RefreshCw className="h-3 w-3" />}
              >
                Retry
              </Button>
            )}
          </div>
        </CardContent>
      </Card>
    );
  }

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <GitCompare className="h-4 w-4 text-clinical-600 animate-pulse" />
            Similar Cases
          </CardTitle>
        </CardHeader>
        <CardContent>
          <SkeletonSimilarCases />
          <p className="text-xs text-gray-500 text-center mt-3 animate-pulse">
            Searching reference cohort with FAISS...
          </p>
        </CardContent>
      </Card>
    );
  }

  if (!cases.length) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <GitCompare className="h-4 w-4 text-gray-400" />
            Similar Cases
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-gray-500">
            <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gray-100 flex items-center justify-center">
              <Database className="h-8 w-8 text-gray-400" />
            </div>
            <p className="text-sm font-medium text-gray-600">
              No similar cases found
            </p>
            <p className="text-xs mt-1.5 text-gray-500 max-w-[200px] mx-auto">
              The reference cohort may be empty or unavailable for comparison.
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <GitCompare className="h-4 w-4 text-clinical-600" />
            Similar Cases
            <Badge variant="default" size="sm" className="font-mono">
              {cases.length}
            </Badge>
          </CardTitle>
        </div>
      </CardHeader>
      <CardContent className="space-y-4 pt-0">
        {/* Outcome Summary */}
        {(outcomeSummary.responders > 0 || outcomeSummary.nonResponders > 0) && (
          <div className="p-3 bg-surface-secondary rounded-lg border border-surface-border">
            <div className="flex items-center gap-2 mb-2">
              <BarChart3 className="h-4 w-4 text-gray-500" />
              <span className="text-xs font-medium text-gray-600 uppercase tracking-wide">
                Outcome Distribution
              </span>
            </div>
            <div className="flex items-center gap-2">
              {/* Visual bar */}
              <div className="flex-1 h-3 bg-gray-200 rounded-full overflow-hidden flex">
                {outcomeSummary.responders > 0 && (
                  <div
                    className="h-full bg-status-positive"
                    style={{
                      width: `${(outcomeSummary.responders / cases.length) * 100}%`,
                    }}
                  />
                )}
                {outcomeSummary.nonResponders > 0 && (
                  <div
                    className="h-full bg-status-negative"
                    style={{
                      width: `${(outcomeSummary.nonResponders / cases.length) * 100}%`,
                    }}
                  />
                )}
                {outcomeSummary.unknown > 0 && (
                  <div
                    className="h-full bg-gray-400"
                    style={{
                      width: `${(outcomeSummary.unknown / cases.length) * 100}%`,
                    }}
                  />
                )}
              </div>
            </div>
            <div className="flex items-center gap-4 mt-2 text-xs">
              {outcomeSummary.responders > 0 && (
                <div className="flex items-center gap-1">
                  <div className="w-2 h-2 rounded-full bg-status-positive" />
                  <span className="text-gray-600">
                    {outcomeSummary.responders} {positiveLabel}
                  </span>
                </div>
              )}
              {outcomeSummary.nonResponders > 0 && (
                <div className="flex items-center gap-1">
                  <div className="w-2 h-2 rounded-full bg-status-negative" />
                  <span className="text-gray-600">
                    {outcomeSummary.nonResponders} {negativeLabel}
                  </span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* View Mode Toggle */}
        <div className="flex items-center justify-end gap-2">
          <span className="text-xs text-gray-500">View:</span>
          <div className="flex items-center gap-1 bg-surface-secondary rounded-lg p-0.5">
            <button
              onClick={() => setViewMode("grouped")}
              className={cn(
                "px-2 py-1 rounded text-xs font-medium transition-all",
                viewMode === "grouped"
                  ? "bg-white shadow-clinical text-clinical-700"
                  : "text-gray-500 hover:text-gray-700"
              )}
            >
              By Outcome
            </button>
            <button
              onClick={() => setViewMode("list")}
              className={cn(
                "px-2 py-1 rounded text-xs font-medium transition-all",
                viewMode === "list"
                  ? "bg-white shadow-clinical text-clinical-700"
                  : "text-gray-500 hover:text-gray-700"
              )}
            >
              All
            </button>
          </div>
        </div>

        {/* Grouped View - Cases by Outcome */}
        {viewMode === "grouped" ? (
          <div className="space-y-4">
            {/* Similar Positive Class */}
            {groupedCases.responders.length > 0 && (
              <div className="space-y-2">
                <div className="flex items-center gap-2 pb-1 border-b border-green-200">
                  <CheckCircle className="h-4 w-4 text-green-600" />
                  <span className="text-sm font-semibold text-green-800">
                    Similar {positiveLabel}s (N={groupedCases.responders.length})
                  </span>
                </div>
                <div className="space-y-2 pl-1">
                  {groupedCases.responders.map((similarCase, index) => (
                    <SimilarCaseItem
                      key={`resp-${similarCase.caseId || similarCase.slideId}-${index}`}
                      case_={similarCase}
                      rank={index + 1}
                      isExpanded={expandedCase === (similarCase.caseId || similarCase.slideId)}
                      onToggleExpand={() => {
                        const caseKey = similarCase.caseId || similarCase.slideId;
                        setExpandedCase(expandedCase === caseKey ? null : caseKey);
                      }}
                      onViewCase={() =>
                        onCaseClick?.(similarCase.caseId || similarCase.slideId || "")
                      }
                    />
                  ))}
                </div>
              </div>
            )}

            {/* Similar Negative Class */}
            {groupedCases.nonResponders.length > 0 && (
              <div className="space-y-2">
                <div className="flex items-center gap-2 pb-1 border-b border-red-200">
                  <XCircle className="h-4 w-4 text-red-600" />
                  <span className="text-sm font-semibold text-red-800">
                    Similar {negativeLabel}s (N={groupedCases.nonResponders.length})
                  </span>
                </div>
                <div className="space-y-2 pl-1">
                  {groupedCases.nonResponders.map((similarCase, index) => (
                    <SimilarCaseItem
                      key={`nonresp-${similarCase.caseId || similarCase.slideId}-${index}`}
                      case_={similarCase}
                      rank={index + 1}
                      isExpanded={expandedCase === (similarCase.caseId || similarCase.slideId)}
                      onToggleExpand={() => {
                        const caseKey = similarCase.caseId || similarCase.slideId;
                        setExpandedCase(expandedCase === caseKey ? null : caseKey);
                      }}
                      onViewCase={() =>
                        onCaseClick?.(similarCase.caseId || similarCase.slideId || "")
                      }
                    />
                  ))}
                </div>
              </div>
            )}

            {/* Unknown Outcome */}
            {groupedCases.unknown.length > 0 && (
              <div className="space-y-2">
                <div className="flex items-center gap-2 pb-1 border-b border-gray-200">
                  <HelpCircle className="h-4 w-4 text-gray-500" />
                  <span className="text-sm font-semibold text-gray-600">
                    Unknown Outcome (N={groupedCases.unknown.length})
                  </span>
                </div>
                <div className="space-y-2 pl-1">
                  {groupedCases.unknown.map((similarCase, index) => (
                    <SimilarCaseItem
                      key={`unk-${similarCase.caseId || similarCase.slideId}-${index}`}
                      case_={similarCase}
                      rank={index + 1}
                      isExpanded={expandedCase === (similarCase.caseId || similarCase.slideId)}
                      onToggleExpand={() => {
                        const caseKey = similarCase.caseId || similarCase.slideId;
                        setExpandedCase(expandedCase === caseKey ? null : caseKey);
                      }}
                      onViewCase={() =>
                        onCaseClick?.(similarCase.caseId || similarCase.slideId || "")
                      }
                    />
                  ))}
                </div>
              </div>
            )}

            {/* Outcome Comparison Note */}
            {groupedCases.responders.length > 0 && groupedCases.nonResponders.length > 0 && (
              <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
                <p className="text-xs text-blue-800 leading-relaxed">
                  <strong>Comparison note:</strong> This case shares morphological features with both 
                  {positiveLabel.toLowerCase()} and {negativeLabel.toLowerCase()} cases. Key differentiating features may include tumor 
                  cellularity, stromal patterns, and inflammatory infiltrate distribution.
                </p>
              </div>
            )}
          </div>
        ) : (
          <>
            {/* List View - All Cases */}
            <div className="space-y-2">
              {visibleCases.map((similarCase, index) => (
                <SimilarCaseItem
                  key={`${similarCase.caseId || similarCase.slideId}-${similarCase.patchId || index}`}
                  case_={similarCase}
                  rank={index + 1}
                  isExpanded={expandedCase === (similarCase.caseId || similarCase.slideId)}
                  onToggleExpand={() => {
                    const caseKey = similarCase.caseId || similarCase.slideId;
                    setExpandedCase(expandedCase === caseKey ? null : caseKey);
                  }}
                  onViewCase={() =>
                    onCaseClick?.(similarCase.caseId || similarCase.slideId || "")
                  }
                />
              ))}
            </div>

            {/* Show More/Less */}
            {cases.length > 5 && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowAll(!showAll)}
                className="w-full text-gray-600 hover:text-gray-900"
              >
                {showAll ? (
                  <>
                    <ChevronUp className="h-4 w-4 mr-1" />
                    Show Less
                  </>
                ) : (
                  <>
                    <ChevronDown className="h-4 w-4 mr-1" />
                    Show {cases.length - 5} More Cases
                  </>
                )}
              </Button>
            )}
          </>
        )}

        {/* Info */}
        <div className="pt-3 border-t border-gray-100">
          <p className="text-xs text-gray-500 leading-relaxed">
            Similar cases retrieved using FAISS vector similarity on patch
            embeddings. Lower distance indicates higher morphological similarity.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}

// Similar Case Item Component
interface SimilarCaseItemProps {
  case_: SimilarCase;
  rank: number;
  isExpanded: boolean;
  onToggleExpand: () => void;
  onViewCase: () => void;
}

function SimilarCaseItem({
  case_,
  rank,
  isExpanded,
  onToggleExpand,
  onViewCase,
}: SimilarCaseItemProps) {
  const [thumbnailError, setThumbnailError] = useState(false);
  const [thumbnailLoading, setThumbnailLoading] = useState(true);
  
  const distance = case_.distance ?? 0;
  // Use pre-computed similarity from backend
  // The /api/similar endpoint returns cosine similarity (0-1, typically 0.85-0.98)
  // The old /api/analyze endpoint returned 1/(1+L2_dist) which gives tiny values
  const rawSimilarity = case_.similarity ?? (distance > 0 ? 1.0 / (1.0 + distance) : 0);
  // Auto-detect scale: cosine similarity is typically > 0.5, L2-based is << 0.1
  const similarityScore = rawSimilarity > 0.1
    ? Math.round(rawSimilarity * 100)  // Cosine similarity — already 0-1 scale
    : Math.max(0, Math.min(100, Math.round(60 + 35 * Math.min(1, rawSimilarity / 0.003))));  // L2-based legacy

  const isResponder =
    case_.label?.toLowerCase().includes("positive") ||
    case_.label?.toLowerCase().includes("responder");
  const isNonResponder =
    case_.label?.toLowerCase().includes("negative") ||
    case_.label?.toLowerCase().includes("non");

  return (
    <div
      className={cn(
        "border rounded-lg transition-all overflow-hidden",
        isExpanded ? "border-clinical-300 bg-clinical-50/50" : "border-gray-200 bg-white"
      )}
    >
      {/* Main Row */}
      <button
        onClick={onToggleExpand}
        className="w-full flex items-center gap-3 p-3 text-left hover:bg-gray-50/50 transition-colors"
      >
        {/* Thumbnail */}
        <div className="relative w-12 h-12 rounded-lg overflow-hidden shrink-0 border border-gray-200">
          {case_.thumbnailUrl && !thumbnailError ? (
            <>
              {thumbnailLoading && (
                <div className="absolute inset-0 bg-gray-100 flex items-center justify-center animate-pulse">
                  <Database className="h-5 w-5 text-gray-300" />
                </div>
              )}
              <img
                src={case_.thumbnailUrl}
                alt={`Similar case ${rank}`}
                className={cn(
                  "w-full h-full object-cover transition-opacity",
                  thumbnailLoading ? "opacity-0" : "opacity-100"
                )}
                onLoad={() => setThumbnailLoading(false)}
                onError={() => {
                  setThumbnailError(true);
                  setThumbnailLoading(false);
                }}
              />
            </>
          ) : (
            <div className={cn(
              "w-full h-full flex items-center justify-center",
              isResponder ? "bg-green-50" : isNonResponder ? "bg-red-50" : "bg-gray-100"
            )}>
              <Database className={cn(
                "h-5 w-5",
                isResponder ? "text-green-400" : isNonResponder ? "text-red-400" : "text-gray-400"
              )} />
            </div>
          )}
          <div className="absolute top-0.5 left-0.5 bg-navy-900/80 text-white text-2xs font-bold px-1 py-0.5 rounded">
            #{rank}
          </div>
        </div>

        {/* Info */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between mb-1">
            <span className="text-sm font-medium text-gray-900 truncate">
              Case {(case_.caseId || case_.slideId || "unknown").slice(0, 12)}
            </span>
            {case_.label && (
              <Badge
                variant={
                  isResponder ? "success" : isNonResponder ? "danger" : "default"
                }
                size="sm"
              >
                {case_.label}
              </Badge>
            )}
          </div>
          <div className="flex items-center gap-3">
            {/* Distance score */}
            <div className="flex items-center gap-1.5">
              <span className="text-xs text-gray-500">Similarity:</span>
              <div className="flex items-center gap-1">
                <div className="w-16 h-1.5 bg-gray-200 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-clinical-500 rounded-full transition-all"
                    style={{ width: `${similarityScore}%` }}
                  />
                </div>
                <span className="text-xs font-mono text-gray-600">
                  {similarityScore}%
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Expand Icon */}
        <div className="shrink-0">
          {isExpanded ? (
            <ChevronUp className="h-4 w-4 text-gray-400" />
          ) : (
            <ChevronDown className="h-4 w-4 text-gray-400" />
          )}
        </div>
      </button>

      {/* Expanded Details */}
      {isExpanded && (
        <div className="px-3 pb-3 pt-0 border-t border-gray-100 animate-fade-in">
          <div className="grid grid-cols-2 gap-3 text-xs mt-3">
            <div className="space-y-1">
              <span className="text-gray-500 font-medium">Slide ID</span>
              <p className="font-mono text-gray-700 truncate">
                {(case_.slideId || "unknown").slice(0, 20)}
              </p>
            </div>
            <div className="space-y-1">
              <span className="text-gray-500 font-medium">Distance</span>
              <p className="font-mono text-gray-700">
                {formatDistance(case_.distance)}
              </p>
            </div>
            {case_.patchId && (
              <div className="space-y-1">
                <span className="text-gray-500 font-medium">Patch ID</span>
                <p className="font-mono text-gray-700">
                  {case_.patchId.slice(0, 12)}
                </p>
              </div>
            )}
            {case_.coordinates && (
              <div className="space-y-1">
                <span className="text-gray-500 font-medium">Coordinates</span>
                <p className="font-mono text-gray-700 flex items-center gap-1">
                  <MapPin className="h-3 w-3" />
                  ({case_.coordinates.x}, {case_.coordinates.y})
                </p>
              </div>
            )}
          </div>
          <Button
            variant="primary"
            size="sm"
            onClick={(e) => {
              e.stopPropagation();
              onViewCase();
            }}
            className="w-full mt-3"
          >
            <ArrowRight className="h-3.5 w-3.5 mr-1.5" />
            Switch to This Case
          </Button>
        </div>
      )}
    </div>
  );
}
