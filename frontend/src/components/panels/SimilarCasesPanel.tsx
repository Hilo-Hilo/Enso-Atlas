"use client";

import React, { useState } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
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
} from "lucide-react";
import type { SimilarCase } from "@/types";

interface SimilarCasesPanelProps {
  cases: SimilarCase[];
  isLoading?: boolean;
  onCaseClick?: (caseId: string) => void;
}

export function SimilarCasesPanel({
  cases,
  isLoading,
  onCaseClick,
}: SimilarCasesPanelProps) {
  const [expandedCase, setExpandedCase] = useState<string | null>(null);
  const [showAll, setShowAll] = useState(false);

  const visibleCases = showAll ? cases : cases.slice(0, 5);

  // Calculate outcome summary
  const outcomeSummary = cases.reduce(
    (acc, c) => {
      if (c.label) {
        const isResponder =
          c.label.toLowerCase().includes("positive") ||
          c.label.toLowerCase().includes("responder");
        const isNonResponder =
          c.label.toLowerCase().includes("negative") ||
          c.label.toLowerCase().includes("non");
        if (isResponder) acc.responders++;
        else if (isNonResponder) acc.nonResponders++;
        else acc.unknown++;
      } else {
        acc.unknown++;
      }
      return acc;
    },
    { responders: 0, nonResponders: 0, unknown: 0 }
  );

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <GitCompare className="h-4 w-4 text-clinical-600" />
            Similar Cases
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {[...Array(3)].map((_, i) => (
              <div
                key={i}
                className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg animate-pulse"
              >
                <div className="w-12 h-12 bg-gray-200 rounded-lg" />
                <div className="flex-1 space-y-2">
                  <div className="h-4 bg-gray-200 rounded w-3/4" />
                  <div className="h-3 bg-gray-200 rounded w-1/2" />
                </div>
              </div>
            ))}
          </div>
          <p className="text-xs text-gray-500 text-center mt-3">
            Searching reference cohort...
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
                    {outcomeSummary.responders} Responder
                    {outcomeSummary.responders !== 1 ? "s" : ""}
                  </span>
                </div>
              )}
              {outcomeSummary.nonResponders > 0 && (
                <div className="flex items-center gap-1">
                  <div className="w-2 h-2 rounded-full bg-status-negative" />
                  <span className="text-gray-600">
                    {outcomeSummary.nonResponders} Non-Responder
                    {outcomeSummary.nonResponders !== 1 ? "s" : ""}
                  </span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Cases List */}
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
  const distance = case_.distance ?? 0;
  const similarityScore = Math.max(0, Math.min(100, Math.round((1 - distance) * 100)));

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
          {case_.thumbnailUrl ? (
            <img
              src={case_.thumbnailUrl}
              alt={`Similar case ${rank}`}
              className="w-full h-full object-cover"
            />
          ) : (
            <div className="w-full h-full bg-gray-100 flex items-center justify-center">
              <Database className="h-5 w-5 text-gray-400" />
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
            variant="secondary"
            size="sm"
            onClick={(e) => {
              e.stopPropagation();
              onViewCase();
            }}
            className="w-full mt-3"
          >
            <ExternalLink className="h-3.5 w-3.5 mr-1.5" />
            View Case Details
          </Button>
        </div>
      )}
    </div>
  );
}
