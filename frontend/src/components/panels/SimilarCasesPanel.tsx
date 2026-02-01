"use client";

import React, { useState } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { cn, formatDistance } from "@/lib/utils";
import {
  Search,
  ArrowRight,
  ChevronDown,
  ChevronUp,
  Database,
  ExternalLink,
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

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Search className="h-4 w-4" />
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
                <div className="w-12 h-12 bg-gray-200 rounded" />
                <div className="flex-1 space-y-2">
                  <div className="h-4 bg-gray-200 rounded w-3/4" />
                  <div className="h-3 bg-gray-200 rounded w-1/2" />
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!cases.length) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Search className="h-4 w-4" />
            Similar Cases
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-gray-500">
            <Database className="h-8 w-8 mx-auto mb-2 text-gray-400" />
            <p className="text-sm">No similar cases found.</p>
            <p className="text-xs mt-1">
              The reference cohort may be empty or unavailable.
            </p>
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
            <Search className="h-4 w-4" />
            Similar Cases
            <Badge variant="default" size="sm">
              {cases.length} matches
            </Badge>
          </CardTitle>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {/* Cases List */}
        <div className="space-y-2">
          {visibleCases.map((similarCase, index) => (
            <SimilarCaseItem
              key={`${similarCase.caseId}-${similarCase.patchId}`}
              case_={similarCase}
              rank={index + 1}
              isExpanded={expandedCase === similarCase.caseId}
              onToggleExpand={() =>
                setExpandedCase(
                  expandedCase === similarCase.caseId
                    ? null
                    : similarCase.caseId
                )
              }
              onViewCase={() => onCaseClick?.(similarCase.caseId)}
            />
          ))}
        </div>

        {/* Show More/Less */}
        {cases.length > 5 && (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowAll(!showAll)}
            className="w-full"
          >
            {showAll ? (
              <>
                <ChevronUp className="h-4 w-4 mr-1" />
                Show Less
              </>
            ) : (
              <>
                <ChevronDown className="h-4 w-4 mr-1" />
                Show {cases.length - 5} More
              </>
            )}
          </Button>
        )}

        {/* Info */}
        <div className="pt-2 border-t border-gray-100">
          <p className="text-xs text-gray-500">
            Similar cases are retrieved using FAISS similarity search on patch
            embeddings. Lower distance scores indicate higher similarity.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}

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
  // Calculate similarity score (inverse of distance, normalized)
  const similarityScore = Math.max(0, Math.min(100, 100 - case_.distance * 100));

  return (
    <div
      className={cn(
        "border rounded-lg transition-all overflow-hidden",
        isExpanded ? "border-clinical-300 bg-clinical-50" : "border-gray-200"
      )}
    >
      {/* Main Row */}
      <button
        onClick={onToggleExpand}
        className="w-full flex items-center gap-3 p-3 text-left hover:bg-gray-50"
      >
        {/* Thumbnail */}
        <div className="relative w-12 h-12 rounded overflow-hidden shrink-0 border border-gray-200">
          <img
            src={case_.thumbnailUrl}
            alt={`Similar case ${rank}`}
            className="w-full h-full object-cover"
          />
        </div>

        {/* Info */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-gray-900 truncate">
              Case {case_.caseId.slice(0, 12)}
            </span>
            {case_.label && (
              <Badge
                variant={
                  case_.label.toLowerCase().includes("positive") ||
                  case_.label.toLowerCase().includes("responder")
                    ? "success"
                    : case_.label.toLowerCase().includes("negative") ||
                      case_.label.toLowerCase().includes("non")
                    ? "danger"
                    : "default"
                }
                size="sm"
              >
                {case_.label}
              </Badge>
            )}
          </div>
          <div className="flex items-center gap-3 mt-1">
            <span className="text-xs text-gray-500">
              Distance: {formatDistance(case_.distance)}
            </span>
            <div className="flex-1 h-1.5 bg-gray-200 rounded-full max-w-[80px]">
              <div
                className="h-full bg-clinical-500 rounded-full"
                style={{ width: `${similarityScore}%` }}
              />
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
        <div className="px-3 pb-3 pt-0 border-t border-gray-100">
          <div className="grid grid-cols-2 gap-2 text-xs mt-2">
            <div>
              <span className="text-gray-500">Slide ID:</span>
              <span className="ml-1 font-mono text-gray-700">
                {case_.slideId.slice(0, 16)}
              </span>
            </div>
            <div>
              <span className="text-gray-500">Patch:</span>
              <span className="ml-1 font-mono text-gray-700">
                {case_.patchId.slice(0, 8)}
              </span>
            </div>
            <div className="col-span-2">
              <span className="text-gray-500">Coordinates:</span>
              <span className="ml-1 font-mono text-gray-700">
                ({case_.coordinates.x}, {case_.coordinates.y})
              </span>
            </div>
          </div>
          <Button
            variant="secondary"
            size="sm"
            onClick={onViewCase}
            className="w-full mt-3"
          >
            <ExternalLink className="h-3.5 w-3.5 mr-1.5" />
            View Case
          </Button>
        </div>
      )}
    </div>
  );
}
